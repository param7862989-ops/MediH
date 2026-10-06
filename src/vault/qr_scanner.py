"""Multi-Input QR Code Scanner, Optical Decoder, and Cryptographic Verifier for MediHaven.

Decodes visual QR codes from camera feeds, image files, Base64 data URIs, or raw token strings.
Executes multi-layered security validation (HMAC signature check, expiration, revocation, and use quota).
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import cv2
import jwt
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import sqlite3

from src.database.db import get_connection, get_db_connection
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.vault.scanner")


def extract_qr_text_from_image(image_input: Union[bytes, str, np.ndarray, Path, Image.Image]) -> Optional[str]:
    """Decodes QR code text payload from an image in various input formats.

    Args:
        image_input: File path, raw bytes, Base64 Data URI, PIL Image, or OpenCV numpy array.

    Returns:
        Decoded text string from the QR code, or None if no QR code was detected.
    """
    try:
        # 1. Base64 Data URI string
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            header, b64_part = image_input.split(",", 1)
            raw_bytes = base64.b64decode(b64_part)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        # 2. File path string or Path object
        elif isinstance(image_input, (str, Path)) and Path(str(image_input)).exists():
            pil_img = Image.open(str(image_input))
        # 3. Raw image bytes
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
        # 4. PIL Image
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        # 5. OpenCV numpy array frame
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            return None

        # Try pyzbar first for robust multi-angle decoding
        decoded_objs = pyzbar_decode(pil_img)
        if decoded_objs:
            return decoded_objs[0].data.decode("utf-8")

        # Fallback to OpenCV QRCodeDetector
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        text, points, _ = detector.detectAndDecode(cv_img)
        if text:
            return text

    except Exception as e:
        logger.warning(f"Error while extracting QR code from image: {e}")

    return None


def validate_vault_token(
    token_str: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Validates a signed JWT token against cryptographic keys and database revocation state.

    Args:
        token_str: Signed JWT string.
        conn: Optional active database connection.

    Returns:
        Dictionary with validation results:
        {
            "is_valid": bool,
            "access_status": "GRANTED" | "INVALID_SIGNATURE" | "EXPIRED" | "REVOKED" | "MAX_USES_EXCEEDED" | "NOT_FOUND",
            "error_message": Optional[str],
            "claims": Optional[Dict[str, Any]],
            "token_record": Optional[Dict[str, Any]],
        }
    """
    cleaned_token = token_str.strip()

    # Step 1: Cryptographic JWT Signature & Expiration Verification
    try:
        claims = jwt.decode(
            jwt=cleaned_token,
            key=Config.JWT_SECRET_KEY,
            algorithms=[Config.JWT_ALGORITHM],
            options={"require": ["jti", "sub", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        return {
            "is_valid": False,
            "access_status": "EXPIRED",
            "error_message": "QR access token has expired. Please request a new pass from the patient.",
            "claims": None,
            "token_record": None,
        }
    except (jwt.InvalidSignatureError, jwt.DecodeError):
        return {
            "is_valid": False,
            "access_status": "INVALID_SIGNATURE",
            "error_message": "Cryptographic signature verification failed. Token is tampered or forged.",
            "claims": None,
            "token_record": None,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "access_status": "INVALID_PAYLOAD",
            "error_message": f"Malformed token payload: {e}",
            "claims": None,
            "token_record": None,
        }

    token_hash = claims.get("jti")
    patient_id = int(claims.get("sub")) if claims.get("sub") else None

    # Step 2: Stateful Database Checks (Revocation, Expiry, Single-Use Quota)
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        query = """
            SELECT token_id, patient_id, token_hash, scope_json, issued_at, 
                   expires_at, max_uses, use_count, revoked, qr_payload
            FROM vault_access_tokens
            WHERE token_hash = ?
        """
        cursor.execute(query, (token_hash,))
        row = cursor.fetchone()

        if not row:
            return {
                "is_valid": False,
                "access_status": "NOT_FOUND",
                "error_message": "Token record not found in hospital registry.",
                "claims": claims,
                "token_record": None,
            }

        # Check revocation
        if row["revoked"] == 1:
            return {
                "is_valid": False,
                "access_status": "REVOKED",
                "error_message": "Access pass was revoked by the patient.",
                "claims": claims,
                "token_record": dict(row),
            }

        # Check database expiration timestamp
        expires_at_dt = datetime.fromisoformat(row["expires_at"])
        if datetime.now() > expires_at_dt:
            return {
                "is_valid": False,
                "access_status": "EXPIRED",
                "error_message": "Token expiration time has passed.",
                "claims": claims,
                "token_record": dict(row),
            }

        # Check maximum use quota
        max_uses = row["max_uses"]
        use_count = row["use_count"]
        if max_uses is not None and use_count >= max_uses:
            return {
                "is_valid": False,
                "access_status": "MAX_USES_EXCEEDED",
                "error_message": f"Pass has reached its maximum scan limit ({max_uses} use(s)).",
                "claims": claims,
                "token_record": dict(row),
            }

        # Success: All checks passed
        return {
            "is_valid": True,
            "access_status": "GRANTED",
            "error_message": None,
            "claims": claims,
            "token_record": dict(row),
        }
    finally:
        if should_close and conn:
            conn.close()


def scan_and_validate_qr(
    raw_input: Union[str, bytes, np.ndarray, Path, Image.Image],
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """High-level function: parses image or string input, extracts QR, and validates token.

    Args:
        raw_input: Direct JWT token string OR visual image containing QR code.
        conn: Optional active database connection.

    Returns:
        Validation response dictionary.
    """
    token_str: Optional[str] = None

    # Check if raw_input is already a clean JWT string (3 dot-separated base64 segments)
    if isinstance(raw_input, str) and raw_input.count(".") == 2 and not raw_input.startswith("data:"):
        token_str = raw_input.strip()
    else:
        # Otherwise decode visual QR code from image
        token_str = extract_qr_text_from_image(raw_input)

    if not token_str:
        return {
            "is_valid": False,
            "access_status": "NO_QR_DETECTED",
            "error_message": "No valid QR code or token detected in input.",
            "claims": None,
            "token_record": None,
        }

    return validate_vault_token(token_str, conn=conn)

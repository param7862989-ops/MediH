"""Patient Medical Vault and cryptographic QR code sharing subsystem."""

from src.vault.vault_service import (
    VALID_VAULT_CATEGORIES,
    add_vault_record,
    get_patient_vault_records,
    get_vault_record_by_id,
    delete_vault_record,
    get_vault_category_summary,
)
from src.vault.qr_generator import (
    generate_vault_access_token,
    revoke_vault_access_token,
    get_patient_active_tokens,
    render_qr_code_base64,
)
from src.vault.qr_scanner import (
    extract_qr_text_from_image,
    validate_vault_token,
    scan_and_validate_qr,
)
from src.vault.access_logger import (
    log_vault_access,
    get_patient_access_audit_log,
)
from src.vault.access_controller import (
    access_vault_with_token,
)

__all__ = [
    "VALID_VAULT_CATEGORIES",
    "add_vault_record",
    "get_patient_vault_records",
    "get_vault_record_by_id",
    "delete_vault_record",
    "get_vault_category_summary",
    "generate_vault_access_token",
    "revoke_vault_access_token",
    "get_patient_active_tokens",
    "render_qr_code_base64",
    "extract_qr_text_from_image",
    "validate_vault_token",
    "scan_and_validate_qr",
    "log_vault_access",
    "get_patient_access_audit_log",
    "access_vault_with_token",
]

"""Measures MediHaven's production server (wsgi.py under Waitress, 8 threads) on a temporary copy of the
synthetic demo database. Throwaway secrets; the persistent database is only read (backup API).

Workload: for each endpoint and each client concurrency level, N requests spread over C client threads
sharing one authenticated session. Records per-request latency (wall clock), throughput, error count,
and the server process working set (memory) via PowerShell Get-Process.
Writes results to perf_results.json next to this script.
"""
import http.cookiejar
import json
import os
import secrets
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import closing
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\OneDrive\Desktop\MediH")
OUT = Path(__file__).with_name("perf_results.json")
tmp = Path(tempfile.mkdtemp(prefix="medihaven_perf_"))
with closing(sqlite3.connect((ROOT / "database" / "medihaven.db").resolve().as_uri() + "?mode=ro", uri=True)) as src, \
        closing(sqlite3.connect(tmp / "medihaven.db")) as dst:
    src.backup(dst)
shutil.copytree(ROOT / "models_store", tmp / "models_store")
shutil.copytree(ROOT / "data" / "processed", tmp / "processed")
env = dict(os.environ)
env.update({"DATA_MODE": "synthetic", "FLASK_ENV": "production", "SESSION_COOKIE_SECURE": "0", "DEMO_LOGINS": "1",
            "FLASK_SECRET_KEY": secrets.token_hex(32), "JWT_SECRET_KEY": secrets.token_hex(32),
            "DATABASE_PATH": str(tmp / "medihaven.db"), "MODELS_STORE_DIR": str(tmp / "models_store"),
            "PROCESSED_DATA_DIR": str(tmp / "processed"), "LOG_FILE": str(tmp / "app.log"),
            "AUDIT_LOG_FILE": str(tmp / "audit.log"), "GEMINI_API_KEY": "", "LOG_LEVEL": "WARNING", "FLASK_DEBUG": ""})
PORT = 8790
proc = subprocess.Popen([str(ROOT / ".venv/Scripts/waitress-serve.exe"), f"--listen=127.0.0.1:{PORT}", "--threads=8",
                         "wsgi:app"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{PORT}"


def working_set_mb(pid):
    """Memory and CPU of the launcher plus its child processes (waitress-serve.exe starts python.exe)."""
    # Whole process tree: waitress-serve.exe -> venv python.exe redirector -> real python.exe
    script = (f"$all = Get-CimInstance Win32_Process; $ids = @({pid}); $frontier = @({pid}); "
              "while ($frontier.Count -gt 0) { $next = @($all | Where-Object { $frontier -contains $_.ParentProcessId } "
              "| ForEach-Object { $_.ProcessId }); $ids += $next; $frontier = $next }; $p = Get-Process -Id $ids; "
              "($p | Measure-Object WorkingSet64 -Sum).Sum; ($p | Measure-Object CPU -Sum).Sum")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout.split()
    return round(int(out[0]) / 1024 / 1024, 1), float(out[1])


try:
    for _ in range(300):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    req = urllib.request.Request(base + "/api/auth/login", method="POST",
                                 data=json.dumps({"username": "physician1", "password": "Physician@123"}).encode(),
                                 headers={"Content-Type": "application/json"})
    assert opener.open(req).status == 200
    cookie = "; ".join(f"{c.name}={c.value}" for c in jar)

    idle_mb, _ = working_set_mb(proc.pid)
    endpoints = {
        "health": "/api/health",
        "patient_list": "/api/patients",
        "predict": "/api/patients/{pid}/predict",
        "vitals_72h": "/api/patients/{pid}/vitals?hours=72",
    }
    results = {"environment": {"server": "Waitress 3.0.2, 8 worker threads, wsgi.py (FLASK_ENV=production)",
                               "client": "Python urllib, same machine (loopback)", "python": sys.version.split()[0],
                               "dataset": "synthetic demo copy: 100 patients"},
               "idle_working_set_mb": idle_mb, "runs": []}
    for name, path in endpoints.items():
        for clients in (1, 4, 8, 16):
            total = 200
            latencies, errors, lock = [], [0], threading.Lock()

            def worker(count, offset):
                for i in range(count):
                    url = base + path.format(pid=1 + (offset + i) % 100)
                    r = urllib.request.Request(url, headers={"Cookie": cookie})
                    t0 = time.perf_counter()
                    try:
                        with urllib.request.urlopen(r, timeout=60) as resp:
                            resp.read()
                            ok = resp.status == 200
                    except Exception:
                        ok = False
                    dt = (time.perf_counter() - t0) * 1000
                    with lock:
                        latencies.append(dt)
                        if not ok:
                            errors[0] += 1

            per = total // clients
            threads = [threading.Thread(target=worker, args=(per, k * per)) for k in range(clients)]
            _, cpu_before = working_set_mb(proc.pid)
            start = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.perf_counter() - start
            mem_mb, cpu_after = working_set_mb(proc.pid)
            lat = sorted(latencies)
            run = {"endpoint": name, "clients": clients, "requests": len(lat), "errors": errors[0],
                   "throughput_rps": round(len(lat) / elapsed, 1),
                   "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": round(lat[len(lat) // 2], 1),
                                  "p95": round(lat[int(len(lat) * 0.95) - 1], 1), "max": round(lat[-1], 1)},
                   "server_cpu_seconds": round(cpu_after - cpu_before, 2), "working_set_mb_after": mem_mb}
            results["runs"].append(run)
            print(f"{name:13s} c={clients:2d} rps={run['throughput_rps']:7.1f} p50={run['latency_ms']['p50']:7.1f}ms "
                  f"p95={run['latency_ms']['p95']:7.1f}ms err={errors[0]} mem={mem_mb}MB")
    results["database_size_mb"] = round((tmp / "medihaven.db").stat().st_size / 1024 / 1024, 2)
    results["models_store_mb"] = round(sum(f.stat().st_size for f in (tmp / "models_store").iterdir()) / 1024 / 1024, 2)
    OUT.write_text(json.dumps(results, indent=2))
    print("idle MB", idle_mb, "| db MB", results["database_size_mb"], "| models MB", results["models_store_mb"])
finally:
    proc.terminate()
    try:
        proc.wait(10)
    except Exception:
        proc.kill()
    shutil.rmtree(tmp, ignore_errors=True)

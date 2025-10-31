"""Environment diagnostic helper.

Prints Python, platform, and attempts to import numpy and scipy with timings.
Use this to help debug import hangs seen during training runs.
"""
import sys
import time
import platform
import subprocess


def try_import(module_name: str):
    t0 = time.time()
    try:
        mod = __import__(module_name)
        ok = True
        err = None
    except Exception as e:
        mod = None
        ok = False
        err = e
    t1 = time.time()
    return ok, mod, err, t1 - t0


def main():
    print("Python:", sys.version.replace('\n', ' '))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    try:
        # On Windows this may call system utilities; capture safely
        ver = subprocess.check_output(["cmd", "/c", "ver"], stderr=subprocess.STDOUT, encoding="utf-8")
        print("cmd /c ver ->", ver.strip())
    except Exception as e:
        print("Could not run 'cmd /c ver':", e)

    for m in ("numpy", "scipy", "sklearn", "pandas"):
        ok, mod, err, dt = try_import(m)
        if ok:
            try:
                ver = getattr(mod, "__version__", "<unknown>")
            except Exception:
                ver = "<unknown>"
            print(f"Imported {m} (v{ver}) in {dt:.3f}s")
        else:
            print(f"Failed to import {m} after {dt:.3f}s: {err!r}")

    print("Done diagnostics.")

if __name__ == '__main__':
    main()

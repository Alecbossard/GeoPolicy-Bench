"""Terminate only a subprocess we launched, including Windows venv launchers."""

import psutil


def stop_owned_job(proc):
    if proc.poll() is not None:
        return
    try:
        parent = psutil.Process(proc.pid)
        targets = list(reversed(parent.children(recursive=True))) + [parent]
    except psutil.NoSuchProcess:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    proc.wait(timeout=10)

"""Linux-only isolated CPU test helper; never used by the live controller."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .allocation import ProcessSupervisor
from .process_state import owned_group_state


def main():
    if sys.platform!="linux":
        raise SystemExit("Linux-only owned CPU stand-in")
    # Subreaper adoption is confined to this disposable helper process; it
    # grants no privilege and changes no host security or scheduler setting.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),"stand-in subreaper setup failed")
    with tempfile.TemporaryDirectory() as path:
        ready=Path(path)/"ready.json"
        code="""import os,signal,time,sys,json
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
kid=os.fork()
if kid==0:
    time.sleep(30)
else:
    Path(sys.argv[1]).write_text(json.dumps({'child':kid}))
    time.sleep(30)
"""
        leader=subprocess.Popen([sys.executable,"-u","-c",code,str(ready)],stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        child=None
        try:
            deadline=time.monotonic()+3
            while not ready.exists() and time.monotonic()<deadline:
                time.sleep(.01)
            if not ready.exists():
                raise TimeoutError("Owned stand-in readiness")
            child=json.loads(ready.read_bytes())["child"]
            engine=ProcessSupervisor(term_grace_seconds=.1,kill_grace_seconds=.5)
            engine.stop(leader,time.time()+2)
            result=owned_group_state(leader.pid)
            if result["state"]!="zombie_only" or leader.poll() is None:
                raise AssertionError(result)
            print(json.dumps(dict(state=result["state"],leader_reaped=True,actual_model_calls=0)))
        finally:
            if leader.poll() is None:
                try:
                    os.killpg(leader.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait(timeout=1)
            if child is not None:
                try:
                    # Known fork child in this helper's owned group only.
                    os.kill(child,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(child,0)
                except ChildProcessError:
                    pass


if __name__=="__main__":
    main()

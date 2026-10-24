"""Read-only Linux membership/state inspection for one owned process group."""
from __future__ import annotations

import os
from pathlib import Path


def owned_group_state(pgid,proc_root=Path("/proc"),getpgid=None):
    """Distinguish absent, live, zombie-only and unknown; never signals/reaps.

    Only stat data for members of the requested group is read. Unreadable or
    inconsistent membership/state is unknown, never accepted as dead. The group
    must also retain the owned session ID created by start_new_session=True.
    """
    getpgid=getpgid or getattr(os,"getpgid",None)
    live,zombies,unknown=[],[],[]
    if getpgid is None:
        return dict(state="unknown",live_pids=[],zombie_pids=[],unknown_pids=["unsupported_platform"])
    try:
        candidates=list(Path(proc_root).iterdir())
    except OSError:
        return dict(state="unknown",live_pids=[],zombie_pids=[],unknown_pids=["proc_unreadable"])
    for directory in candidates:
        if not directory.name.isdigit():
            continue
        pid=int(directory.name)
        try:
            if getpgid(pid)!=pgid:
                continue
        except ProcessLookupError:
            continue
        except OSError:
            unknown.append(pid)
            continue
        try:
            raw=(directory/"stat").read_text(encoding="utf-8")
            close=raw.rfind(")")
            if close<0 or int(raw.split("(",1)[0].strip())!=pid:
                raise ValueError("Malformed stat identity")
            fields=raw[close+1:].split()
            state=fields[0]
            group,session=int(fields[2]),int(fields[3])
            if group!=pgid or session!=pgid:
                raise ValueError("Owned group/session identity changed")
            if state in ("Z","X","x"):
                zombies.append(pid)
            elif state in ("R","S","D","T","t","W","P","I","K"):
                live.append(pid)
            else:
                unknown.append(pid)
        except (FileNotFoundError,ProcessLookupError):
            continue
        except (OSError,ValueError,IndexError,UnicodeError):
            unknown.append(pid)
    state="unknown" if unknown else "live" if live else "zombie_only" if zombies else "absent"
    return dict(state=state,live_pids=sorted(live),zombie_pids=sorted(zombies),unknown_pids=sorted(unknown,key=str))

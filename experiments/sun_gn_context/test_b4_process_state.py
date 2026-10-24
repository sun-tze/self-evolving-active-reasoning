"""Portable proc-state fixtures and a Linux-only owned zombie stand-in."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import allocation
from .process_state import owned_group_state


class B4StateTests(unittest.TestCase):
    def proc(self,root,pid,state,group=4321,session=4321):
        directory=Path(root)/str(pid)
        directory.mkdir()
        (directory/"stat").write_text(f"{pid} (test name with ) parens) {state} 1 {group} {session} 0 0\n")

    def test_zombie_only_live_and_mixed_groups_are_distinct(self):
        with tempfile.TemporaryDirectory() as path:
            self.proc(path,5001,"Z")
            self.assertEqual(owned_group_state(4321,Path(path),lambda pid:4321)["state"],"zombie_only")
            self.proc(path,5002,"S")
            observed=owned_group_state(4321,Path(path),lambda pid:4321)
            self.assertEqual(observed["state"],"live")
            self.assertEqual(observed["live_pids"],[5002])
            self.assertEqual(observed["zombie_pids"],[5001])

    def test_unknown_or_changed_identity_never_counts_as_dead(self):
        with tempfile.TemporaryDirectory() as path:
            self.proc(path,5001,"Z",session=9999)
            self.assertEqual(owned_group_state(4321,Path(path),lambda pid:4321)["state"],"unknown")
            (Path(path)/"5001/stat").write_text("malformed")
            self.assertEqual(owned_group_state(4321,Path(path),lambda pid:4321)["state"],"unknown")
            def denied(pid):
                raise PermissionError()
            self.assertEqual(owned_group_state(4321,Path(path),denied)["state"],"unknown")

    def test_unrelated_group_stat_is_not_read(self):
        with tempfile.TemporaryDirectory() as path:
            (Path(path)/"9999").mkdir() # no readable stat is required for unrelated group.
            self.proc(path,5001,"Z")
            result=owned_group_state(4321,Path(path),lambda pid:9999 if pid==9999 else 4321)
            self.assertEqual(result["state"],"zombie_only")
            self.assertEqual(result["unknown_pids"],[])

    def test_stop_accepts_stable_zombies_but_rejects_live_and_unknown(self):
        class DeadLeader:
            pid=4321
            def wait(self,timeout):
                return -9
            def poll(self):
                return -9
        for state in ("zombie_only","live","unknown"):
            with self.subTest(state=state):
                now=[1000.]
                def sleep(seconds):
                    now[0]+=seconds
                observed=dict(state=state,live_pids=[5001] if state=="live" else [],
                              zombie_pids=[5001] if state=="zombie_only" else [],unknown_pids=[5001] if state=="unknown" else [])
                signals=[]
                with patch.object(allocation.os,"killpg",side_effect=lambda pid,sig:signals.append((pid,sig)),create=True), \
                     patch.object(allocation.signal,"SIGKILL",9,create=True),patch.object(allocation,"owned_group_state",return_value=observed):
                    engine=allocation.ProcessSupervisor(clock=lambda:now[0],sleep=sleep,kill_grace_seconds=.2)
                    if state=="zombie_only":
                        engine.stop(DeadLeader(),1001)
                        self.assertEqual(engine.cleanup_observations[-1]["state"],"zombie_only")
                    else:
                        with self.assertRaises(TimeoutError):
                            engine.stop(DeadLeader(),1001)
                    self.assertTrue(all(pid==4321 for pid,sig in signals))


@unittest.skipUnless(sys.platform=="linux" and Path("/proc").is_dir(),
                     "Linux proc/subreaper stand-in unavailable on Windows; no workaround attempted")
class B4LinuxZombieTests(unittest.TestCase):
    def test_owned_unreaped_zombie_grandchild_is_dead_not_live(self):
        result=subprocess.run([sys.executable,"-m","experiments.sun_gn_context.b4_zombie_standin"],
                              capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)
        record=json.loads(result.stdout)
        self.assertEqual(record["state"],"zombie_only")
        self.assertTrue(record["leader_reaped"])
        self.assertEqual(record["actual_model_calls"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)

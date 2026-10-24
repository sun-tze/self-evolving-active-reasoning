"""Owned CPU sleeper stand-ins only. Real POSIX cases are skipped on Windows."""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import allocation
from .allocation_budget import AllocationContext,validate_prior_allocation
from .core import IntegrityError


class B2MechanicalTests(unittest.TestCase):
    def test_cancellation_records_then_raises_only_at_checkpoint(self):
        cancel=allocation.Cancellation()
        cancel.handler(signal.SIGTERM,None)
        cancel.handler(signal.SIGTERM,None)
        self.assertTrue(cancel.requested)
        with self.assertRaises(TimeoutError):
            cancel.check()

    def test_interrupted_TERM_wait_still_kills_and_confirms_owned_group(self):
        signals=[]
        class Child:
            pid=4321
            killed=False
            def wait(self,timeout):
                if not self.killed:
                    raise TimeoutError("synthetic late cancellation")
                return -9
            def poll(self):
                return -9 if self.killed else None
        child=Child()
        def killpg(pid,number):
            self.assertEqual(pid,child.pid)
            signals.append(number)
            if number==9:
                child.killed=True
            if number==0:
                raise ProcessLookupError()
        with patch.object(allocation.os,"killpg",side_effect=killpg,create=True),patch.object(signal,"SIGKILL",9,create=True):
            allocation.ProcessSupervisor(clock=lambda:1000).stop(child,1100)
        self.assertTrue(child.killed)
        self.assertEqual(signals,[signal.SIGTERM,9,0])


@unittest.skipUnless(os.name=="posix" and hasattr(signal,"setitimer") and hasattr(os,"killpg"),
                     "Real POSIX signal/process-group tests unavailable on Windows; no workaround attempted")
class B2PosixStandInTests(unittest.TestCase):
    def scenario(self,mode):
        class StandInEngine(allocation.ProcessSupervisor):
            def __init__(self,root):
                super().__init__(term_grace_seconds=.35,kill_grace_seconds=.5)
                self.root=root
                self.children=[]
                self.timers=[]
                self.injected=False
            def start(self,command,log_path,cwd,env):
                ready=self.root/f"ready-{len(self.children)}"
                code="import signal,time,sys;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text('ready');time.sleep(30)"
                child=subprocess.Popen([sys.executable,"-u","-c",code,str(ready)],
                                       stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                       start_new_session=True)
                self.children.append(child)
                deadline=time.monotonic()+3
                while not ready.exists() and time.monotonic()<deadline:
                    if child.poll() is not None:
                        raise RuntimeError("CPU stand-in exited before readiness")
                    time.sleep(.01)
                if not ready.exists():
                    raise TimeoutError("CPU stand-in readiness")
                return child
            def ready(self,*args):
                return
            def wait_controller(self,*args):
                if mode=="alarm_then_late_TERM":
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline:
                        self.cancellation.check()
                        time.sleep(.01)
                    raise AssertionError("Expected real SIGALRM was not delivered")
                target=self.root/"output/controller"
                target.mkdir()
                (target/"stage_one_complete.json").write_text('{"completed":true}')
                return 0
            def stop(self,child,deadline):
                if child is not None and mode!="control" and not self.injected:
                    self.injected=True
                    for delay in (.05,.10):
                        timer=threading.Timer(delay,lambda:os.kill(os.getpid(),signal.SIGTERM))
                        self.timers.append(timer)
                        timer.start()
                return super().stop(child,deadline)
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            ledger=root/"ledger"
            ledger.mkdir()
            release=root/"release.json"
            runtime=root/"runtime.json"
            release.write_text("{}")
            runtime.write_text("{}")
            args=SimpleNamespace(stage="stage-one",output=root/"output",ledger=ledger,release=release,runtime=runtime,
                                 vllm=Path("unused"),snapshot=root/"unused",tokenizer=root/"unused",port=18000,
                                 code=root,python=Path(sys.executable),arbench=root/"unused",prior=None)
            engine=StandInEngine(root)
            now=time.time()
            context=AllocationContext("stage-one","101","synthetic_exp",now,now,"digest")
            def raising_handler(number,frame):
                raise TimeoutError("synthetic original cancellation handler")
            previous={number:signal.signal(number,raising_handler) for number in (signal.SIGTERM,signal.SIGALRM)}
            original_write=allocation.write_exclusive
            original_guard=allocation.defer_cleanup_cancellation
            receipt_cancellations=[]
            def write_with_late_TERM(file,document):
                if mode!="control" and Path(file).name in ("allocation_receipt.json","stage-one.receipt.json"):
                    os.kill(os.getpid(),signal.SIGTERM)
                    receipt_cancellations.append(Path(file).name)
                return original_write(file,document)
            @contextmanager
            def guard_with_pre_entry_TERM():
                if mode=="pre_guard_TERM":
                    os.kill(os.getpid(),signal.SIGTERM)
                with original_guard() as protected:
                    yield protected
            try:
                if mode=="alarm_then_late_TERM":
                    # Start the alarm only after the two CPU stand-ins are ready.
                    original_wait=engine.wait_controller
                    def arm_alarm_then_wait(*args):
                        signal.setitimer(signal.ITIMER_REAL,.05)
                        return original_wait(*args)
                    engine.wait_controller=arm_alarm_then_wait
                with patch.object(allocation,"write_exclusive",side_effect=write_with_late_TERM), \
                     patch.object(allocation,"defer_cleanup_cancellation",side_effect=guard_with_pre_entry_TERM):
                    receipt=allocation.supervise(args,context,engine)
                self.assertEqual(len(engine.children),2)
                self.assertTrue(all(child.poll() is not None for child in engine.children))
                self.assertEqual(receipt["cleanup_errors"],[])
                self.assertEqual(receipt["cleanup_signal_protection"],"posix_deferred")
                self.assertEqual(receipt["status"],"timed_out" if mode=="alarm_then_late_TERM" else "completed")
                for file in (root/"output/allocation_receipt.json",ledger/"stage-one.receipt.json"):
                    self.assertEqual(json.loads(file.read_bytes()),receipt)
                if mode=="alarm_then_late_TERM":
                    with self.assertRaises(IntegrityError):
                        validate_prior_allocation(receipt,"synthetic_exp","digest","102")
                else:
                    validate_prior_allocation(receipt,"synthetic_exp","digest","102")
                if mode!="control":
                    self.assertEqual(receipt_cancellations,["allocation_receipt.json","stage-one.receipt.json"])
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for timer in engine.timers:
                    timer.join(timeout=1)
                for child in engine.children:
                    if child.poll() is None:
                        try:
                            os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=1)
                for number,handler in previous.items():
                    signal.signal(number,handler)

    def test_control_children_dead_and_durable_completed_receipt(self):
        self.scenario("control")

    def test_repeated_late_TERM_during_cleanup_and_terminal_receipt(self):
        self.scenario("late_TERM")

    def test_alarm_then_repeated_TERM_cleanup_stays_bounded_and_rejected(self):
        self.scenario("alarm_then_late_TERM")

    def test_TERM_immediately_before_cleanup_guard_cannot_escape(self):
        self.scenario("pre_guard_TERM")


if __name__=="__main__":
    unittest.main(verbosity=2)

"""Run unchanged legacy tests and real adapter tests; report platform skips explicitly."""
import argparse
import importlib.util
from pathlib import Path
import platform
import tempfile
import unittest
from common import HERE, write_json


def main(out):
    spec=importlib.util.spec_from_file_location("legacy_test_probe", HERE.parent/"sun_writer_probe/tests/test_probe.py")
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    reason=None
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"target").write_bytes(b"x")
        try: (root/"link").symlink_to(root/"target")
        except OSError as exc:
            if platform.system()=="Windows" and getattr(exc,"winerror",None)==1314:
                reason="Windows lacks symlink privilege (WinError 1314); run this unchanged test on Linux."
            else: raise
    if reason:
        name="test_symlink_escaping_manifest_directory_is_rejected"
        setattr(legacy.ProbeTests,name,unittest.skip(reason)(getattr(legacy.ProbeTests,name)))
    summaries={}
    for name,suite in [("legacy_synthetic",unittest.defaultTestLoader.loadTestsFromModule(legacy)),
                       ("real_adapter",unittest.defaultTestLoader.discover(str(HERE/"tests")))]:
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        summaries[name]={"run":result.testsRun,"passed":result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                         "failures":len(result.failures),"errors":len(result.errors),"skipped":[{"test":str(t),"reason":why} for t,why in result.skipped]}
    report={"python":platform.python_version(),"platform":platform.platform(),"suites":summaries}
    write_json(out,report)
    if any(s["failures"] or s["errors"] for s in summaries.values()): raise SystemExit(1)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True);main(p.parse_args().out)

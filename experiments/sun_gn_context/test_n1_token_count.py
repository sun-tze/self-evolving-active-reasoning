"""Exact chat-template counts, including mapping-return and overflow controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from .core import Assets,IntegrityError
from .journal import CallBoundary,Journal,read_journal
from .runtime import PinnedTokenizer

ASSETS=Assets.load()
TOKENIZER_PATH=os.environ.get("SUN_GN_TOKENIZER")


class N1ShapeTests(unittest.TestCase):
    def test_mapping_or_batched_return_is_rejected_even_if_backend_ignores_flag(self):
        for returned in ({"input_ids":[1,2,3],"attention_mask":[1,1,1]},[[1,2,3]],[1,True]):
            tokenizer=PinnedTokenizer.__new__(PinnedTokenizer)
            tokenizer.lock=threading.Lock()
            tokenizer.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:returned)
            with self.assertRaises(IntegrityError):
                tokenizer.chat_count([dict(role="user",content="synthetic")])


@unittest.skipUnless(TOKENIZER_PATH,"SUN_GN_TOKENIZER not supplied")
class N1GenuineTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=PinnedTokenizer(Path(TOKENIZER_PATH))

    def test_exact_counts_equal_render_then_encode_and_explicit_ID_lists(self):
        cases={
            "B0_actor":([dict(role="system",content=ASSETS.system("B0")),dict(role="user",content=ASSETS.prompts["guess_prompt"].format(turn=1))],221),
            "S_actor":([dict(role="system",content=ASSETS.system("S",ASSETS.m0)),dict(role="user",content=ASSETS.prompts["guess_prompt"].format(turn=1))],328),
            "curator":(ASSETS.curator_messages(ASSETS.m0,[]),442)}
        for name,(messages,expected) in cases.items():
            with self.subTest(case=name):
                backend=self.tokenizer.tokenizer
                rendered=backend.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                reference=backend.encode(rendered,add_special_tokens=False)
                explicit=backend.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                self.assertIsInstance(explicit,list)
                self.assertEqual(explicit,reference)
                self.assertEqual(len(reference),expected)
                self.assertEqual(self.tokenizer.chat_count(messages),len(reference))
                # This assertion detects len(mapping), unlike a cap-only check.
                self.assertGreater(self.tokenizer.chat_count(messages),2)

    def test_actual_oversized_chat_stops_before_transport_without_truncation(self):
        messages=[dict(role="user",content="a "*9000)]
        expected=len(self.tokenizer.tokenizer.encode(self.tokenizer.tokenizer.apply_chat_template(
            messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
        self.assertGreater(expected,8064)
        self.assertEqual(self.tokenizer.chat_count(messages),expected)
        calls=[]
        with tempfile.TemporaryDirectory() as path:
            journal=Journal(Path(path)/"journal.jsonl",dict(mode="synthetic_N1_overflow"))
            boundary=CallBoundary(lambda request:calls.append(request),self.tokenizer,journal,
                                  actor_limit=1,curator_limit=0,input_limit=94000000,output_limit=128,
                                  deadline=time.monotonic()+3600,reserve_seconds=120)
            with self.assertRaisesRegex(IntegrityError,"Context overflow"):
                boundary.call(messages,1,"actor",{})
            journal.close()
            rows=read_journal(Path(path)/"journal.jsonl")
            self.assertEqual(calls,[])
            self.assertFalse(any(row["event"]=="call_intent" for row in rows))


if __name__=="__main__":
    unittest.main(verbosity=2)

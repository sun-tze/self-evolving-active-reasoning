"""Minimal OpenAI-compatible chat transport for a local vLLM server; stdlib only.

http://127.0.0.1 only, no proxy, no redirect, no retry, fixed timeout. Returns the
five-key raw dict that ConcurrentBoundary checks. Creating an instance opens no
socket; each call makes exactly one HTTP request. Any HTTP error, timeout or
malformed body raises, which the boundary records as uncertain cost and a halt.
"""
import json
import urllib.parse
import urllib.request

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REQUEST_KEYS = frozenset({'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'seed'})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class LoopbackChatTransport:
    def __init__(self, base_url, *, timeout=120, model=MODEL):
        parts = urllib.parse.urlsplit(base_url)
        if (parts.scheme != 'http' or parts.hostname != '127.0.0.1' or parts.port is None or
                parts.query or parts.fragment or parts.username or parts.password):
            raise ValueError('http://127.0.0.1:PORT base URL required')
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError('Finite positive timeout required')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.timeout, self.model = timeout, model
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def __call__(self, request):
        if set(request) != REQUEST_KEYS or request['model'] != self.model:
            raise ValueError('Unexpected request keys or model')
        http = urllib.request.Request(self.url, data=json.dumps(request).encode('utf-8'),
                                      headers={'Content-Type': 'application/json'}, method='POST')
        with self._opener.open(http, timeout=self.timeout) as response:
            if response.status != 200:
                raise RuntimeError('HTTP status ' + str(response.status))
            data = json.loads(response.read().decode('utf-8'))
        if data.get('model') != self.model or len(data.get('choices') or []) != 1:
            raise RuntimeError('Unexpected served model or choice count')
        choice, usage = data['choices'][0], data['usage']
        return {'text': choice['message']['content'], 'finish_reason': choice['finish_reason'],
                'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens'], 'response_id': data['id']}

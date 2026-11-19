"""Send one image to the LAN perception server (Windows or Ubuntu)."""
import argparse
import json
import math
import mimetypes
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--group', required=True)
    parser.add_argument('--image', type=Path, required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt', help='Short question supplied directly')
    prompts.add_argument('--prompt-file', type=Path, help='Read a question from a UTF-8 text file')
    parser.add_argument('--read-timeout', type=float, default=75)
    args = parser.parse_args()
    if not math.isfinite(args.read_timeout) or args.read_timeout <= 0:
        parser.error('--read-timeout must be finite and positive')
    try:
        # utf-8-sig also accepts the optional BOM written by some Windows editors.
        prompt = args.prompt_file.read_text(encoding='utf-8-sig') if args.prompt_file else args.prompt
        if not prompt.strip():
            raise ValueError('Prompt must not be blank.')
        started = time.monotonic()
        with args.image.open('rb') as image:
            upload = (args.image.name, image, mimetypes.guess_type(args.image.name)[0] or 'application/octet-stream')
            response = httpx.post(
                args.server.rstrip('/') + '/analyze',
                data={'group_id': args.group, 'prompt': prompt},
                files={'image': upload},
                timeout=httpx.Timeout(args.read_timeout, connect=5),
            )
        print(f'HTTP {response.status_code} {response.reason_phrase}')
        print(f'Client elapsed: {time.monotonic() - started:.3f} s')
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except ValueError:
            print(response.text)
        if response.is_error:
            print('Server rejected or could not complete the request.', file=sys.stderr)
            if 'retry-after' in response.headers:
                print(f'Retry after {response.headers["retry-after"]} seconds.', file=sys.stderr)
            return 1
        return 0
    except (OSError, httpx.HTTPError, ValueError) as error:
        print(f'Client error ({type(error).__name__}): {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

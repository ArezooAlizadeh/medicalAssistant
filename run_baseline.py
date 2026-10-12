"""Run the reviewed questions against the existing local /ask/ endpoint.

Uses only Python's standard library. Does not grade or change the RAG system.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/ask/')
    parser.add_argument('--dataset', type=Path,
                        default=Path(__file__).with_name('neuroscience_cases_draft.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    timestamp = datetime.now(timezone.utc)
    output = args.output or Path.cwd() / (
        'v4_baseline_' + timestamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json'
    )
    report = {
        'started_at_utc': timestamp.isoformat(),
        'endpoint': args.url,
        'dataset_version': dataset['dataset_version'],
        'source_sha256': dataset['source_sha256'],
        'scope_note': dataset['scope'],
        'scoring_status': 'Not scored. Verify corpus alignment and review outputs.',
        'results': [],
    }
    for case in dataset['cases']:
        request = Request(
            args.url,
            data=urlencode({'question': case['question']}).encode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            method='POST',
        )
        record = dict(case)
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=180) as response:
                record['http_status'] = response.status
                record['api_output'] = json.loads(response.read().decode('utf-8'))
        except HTTPError as error:
            record['http_status'] = error.code
            record['error'] = str(error)
        except (URLError, TimeoutError, OSError, ValueError) as error:
            record['error'] = str(error)
        record['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        record['human_score'] = None
        report['results'].append(record)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')
        result = record.get('api_output', {}).get('response', record.get('error', ''))
        print(f"{case['id']} ({record['elapsed_seconds']}s): {result}", flush=True)
        if 'api_output' not in record:
            print('Stopped after request failure. Fix the endpoint, then rerun.', flush=True)
            break
    print(f'\nSaved results: {output.resolve()}')


if __name__ == '__main__':
    main()

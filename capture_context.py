"""Capture a fresh V3 graph run, including retrieved passages and node updates.

Place beside neuroscience_cases_draft.json in medicalAssistant. Run with the
server virtual environment. No API or graph implementation is modified.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def serialize(value):
    if hasattr(value, 'page_content') and hasattr(value, 'metadata'):
        return {'page_content': value.page_content,
                'metadata': serialize(value.metadata)}
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'server')
    parser.add_argument('--dataset', type=Path,
                        default=Path(__file__).resolve().with_name(
                            'neuroscience_cases_draft.json'))
    args = parser.parse_args()
    server = args.server_dir.resolve()
    graph_file = server / 'modules' / 'rag_graph.py'
    if not graph_file.is_file():
        parser.error('Place this script in medicalAssistant, or specify --server-dir.')
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    # Existing modules use paths relative to server, including the BM25 corpus.
    os.chdir(server)
    sys.path.insert(0, str(server))
    from modules.rag_graph import rag_graph

    stamp = datetime.now(timezone.utc)
    output = args.dataset.resolve().parent / (
        'v4_context_' + stamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    report = {
        'started_at_utc': stamp.isoformat(),
        'execution_mode': 'Direct V3 graph, fresh run; not the earlier HTTP run',
        'langgraph_version': version('langgraph'),
        'graph_file_sha256': hashlib.sha256(graph_file.read_bytes()).hexdigest(),
        'dataset_version': dataset['dataset_version'],
        'source_sha256': dataset['source_sha256'],
        'scope_note': dataset['scope'],
        'notes': [
            'Node events contain retrieved text even when context is rejected.',
            'Service-style output below is reconstructed, not an HTTP response.',
            'Elapsed time is direct graph runtime, not API response latency.',
            'No automated quality scores assigned.',
        ],
        'results': [],
    }
    for case in dataset['cases']:
        state = {'original_question': case['question'],
                 'search_query': case['question']}
        record = dict(case, events=[])
        started = time.perf_counter()
        try:
            # Single-mode default stream format: {node_name: partial_update}.
            # Plain dictionary replacement matches V3's reducer-free state.
            for event in rag_graph.stream(state, stream_mode='updates'):
                for node, update in event.items():
                    if not isinstance(update, dict):
                        raise ValueError(f'Unexpected update from {node}: {update!r}')
                    record['events'].append({'node': node,
                                             'update': serialize(update)})
                    state.update(update)
            if not record['events'] or record['events'][-1]['node'] not in (
                    'generate', 'fallback') or 'answer' not in state:
                raise ValueError('Graph did not reach an expected terminal node.')
            record['final_state'] = serialize(state)
            documents = state.get('documents', []) if state.get(
                'context_relevant', False) else []
            record['service_style_output'] = {
                'response': state.get('answer', ''),
                'sources': [doc.metadata.get('source', '') for doc in documents],
            }
            record['original_question_preserved'] = (
                state.get('original_question') == case['question'])
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            record['partial_state'] = serialize(state)
        record['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        report['results'].append(record)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')
        route = ' -> '.join(e['node'] for e in record['events'])
        print(f"{case['id']}: {route}", flush=True)
        if 'error' in record:
            print(record['error'], flush=True)
            print('Stopped after failure; partial results have been saved.')
            break
    print(f'\nSaved context results: {output}')


if __name__ == '__main__':
    main()

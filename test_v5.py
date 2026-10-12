"""Offline V5 unit tests of the project's actual function bodies.

Run from medicalAssistant: python test_v5.py
Only function definitions and MAX_REWRITES are loaded from source. Module
initialization, models, indexes and the live graph are deliberately not loaded.
These tests cover function contracts, not LangGraph wiring or API integration.
"""
import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def load_functions(path, names, bindings, constants=()):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
    if found != set(names):
        raise RuntimeError(f'Missing functions in {path}: {set(names) - found}')
    namespace = dict(bindings)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    print(f'Source: {path}\nSHA-256: {hashlib.sha256(source.encode()).hexdigest()}')
    return namespace


class GraphContracts(unittest.TestCase):
    def setUp(self):
        self.g = dict(GRAPH)
        # Reload functions so each test has independent global bindings.
        exec(GRAPH_CODE, self.g)
        self.g['grader_chain'] = Mock()
        self.g['rewrite_chain'] = Mock()
        self.g['retrieve_hybrid_documents'] = Mock()
        self.g['rerank_documents'] = Mock()
        self.g['get_llm_chain'] = Mock()
        self.doc = SimpleNamespace(page_content='Supporting evidence.', metadata={'source': 'paper.pdf'})

    def test_yes_generates_before_rewrite_limit(self):
        self.assertEqual(self.g['route_after_grading']({'context_relevant': True}), 'generate')

    def test_yes_generates_at_rewrite_limit(self):
        self.assertEqual(self.g['route_after_grading']({'context_relevant': True,
            'rewrite_count': self.g['MAX_REWRITES']}), 'generate')

    def test_no_rewrites_before_limit(self):
        self.assertEqual(self.g['route_after_grading']({'context_relevant': False}), 'rewrite')

    def test_no_falls_back_at_limit(self):
        self.assertEqual(self.g['route_after_grading']({'context_relevant': False,
            'rewrite_count': self.g['MAX_REWRITES']}), 'fallback')

    def test_zero_cap_disables_rewriting(self):
        self.g['MAX_REWRITES'] = 0
        self.assertEqual(self.g['route_after_grading']({'context_relevant': False}), 'fallback')

    def test_retrieve_uses_search_query(self):
        self.g['retrieve_hybrid_documents'].return_value = [self.doc]
        self.assertEqual(self.g['retrieve_node']({'search_query': 'rewritten'}), {'documents': [self.doc]})
        self.g['retrieve_hybrid_documents'].assert_called_once_with('rewritten')

    def test_rerank_receives_candidates_and_search_query(self):
        self.g['rerank_documents'].return_value = [self.doc]
        self.assertEqual(self.g['rerank_node']({'search_query': 'rewritten', 'documents': [self.doc]}),
            {'documents': [self.doc]})
        self.g['rerank_documents'].assert_called_once_with('rewritten', [self.doc])

    def test_grade_uses_original_question_and_selected_evidence(self):
        other = SimpleNamespace(page_content='Second passage.', metadata={})
        self.g['grader_chain'].invoke.return_value = '  yes\n'
        update = self.g['grade_context_node']({'original_question': 'original',
            'search_query': 'rewritten', 'documents': [self.doc, other]})
        self.assertEqual(update, {'context_relevant': True})
        self.g['grader_chain'].invoke.assert_called_once_with({
            'question': 'original', 'context': 'Supporting evidence.\n\nSecond passage.'})

    def test_no_and_non_yes_grades_are_rejected(self):
        for grade in ['NO', 'YES because it is relevant', '', 'MAYBE']:
            with self.subTest(grade=grade):
                self.g['grader_chain'].invoke.return_value = grade
                self.assertFalse(self.g['grade_context_node']({
                    'original_question': 'original', 'documents': [self.doc]})['context_relevant'])

    def test_rewrite_preserves_original_and_increments_count(self):
        self.g['rewrite_chain'].invoke.return_value = '  better query\n'
        state = {'original_question': 'original', 'search_query': 'current', 'rewrite_count': 0}
        before = dict(state)
        update = self.g['rewrite_node'](state)
        self.assertEqual(state, before)
        self.assertEqual(update, {'search_query': 'better query', 'rewrite_count': 1})
        state.update(update)
        self.assertEqual(state['original_question'], 'original')
        self.g['rewrite_chain'].invoke.assert_called_once_with({'question': 'current'})

    def test_rewrite_initializes_missing_count(self):
        self.g['rewrite_chain'].invoke.return_value = 'better query'
        self.assertEqual(self.g['rewrite_node']({'search_query': 'current'})['rewrite_count'], 1)

    def test_generation_uses_original_question_and_selected_documents(self):
        chain = self.g['get_llm_chain'].return_value
        chain.invoke.return_value = {'result': 'Supported answer.'}
        self.assertEqual(self.g['generate_node']({'original_question': 'original',
            'search_query': 'rewritten', 'documents': [self.doc]}), {'answer': 'Supported answer.'})
        chain.invoke.assert_called_once_with({'query': 'original', 'documents': [self.doc]})

    def test_fallback_returns_fixed_refusal_without_llm_call(self):
        self.assertEqual(self.g['fallback_node']({})['answer'],
            'I could not find enough relevant information in the uploaded biomedical literature to answer this question reliably.')
        self.g['get_llm_chain'].assert_not_called()

    def test_second_rejection_stops_after_one_rewrite(self):
        self.g['grader_chain'].invoke.return_value = 'NO'
        self.g['rewrite_chain'].invoke.return_value = 'better query'
        self.g['retrieve_hybrid_documents'].return_value = [self.doc]
        self.g['rerank_documents'].return_value = [self.doc]
        state = {'original_question': 'original', 'search_query': 'original'}
        for attempt in range(2):
            for node in ['retrieve_node', 'rerank_node', 'grade_context_node']:
                state.update(self.g[node](state))
            route = self.g['route_after_grading'](state)
            if attempt == 0:
                self.assertEqual(route, 'rewrite')
                state.update(self.g['rewrite_node'](state))
            else:
                self.assertEqual(route, 'fallback')
                state.update(self.g['fallback_node'](state))
        self.g['rewrite_chain'].invoke.assert_called_once()
        self.assertEqual(state['original_question'], 'original')
        self.assertEqual(state['rewrite_count'], 1)


class ServiceContracts(unittest.TestCase):
    def setUp(self):
        self.graph = Mock()
        self.s = dict(SERVICE)
        exec(SERVICE_CODE, self.s)
        self.s['rag_graph'] = self.graph
        self.doc = SimpleNamespace(metadata={'source': 'paper.pdf'})

    def test_service_initializes_original_and_search_question(self):
        self.graph.invoke.return_value = {'answer': 'answer', 'context_relevant': True, 'documents': [self.doc]}
        self.assertEqual(self.s['answer_question']('question'), {'response': 'answer', 'sources': ['paper.pdf']})
        self.graph.invoke.assert_called_once_with({'original_question': 'question', 'search_query': 'question'})

    def test_rejected_context_has_no_public_sources(self):
        self.graph.invoke.return_value = {'answer': 'refusal', 'context_relevant': False, 'documents': [self.doc]}
        self.assertEqual(self.s['answer_question']('question'), {'response': 'refusal', 'sources': []})

    def test_missing_relevance_has_no_public_sources(self):
        self.graph.invoke.return_value = {'answer': 'refusal', 'documents': [self.doc]}
        self.assertEqual(self.s['answer_question']('question')['sources'], [])

    def test_missing_source_metadata_does_not_crash(self):
        self.graph.invoke.return_value = {'answer': 'answer', 'context_relevant': True,
            'documents': [SimpleNamespace(metadata={})]}
        self.assertEqual(self.s['answer_question']('question')['sources'], [''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-dir', type=Path, default=Path(__file__).resolve().parent / 'server')
    args = parser.parse_args()
    graph_path = args.server_dir.resolve() / 'modules/rag_graph.py'
    service_path = args.server_dir.resolve() / 'services/rag_service.py'
    names = ['retrieve_node', 'rerank_node', 'grade_context_node', 'route_after_grading',
             'rewrite_node', 'generate_node', 'fallback_node']
    GRAPH = load_functions(graph_path, names, {'GraphState': dict}, ('MAX_REWRITES',))
    if GRAPH.get('MAX_REWRITES') != 1:
        raise RuntimeError('This V5 suite expects the V3 default MAX_REWRITES = 1.')
    GRAPH_CODE = compile(ast.Module(body=[n for n in ast.parse(graph_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(graph_path), 'exec')
    SERVICE = load_functions(service_path, ['answer_question'], {})
    SERVICE_CODE = compile(ast.Module(body=[n for n in ast.parse(service_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'answer_question'], type_ignores=[]), str(service_path), 'exec')
    unittest.main(argv=['test_v5.py', '-v'])

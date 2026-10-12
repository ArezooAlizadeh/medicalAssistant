"""Offline integration tests of the actual compiled V3 graph.

Run from medicalAssistant using the server environment: python test_v5_graph.py
Requires the project's installed LangGraph/LangChain packages. Model, retrieval,
reranking and dotenv initialization are replaced before graph source is loaded.
No API calls, index loading, model downloads or credential loading are needed.
This tests graph composition and the service adapter, not the HTTP endpoint.
"""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_graph_and_service(server):
    from langchain_core.runnables import RunnableLambda
    replacements = {}
    # Provide a package without executing the project's modules/__init__.py.
    package = ModuleType('modules')
    package.__path__ = [str(server / 'modules')]
    replacements['modules'] = package
    for name, attribute in [('modules.hybrid_retriever', 'retrieve_hybrid_documents'),
                            ('modules.reranker', 'rerank_documents'),
                            ('modules.llm', 'get_llm_chain')]:
        module = ModuleType(name)
        setattr(module, attribute, Mock(name=attribute))
        replacements[name] = module
    groq = ModuleType('langchain_groq')
    groq.ChatGroq = lambda **kwargs: RunnableLambda(lambda value: 'NO')
    replacements['langchain_groq'] = groq
    dotenv = ModuleType('dotenv')
    dotenv.load_dotenv = lambda *args, **kwargs: False
    replacements['dotenv'] = dotenv
    with patch.dict(sys.modules, replacements):
        graph = load_file('v5_graph_under_test', server / 'modules/rag_graph.py')
        with patch.dict(sys.modules, {'modules.rag_graph': graph}):
            service = load_file('v5_service_under_test', server / 'services/rag_service.py')
    return graph, service


class CompiledGraphTests(unittest.TestCase):
    def setUp(self):
        from langchain_core.documents import Document
        self.document = Document(page_content='Reviewed supporting passage.',
                                 metadata={'source': 'test-paper.pdf'})
        GRAPH.MAX_REWRITES = 1
        GRAPH.retrieve_hybrid_documents.reset_mock(return_value=True, side_effect=True)
        GRAPH.retrieve_hybrid_documents.return_value = [self.document]
        GRAPH.rerank_documents.reset_mock(return_value=True, side_effect=True)
        GRAPH.rerank_documents.return_value = [self.document]
        GRAPH.grader_chain = Mock()
        GRAPH.rewrite_chain = Mock()
        GRAPH.rewrite_chain.invoke.return_value = 'better query'
        GRAPH.get_llm_chain.reset_mock(return_value=True, side_effect=True)
        self.generation = Mock()
        self.generation.invoke.return_value = {'result': 'Supported test answer.'}
        GRAPH.get_llm_chain.return_value = self.generation

    def run_trace(self):
        state = {'original_question': 'original question', 'search_query': 'original question'}
        nodes = []
        for event in GRAPH.rag_graph.stream(state, stream_mode='updates'):
            for node, update in event.items():
                nodes.append(node)
                state.update(update)
        self.assertEqual(state['original_question'], 'original question')
        return nodes, state

    def test_normal_path_generates_without_rewriting(self):
        GRAPH.grader_chain.invoke.return_value = 'YES'
        nodes, state = self.run_trace()
        self.assertEqual(nodes, ['retrieve', 'rerank', 'grade_context', 'generate'])
        self.assertEqual(state['answer'], 'Supported test answer.')
        GRAPH.rewrite_chain.invoke.assert_not_called()
        self.generation.invoke.assert_called_once_with({
            'query': 'original question', 'documents': [self.document]})

    def test_rewrite_success_generates_even_at_limit(self):
        GRAPH.grader_chain.invoke.side_effect = ['NO', 'YES']
        nodes, state = self.run_trace()
        self.assertEqual(nodes, ['retrieve', 'rerank', 'grade_context', 'rewrite',
                                 'retrieve', 'rerank', 'grade_context', 'generate'])
        self.assertEqual(state['rewrite_count'], 1)
        self.assertEqual(state['search_query'], 'better query')
        self.assertTrue(state['context_relevant'])
        self.assertEqual([c.args[0] for c in GRAPH.retrieve_hybrid_documents.call_args_list],
                         ['original question', 'better query'])
        self.assertEqual([c.args[0]['question'] for c in GRAPH.grader_chain.invoke.call_args_list],
                         ['original question', 'original question'])
        GRAPH.rewrite_chain.invoke.assert_called_once()
        self.generation.invoke.assert_called_once_with({
            'query': 'original question', 'documents': [self.document]})

    def test_two_rejections_reach_fallback_without_generation(self):
        GRAPH.grader_chain.invoke.return_value = 'NO'
        nodes, state = self.run_trace()
        self.assertEqual(nodes, ['retrieve', 'rerank', 'grade_context', 'rewrite',
                                 'retrieve', 'rerank', 'grade_context', 'fallback'])
        self.assertEqual(state['rewrite_count'], 1)
        self.assertFalse(state['context_relevant'])
        self.assertIn('could not find enough relevant information', state['answer'])
        self.assertEqual(GRAPH.retrieve_hybrid_documents.call_count, 2)
        self.assertEqual(GRAPH.grader_chain.invoke.call_count, 2)
        GRAPH.rewrite_chain.invoke.assert_called_once()
        GRAPH.get_llm_chain.assert_not_called()

    def test_zero_cap_falls_back_without_second_search(self):
        GRAPH.MAX_REWRITES = 0
        GRAPH.grader_chain.invoke.return_value = 'NO'
        nodes, state = self.run_trace()
        self.assertEqual(nodes, ['retrieve', 'rerank', 'grade_context', 'fallback'])
        GRAPH.retrieve_hybrid_documents.assert_called_once_with('original question')
        GRAPH.rewrite_chain.invoke.assert_not_called()
        GRAPH.get_llm_chain.assert_not_called()

    def test_service_returns_sources_after_real_graph_success(self):
        GRAPH.grader_chain.invoke.return_value = 'YES'
        result = SERVICE.answer_question('original question')
        self.assertEqual(result, {'response': 'Supported test answer.', 'sources': ['test-paper.pdf']})

    def test_service_hides_rejected_documents_after_real_graph_fallback(self):
        GRAPH.grader_chain.invoke.return_value = 'NO'
        result = SERVICE.answer_question('original question')
        self.assertEqual(result['sources'], [])
        self.assertIn('could not find enough relevant information', result['response'])
        GRAPH.rewrite_chain.invoke.assert_called_once()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-dir', type=Path, default=Path(__file__).resolve().parent / 'server')
    args = parser.parse_args()
    server = args.server_dir.resolve()
    for relative in ['modules/rag_graph.py', 'services/rag_service.py']:
        path = server / relative
        print(f'Source: {path}\nSHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}', flush=True)
    print(f'LangGraph: {version("langgraph")}', flush=True)
    GRAPH, SERVICE = load_graph_and_service(server)
    unittest.main(argv=['test_v5_graph.py', '-v'])

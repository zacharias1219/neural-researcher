"""Tests for the coverage agent's domain clustering logic."""
import pytest
from neuralresearcher.state import Paper, Claim, CoverageCluster
from neuralresearcher.agents.coverage import _classify_paper, DOMAIN_TAXONOMY


def _make_paper(id, title, abstract, methods=None, datasets=None):
    return Paper(
        id=id, title=title, authors=["Author"], venue="arXiv",
        year=2024, url=f"http://test/{id}", abstract=abstract,
        methods=methods or [], datasets=datasets or [],
    )


class TestClassifyPaper:
    def test_vision_paper(self):
        paper = _make_paper("p1", "Vision Transformer for Image Classification",
                           "We propose a ViT model for image classification on ImageNet.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "vision" in domains

    def test_nlp_paper(self):
        paper = _make_paper("p2", "Language Model Pretraining",
                           "We train a transformer language model on the Pile dataset.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "nlp" in domains

    def test_speech_paper(self):
        paper = _make_paper("p3", "Keyword Spotting on MCUs",
                           "We deploy an audio model for speech recognition on LibriSpeech.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "speech" in domains

    def test_rl_paper(self):
        paper = _make_paper("p4", "Multi-Agent Reinforcement Learning",
                           "We use PPO for policy optimization in a multi-agent environment.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "rl" in domains

    def test_edge_paper(self):
        paper = _make_paper("p5", "TinyML Deployment",
                           "We deploy a quantized model on an ARM Cortex-M microcontroller.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "edge" in domains

    def test_multi_domain_paper(self):
        """A paper about edge vision should match both domains."""
        paper = _make_paper("p6", "Efficient Vision on MCU",
                           "We run image classification on a microcontroller using quantization.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "vision" in domains
        assert "edge" in domains

    def test_uncategorized_paper(self):
        """A paper with no matching keywords should be 'uncategorized'."""
        paper = _make_paper("p7", "Novel Algebraic Topology Results",
                           "We prove new theorems about homological algebra.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert domains == ["uncategorized"]

    def test_methods_and_datasets_used_for_classification(self):
        """Methods and datasets fields should also be searched."""
        paper = _make_paper("p8", "Efficient Model",
                           "We propose an efficient model.",
                           methods=["Mamba SSM"], datasets=["LibriSpeech"])
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "speech" in domains

    def test_time_series_paper(self):
        paper = _make_paper("p9", "Forecasting with SSMs",
                           "We apply state-space models to time series forecasting.")
        domains = _classify_paper(paper, DOMAIN_TAXONOMY)
        assert "time_series" in domains

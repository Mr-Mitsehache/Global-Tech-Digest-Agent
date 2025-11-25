# app/graphs/digest_graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.states.digest_state import DigestState
from app.agents.collector_agent import collector_node
from app.agents.classifier_agent import classifier_node
from app.agents.summarizer_agent import summarizer_node
import logging
logger = logging.getLogger(__name__)


def build_digest_graph():
    """
    ประกอบ StateGraph 3 node:
    collector -> classifier -> summarizer -> END
    """
    logger.info("Building digest graph")
    graph = StateGraph(DigestState)

    logger.info("Register node: collector")
    graph.add_node("collector", collector_node)
    logger.info("Register node: classifier")
    graph.add_node("classifier", classifier_node)
    logger.info("Register node: summarizer")
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("collector")
    graph.add_edge("collector", "classifier")
    graph.add_edge("classifier", "summarizer")
    graph.add_edge("summarizer", END)

    logger.info("Digest graph compiled")
    return graph


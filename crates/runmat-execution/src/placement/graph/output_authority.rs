//! Canonical graph-output facts carried by placement signatures and feedback.

use serde::Serialize;

use super::{PlacementGraph, PlacementGraphOutputFact, SourceOutputOrdinal};
use crate::Digest;

#[derive(Serialize)]
struct Candidate<'a> {
    identity: &'a str,
    outputs: Option<&'a [PlacementGraphOutputFact]>,
}

#[derive(Serialize)]
struct Node<'a> {
    identity: &'a str,
    candidates: Vec<Candidate<'a>>,
}

#[derive(Serialize)]
struct Edge {
    from: u32,
    to: u32,
    source_output: Option<SourceOutputOrdinal>,
}

#[derive(Serialize)]
struct Projection<'a> {
    domain: &'static str,
    nodes: Vec<Node<'a>>,
    edges: Vec<Edge>,
}

pub(super) fn digest(graph: &PlacementGraph) -> Option<Digest> {
    if !graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .any(|candidate| candidate.output_facts.is_some())
        && !graph.edges.iter().any(|edge| edge.source_output.is_some())
    {
        return None;
    }
    let projection = Projection {
        domain: "placement-graph-output-authority/v1",
        nodes: graph
            .nodes
            .iter()
            .map(|node| Node {
                identity: &node.identity,
                candidates: node
                    .candidates
                    .iter()
                    .map(|candidate| Candidate {
                        identity: &candidate.descriptor.identity,
                        outputs: candidate.output_facts.as_deref(),
                    })
                    .collect(),
            })
            .collect(),
        edges: graph
            .edges
            .iter()
            .map(|edge| Edge {
                from: edge.from,
                to: edge.to,
                source_output: edge.source_output,
            })
            .collect(),
    };
    Some(Digest::sha256_serialized(&projection).expect("bounded graph output facts serialize"))
}

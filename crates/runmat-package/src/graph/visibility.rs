use super::PackageGraph;
use crate::ContentDigest;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisibilityResolution {
    Found(ContentDigest),
    NotVisible,
    Ambiguous(Vec<ContentDigest>),
}

pub(super) fn resolve(
    graph: &PackageGraph,
    requester: &ContentDigest,
    candidates: impl IntoIterator<Item = ContentDigest>,
) -> VisibilityResolution {
    let candidates = candidates.into_iter().collect::<BTreeSet<_>>();
    if candidates.contains(&graph.root) {
        return VisibilityResolution::Found(graph.root.clone());
    }
    let visible = graph
        .edges
        .iter()
        .filter(|edge| &edge.from == requester && candidates.contains(&edge.to))
        .map(|edge| edge.to.clone())
        .collect::<BTreeSet<_>>();
    match visible.len() {
        0 => VisibilityResolution::NotVisible,
        1 => VisibilityResolution::Found(visible.into_iter().next().unwrap()),
        _ => VisibilityResolution::Ambiguous(visible.into_iter().collect()),
    }
}

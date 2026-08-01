use super::{GraphEdge, GraphPackage};
use crate::{ContentDigest, GraphError};
use serde::Serialize;
use std::collections::BTreeMap;

pub(super) fn compute_graph_digest(
    root: &ContentDigest,
    packages: &BTreeMap<ContentDigest, GraphPackage>,
    edges: &[GraphEdge],
) -> Result<ContentDigest, GraphError> {
    #[derive(Serialize)]
    struct Input<'a> {
        format: &'static str,
        root: &'a ContentDigest,
        packages: &'a BTreeMap<ContentDigest, GraphPackage>,
        edges: &'a [GraphEdge],
    }
    serde_json::to_vec(&Input {
        format: "runmat-package-graph-v1",
        root,
        packages,
        edges,
    })
    .map(ContentDigest::sha256)
    .map_err(|error| GraphError::Invalid(format!("cannot encode graph digest input: {error}")))
}

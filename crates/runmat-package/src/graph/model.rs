use super::{digest::compute_graph_digest, GraphEdge, VisibilityResolution};
use crate::{
    CanonicalPackageId, ContentDigest, GraphError, HostCapability, PackageAlias, PackageInstanceId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphPackage {
    pub instance: PackageInstanceId,
    pub local_name: String,
    pub required_capabilities: BTreeSet<HostCapability>,
    pub singleton: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageGraph {
    pub root: ContentDigest,
    pub packages: BTreeMap<ContentDigest, GraphPackage>,
    pub edges: Vec<GraphEdge>,
    pub graph_digest: ContentDigest,
}

impl PackageGraph {
    pub(crate) fn finish(
        root: ContentDigest,
        packages: BTreeMap<ContentDigest, GraphPackage>,
        mut edges: Vec<GraphEdge>,
    ) -> Result<Self, GraphError> {
        if !packages.contains_key(&root) {
            return Err(GraphError::Invalid(
                "root package instance is missing".to_string(),
            ));
        }
        edges.sort();
        let graph_digest = compute_graph_digest(&root, &packages, &edges)?;
        Ok(Self {
            root,
            packages,
            edges,
            graph_digest,
        })
    }

    pub fn dependency(&self, from: &ContentDigest, alias: &PackageAlias) -> Option<&GraphPackage> {
        self.edges
            .iter()
            .find(|edge| &edge.from == from && &edge.alias == alias)
            .and_then(|edge| self.packages.get(&edge.to))
    }

    pub fn resolve_visible_candidates(
        &self,
        requester: &ContentDigest,
        candidates: impl IntoIterator<Item = ContentDigest>,
    ) -> VisibilityResolution {
        super::visibility::resolve(self, requester, candidates)
    }

    pub fn instances_of(&self, package: &CanonicalPackageId) -> Vec<&GraphPackage> {
        self.packages
            .values()
            .filter(|candidate| &candidate.instance.package == package)
            .collect()
    }
}

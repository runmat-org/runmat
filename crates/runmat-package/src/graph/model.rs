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

    pub(crate) fn validate_digest(&self) -> Result<(), GraphError> {
        if !self.packages.contains_key(&self.root) {
            return Err(GraphError::Invalid(
                "root package instance is missing".to_string(),
            ));
        }
        for (identity, package) in &self.packages {
            if identity != &package.instance.identity_digest {
                return Err(GraphError::Invalid(format!(
                    "package map key {identity} does not match its instance identity {}",
                    package.instance.identity_digest
                )));
            }
        }
        for edge in &self.edges {
            if !self.packages.contains_key(&edge.from) || !self.packages.contains_key(&edge.to) {
                return Err(GraphError::Invalid(format!(
                    "dependency edge `{}` references an absent package instance",
                    edge.alias
                )));
            }
        }
        if self.edges.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(GraphError::Invalid(
                "dependency edges are not in canonical order".to_string(),
            ));
        }
        let mut aliases = BTreeSet::new();
        for edge in &self.edges {
            if !aliases.insert((edge.from.clone(), edge.alias.clone())) {
                return Err(GraphError::Invalid(format!(
                    "package {} has duplicate dependency alias `{}`",
                    edge.from, edge.alias
                )));
            }
        }
        let expected =
            super::digest::compute_graph_digest(&self.root, &self.packages, &self.edges)?;
        if expected != self.graph_digest {
            return Err(GraphError::Invalid(
                "package graph digest does not match its canonical contents".to_string(),
            ));
        }
        Ok(())
    }
}

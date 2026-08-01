use super::validate::{canonicalized, compute_graph_digest, validate_lock};
use crate::{
    CanonicalPackageId, ContentDigest, DependencyGroup, HostCapability, LockError, PackageAlias,
    PackageInstanceId, TargetPredicate,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const LOCK_SCHEMA_VERSION: u32 = 1;
pub const RESOLVER_FORMAT_VERSION: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootLock {
    pub manifest_digest: ContentDigest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<CanonicalPackageId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockSelection {
    pub target: String,
    pub groups: BTreeSet<DependencyGroup>,
    pub root_features: BTreeSet<String>,
    pub host_capabilities: BTreeSet<HostCapability>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedPackage {
    pub instance: PackageInstanceId,
    pub features: BTreeSet<String>,
    pub required_capabilities: BTreeSet<HostCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runmat_version: Option<String>,
    pub singleton: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedEdge {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<ContentDigest>,
    pub alias: PackageAlias,
    pub to: ContentDigest,
    pub group: DependencyGroup,
    pub optional: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageLock {
    pub schema_version: u32,
    pub resolver_version: String,
    pub root: RootLock,
    pub selection: LockSelection,
    pub packages: Vec<LockedPackage>,
    pub edges: Vec<LockedEdge>,
    pub graph_digest: ContentDigest,
}

impl PackageLock {
    pub fn new(
        root: RootLock,
        selection: LockSelection,
        packages: Vec<LockedPackage>,
        edges: Vec<LockedEdge>,
    ) -> Result<Self, LockError> {
        let mut lock = Self {
            schema_version: LOCK_SCHEMA_VERSION,
            resolver_version: RESOLVER_FORMAT_VERSION.to_string(),
            root,
            selection,
            packages,
            edges,
            graph_digest: ContentDigest::sha256([]),
        };
        lock = canonicalized(lock);
        lock.graph_digest = compute_graph_digest(&lock)?;
        validate_lock(&lock)?;
        Ok(lock)
    }

    pub fn validate(&self) -> Result<(), LockError> {
        validate_lock(self)
    }
}

use crate::{
    CanonicalPackageId, DependencyGroup, HostCapability, PackageAlias, TargetEnvironment,
    TargetPredicate,
};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionRequirement {
    pub alias: PackageAlias,
    pub package: CanonicalPackageId,
    pub version: VersionReq,
    pub group: DependencyGroup,
    pub optional: bool,
    pub default_features: bool,
    pub features: BTreeSet<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionRequest {
    pub root: String,
    pub requirements: Vec<ResolutionRequirement>,
    pub groups: BTreeSet<DependencyGroup>,
    pub root_features: BTreeSet<String>,
    pub environment: TargetEnvironment,
    pub runmat_version: Version,
    pub offline: bool,
    pub allowed_capabilities: BTreeSet<HostCapability>,
}

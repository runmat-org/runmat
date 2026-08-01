use crate::{
    DependencyGroup, FrozenProject, GitAcquisitionIntent, GitAcquisitionPlan, GitAcquisitionPolicy,
    GitSourceId, HostCapability, LockSelection, PackageLock, PathLockDecision,
};
use std::collections::BTreeSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitPackageMount {
    pub source: GitSourceId,
    pub root: PathBuf,
}

pub trait GitPackageProvider {
    fn acquire<'a>(
        &'a self,
        plan: &'a GitAcquisitionPlan,
    ) -> Pin<Box<dyn Future<Output = Result<GitPackageMount, String>> + 'a>>;
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectResolveOptions {
    pub target: String,
    pub groups: BTreeSet<DependencyGroup>,
    pub root_features: BTreeSet<String>,
    pub host_capabilities: BTreeSet<HostCapability>,
    pub git_intent: GitAcquisitionIntent,
    pub git_policy: GitAcquisitionPolicy,
}

impl ProjectResolveOptions {
    pub fn lock_selection(&self) -> LockSelection {
        LockSelection {
            target: self.target.clone(),
            groups: self.groups.clone(),
            root_features: self.root_features.clone(),
            host_capabilities: self.host_capabilities.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedProject {
    pub frozen: FrozenProject,
    pub lock: PackageLock,
    pub lock_decision: PathLockDecision,
    pub acquired_git_sources: Vec<GitSourceId>,
    pub source_inventories: Vec<crate::SourceInventory>,
}

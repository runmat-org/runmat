use crate::{
    DependencyGroup, FrozenProject, GitAcquisitionPlan, GitSourceId, HostCapability, LockSelection,
    PackageLock, PathLockDecision, RegistryAcquisitionPlan, RegistryCandidatePlan,
    RegistryCandidateRecord, RegistrySourceId, ServerProjectAcquisitionPlan, ServerProjectSourceId,
    SourceAcquisitionIntent, SourceAcquisitionPolicy,
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

pub trait PackageSourceProvider {
    fn acquire_git<'a>(
        &'a self,
        plan: &'a GitAcquisitionPlan,
    ) -> Pin<Box<dyn Future<Output = Result<GitPackageMount, String>> + 'a>>;

    fn acquire_server_project<'a>(
        &'a self,
        _plan: &'a ServerProjectAcquisitionPlan,
    ) -> Pin<Box<dyn Future<Output = Result<ServerProjectPackageMount, String>> + 'a>> {
        Box::pin(async { Err("Server project snapshot acquisition is not configured".to_string()) })
    }

    fn acquire_registry<'a>(
        &'a self,
        _plan: &'a RegistryAcquisitionPlan,
    ) -> Pin<Box<dyn Future<Output = Result<RegistryPackageMount, String>> + 'a>> {
        Box::pin(async { Err("registry package acquisition is not configured".to_string()) })
    }

    fn registry_candidates<'a>(
        &'a self,
        _plan: &'a RegistryCandidatePlan,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RegistryCandidateRecord>, String>> + 'a>> {
        Box::pin(async { Err("registry candidate acquisition is not configured".to_string()) })
    }
}

pub use PackageSourceProvider as GitPackageProvider;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerProjectPackageMount {
    pub source: ServerProjectSourceId,
    pub root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryPackageMount {
    pub source: RegistrySourceId,
    pub root: PathBuf,
    pub metadata: Option<crate::RegistryReleaseMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectResolveOptions {
    pub target: String,
    pub default_server_origin: String,
    pub default_registry_index: String,
    pub groups: BTreeSet<DependencyGroup>,
    pub root_features: BTreeSet<String>,
    pub host_capabilities: BTreeSet<HostCapability>,
    pub source_intent: SourceAcquisitionIntent,
    pub source_policy: SourceAcquisitionPolicy,
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
    pub acquired_server_sources: Vec<ServerProjectSourceId>,
    pub acquired_registry_sources: Vec<RegistrySourceId>,
    pub source_inventories: Vec<crate::SourceInventory>,
}

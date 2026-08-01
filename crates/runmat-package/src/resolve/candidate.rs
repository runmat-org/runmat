use super::ResolutionRequirement;
use crate::{
    CanonicalPackageId, HostCapability, PackageInstanceId, RegistryId, ResolveError,
    TargetPredicate,
};
use semver::VersionReq;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateMetadata {
    pub instance: PackageInstanceId,
    pub dependencies: Vec<ResolutionRequirement>,
    pub features: BTreeMap<String, BTreeSet<String>>,
    pub required_capabilities: BTreeSet<HostCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runmat_version: Option<VersionReq>,
    pub singleton: bool,
    pub yanked: bool,
    pub available_offline: bool,
    pub target_artifacts: BTreeSet<TargetPredicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry_metadata: Option<crate::RegistryReleaseMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateQuery {
    pub package: CanonicalPackageId,
    pub source_registry: RegistryId,
    pub offline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceSelectionPolicy {
    pub replacements: BTreeMap<RegistryId, RegistryId>,
    pub offline: bool,
}

pub trait CandidateProvider {
    fn candidates<'a>(
        &'a self,
        query: &'a CandidateQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CandidateMetadata>, ResolveError>> + 'a>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandidateIndex {
    packages: BTreeMap<CanonicalPackageId, Vec<CandidateMetadata>>,
}

impl CandidateIndex {
    pub fn insert(&mut self, candidate: CandidateMetadata) {
        self.packages
            .entry(candidate.instance.package.clone())
            .or_default()
            .push(candidate);
    }

    pub fn candidates(&self, package: &CanonicalPackageId) -> &[CandidateMetadata] {
        self.packages.get(package).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn package_ids(&self) -> impl Iterator<Item = &CanonicalPackageId> {
        self.packages.keys()
    }
}

pub async fn acquire_candidates(
    provider: &dyn CandidateProvider,
    packages: impl IntoIterator<Item = CanonicalPackageId>,
    offline: bool,
) -> Result<CandidateIndex, ResolveError> {
    acquire_candidates_with_policy(
        provider,
        packages,
        &SourceSelectionPolicy {
            replacements: BTreeMap::new(),
            offline,
        },
    )
    .await
}

pub async fn acquire_candidates_with_policy(
    provider: &dyn CandidateProvider,
    packages: impl IntoIterator<Item = CanonicalPackageId>,
    policy: &SourceSelectionPolicy,
) -> Result<CandidateIndex, ResolveError> {
    let mut index = CandidateIndex::default();
    let mut pending = packages.into_iter().collect::<BTreeSet<_>>();
    let mut queried = BTreeSet::new();
    while let Some(package) = pending.pop_first() {
        if !queried.insert(package.clone()) {
            continue;
        }
        let source_registry = replacement_registry(package.registry(), &policy.replacements)?;
        let mut candidates = provider
            .candidates(&CandidateQuery {
                package: package.clone(),
                source_registry,
                offline: policy.offline,
            })
            .await?;
        candidates.sort_by(|left, right| {
            right
                .instance
                .version
                .cmp(&left.instance.version)
                .then_with(|| {
                    left.instance
                        .identity_digest
                        .cmp(&right.instance.identity_digest)
                })
        });
        for candidate in candidates {
            if candidate.instance.package != package {
                return Err(ResolveError::Provider(format!(
                    "provider returned {} for query {package}",
                    candidate.instance.package
                )));
            }
            pending.extend(
                candidate
                    .dependencies
                    .iter()
                    .map(|dependency| dependency.package.clone()),
            );
            index.insert(candidate);
        }
    }
    Ok(index)
}

fn replacement_registry(
    source: &RegistryId,
    replacements: &BTreeMap<RegistryId, RegistryId>,
) -> Result<RegistryId, ResolveError> {
    let mut current = source.clone();
    let mut visited = BTreeSet::new();
    while let Some(replacement) = replacements.get(&current) {
        if !visited.insert(current.clone()) {
            return Err(ResolveError::Provider(format!(
                "registry source replacement cycle begins at `{source}`"
            )));
        }
        current = replacement.clone();
    }
    Ok(current)
}

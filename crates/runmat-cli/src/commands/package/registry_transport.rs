use runmat_package::{
    CanonicalPackageId, ContentDigest, DependencyGroup, PackageVersion, RegistryAcquisitionPlan,
    RegistryCandidatePlan, RegistryCandidateRecord, RegistryOrigin, RegistryPackageReference,
    RegistryReleaseDependency, RegistryReleaseId, RegistryReleaseMetadata, RegistrySourceId,
};
use runmat_package_cache_native::registry::{RegistryArtifactTransfer, RegistryTransport};
use runmat_server_client::auth::{resolve_auth_token, resolve_server_url, RemoteConfig};
use runmat_server_client::packages::{
    RegistryCandidate, RegistryCandidateOutcome, RegistryClient, RegistryReleaseCore,
    RegistryReleaseMetadata as RegistryReleaseResponse, RegistryReleaseOutcome,
};
use std::str::FromStr;

const MAX_REGISTRY_ARTIFACT_BYTES: u64 = 6 * 1024 * 1024 * 1024;

#[derive(Debug, Default)]
pub(super) struct RunMatRegistryTransport;

impl RegistryTransport for RunMatRegistryTransport {
    fn candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<RegistryCandidateRecord>, String>> + 'a>,
    > {
        Box::pin(async move {
            let client = registry_client(&plan.index).await?;
            let outcome = client
                .candidates(plan.package.organization(), plan.package.name(), None)
                .await
                .map_err(|error| error.to_string())?;
            let response = match outcome {
                RegistryCandidateOutcome::Candidates(response) => response,
                RegistryCandidateOutcome::NotModified => {
                    return Err(
                        "unexpected not-modified candidate response without an ETag".to_string()
                    );
                }
            };
            response
                .candidates
                .iter()
                .map(|candidate| candidate_record(&plan.package, candidate))
                .collect()
        })
    }

    fn fetch<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<RegistryArtifactTransfer, String>> + 'a>,
    > {
        Box::pin(async move {
            let client = registry_client(&plan.index).await?;
            let outcome = if let Some(expected) = &plan.expected {
                client.resolve_exact(expected.release.as_str(), None).await
            } else {
                client
                    .resolve(
                        plan.package.organization(),
                        plan.package.name(),
                        &plan.requirement.to_string(),
                        None,
                    )
                    .await
            }
            .map_err(|error| error.to_string())?;
            let response = match outcome {
                RegistryReleaseOutcome::Release(response) => response,
                RegistryReleaseOutcome::NotModified => {
                    return Err(
                        "unexpected not-modified registry response without an ETag".to_string()
                    );
                }
            };
            let source = source_from_metadata(plan, &response.metadata)?;
            let metadata = release_metadata(&response.metadata.release)?;
            let artifact_bytes = client
                .download_artifact(&response.metadata.artifact, MAX_REGISTRY_ARTIFACT_BYTES)
                .await
                .map_err(|error| error.to_string())?;
            Ok(RegistryArtifactTransfer {
                package_id: response.metadata.release.package_id.clone(),
                source,
                metadata,
                artifact_bytes,
            })
        })
    }
}

async fn registry_client(index: &str) -> Result<RegistryClient, String> {
    let mut config = RemoteConfig::load().map_err(|error| error.to_string())?;
    let configured = resolve_server_url(&config, None).map_err(|error| error.to_string())?;
    let token = if configured.trim_end_matches('/') == index {
        resolve_auth_token(&mut config, &configured).await.ok()
    } else {
        None
    };
    RegistryClient::new(index, token).map_err(|error| error.to_string())
}

fn candidate_record(
    package: &CanonicalPackageId,
    candidate: &RegistryCandidate,
) -> Result<RegistryCandidateRecord, String> {
    let source = source_from_parts(
        package,
        &candidate.release,
        &candidate.artifact.digest,
        &candidate.artifact.tree_digest,
    )?;
    let metadata = release_metadata(&candidate.release)?;
    metadata.verify_supply_chain(&candidate.release.package_id, &source)?;
    Ok(RegistryCandidateRecord {
        source,
        metadata,
        yanked: candidate.yanked,
    })
}

fn release_metadata(metadata: &RegistryReleaseCore) -> Result<RegistryReleaseMetadata, String> {
    Ok(RegistryReleaseMetadata {
        singleton: metadata.singleton,
        runmat_requirement: metadata.runmat_requirement.clone(),
        dependencies: metadata
            .dependencies
            .iter()
            .map(|dependency| {
                Ok(RegistryReleaseDependency {
                    alias: dependency.alias.clone(),
                    package: RegistryPackageReference {
                        registry: RegistryOrigin::from_str(&dependency.registry)
                            .map_err(|error| error.to_string())?,
                        namespace: dependency.namespace.clone(),
                        name: dependency.name.clone(),
                    },
                    requirement: dependency.requirement.clone(),
                    group: match dependency.group.as_str() {
                        "runtime" => DependencyGroup::Runtime,
                        "development" => DependencyGroup::Development,
                        "test" => DependencyGroup::Test,
                        _ => return Err("registry dependency group is invalid".to_string()),
                    },
                    target: dependency.target.clone(),
                    optional: dependency.optional,
                    default_features: dependency.default_features,
                    features: dependency.features.clone(),
                })
            })
            .collect::<Result<_, String>>()?,
        features: metadata.features.clone(),
        required_capabilities: metadata.required_capabilities.clone(),
        optional_capabilities: metadata.optional_capabilities.clone(),
        readme_digest: metadata.readme_digest.clone(),
        license: metadata.license.clone(),
        supply_chain: metadata.supply_chain.clone(),
    })
}

fn source_from_metadata(
    plan: &RegistryAcquisitionPlan,
    metadata: &RegistryReleaseResponse,
) -> Result<RegistrySourceId, String> {
    if metadata.release.namespace != plan.package.organization()
        || metadata.release.name != plan.package.name()
    {
        return Err("registry returned a different package identity".to_string());
    }
    source_from_parts(
        &plan.package,
        &metadata.release,
        &metadata.artifact.digest,
        &metadata.artifact.tree_digest,
    )
}

fn source_from_parts(
    package: &CanonicalPackageId,
    metadata: &RegistryReleaseCore,
    artifact_digest: &str,
    tree_digest: &str,
) -> Result<RegistrySourceId, String> {
    Ok(RegistrySourceId {
        registry_origin: RegistryOrigin::from_str(&metadata.registry)
            .map_err(|error| error.to_string())?,
        package: package.clone(),
        release: RegistryReleaseId::from_str(&metadata.release_id)
            .map_err(|error| error.to_string())?,
        version: PackageVersion::from_str(&metadata.version).map_err(|error| error.to_string())?,
        release_digest: ContentDigest::from_str(&metadata.release_digest)
            .map_err(|error| error.to_string())?,
        artifact_digest: ContentDigest::from_str(artifact_digest)
            .map_err(|error| error.to_string())?,
        tree_digest: ContentDigest::from_str(tree_digest).map_err(|error| error.to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_package::{CanonicalPackageId, RegistryId, SourceAcquisitionPolicy};

    #[test]
    fn metadata_conversion_rejects_identity_substitution() {
        let plan = runmat_package::plan_registry_acquisition(
            RegistryId::default(),
            "https://api.runmat.test",
            CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
            "*".parse().unwrap(),
            None,
            runmat_package::SourceAcquisitionIntent::Execute,
            SourceAcquisitionPolicy::default(),
        )
        .unwrap();
        let metadata = RegistryReleaseResponse {
            release: RegistryReleaseCore {
                package_id: "pkg_0123456789abcdef0123456789abcdef".to_string(),
                release_id: "rel_0123456789abcdef0123456789abcdef".to_string(),
                registry: "https://packages.runmat.test".to_string(),
                namespace: "other".to_string(),
                name: "tools".to_string(),
                version: "1.0.0".to_string(),
                release_digest: format!("sha256:{}", "a".repeat(64)),
                singleton: false,
                runmat_requirement: None,
                features: Default::default(),
                required_capabilities: Vec::new(),
                optional_capabilities: Vec::new(),
                readme_digest: None,
                license: None,
                dependencies: Vec::new(),
                advisories: Vec::new(),
                supply_chain: None,
            },
            artifact: runmat_server_client::packages::RegistryArtifact {
                id: "art_0123456789abcdef0123456789abcdef".to_string(),
                digest: format!("sha256:{}", "b".repeat(64)),
                tree_digest: format!("sha256:{}", "c".repeat(64)),
                byte_len: 1,
                media_type: "application/vnd.runmat.package+json".to_string(),
                download_url: "/artifact".to_string(),
                expires_at: 1,
            },
        };
        assert!(source_from_metadata(&plan, &metadata).is_err());
    }
}

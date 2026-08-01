use super::{shared, JsPackageSourceProvider};
use runmat_package::{
    CanonicalPackageId, ContentDigest, DependencyGroup, PackageVersion, RegistryAcquisitionPlan,
    RegistryCandidatePlan, RegistryCandidateRecord, RegistryOrigin, RegistryPackageMount,
    RegistryPackageReference, RegistryReleaseDependency, RegistryReleaseId,
    RegistryReleaseMetadata, RegistrySourceId,
};
use runmat_package_cache::{CacheBackend, CacheError, CommitOutcome};
use std::collections::BTreeMap;
use std::str::FromStr;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CandidateListWire {
    candidates: Vec<CandidateWire>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CandidateWire {
    #[serde(flatten)]
    release: ReleaseCoreWire,
    artifact: CandidateArtifactWire,
    yanked: bool,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResolveWire {
    #[serde(flatten)]
    release: ReleaseCoreWire,
    artifact: ArtifactWire,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleaseCoreWire {
    package_id: String,
    release_id: String,
    registry: String,
    namespace: String,
    name: String,
    version: String,
    release_digest: String,
    singleton: bool,
    runmat_requirement: Option<String>,
    features: BTreeMap<String, Vec<String>>,
    required_capabilities: Vec<String>,
    optional_capabilities: Vec<String>,
    readme_digest: Option<String>,
    license: Option<String>,
    dependencies: Vec<DependencyWire>,
    advisories: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DependencyWire {
    alias: String,
    registry: String,
    namespace: String,
    name: String,
    requirement: String,
    group: String,
    target: Option<String>,
    optional: bool,
    default_features: bool,
    features: Vec<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CandidateArtifactWire {
    #[serde(rename = "id")]
    _id: String,
    digest: String,
    tree_digest: String,
    #[serde(rename = "byteLen")]
    _byte_len: u64,
    #[serde(rename = "mediaType")]
    _media_type: String,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ArtifactWire {
    #[serde(rename = "id")]
    _id: String,
    digest: String,
    tree_digest: String,
    byte_len: u64,
    #[serde(rename = "mediaType")]
    _media_type: String,
    #[serde(rename = "downloadUrl")]
    _download_url: String,
    #[serde(rename = "expiresAt")]
    _expires_at: i64,
}

pub(super) async fn candidates(
    provider: &JsPackageSourceProvider,
    plan: &RegistryCandidatePlan,
) -> Result<Vec<RegistryCandidateRecord>, String> {
    if !plan.allow_network {
        return Err("registry candidates are unavailable without network access".to_string());
    }
    let plan_value = serde_wasm_bindgen::to_value(plan)
        .map_err(|error| format!("registry candidate plan serialization failed: {error}"))?;
    let value =
        shared::call_provider(&provider.bindings, "fetchRegistryCandidates", &plan_value).await?;
    let list: CandidateListWire = serde_wasm_bindgen::from_value(value)
        .map_err(|error| format!("registry candidate response parse failed: {error}"))?;
    list.candidates
        .into_iter()
        .map(|candidate| {
            let source = source(
                &plan.package,
                &candidate.release,
                &candidate.artifact.digest,
                &candidate.artifact.tree_digest,
            )?;
            let metadata = metadata(candidate.release)?;
            metadata.validate_source(&source)?;
            Ok(RegistryCandidateRecord {
                source,
                metadata,
                yanked: candidate.yanked,
            })
        })
        .collect()
}

pub(super) async fn acquire(
    provider: &JsPackageSourceProvider,
    plan: &RegistryAcquisitionPlan,
) -> Result<RegistryPackageMount, String> {
    let cache = shared::cache(provider)?;
    let mut snapshot = None;
    let mut release_metadata = None;
    if let Some(expected) = &plan.expected {
        match shared::lease_expected(provider, &cache, &expected.tree_digest).await {
            Ok(Some(lease)) => {
                match runmat_package_cache::load_registry_snapshot(&cache, expected.clone()).await {
                    Ok(cached) => {
                        provider.temporary_leases.borrow_mut().push(lease);
                        snapshot = Some(cached);
                    }
                    Err(CacheError::Miss(_)) => {
                        let _ = runmat_package_cache::release_lease(&cache, &lease, 16).await;
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
            Ok(None) => unreachable!("expected registry tree is a lease root"),
            Err(CacheError::Miss(_)) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    if snapshot.is_none() {
        if !plan.allow_network {
            return Err(format!(
                "package cache miss for registry package `{}` while network access is disabled",
                plan.package
            ));
        }
        let plan_value = serde_wasm_bindgen::to_value(plan)
            .map_err(|error| format!("registry acquisition plan serialization failed: {error}"))?;
        let transfer =
            shared::call_provider(&provider.bindings, "fetchRegistryRelease", &plan_value).await?;
        let release_value = js_sys::Reflect::get(&transfer, &JsValue::from_str("release"))
            .map_err(shared::js_error)?;
        let release: ResolveWire = serde_wasm_bindgen::from_value(release_value)
            .map_err(|error| format!("registry release response parse failed: {error}"))?;
        let bytes_value = js_sys::Reflect::get(&transfer, &JsValue::from_str("artifactBytes"))
            .map_err(shared::js_error)?;
        if !bytes_value.is_instance_of::<js_sys::Uint8Array>() {
            return Err("registry artifactBytes must be a Uint8Array".to_string());
        }
        let artifact_bytes = js_sys::Uint8Array::new(&bytes_value).to_vec();
        if artifact_bytes.len() as u64 != release.artifact.byte_len {
            return Err("registry artifact length differs from signed metadata".to_string());
        }
        let acquired_source = source(
            &plan.package,
            &release.release,
            &release.artifact.digest,
            &release.artifact.tree_digest,
        )?;
        let acquired_metadata = metadata(release.release)?;
        acquired_metadata.validate_source(&acquired_source)?;
        let acquired = runmat_package_cache::RegistryArtifactInventory::decode_snapshot(
            &artifact_bytes,
            acquired_source,
            runmat_package_cache::ArchiveLimits::default(),
        )
        .map_err(|error| error.to_string())?;
        runmat_package::validate_registry_acquisition(plan, &acquired.source)
            .map_err(|error| error.to_string())?;
        loop {
            let current = cache.snapshot().await.map_err(|error| error.to_string())?;
            let transaction = runmat_package_cache::cache_registry_snapshot(
                current.revision,
                current.state,
                &acquired,
                js_sys::Date::now().max(0.0) as u64,
            )
            .map_err(|error| error.to_string())?;
            match cache
                .commit(transaction)
                .await
                .map_err(|error| error.to_string())?
            {
                CommitOutcome::Committed(_) => break,
                CommitOutcome::Conflict { .. } => continue,
            }
        }
        release_metadata = Some(acquired_metadata);
        snapshot = Some(acquired);
    }
    let snapshot = snapshot.expect("cache hit or fetched registry snapshot");
    runmat_package::validate_registry_acquisition(plan, &snapshot.source)
        .map_err(|error| error.to_string())?;
    shared::retain_tree(provider, &cache, &snapshot.tree.digest).await?;
    let root = shared::mount(provider, &snapshot).await?;
    Ok(RegistryPackageMount {
        source: snapshot.source,
        root,
        metadata: release_metadata,
    })
}

fn source(
    package: &CanonicalPackageId,
    release: &ReleaseCoreWire,
    artifact_digest: &str,
    tree_digest: &str,
) -> Result<RegistrySourceId, String> {
    if release.namespace != package.organization() || release.name != package.name() {
        return Err("registry returned a different package identity".to_string());
    }
    Ok(RegistrySourceId {
        registry_origin: RegistryOrigin::from_str(&release.registry)
            .map_err(|error| error.to_string())?,
        package: package.clone(),
        release: RegistryReleaseId::from_str(&release.release_id)
            .map_err(|error| error.to_string())?,
        version: PackageVersion::from_str(&release.version).map_err(|error| error.to_string())?,
        release_digest: ContentDigest::from_str(&release.release_digest)
            .map_err(|error| error.to_string())?,
        artifact_digest: ContentDigest::from_str(artifact_digest)
            .map_err(|error| error.to_string())?,
        tree_digest: ContentDigest::from_str(tree_digest).map_err(|error| error.to_string())?,
    })
}

fn metadata(release: ReleaseCoreWire) -> Result<RegistryReleaseMetadata, String> {
    let _ = release.package_id;
    let _ = release.advisories;
    Ok(RegistryReleaseMetadata {
        singleton: release.singleton,
        runmat_requirement: release.runmat_requirement,
        dependencies: release
            .dependencies
            .into_iter()
            .map(|dependency| {
                Ok(RegistryReleaseDependency {
                    alias: dependency.alias,
                    package: RegistryPackageReference {
                        registry: RegistryOrigin::from_str(&dependency.registry)
                            .map_err(|error| error.to_string())?,
                        namespace: dependency.namespace,
                        name: dependency.name,
                    },
                    requirement: dependency.requirement,
                    group: match dependency.group.as_str() {
                        "runtime" => DependencyGroup::Runtime,
                        "development" => DependencyGroup::Development,
                        "test" => DependencyGroup::Test,
                        _ => return Err("registry dependency group is invalid".to_string()),
                    },
                    target: dependency.target,
                    optional: dependency.optional,
                    default_features: dependency.default_features,
                    features: dependency.features,
                })
            })
            .collect::<Result<_, String>>()?,
        features: release.features,
        required_capabilities: release.required_capabilities,
        optional_capabilities: release.optional_capabilities,
        readme_digest: release.readme_digest,
        license: release.license,
    })
}

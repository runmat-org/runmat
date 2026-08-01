mod cache;
mod provider;

pub use cache::{
    package_cache_acquire_lease, package_cache_gc, package_cache_release_lease,
    package_cache_renew_lease, package_cache_status,
};

use provider::JsPackageSourceProvider;
use wasm_bindgen::prelude::*;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GitPlanRequest {
    repository: String,
    selector: runmat_package::GitSelector,
    #[serde(default = "default_subdir")]
    subdir: String,
    locked_source: Option<runmat_package::GitSourceId>,
    intent: runmat_package::GitAcquisitionIntent,
    #[serde(default)]
    policy: runmat_package::GitAcquisitionPolicy,
}

fn default_subdir() -> String {
    ".".to_string()
}

#[wasm_bindgen(js_name = planGitAcquisition)]
pub fn plan_git_acquisition(value: JsValue) -> Result<JsValue, JsValue> {
    let request: GitPlanRequest = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("Git plan request parse failed: {error}")))?;
    let repository = runmat_package::GitRepositoryUrl::new(&request.repository)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let subdir = runmat_package::NormalizedRelativePath::new(&request.subdir)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let plan = runmat_package::plan_git_acquisition(
        repository,
        request.selector,
        subdir,
        request.locked_source.as_ref(),
        request.intent,
        request.policy,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&plan)
        .map_err(|error| JsValue::from_str(&format!("Git plan serialization failed: {error}")))
}

#[wasm_bindgen(js_name = buildGitSnapshot)]
pub fn build_git_snapshot(
    repository: &str,
    subdir: &str,
    value: JsValue,
) -> Result<JsValue, JsValue> {
    let inventory: runmat_package_cache::GitTreeInventory =
        serde_wasm_bindgen::from_value(value)
            .map_err(|error| JsValue::from_str(&format!("Git inventory parse failed: {error}")))?;
    let snapshot = inventory
        .into_snapshot(
            repository,
            subdir,
            runmat_package_cache::ArchiveLimits::default(),
        )
        .map_err(|error| JsValue::from_str(&format!("Git inventory validation failed: {error}")))?;
    serde_wasm_bindgen::to_value(&snapshot)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot serialization failed: {error}")))
}

#[wasm_bindgen(js_name = validateGitSnapshot)]
pub fn validate_git_snapshot(value: JsValue) -> Result<JsValue, JsValue> {
    let snapshot: runmat_package_cache::GitSnapshot = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot parse failed: {error}")))?;
    snapshot
        .validate()
        .map_err(|error| JsValue::from_str(&format!("Git snapshot validation failed: {error}")))?;
    serde_wasm_bindgen::to_value(&snapshot)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot serialization failed: {error}")))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProjectResolveRequest {
    manifest_path: String,
    existing_lock: Option<runmat_package::PackageLock>,
    options: runmat_package::ProjectResolveOptions,
}

#[wasm_bindgen(js_name = resolveProject)]
pub async fn resolve_project(
    request: JsValue,
    provider: JsValue,
    filesystem: JsValue,
) -> Result<JsValue, JsValue> {
    let request: ProjectResolveRequest =
        serde_wasm_bindgen::from_value(request).map_err(|error| {
            JsValue::from_str(&format!("project resolve request parse failed: {error}"))
        })?;
    let provider = JsPackageSourceProvider::new(provider)?;
    crate::runtime::filesystem::install_js_fs_provider(&filesystem).map_err(|error| {
        JsValue::from_str(&format!(
            "filesystem provider installation failed: {}",
            provider::shared::js_error(error)
        ))
    })?;
    let resolved = runmat_package::resolve_project_async(
        std::path::Path::new(&request.manifest_path),
        request.existing_lock.as_ref(),
        request.options,
        &provider,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let cache_bindings =
        js_sys::Reflect::get(&provider.bindings, &JsValue::from_str("packageCache"))?;
    let cache = crate::runtime::package_cache::JsPackageCacheBackend::new(&cache_bindings)?;
    let cached_trees = resolved
        .acquired_git_sources
        .iter()
        .map(|source| source.tree_digest.clone())
        .chain(
            resolved
                .acquired_server_sources
                .iter()
                .map(|source| source.tree_digest.clone()),
        )
        .chain(
            resolved
                .acquired_registry_sources
                .iter()
                .map(|source| source.tree_digest.clone()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    for inventory in &resolved.source_inventories {
        if cached_trees.contains(&inventory.tree_digest) {
            runmat_package_cache::publish_source_inventory(
                &cache,
                inventory,
                js_sys::Date::now().max(0.0) as u64,
                16,
            )
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        }
    }
    let cache_lease = cache::acquire_for_objects(
        &cache,
        format!("{}-graph", provider.lease_owner),
        provider.lease_owner.clone(),
        cached_trees,
        120_000,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let temporary_leases = provider.temporary_leases.take();
    for temporary in temporary_leases {
        runmat_package_cache::release_lease(&cache, &temporary, 16)
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    }
    #[derive(serde::Serialize)]
    struct BrowserResolveResult {
        #[serde(flatten)]
        resolved: runmat_package::ResolvedProject,
        cache_lease: Option<runmat_package_cache::Lease>,
    }
    serde_wasm_bindgen::to_value(&BrowserResolveResult {
        resolved,
        cache_lease,
    })
    .map_err(|error| JsValue::from_str(&format!("project result serialization failed: {error}")))
}

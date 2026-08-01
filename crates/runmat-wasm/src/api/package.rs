mod cache;

pub use cache::{
    package_cache_acquire_lease, package_cache_gc, package_cache_release_lease,
    package_cache_renew_lease, package_cache_status,
};

use runmat_package_cache::{CacheBackend, CacheError, CommitOutcome};
use std::cell::{Cell, RefCell};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

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

struct JsGitPackageProvider {
    bindings: JsValue,
    lease_owner: String,
    temporary_leases: RefCell<Vec<runmat_package_cache::Lease>>,
    next_lease: Cell<u32>,
}

impl JsGitPackageProvider {
    fn new(bindings: JsValue) -> Result<Self, JsValue> {
        let lease_owner = js_sys::Reflect::get(&bindings, &JsValue::from_str("leaseOwner"))?
            .as_string()
            .ok_or_else(|| JsValue::from_str("package provider leaseOwner must be a string"))?;
        runmat_package_cache::LeaseOwner::new(lease_owner.clone()).map_err(JsValue::from_str)?;
        Ok(Self {
            bindings,
            lease_owner,
            temporary_leases: RefCell::new(Vec::new()),
            next_lease: Cell::new(0),
        })
    }

    fn temporary_lease_id(&self) -> String {
        let next = self.next_lease.get();
        self.next_lease.set(next.saturating_add(1));
        format!("{}-acquire-{next}", self.lease_owner)
    }
}

impl runmat_package::GitPackageProvider for JsGitPackageProvider {
    fn acquire<'a>(
        &'a self,
        plan: &'a runmat_package::GitAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<runmat_package::GitPackageMount, String>> + 'a>,
    > {
        Box::pin(async move {
            let cache_bindings =
                js_sys::Reflect::get(&self.bindings, &JsValue::from_str("packageCache"))
                    .map_err(js_error)?;
            let cache = crate::runtime::package_cache::JsPackageCacheBackend::new(&cache_bindings)
                .map_err(js_error)?;
            let mut snapshot = None;
            if let Some(expected) = &plan.expected {
                match cache::acquire_for_objects(
                    &cache,
                    self.temporary_lease_id(),
                    self.lease_owner.clone(),
                    [expected.tree_digest.clone()].into_iter().collect(),
                    60_000,
                )
                .await
                {
                    Ok(Some(lease)) => {
                        match runmat_package_cache::load_git_snapshot(&cache, expected.clone())
                            .await
                        {
                            Ok(cached) => {
                                self.temporary_leases.borrow_mut().push(lease);
                                snapshot = Some(cached);
                            }
                            Err(CacheError::Miss(_)) => {
                                let _ =
                                    runmat_package_cache::release_lease(&cache, &lease, 16).await;
                            }
                            Err(error) => return Err(error.to_string()),
                        }
                    }
                    Ok(None) => unreachable!("expected Git tree is a lease root"),
                    Err(CacheError::Miss(_)) => {}
                    Err(error) => return Err(error.to_string()),
                }
            }
            if snapshot.is_none() {
                if !plan.allow_network {
                    return Err(format!(
                        "package cache miss for Git source `{}` while network access is disabled",
                        plan.repository
                    ));
                }
                let plan_value = serde_wasm_bindgen::to_value(plan).map_err(|error| {
                    format!("Git acquisition plan serialization failed: {error}")
                })?;
                let inventory_value =
                    call_provider(&self.bindings, "fetchGitInventory", &plan_value).await?;
                let inventory: runmat_package_cache::GitTreeInventory =
                    serde_wasm_bindgen::from_value(inventory_value)
                        .map_err(|error| format!("Git inventory parse failed: {error}"))?;
                let acquired = inventory
                    .into_snapshot(
                        plan.repository.as_str(),
                        plan.subdir.as_str(),
                        runmat_package_cache::ArchiveLimits::default(),
                    )
                    .map_err(|error| error.to_string())?;
                runmat_package::validate_git_acquisition(plan, &acquired.source)
                    .map_err(|error| error.to_string())?;
                loop {
                    let current = cache.snapshot().await.map_err(|error| error.to_string())?;
                    let transaction = runmat_package_cache::cache_git_snapshot(
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
                snapshot = Some(acquired);
            }
            let snapshot = snapshot.expect("cache hit or fetched snapshot");
            runmat_package::validate_git_acquisition(plan, &snapshot.source)
                .map_err(|error| error.to_string())?;
            if !self
                .temporary_leases
                .borrow()
                .iter()
                .any(|lease| lease.objects.contains(&snapshot.tree.digest))
            {
                let temporary = cache::acquire_for_objects(
                    &cache,
                    self.temporary_lease_id(),
                    self.lease_owner.clone(),
                    [snapshot.tree.digest.clone()].into_iter().collect(),
                    60_000,
                )
                .await
                .map_err(|error| error.to_string())?
                .expect("Git snapshot has one tree lease root");
                self.temporary_leases.borrow_mut().push(temporary);
            }
            let snapshot_value = serde_wasm_bindgen::to_value(&snapshot)
                .map_err(|error| format!("Git snapshot serialization failed: {error}"))?;
            let root = call_provider(&self.bindings, "mountGitSnapshot", &snapshot_value)
                .await?
                .as_string()
                .ok_or_else(|| {
                    "package provider mountGitSnapshot must return a root path".to_string()
                })?;
            Ok(runmat_package::GitPackageMount {
                source: snapshot.source,
                root: root.into(),
            })
        })
    }
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
    let provider = JsGitPackageProvider::new(provider)?;
    crate::runtime::filesystem::install_js_fs_provider(&filesystem).map_err(|error| {
        JsValue::from_str(&format!(
            "filesystem provider installation failed: {}",
            js_error(error)
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
    let git_trees = resolved
        .acquired_git_sources
        .iter()
        .map(|source| source.tree_digest.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for inventory in &resolved.source_inventories {
        if git_trees.contains(&inventory.tree_digest) {
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
        git_trees,
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

fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|message| message.as_string())
        })
        .unwrap_or_else(|| "JavaScript package provider failed".to_string())
}

async fn call_provider(provider: &JsValue, name: &str, input: &JsValue) -> Result<JsValue, String> {
    let function = js_sys::Reflect::get(provider, &JsValue::from_str(name))
        .map_err(js_error)?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| format!("package provider {name} must be a function"))?;
    let value = function.call1(provider, input).map_err(js_error)?;
    JsFuture::from(js_sys::Promise::resolve(&value))
        .await
        .map_err(js_error)
}

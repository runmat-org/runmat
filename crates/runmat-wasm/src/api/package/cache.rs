use runmat_package::ContentDigest;
use runmat_package_cache::{
    acquire_lease, release_lease, renew_lease, CacheBackend, CacheError, Lease, LeaseId, LeaseOwner,
};
use std::collections::BTreeSet;
use wasm_bindgen::prelude::*;

const TRANSACTION_RETRIES: usize = 16;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AcquireLeaseRequest {
    id: String,
    owner: String,
    objects: BTreeSet<ContentDigest>,
    ttl_ms: u64,
}

pub(super) async fn acquire_for_objects<B: CacheBackend>(
    backend: &B,
    id: String,
    owner: String,
    objects: BTreeSet<ContentDigest>,
    ttl_ms: u64,
) -> Result<Option<Lease>, CacheError> {
    if objects.is_empty() {
        return Ok(None);
    }
    acquire_lease(
        backend,
        LeaseId::new(id).map_err(|error| CacheError::Lease(error.to_string()))?,
        LeaseOwner::new(owner).map_err(|error| CacheError::Lease(error.to_string()))?,
        objects,
        now_ms(),
        ttl_ms,
        TRANSACTION_RETRIES,
    )
    .await
    .map(Some)
}

#[wasm_bindgen(js_name = packageCacheAcquireLease)]
pub async fn package_cache_acquire_lease(
    provider: JsValue,
    request: JsValue,
) -> Result<JsValue, JsValue> {
    let request: AcquireLeaseRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("lease request parse failed: {error}")))?;
    let backend = crate::runtime::package_cache::JsPackageCacheBackend::new(&provider)?;
    let lease = acquire_for_objects(
        &backend,
        request.id,
        request.owner,
        request.objects,
        request.ttl_ms,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?
    .ok_or_else(|| JsValue::from_str("lease request must contain at least one object"))?;
    serde_wasm_bindgen::to_value(&lease).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = packageCacheRenewLease)]
pub async fn package_cache_renew_lease(
    provider: JsValue,
    value: JsValue,
    ttl_ms: u64,
) -> Result<JsValue, JsValue> {
    let lease: Lease = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("lease parse failed: {error}")))?;
    let backend = crate::runtime::package_cache::JsPackageCacheBackend::new(&provider)?;
    let renewed = renew_lease(&backend, &lease, now_ms(), ttl_ms, TRANSACTION_RETRIES)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&renewed).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = packageCacheReleaseLease)]
pub async fn package_cache_release_lease(provider: JsValue, value: JsValue) -> Result<(), JsValue> {
    let lease: Lease = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("lease parse failed: {error}")))?;
    let backend = crate::runtime::package_cache::JsPackageCacheBackend::new(&provider)?;
    release_lease(&backend, &lease, TRANSACTION_RETRIES)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = packageCacheStatus)]
pub async fn package_cache_status(provider: JsValue) -> Result<JsValue, JsValue> {
    let backend = crate::runtime::package_cache::JsPackageCacheBackend::new(&provider)?;
    let snapshot = backend
        .snapshot()
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&runmat_package_cache::CacheStatus::from_state(
        &snapshot.state,
    ))
    .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = packageCacheGc)]
pub async fn package_cache_gc(
    provider: JsValue,
    target_bytes: u64,
    retain_recent_ms: u64,
) -> Result<JsValue, JsValue> {
    let backend = crate::runtime::package_cache::JsPackageCacheBackend::new(&provider)?;
    let plan = runmat_package_cache::execute_gc(
        &backend,
        runmat_package_cache::GcPolicy {
            now_ms: now_ms(),
            retain_recent_ms,
            target_bytes,
        },
        TRANSACTION_RETRIES,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&plan).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn now_ms() -> u64 {
    js_sys::Date::now().max(0.0) as u64
}

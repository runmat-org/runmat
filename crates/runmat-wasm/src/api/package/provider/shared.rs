use super::JsPackageSourceProvider;
use runmat_package::ContentDigest;
use runmat_package_cache::{CacheError, Lease};
use std::collections::BTreeSet;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

pub(super) fn cache(
    provider: &JsPackageSourceProvider,
) -> Result<crate::runtime::package_cache::JsPackageCacheBackend, String> {
    let bindings = js_sys::Reflect::get(&provider.bindings, &JsValue::from_str("packageCache"))
        .map_err(js_error)?;
    crate::runtime::package_cache::JsPackageCacheBackend::new(&bindings).map_err(js_error)
}

pub(super) async fn lease_expected(
    provider: &JsPackageSourceProvider,
    cache: &crate::runtime::package_cache::JsPackageCacheBackend,
    digest: &ContentDigest,
) -> Result<Option<Lease>, CacheError> {
    super::super::cache::acquire_for_objects(
        cache,
        provider.temporary_lease_id(),
        provider.lease_owner.clone(),
        [digest.clone()].into_iter().collect(),
        60_000,
    )
    .await
}

pub(super) async fn retain_tree(
    provider: &JsPackageSourceProvider,
    cache: &crate::runtime::package_cache::JsPackageCacheBackend,
    digest: &ContentDigest,
) -> Result<(), String> {
    if provider
        .temporary_leases
        .borrow()
        .iter()
        .any(|lease| lease.objects.contains(digest))
    {
        return Ok(());
    }
    let lease = super::super::cache::acquire_for_objects(
        cache,
        provider.temporary_lease_id(),
        provider.lease_owner.clone(),
        [digest.clone()].into_iter().collect::<BTreeSet<_>>(),
        60_000,
    )
    .await
    .map_err(|error| error.to_string())?
    .expect("package snapshot has one tree lease root");
    provider.temporary_leases.borrow_mut().push(lease);
    Ok(())
}

pub(super) async fn mount<T: serde::Serialize>(
    provider: &JsPackageSourceProvider,
    snapshot: &T,
) -> Result<std::path::PathBuf, String> {
    let value = serde_wasm_bindgen::to_value(snapshot)
        .map_err(|error| format!("package snapshot serialization failed: {error}"))?;
    call_provider(&provider.bindings, "mountPackageSnapshot", &value)
        .await?
        .as_string()
        .map(Into::into)
        .ok_or_else(|| "package provider mountPackageSnapshot must return a root path".to_string())
}

pub(super) async fn mount_private<T: serde::Serialize>(
    provider: &JsPackageSourceProvider,
    snapshot: &T,
) -> Result<std::path::PathBuf, String> {
    let value = serde_wasm_bindgen::to_value(snapshot)
        .map_err(|error| format!("private package snapshot serialization failed: {error}"))?;
    call_provider(&provider.bindings, "mountPrivatePackageSnapshot", &value)
        .await?
        .as_string()
        .map(Into::into)
        .ok_or_else(|| {
            "package provider mountPrivatePackageSnapshot must return a root path".to_string()
        })
}

pub(super) async fn call_provider(
    provider: &JsValue,
    name: &str,
    input: &JsValue,
) -> Result<JsValue, String> {
    let function = js_sys::Reflect::get(provider, &JsValue::from_str(name))
        .map_err(js_error)?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| format!("package provider {name} must be a function"))?;
    let value = function.call1(provider, input).map_err(js_error)?;
    JsFuture::from(js_sys::Promise::resolve(&value))
        .await
        .map_err(js_error)
}

pub(crate) fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|message| message.as_string())
        })
        .unwrap_or_else(|| "JavaScript package provider failed".to_string())
}

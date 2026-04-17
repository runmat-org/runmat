use runmat_accelerate::{initialize_acceleration_provider_with, AccelerateProviderPreference};
use runmat_accelerate_api::{provider, ProviderPrecision};
use serde::Serialize;
use wasm_bindgen::prelude::JsValue;
use wasm_bindgen::JsCast;

use crate::runtime::config::SessionConfig;
use crate::wire::errors::js_error;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GpuStatus {
    pub(crate) requested: bool,
    pub(crate) active: bool,
    pub(crate) error: Option<String>,
    pub(crate) adapter: Option<GpuAdapterInfo>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GpuAdapterInfo {
    pub(crate) device_id: u32,
    pub(crate) name: String,
    pub(crate) vendor: String,
    pub(crate) backend: Option<String>,
    pub(crate) memory_bytes: Option<u64>,
    pub(crate) precision: Option<String>,
}

pub(crate) fn install_cpu_provider(config: &SessionConfig) {
    let mut options = config.to_accel_options();
    options.enabled = false;
    options.provider = AccelerateProviderPreference::InProcess;
    initialize_acceleration_provider_with(&options);
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) async fn initialize_gpu_provider(config: &SessionConfig) -> Result<(), JsValue> {
    use runmat_accelerate::initialize_wgpu_provider_async;

    let options = config.to_accel_options();
    initialize_wgpu_provider_async(&options)
        .await
        .map_err(|err| js_error(&format!("Failed to initialize WebGPU provider: {err}")))
}

#[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
pub(crate) async fn initialize_gpu_provider(_config: &SessionConfig) -> Result<(), JsValue> {
    Err(js_error(
        "runmat-wasm was built without GPU support or on a non-wasm32 target",
    ))
}

pub(crate) fn capture_gpu_adapter_info() -> Option<GpuAdapterInfo> {
    let provider = provider()?;
    let info = provider.device_info_struct();
    let precision = provider_precision_label(provider.precision()).map(str::to_string);
    Some(GpuAdapterInfo {
        device_id: info.device_id,
        name: info.name,
        vendor: info.vendor,
        backend: info.backend,
        memory_bytes: info.memory_bytes,
        precision,
    })
}

fn provider_precision_label(precision: ProviderPrecision) -> Option<&'static str> {
    match precision {
        ProviderPrecision::F32 => Some("single"),
        ProviderPrecision::F64 => Some("double"),
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn capture_memory_usage() -> Result<(u64, u32), JsValue> {
    let memory = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .map_err(|_| js_error("Active wasm memory handle unavailable"))?;
    let buffer: js_sys::ArrayBuffer = memory
        .buffer()
        .dyn_into()
        .map_err(|_| js_error("Active wasm memory buffer unavailable"))?;
    let bytes = buffer.byte_length() as u64;
    let pages = (bytes / 65_536) as u32;
    Ok((bytes, pages))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn capture_memory_usage() -> Result<(u64, u32), JsValue> {
    Ok((0, 0))
}

use runmat_accelerate::{initialize_acceleration_provider_with, AccelerateProviderPreference};
use runmat_accelerate_api::{provider, ProviderPrecision};
use serde::Serialize;
use wasm_bindgen::prelude::JsValue;
use wasm_bindgen::JsCast;

use crate::runtime::config::SessionConfig;
use crate::wire::errors::js_error;

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserPhysicalPolicy {
    enabled: bool,
    provider: runmat_config::runtime::AccelerateProviderPreference,
    power: runmat_config::runtime::AccelPowerPreference,
    force_fallback: bool,
    allow_fallback: bool,
}

#[cfg(any(target_arch = "wasm32", test))]
impl From<&runmat_config::runtime::AccelerateConfig> for BrowserPhysicalPolicy {
    fn from(config: &runmat_config::runtime::AccelerateConfig) -> Self {
        Self {
            enabled: config.enabled,
            provider: config.provider,
            power: config.wgpu_power_preference,
            force_fallback: config.wgpu_force_fallback_adapter,
            allow_fallback: config.allow_inprocess_fallback,
        }
    }
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Clone, Copy)]
enum BrowserProviderState {
    Pending {
        policy: BrowserPhysicalPolicy,
    },
    Ready {
        policy: BrowserPhysicalPolicy,
        owner: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    },
}

#[cfg(any(target_arch = "wasm32", test))]
fn same_browser_owner(
    left: Option<&dyn runmat_accelerate_api::AccelProvider>,
    right: Option<&dyn runmat_accelerate_api::AccelProvider>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => runmat_accelerate_api::same_provider_instance(left, right),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl BrowserProviderState {
    fn claim(
        previous: Option<Self>,
        requested: BrowserPhysicalPolicy,
        current: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
        retained_foreign_wgpu: bool,
    ) -> Result<Self, String> {
        match previous {
            Some(Self::Pending { .. }) => Err("browser provider startup is pending or previously failed; await initialization or create a new Web Worker/WASM instance".into()),
            Some(Self::Ready { policy, owner }) if policy == requested && same_browser_owner(owner, current) => Ok(Self::Ready { policy, owner }),
            Some(Self::Ready { .. }) => Err("browser physical acceleration settings or provider owner changed; create a new Web Worker/WASM instance to apply them".into()),
            None if current.is_some() || retained_foreign_wgpu => Err("browser acceleration is already owned by another host; use a new Web Worker/WASM instance for these settings".into()),
            None => Ok(Self::Pending { policy: requested }),
        }
    }

    fn complete(
        self,
        current: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    ) -> Result<Self, String> {
        match self {
            Self::Pending { policy } => Ok(Self::Ready { policy, owner: current }),
            Self::Ready { policy, owner } if same_browser_owner(owner, current) => Ok(Self::Ready { policy, owner }),
            Self::Ready { .. } => Err("browser provider owner changed during initialization; create a new Web Worker/WASM instance".into()),
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static BROWSER_PROVIDER: std::cell::Cell<Option<BrowserProviderState>> = const { std::cell::Cell::new(None) };
}

/// Claim only the typed startup tuple, before logging/GC/provider effects. This
/// also closes concurrent async initialization: a different pending request
/// cannot silently inherit whichever physical adapter completes first.
#[cfg(target_arch = "wasm32")]
pub(crate) fn admit_browser_provider(config: &SessionConfig) -> Result<(), String> {
    let requested = BrowserPhysicalPolicy::from(&config.runtime.accelerate);
    BROWSER_PROVIDER.with(|slot| {
        #[cfg(feature = "gpu")]
        let retained_wgpu =
            runmat_accelerate::backend::wgpu::provider::registered_wgpu_provider().is_some();
        #[cfg(not(feature = "gpu"))]
        let retained_wgpu = false;
        slot.set(Some(BrowserProviderState::claim(
            slot.get(),
            requested,
            provider(),
            retained_wgpu,
        )?));
        Ok(())
    })
}

#[cfg(target_arch = "wasm32")]
fn remember_browser_provider() -> Result<(), JsValue> {
    BROWSER_PROVIDER.with(|slot| {
        let selected = slot
            .get()
            .ok_or_else(|| js_error("browser provider startup has no host claim"))?;
        slot.set(Some(
            selected
                .complete(provider())
                .map_err(|error| js_error(&error))?,
        ));
        Ok(())
    })
}

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

pub(crate) fn install_cpu_provider(config: &SessionConfig) -> Result<(), JsValue> {
    #[cfg(all(target_arch = "wasm32", feature = "gpu"))]
    if provider().is_some_and(|owner| owner.wgpu_device_access().is_some()) {
        return Err(js_error(
            "this browser instance retains a GPU owner; create a new worker to select CPU fallback",
        ));
    }
    let mut options = config.to_accel_options();
    options.enabled = false;
    // Explicitly disabled acceleration with no fallback needs no provider;
    // ordinary host computations remain legal. An explicit in-process choice
    // or permitted hardware failure fallback installs the CPU facade.
    if config.runtime.accelerate.enabled
        || options.allow_inprocess_fallback
        || options.provider == AccelerateProviderPreference::InProcess
    {
        options.provider = AccelerateProviderPreference::InProcess;
    }
    initialize_acceleration_provider_with(&options);
    #[cfg(target_arch = "wasm32")]
    remember_browser_provider()?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn validate_webgpu_runtime() -> Result<(), String> {
    fn get_required_member(target: &JsValue, member: &str) -> Result<JsValue, String> {
        js_sys::Reflect::get(target, &JsValue::from_str(member))
            .map_err(|_| format!("missing `{member}` binding"))
            .and_then(|value| {
                if value.is_null() || value.is_undefined() {
                    Err(format!("missing `{member}` binding"))
                } else {
                    Ok(value)
                }
            })
    }

    let global = js_sys::global();
    let navigator = get_required_member(&global, "navigator")?;
    let _gpu = get_required_member(&navigator, "gpu")?;
    let limits_ctor = get_required_member(&global, "GPUSupportedLimits")?;
    let prototype = get_required_member(&limits_ctor, "prototype")?;
    let required_limit_fields = ["maxInterStageShaderVariables"];
    for field in required_limit_fields {
        let has = js_sys::Reflect::has(&prototype, &JsValue::from_str(field)).unwrap_or(false);
        if !has {
            return Err(format!(
                "browser WebGPU limits API missing `{field}` support"
            ));
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn validate_webgpu_runtime() -> Result<(), String> {
    Ok(())
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) async fn initialize_gpu_provider(config: &SessionConfig) -> Result<(), JsValue> {
    use runmat_accelerate::backend::wgpu::provider::{
        register_wgpu_provider_async, WgpuProviderOptions,
    };
    // A previous allowed fallback stays CPU-owned. Never report that facade as
    // an active GPU merely because the API has some provider registered.
    if provider().is_some_and(|owner| owner.wgpu_device_access().is_none()) {
        return Err(js_error("this browser instance retained an in-process fallback; create a new worker to retry GPU initialization"));
    }
    let options = WgpuProviderOptions {
        power_preference: match config.wgpu_power_preference {
            runmat_accelerate::AccelPowerPreference::Auto
            | runmat_accelerate::AccelPowerPreference::HighPerformance => {
                wgpu::PowerPreference::HighPerformance
            }
            runmat_accelerate::AccelPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
        },
        force_fallback_adapter: config.wgpu_force_fallback_adapter,
        ..Default::default()
    };
    register_wgpu_provider_async(options)
        .await
        .map_err(|err| js_error(&format!("Failed to initialize WebGPU provider: {err}")))?;
    runmat_accelerate::configure_auto_offload(config.auto_offload.clone());
    remember_browser_provider()?;
    Ok(())
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

#[cfg(test)]
mod startup_policy_tests {
    use super::*;

    #[test]
    fn physical_owner_key_excludes_live_planner_policy_but_not_adapter_choices() {
        let mut config = runmat_config::runtime::AccelerateConfig::default();
        let initial = BrowserPhysicalPolicy::from(&config);
        config.auto_offload.enabled = false;
        config.auto_offload.calibrate = false;
        assert_eq!(initial, BrowserPhysicalPolicy::from(&config));
        config.wgpu_power_preference = runmat_config::runtime::AccelPowerPreference::LowPower;
        assert_ne!(initial, BrowserPhysicalPolicy::from(&config));
        config.wgpu_power_preference = runmat_config::runtime::AccelPowerPreference::Auto;
        config.wgpu_force_fallback_adapter = true;
        assert_ne!(initial, BrowserPhysicalPolicy::from(&config));

        // Exercise the actual startup transition functions without registering
        // an ambient provider or requiring hardware. GPU kind is established
        // by the real WGPU registration Result, not by this owner protocol.
        let owner: &'static dyn runmat_accelerate_api::AccelProvider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let pending = BrowserProviderState::claim(None, initial, None, false)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(pending, BrowserProviderState::Pending { .. }));
        assert!(BrowserProviderState::claim(Some(pending), initial, None, false).is_err());
        let ready = pending
            .complete(Some(owner))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(BrowserProviderState::claim(Some(ready), initial, Some(owner), true).is_ok());
        assert!(BrowserProviderState::claim(
            Some(ready),
            BrowserPhysicalPolicy::from(&config),
            Some(owner),
            true
        )
        .is_err());
        let other: &'static dyn runmat_accelerate_api::AccelProvider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        assert!(BrowserProviderState::claim(Some(ready), initial, Some(other), false).is_err());
        assert!(BrowserProviderState::claim(None, initial, None, true).is_err());
        let no_provider = pending
            .complete(None)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(BrowserProviderState::claim(Some(no_provider), initial, None, false).is_ok());
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

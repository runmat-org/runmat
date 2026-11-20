//! Measures the browser platform one FEA instance runs on. Admission arithmetic
//! and refusals live in `runmat_core`; this module only reads the platform.
use runmat_core::{BrowserMemoryPlatform, BrowserResourceLimit};
use wasm_bindgen::JsValue;

pub(in crate::api) fn measure() -> BrowserMemoryPlatform {
    BrowserMemoryPlatform {
        module_baseline_bytes: module_baseline_bytes(),
        device_memory_bytes: device_memory_bytes(),
    }
}

/// The memory budget of one instance of this module, the session worker's or a
/// fresh owned worker's.
pub(in crate::api) fn instance_budget_bytes() -> u64 {
    measure().instance_budget_bytes()
}

/// What this instance can still allocate.
pub(in crate::api) fn free_heap_bytes() -> u64 {
    measure().free_bytes(crate::runtime::heap::live_heap_bytes())
}

pub(in crate::api) fn limit_to_js(limit: &BrowserResourceLimit) -> JsValue {
    crate::wire::json_contract_to_js(&limit.diagnostic())
        .unwrap_or_else(|_| crate::wire::errors::js_error(&limit.message()))
}

pub(in crate::api) fn admit(result: Result<(), BrowserResourceLimit>) -> Result<(), JsValue> {
    result.map_err(|limit| limit_to_js(&limit))
}

#[cfg(target_arch = "wasm32")]
fn module_baseline_bytes() -> u64 {
    extern "C" {
        // Defined by the linker at the end of static data and the stack.
        static __heap_base: u8;
    }
    // Only the address of the linker symbol is taken; it is never read.
    std::ptr::addr_of!(__heap_base) as usize as u64
}

#[cfg(not(target_arch = "wasm32"))]
fn module_baseline_bytes() -> u64 {
    0
}

/// `navigator.deviceMemory` (GiB, rounded by the browser) when the browser reports it.
fn device_memory_bytes() -> Option<u64> {
    let navigator =
        js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator")).ok()?;
    let gib = js_sys::Reflect::get(&navigator, &JsValue::from_str("deviceMemory"))
        .ok()?
        .as_f64()?;
    (gib.is_finite() && gib > 0.0).then_some((gib * (1u64 << 30) as f64) as u64)
}

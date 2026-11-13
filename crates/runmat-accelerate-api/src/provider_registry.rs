//! Existing provider selection/address registry, process-wide on native and
//! worker-local on wasm. References follow the existing retained-façade ABI;
//! payload lifetime is independently controlled by allocation leases.
use crate::AccelProvider;
use std::collections::HashMap;

/// Compare retained provider facades under the registry's unique, stable
/// device-namespace and process/worker-lifetime registration contract.
///
/// Both the device namespace and facade data address must agree. Trait-object
/// vtable addresses are not identity: codegen may duplicate or deduplicate
/// vtables. Distinct facade objects are not equivalent merely because they
/// report the same namespace.
///
/// This does not register a provider, authorize execution, authenticate a
/// handle, or compare physical allocation leases. Callers must obtain the
/// actual registered owners and retain their normal handle/liveness checks.
pub fn same_provider_instance(left: &dyn AccelProvider, right: &dyn AccelProvider) -> bool {
    left.device_id() == right.device_id() && std::ptr::addr_eq(left, right)
}

#[derive(Default)]
struct Registry {
    default: Option<&'static dyn AccelProvider>,
    devices: HashMap<u32, &'static dyn AccelProvider>,
}

#[cfg(not(target_arch = "wasm32"))]
static REGISTRY: once_cell::sync::Lazy<std::sync::RwLock<Registry>> =
    once_cell::sync::Lazy::new(|| std::sync::RwLock::new(Registry::default()));
#[cfg(target_arch = "wasm32")]
thread_local! { static REGISTRY: std::cell::RefCell<Registry> = std::cell::RefCell::new(Registry::default()); }

fn read<T>(f: impl FnOnce(&Registry) -> T) -> Option<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        REGISTRY.read().ok().map(|registry| f(&registry))
    }
    #[cfg(target_arch = "wasm32")]
    {
        REGISTRY.with(|registry| registry.try_borrow().ok().map(|registry| f(&registry)))
    }
}
fn write(f: impl FnOnce(&mut Registry)) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(mut registry) = REGISTRY.write() {
        f(&mut registry);
    }
    #[cfg(target_arch = "wasm32")]
    REGISTRY.with(|registry| {
        if let Ok(mut registry) = registry.try_borrow_mut() {
            f(&mut registry);
        }
    });
}

pub(crate) fn default() -> Option<&'static dyn AccelProvider> {
    read(|registry| registry.default).flatten()
}
pub(crate) fn set_default(provider: Option<&'static dyn AccelProvider>) {
    write(|registry| registry.default = provider);
}
pub(crate) fn register(device: u32, provider: &'static dyn AccelProvider) {
    write(|registry| {
        registry.devices.insert(device, provider);
    });
}
pub(crate) fn get(device: u32) -> Option<&'static dyn AccelProvider> {
    read(|registry| registry.devices.get(&device).copied()).flatten()
}
pub(crate) fn all() -> Vec<&'static dyn AccelProvider> {
    read(|registry| {
        let mut devices: Vec<_> = registry
            .devices
            .iter()
            .map(|(&id, &provider)| (id, provider))
            .collect();
        devices.sort_unstable_by_key(|(id, _)| *id);
        devices.into_iter().map(|(_, provider)| provider).collect()
    })
    .unwrap_or_default()
}

/// Return the next registered owner in the same device-id order as `all`,
/// copying only the retained reference out of the registry lock. The caller
/// performs provider authorization after the lock has been released.
pub(crate) fn next_after(prior_device: Option<u32>) -> Option<(u32, &'static dyn AccelProvider)> {
    read(|registry| {
        registry
            .devices
            .iter()
            .filter(|(id, _)| prior_device.is_none_or(|prior| **id > prior))
            .min_by_key(|(id, _)| **id)
            .map(|(&id, &provider)| (id, provider))
    })
    .flatten()
}

#[cfg(test)]
mod tests;

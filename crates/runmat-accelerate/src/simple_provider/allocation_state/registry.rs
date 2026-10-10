use super::{F64Registry, IntegerRegistry, NumericAllocationState, NumericOwner};

#[cfg(not(target_arch = "wasm32"))]
static NUMERIC_REGISTRY: once_cell::sync::OnceCell<NumericOwner> = once_cell::sync::OnceCell::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    // One intentionally retained worker façade supplies stable guard borrows,
    // matching the provider's &'static ABI. Payloads are not leaked: final
    // allocation leases remove their records from this same map.
    static NUMERIC_REGISTRY: std::cell::OnceCell<&'static NumericOwner> = const { std::cell::OnceCell::new() };
}

pub(in crate::simple_provider) fn numeric_registry() -> &'static NumericOwner {
    #[cfg(not(target_arch = "wasm32"))]
    {
        NUMERIC_REGISTRY.get_or_init(|| NumericOwner::new(NumericAllocationState::default()))
    }
    #[cfg(target_arch = "wasm32")]
    {
        NUMERIC_REGISTRY.with(|registry| {
            *registry.get_or_init(|| {
                Box::leak(Box::new(NumericOwner::new(
                    NumericAllocationState::default(),
                )))
            })
        })
    }
}

pub(in crate::simple_provider) fn registry(device: u32) -> F64Registry {
    F64Registry { device }
}
pub(in crate::simple_provider) fn integer_registry(device: u32) -> IntegerRegistry {
    IntegerRegistry { device }
}

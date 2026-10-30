//! Invocation-owned restrictions; these never register, replace or reset a provider.
use runmat_accelerate_api::AccelProvider;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeProviderPreference {
    Auto,
    Wgpu,
    InProcess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeAccelerationPolicy {
    pub enabled: bool,
    pub provider: RuntimeProviderPreference,
    pub automatic_offload: bool,
    pub allow_host_fallback: bool,
    pub calibrate: bool,
}

impl RuntimeAccelerationPolicy {
    pub fn permits_provider(self, provider: &dyn AccelProvider) -> bool {
        self.enabled
            && match self.provider {
                RuntimeProviderPreference::InProcess => false,
                RuntimeProviderPreference::Wgpu => provider.wgpu_device_access().is_some(),
                RuntimeProviderPreference::Auto => true,
            }
    }
    pub fn automatic_offload_enabled(self) -> bool {
        self.enabled
            && self.provider != RuntimeProviderPreference::InProcess
            && self.automatic_offload
    }
}

/// Poll-local query of a narrow immutable projection, not ambient mutable policy.
pub fn current_acceleration_policy() -> Option<RuntimeAccelerationPolicy> {
    super::scope::with_active_runtime_context(|context| {
        context.and_then(|context| context.acceleration_policy())
    })
}

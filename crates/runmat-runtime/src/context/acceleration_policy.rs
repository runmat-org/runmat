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
        self.permits_provider_metadata(provider.wgpu_device_access().is_some())
    }

    /// The same predicate for a source-bound capability's validated owner
    /// metadata. This does not acquire or expose the provider's compute facade.
    pub fn permits_provider_metadata(self, has_wgpu_access: bool) -> bool {
        self.enabled
            && match self.provider {
                RuntimeProviderPreference::InProcess => false,
                RuntimeProviderPreference::Wgpu => has_wgpu_access,
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

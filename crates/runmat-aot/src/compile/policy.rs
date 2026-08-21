use crate::{archive::RuntimeArchiveCapabilities, AotError, AotResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompilationPolicy {
    NativeSpecialized,
    ClosedWorld,
    DynamicRuntime,
    Portable,
}

impl CompilationPolicy {
    pub fn validate(self, capabilities: &RuntimeArchiveCapabilities) -> AotResult<()> {
        match self {
            Self::NativeSpecialized
                if capabilities.static_program_calls && capabilities.runtime_builtins =>
            {
                Ok(())
            }
            Self::NativeSpecialized => Err(AotError::contract(
                "aot.compile.policy.native_specialized",
                "embedded runtime cannot execute target-native program calls and builtins",
            )),
            Self::ClosedWorld if capabilities.closed_world_linking => Ok(()),
            Self::ClosedWorld => Err(AotError::contract(
                "aot.compile.policy.closed_world",
                "closed-world compilation requires the R20 reachability-pruned runtime profile",
            )),
            Self::DynamicRuntime if capabilities.dynamic_source_loading => Ok(()),
            Self::DynamicRuntime => Err(AotError::contract(
                "aot.compile.policy.dynamic_runtime",
                "dynamic-runtime compilation requires an embedded frontend and dynamic source loader",
            )),
            Self::Portable => Err(AotError::contract(
                "aot.compile.policy.portable",
                "portable compilation produces a target-independent artifact rather than a host-linked executable and is not available in this workflow yet",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_profile_admits_native_and_closed_world_policies() {
        let capabilities = RuntimeArchiveCapabilities::standalone_host();
        assert!(CompilationPolicy::NativeSpecialized
            .validate(&capabilities)
            .is_ok());
        assert!(CompilationPolicy::ClosedWorld
            .validate(&capabilities)
            .is_ok());
        for policy in [
            CompilationPolicy::DynamicRuntime,
            CompilationPolicy::Portable,
        ] {
            assert!(policy.validate(&capabilities).is_err());
        }
    }
}

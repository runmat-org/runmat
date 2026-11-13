use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_execution::host::{
    NativeAbiProbe, NativeArchitecture, NativeCpuBaseline, NativeObjectFormat,
    NativeOperatingSystem, NativeTargetIdentity, SelectedNativeTarget,
    SELECTED_NATIVE_TARGET_SCHEMA_VERSION,
};

pub const PORTABLE_WASM_LOWERING_SCHEMA_VERSION: u16 = 1;

/// Versioned lowering domain. Portable browser IR has no native runtime ABI,
/// while native objects require an explicit target-built probe.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeTarget {
    Native {
        target: Box<SelectedNativeTarget>,
        probe: Box<NativeAbiProbe>,
    },
    PortableWasm32 {
        schema_version: u16,
    },
    /// A graceful current-host query result, never admissible Native IR.
    /// Carries no guessed architecture, platform, ABI, or layout facts.
    UnavailableHost,
}

impl NativeTarget {
    pub fn current() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self::PortableWasm32 {
                schema_version: PORTABLE_WASM_LOWERING_SCHEMA_VERSION,
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(probe) = runmat_runtime::native::try_native_abi_probe() else {
                return Self::UnavailableHost;
            };
            let baseline = match probe.architecture {
                NativeArchitecture::X86_64 => NativeCpuBaseline::X86_64V1,
                NativeArchitecture::Aarch64 => NativeCpuBaseline::ArmV8A,
            };
            Self::from_probe(probe, baseline).unwrap_or(Self::UnavailableHost)
        }
    }

    pub fn from_probe(
        probe: NativeAbiProbe,
        baseline: NativeCpuBaseline,
    ) -> NativeCodegenResult<Self> {
        let binding = probe.binding().map_err(contract)?;
        let format = match probe.operating_system {
            NativeOperatingSystem::Linux => NativeObjectFormat::Elf,
            NativeOperatingSystem::Macos => NativeObjectFormat::MachO,
            NativeOperatingSystem::Windows => NativeObjectFormat::Coff,
        };
        let target = SelectedNativeTarget {
            schema_version: SELECTED_NATIVE_TARGET_SCHEMA_VERSION,
            target: NativeTargetIdentity::new(
                probe.architecture,
                probe.operating_system,
                probe.pointer_width,
                binding.identity().map_err(contract)?,
                format,
            )
            .map_err(contract)?,
            cpu_baseline: baseline,
            endianness: probe.endianness,
            environment: probe.environment,
        };
        let result = Self::Native {
            target: Box::new(target),
            probe: Box::new(probe),
        };
        result.validate()?;
        Ok(result)
    }

    /// Structural/object admission does not require the build host's layout.
    /// It does not authenticate probe provenance; selected-pack byte admission
    /// remains a separate required join before deployment/link orchestration.
    pub fn validate(&self) -> NativeCodegenResult<()> {
        match self {
            Self::PortableWasm32 { schema_version }
                if *schema_version == PORTABLE_WASM_LOWERING_SCHEMA_VERSION =>
            {
                Ok(())
            }
            Self::PortableWasm32 { .. } => Err(NativeCodegenError::new(
                "native.target.wasm_schema",
                "unsupported portable WASM lowering schema",
            )),
            Self::UnavailableHost => Err(NativeCodegenError::new(
                "native.target.host_unavailable",
                "current host has no supported native ABI probe/baseline",
            )),
            Self::Native { target, probe } => {
                target.validate().map_err(contract)?;
                let binding = probe.binding().map_err(contract)?;
                if binding.schema_version != runmat_runtime::native::NATIVE_ABI_SCHEMA_VERSION
                    || binding.encoded_version
                        != runmat_runtime::native::NATIVE_ABI_VERSION.encoded()
                    || binding.contract_fingerprint
                        != runmat_runtime::native::native_abi_contract_fingerprint()
                    || binding.identity().map_err(contract)? != target.target.abi
                    || probe.architecture != target.target.architecture
                    || probe.operating_system != target.target.operating_system
                    || probe.environment != target.environment
                    || probe.endianness != target.endianness
                    || probe.pointer_width != target.target.pointer_width
                {
                    return Err(NativeCodegenError::new(
                        "native.target.abi",
                        "selected target and current ABI contract/target probe disagree",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Must precede creating executable memory or invoking a generated entry.
    pub fn require_host_executable(&self) -> NativeCodegenResult<()> {
        self.native_parts()?;
        if self != &Self::current() {
            return Err(NativeCodegenError::new(
                "native.target.host_execution",
                "native execution requires the exact current runtime target and probe",
            ));
        }
        Ok(())
    }
    pub fn native_parts(&self) -> NativeCodegenResult<(&SelectedNativeTarget, &NativeAbiProbe)> {
        self.validate()?;
        match self {
            Self::Native { target, probe } => Ok((target, probe)),
            Self::PortableWasm32 { .. } => Err(NativeCodegenError::new(
                "native.target.native_required",
                "portable WASM lowering metadata cannot produce a native object or executable",
            )),
            Self::UnavailableHost => Err(NativeCodegenError::new(
                "native.target.host_unavailable",
                "current host has no supported native ABI probe/baseline",
            )),
        }
    }
    pub fn execution_identity(&self) -> NativeCodegenResult<NativeTargetIdentity> {
        Ok(self.native_parts()?.0.target.clone())
    }
    /// Native IR target/probe identity, not a key for host-tuned machine code.
    /// Process-local JIT owners are not serialized as portable object products.
    pub fn cache_key(
        &self,
        executable_cache_key: &runmat_execution::Digest,
        selected_entrypoint: runmat_types::FunctionId,
        selected_entrypoint_body: &runmat_execution::Digest,
    ) -> NativeCodegenResult<runmat_execution::Digest> {
        self.validate()?;
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "runmat-native-ir-target-v15",
            crate::NATIVE_IR_SCHEMA_VERSION,
            executable_cache_key,
            selected_entrypoint.0,
            selected_entrypoint_body,
            self,
        ))
        .expect("native target identity serializes"))
    }
}
fn contract(error: runmat_execution::ContractError) -> NativeCodegenError {
    NativeCodegenError::new("native.target.contract", error.to_string())
}

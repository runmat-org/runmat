use crate::{abi::NativeAbiBinding, NativeCodegenError, NativeCodegenResult};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTarget {
    pub architecture: String,
    pub operating_system: String,
    pub pointer_width: u16,
    pub abi: NativeAbiBinding,
}

impl NativeTarget {
    pub fn current() -> Self {
        Self {
            architecture: std::env::consts::ARCH.to_string(),
            operating_system: current_operating_system().to_string(),
            pointer_width: usize::BITS as u16,
            abi: NativeAbiBinding::current(),
        }
    }

    pub fn validate(&self) -> NativeCodegenResult<()> {
        if !valid_target_component(&self.architecture)
            || !valid_target_component(&self.operating_system)
        {
            return Err(NativeCodegenError::new(
                "native.target.identity",
                "target architecture and operating system must be bounded printable ASCII",
            ));
        }
        if !matches!(self.pointer_width, 32 | 64) {
            return Err(NativeCodegenError::new(
                "native.target.pointer_width",
                "only 32-bit and 64-bit pointer targets are supported",
            ));
        }
        if self.architecture == std::env::consts::ARCH
            && self.operating_system == current_operating_system()
            && self.pointer_width == usize::BITS as u16
        {
            self.abi.validate()
        } else {
            Err(NativeCodegenError::new(
                "native.target.cross_layout",
                "cross-target Native IR requires a target-probed runtime ABI layout binding",
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn execution_identity(
        &self,
    ) -> NativeCodegenResult<runmat_execution::host::NativeTargetIdentity> {
        self.validate()?;
        let architecture = match self.architecture.as_str() {
            "x86_64" => runmat_execution::host::NativeArchitecture::X86_64,
            "aarch64" => runmat_execution::host::NativeArchitecture::Aarch64,
            value => {
                return Err(NativeCodegenError::new(
                    "native.target.architecture",
                    format!("unsupported execution architecture {value}"),
                ))
            }
        };
        let (operating_system, object_format) = match self.operating_system.as_str() {
            "macos" => (
                runmat_execution::host::NativeOperatingSystem::Macos,
                runmat_execution::host::NativeObjectFormat::MachO,
            ),
            "linux" => (
                runmat_execution::host::NativeOperatingSystem::Linux,
                runmat_execution::host::NativeObjectFormat::Elf,
            ),
            "windows" => (
                runmat_execution::host::NativeOperatingSystem::Windows,
                runmat_execution::host::NativeObjectFormat::Coff,
            ),
            value => {
                return Err(NativeCodegenError::new(
                    "native.target.operating_system",
                    format!("unsupported execution operating system {value}"),
                ))
            }
        };
        let abi = runmat_execution::host::NativeAbi::new(format!(
            "{}:{}:{}:{}",
            self.abi.schema_version,
            self.abi.encoded_version,
            self.abi.contract_fingerprint,
            self.abi.layout_fingerprint
        ))
        .map_err(|error| NativeCodegenError::new("native.target.abi", error.to_string()))?;
        runmat_execution::host::NativeTargetIdentity::new(
            architecture,
            operating_system,
            self.pointer_width,
            abi,
            object_format,
        )
        .map_err(|error| NativeCodegenError::new("native.target.identity", error.to_string()))
    }

    pub fn cache_key(
        &self,
        executable_cache_key: &runmat_execution::Digest,
    ) -> NativeCodegenResult<runmat_execution::Digest> {
        self.validate()?;
        Ok(runmat_execution::Digest::sha256(format!(
            "runmat-native-ir-v{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}",
            crate::NATIVE_IR_SCHEMA_VERSION,
            executable_cache_key,
            self.architecture,
            self.operating_system,
            self.pointer_width,
            self.abi.schema_version,
            self.abi.encoded_version,
            self.abi.contract_fingerprint,
            self.abi.layout_fingerprint,
        )))
    }
}

fn current_operating_system() -> &'static str {
    if std::env::consts::OS.is_empty() {
        "unknown"
    } else {
        std::env::consts::OS
    }
}

fn valid_target_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

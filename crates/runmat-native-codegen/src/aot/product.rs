use runmat_execution::{Digest, ExecutableIdentity};
use runmat_types::{ProgramFunctionId, ProgramSourceId};

use crate::NativeTarget;

pub const NATIVE_OBJECT_SCHEMA_VERSION: u16 = 2;
pub const AOT_PROGRAM_MANIFEST_SCHEMA_VERSION: u16 = 2;
const MAX_AOT_PROGRAM_FUNCTIONS: usize = 100_000;
pub const AOT_ENTRY_SYMBOL: &str = "runmat_aot_entry";
pub const AOT_RUNTIME_MAIN_SYMBOL: &str = "runmat_aot_main";
pub const AOT_NATIVE_IR_SYMBOL: &str = "runmat_aot_native_ir";
pub const AOT_PROGRAM_SYMBOL: &str = "runmat_aot_program";
pub const AOT_RESUME_POINTS_SYMBOL: &str = "runmat_aot_resume_points";

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AotRuntimeBindingMode {
    Dynamic,
    ClosedWorld,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotBuiltinBinding {
    pub name: String,
    pub variant: String,
    pub native_symbol: String,
}

impl AotBuiltinBinding {
    fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        if !valid_identity(&self.name, 512)
            || !valid_identity(&self.variant, 256)
            || !valid_link_symbol(&self.native_symbol)
        {
            return Err(crate::NativeCodegenError::new(
                "native.aot.builtin_binding",
                "AOT builtin binding identity or native symbol is invalid",
            ));
        }
        Ok(())
    }
}

/// Frontend- and VM-independent callable inventory embedded in a native AOT
/// object. The standalone host needs names and source ownership for exact
/// runtime dispatch, but never bytecode or interpreter frame metadata.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotProgramManifest {
    pub schema_version: u16,
    pub executable: ExecutableIdentity,
    pub native_ir_digest: Digest,
    pub functions: Vec<AotProgramFunction>,
    pub runtime_binding_mode: AotRuntimeBindingMode,
    pub builtin_bindings: Vec<AotBuiltinBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotProgramFunction {
    pub function: ProgramFunctionId,
    pub source: ProgramSourceId,
    pub name: String,
}

impl AotProgramManifest {
    pub fn from_assembly(
        assembly: &crate::NativeAssembly,
        native_ir_digest: Digest,
        runtime_binding_mode: AotRuntimeBindingMode,
        mut builtin_bindings: Vec<AotBuiltinBinding>,
    ) -> Result<Self, crate::NativeCodegenError> {
        let mut functions = assembly
            .functions
            .iter()
            .map(|function| AotProgramFunction {
                function: function.id,
                source: function.source,
                name: function.name.clone(),
            })
            .collect::<Vec<_>>();
        functions.sort_by_key(|function| function.function);
        builtin_bindings
            .sort_by(|left, right| (&left.name, &left.variant).cmp(&(&right.name, &right.variant)));
        let manifest = Self {
            schema_version: AOT_PROGRAM_MANIFEST_SCHEMA_VERSION,
            executable: assembly.executable_identity.clone(),
            native_ir_digest,
            functions,
            runtime_binding_mode,
            builtin_bindings,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        self.executable.program.validate().map_err(|error| {
            crate::NativeCodegenError::new("native.aot.program", error.to_string())
        })?;
        if self.schema_version != AOT_PROGRAM_MANIFEST_SCHEMA_VERSION
            || !valid_identity(&self.executable.root_package, 256)
            || !valid_identity(&self.executable.entrypoint, 512)
            || self.functions.is_empty()
            || self.functions.len() > MAX_AOT_PROGRAM_FUNCTIONS
            || self
                .functions
                .windows(2)
                .any(|pair| pair[0].function >= pair[1].function)
            || !self
                .functions
                .iter()
                .any(|function| function.function == self.executable.entrypoint_function)
            || self.functions.iter().any(|function| {
                function.name.is_empty()
                    || function.name.len() > 512
                    || function.name.chars().any(char::is_control)
            })
            || self
                .builtin_bindings
                .iter()
                .any(|binding| binding.validate().is_err())
            || self
                .builtin_bindings
                .windows(2)
                .any(|pair| (&pair[0].name, &pair[0].variant) >= (&pair[1].name, &pair[1].variant))
            || (self.runtime_binding_mode == AotRuntimeBindingMode::Dynamic
                && !self.builtin_bindings.is_empty())
        {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program",
                "AOT program manifest is not canonical",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, crate::NativeCodegenError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to encode AOT program manifest: {error}"),
            )
        })
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, crate::NativeCodegenError> {
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to decode AOT program manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        if manifest.canonical_bytes()? != bytes {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program",
                "AOT program manifest encoding is not canonical",
            ));
        }
        Ok(manifest)
    }

    pub fn resolve_name(&self, name: &str) -> Option<ProgramFunctionId> {
        self.functions
            .iter()
            .find(|function| function.name == name)
            .map(|function| function.function)
    }
}

fn valid_identity(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeObjectFormat {
    Elf,
    MachO,
    Coff,
}

impl NativeObjectFormat {
    pub fn for_target(target: &NativeTarget) -> Result<Self, crate::NativeCodegenError> {
        match target.operating_system.as_str() {
            "linux" => Ok(Self::Elf),
            "macos" => Ok(Self::MachO),
            "windows" => Ok(Self::Coff),
            _ => Err(crate::NativeCodegenError::new(
                "native.object.format",
                "target does not have a supported native object format",
            )),
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Self::Elf => "elf",
            Self::MachO => "mach-o",
            Self::Coff => "coff",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOptimization {
    None,
    Speed,
    SpeedAndSize,
}

impl NativeOptimization {
    #[cfg(feature = "compiler")]
    pub(super) fn cranelift_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Speed => "speed",
            Self::SpeedAndSize => "speed_and_size",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectFunction {
    pub function: ProgramFunctionId,
    pub symbol: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObjectData {
    pub symbol: String,
    pub bytes: Vec<u8>,
    pub alignment: u64,
}

pub fn embedded_blob(
    symbol: &str,
    bytes: Vec<u8>,
    alignment: u64,
) -> Result<NativeObjectData, crate::NativeCodegenError> {
    if !valid_data_symbol(symbol) || bytes.is_empty() {
        return Err(crate::NativeCodegenError::new(
            "native.object.data",
            "embedded AOT blob has an invalid symbol or empty payload",
        ));
    }
    Ok(NativeObjectData {
        symbol: symbol.to_string(),
        bytes,
        alignment,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectDataDescriptor {
    pub symbol: String,
    pub digest: Digest,
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectManifest {
    pub schema_version: u16,
    pub target: NativeTarget,
    pub object_format: NativeObjectFormat,
    pub executable_cache_key: Digest,
    pub native_cache_key: Digest,
    pub runtime_fingerprint: Digest,
    pub catalog_fingerprint: Digest,
    pub optimization: NativeOptimization,
    pub runtime_binding_mode: AotRuntimeBindingMode,
    pub object_digest: Digest,
    pub object_bytes: u64,
    pub entrypoint: ProgramFunctionId,
    pub functions: Vec<NativeObjectFunction>,
    pub data: Vec<NativeObjectDataDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelocatableNativeObject {
    pub manifest: NativeObjectManifest,
    pub bytes: Vec<u8>,
}

impl RelocatableNativeObject {
    pub fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        if self.manifest.schema_version != NATIVE_OBJECT_SCHEMA_VERSION {
            return Err(crate::NativeCodegenError::new(
                "native.object.schema",
                "native object manifest schema is unsupported",
            ));
        }
        self.manifest.target.validate()?;
        if self.manifest.object_format != NativeObjectFormat::for_target(&self.manifest.target)? {
            return Err(crate::NativeCodegenError::new(
                "native.object.format",
                "native object format does not match its target",
            ));
        }
        let object_bytes = u64::try_from(self.bytes.len()).map_err(|_| {
            crate::NativeCodegenError::new(
                "native.object.size",
                "native object size exceeds the portable manifest limit",
            )
        })?;
        if self
            .manifest
            .runtime_fingerprint
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
            || self
                .manifest
                .catalog_fingerprint
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
        {
            return Err(crate::NativeCodegenError::new(
                "native.object.environment",
                "native object runtime and catalog identities must be present",
            ));
        }
        if self.bytes.is_empty()
            || self.manifest.object_bytes != object_bytes
            || self.manifest.object_digest != Digest::sha256(&self.bytes)
            || self.manifest.functions.is_empty()
            || !self
                .manifest
                .functions
                .iter()
                .any(|function| function.function == self.manifest.entrypoint)
            || self
                .manifest
                .functions
                .windows(2)
                .any(|pair| pair[0].function >= pair[1].function)
            || self.manifest.functions.iter().any(|function| {
                function.symbol != function_symbol(function.function, self.manifest.entrypoint)
            })
            || self
                .manifest
                .data
                .windows(2)
                .any(|pair| pair[0].symbol >= pair[1].symbol)
            || self.manifest.data.iter().any(|data| {
                !valid_data_symbol(&data.symbol)
                    || data.bytes == 0
                    || !matches!(data.alignment, 1 | 2 | 4 | 8 | 16)
            })
        {
            return Err(crate::NativeCodegenError::new(
                "native.object.identity",
                "native object does not match its canonical manifest",
            ));
        }
        Ok(())
    }
}

pub(super) fn valid_data_symbol(symbol: &str) -> bool {
    symbol.starts_with("runmat_aot_")
        && symbol.len() <= 128
        && symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(super) fn function_symbol(
    function: ProgramFunctionId,
    entrypoint: ProgramFunctionId,
) -> String {
    if function == entrypoint {
        AOT_ENTRY_SYMBOL.to_string()
    } else {
        format!("runmat_native_f{}", function.0)
    }
}

fn valid_link_symbol(symbol: &str) -> bool {
    symbol.starts_with("runmat_")
        && symbol.len() <= 2_048
        && symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

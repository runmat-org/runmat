use runmat_execution::Digest;
use runmat_types::ProgramFunctionId;

use crate::NativeTarget;

pub const NATIVE_OBJECT_SCHEMA_VERSION: u16 = 1;
pub const AOT_ENTRY_SYMBOL: &str = "runmat_aot_entry";
pub const AOT_RUNTIME_MAIN_SYMBOL: &str = "runmat_aot_main";
pub const AOT_NATIVE_IR_SYMBOL: &str = "runmat_aot_native_ir";
pub const AOT_PROGRAM_SYMBOL: &str = "runmat_aot_program";
pub const AOT_RESUME_POINTS_SYMBOL: &str = "runmat_aot_resume_points";

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
                function.symbol
                    != super::object::function_symbol(function.function, self.manifest.entrypoint)
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

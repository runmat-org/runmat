use super::NativeFunction;
use crate::NativeTarget;
use runmat_execution::{Digest, ExecutableIdentity, ProgramRevision};
use runmat_types::{CapabilitySet, InteropManifest, ParallelManifest, RegionContract};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRequirements {
    pub capabilities: CapabilitySet,
    pub regions: Vec<RegionContract>,
    pub interop: InteropManifest,
    pub parallel: ParallelManifest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAssembly {
    pub schema_version: u16,
    pub executable_identity: ExecutableIdentity,
    pub program: ProgramRevision,
    pub executable_cache_key: Digest,
    pub native_cache_key: Digest,
    pub target: NativeTarget,
    pub requirements: NativeRequirements,
    pub entrypoints: Vec<runmat_types::ProgramFunctionId>,
    pub functions: Vec<NativeFunction>,
}

const MAX_NATIVE_IR_BYTES: usize = 256 * 1024 * 1024;

#[derive(Deserialize)]
struct NativeAssemblyAdmission {
    schema_version: u16,
}

impl NativeAssembly {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, crate::NativeCodegenError> {
        self.verify()?;
        let value = serde_json::to_value(self).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })?;
        serde_json::to_vec(&value).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })
    }

    /// Admit the Native IR revision before decoding the revision-sensitive
    /// function and instruction representation.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, crate::NativeCodegenError> {
        if bytes.len() > MAX_NATIVE_IR_BYTES {
            return Err(crate::NativeCodegenError::new(
                "native.ir.bounds",
                format!("Native IR exceeds its {MAX_NATIVE_IR_BYTES}-byte bound"),
            ));
        }
        let admission: NativeAssemblyAdmission =
            serde_json::from_slice(bytes).map_err(|error| {
                crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
            })?;
        if admission.schema_version != crate::NATIVE_IR_SCHEMA_VERSION {
            return Err(crate::NativeCodegenError::new(
                "native.ir.schema_version",
                format!(
                    "unsupported Native IR schema version {}; expected {}. Rebuild the program with this RunMat version",
                    admission.schema_version,
                    crate::NATIVE_IR_SCHEMA_VERSION
                ),
            ));
        }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })?;
        let canonical = serde_json::to_vec(&value).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })?;
        if canonical != bytes {
            return Err(crate::NativeCodegenError::new(
                "native.ir.encoding",
                "Native IR encoding is valid JSON but not canonical RunMat JSON",
            ));
        }
        let assembly: Self = serde_json::from_value(value.clone()).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })?;
        let normalized = serde_json::to_value(&assembly).map_err(|error| {
            crate::NativeCodegenError::new("native.ir.encoding", error.to_string())
        })?;
        if normalized != value {
            return Err(crate::NativeCodegenError::new(
                "native.ir.encoding",
                "Native IR omits or aliases fields from the current canonical representation",
            ));
        }
        assembly.verify()?;
        Ok(assembly)
    }
}

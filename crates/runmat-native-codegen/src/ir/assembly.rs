use super::NativeFunction;
use crate::NativeTarget;
use runmat_execution::{Digest, ExecutableIdentity, ProgramRevision};
use runmat_types::{CapabilitySet, EffectSet, InteropManifest, ParallelManifest, RegionContract};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
    /// Digest of the admitted AnalysisStore component from which operation
    /// contracts were projected. This is checked against the executable
    /// manifest; it is not self-issued Native IR authority.
    pub analysis_component: Digest,
    /// Canonical digest of the exact admitted AnalysisStore projection used
    /// for the retained Native functions. Consumers must verify this field
    /// against the independently owned AnalysisStore; it is not authority by
    /// itself.
    pub analysis_projection: Digest,
    pub target: NativeTarget,
    pub requirements: NativeRequirements,
    /// Compiler/session identity selected for this product. Portable artifact
    /// identity remains in `executable_identity.entrypoint_function`.
    pub selected_entrypoint: runmat_types::FunctionId,
    pub selected_portable_entrypoint: runmat_types::ProgramFunctionId,
    pub selected_entrypoint_body: Digest,
    pub entrypoints: Vec<runmat_types::FunctionId>,
    pub functions: Vec<NativeFunction>,
    /// Immutable class parent and constructor declarations lowered from the
    /// same canonical MIR product as these functions.
    pub class_declarations: Vec<super::NativeClassDeclaration>,
}

const MAX_NATIVE_IR_BYTES: usize = 256 * 1024 * 1024;

#[derive(Deserialize)]
struct NativeAssemblyAdmission {
    schema_version: u16,
}

impl NativeAssembly {
    /// Exact retained Native IR effect demand used by selection and execution
    /// validation. The CLI must not rediscover this semantic inventory.
    pub fn required_effects(&self) -> EffectSet {
        let mut effects = BTreeSet::new();
        for instruction in self
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
        {
            effects.extend(instruction.effects.0.iter().copied());
        }
        for terminator in self
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .map(|block| &block.terminator)
        {
            effects.extend(terminator.effects.0.iter().copied());
        }
        for region in &self.requirements.regions {
            effects.extend(region.effects.0.iter().copied());
        }
        for region in &self.requirements.parallel.parfor_regions {
            effects.extend(region.effects.0.iter().copied());
        }
        for region in &self.requirements.parallel.spmd_regions {
            effects.extend(region.effects.0.iter().copied());
        }
        EffectSet(effects)
    }

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

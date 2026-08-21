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

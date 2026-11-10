//! Portable root contract for a whole exact-CAD structural study.
//!
//! The root is a driver-owned orchestration request. Workers receive only the immutable exact
//! meshing and analysis host programs selected by the driver; they never schedule nested work.

mod contract;
mod publication;
mod region_resolution;

#[cfg(all(test, feature = "occt-native"))]
pub(crate) use publication::assert_foreign_thermal_substitution_rejected;

pub(crate) use contract::structural_workload_inventory_digest;
pub use contract::{
    decode_exact_structural_study_root, prepare_exact_structural_study_root,
    DecodedExactStructuralStudyRoot, ExactStructuralStudyRootPreparation,
    PreparedExactStructuralStudyRoot, RequiredStudyProgram,
};
pub use publication::{
    import_exact_structural_study_result, import_structural_study_result,
    ExactStructuralBucklingAnalysisResult, ExactStructuralFrequencyAnalysisResult,
    ExactStructuralModalResponseSpectrumAnalysisResult,
    ExactStructuralModalTransientAnalysisResult, ExactStructuralPrestressedFrequencyAnalysisResult,
    ExactStructuralPrestressedModalAnalysisResult, ExactStructuralRandomResponseAnalysisResult,
    ExactStructuralStudyAnalysisResult, ExactStructuralStudyResult,
    ExactStructuralStudyResultExpectation,
};
pub(crate) use publication::{
    package_exact_structural_study_result, ExactStructuralAnalysisPublication,
};
pub(crate) use region_resolution::prepare_topology_resolution_objects;
pub use region_resolution::TopologyRegionResolution;

use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;

use crate::{NativeExecutionError, NativeExecutionResult};

pub const EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE: &str = "exact-structural-study-root";
pub const EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE: &str = "portable-exact-structural-study-root";

const ROOT_PROGRAM_BYTES: &[u8] = b"runmat-exact-structural-study-root\0";
const ROOT_PREFIX: &[u8] = b"runmat-exact-structural-study-root/v1\0";
const ROOT_DOMAIN: &str = "exact-structural-study-root";
const ROOT_MEDIA_TYPE: &str = "application/vnd.runmat.exact-structural-study-root+cbor";
const ROOT_VALUE_SCHEMA: &str = "runmat.analysis.exact-structural-study-root";
const PUBLICATION_PREFIX: &[u8] = b"runmat-exact-structural-study-publication/v3\0";
const PUBLICATION_DOMAIN: &str = "exact-structural-study-publication-v3";
const PUBLICATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.exact-structural-study-publication.v3+cbor";
const PUBLICATION_VALUE_SCHEMA: &str = "runmat.analysis.exact-structural-study-publication.v3";
const REGION_RESOLUTION_PREFIX: &[u8] = b"runmat-topology-region-resolution/v1\0";
const REGION_RESOLUTION_DOMAIN: &str = "topology-region-resolution";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);

fn read(source: &impl CacheImport, reference: &ValueRef) -> NativeExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified(reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("study root object is unavailable"))?;
    if reference.logical_digest != Digest::sha256(&bytes)
        || reference.encoded_length != bytes.len() as u64
    {
        return Err(protocol("study root object differs from its reference"));
    }
    Ok(bytes)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
pub(crate) mod tests;

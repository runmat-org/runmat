//! Portable driver-owned root contract for an exact-CAD structural study sweep.

mod contract;
mod publication;

pub use contract::{
    decode_exact_structural_sweep_root, prepare_exact_structural_sweep_root,
    DecodedExactStructuralSweepRoot, ExactStructuralSweepRootPreparation,
    PreparedExactStructuralSweepRoot,
};
pub(crate) use publication::package_exact_structural_sweep_result;
pub(crate) use publication::ExactStructuralSweepRemoteOutcome;
pub use publication::{
    import_exact_structural_sweep_result, ExactStructuralSweepEntryResult,
    ExactStructuralSweepResult, ExactStructuralSweepResultExpectation,
    ExactStructuralSweepTerminal,
};

use runmat_canonical_codec::CanonicalLimits;

pub const EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE: &str = "exact-structural-sweep-root";
pub const EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE: &str = "portable-exact-structural-sweep-root";

const ROOT_PROGRAM_BYTES: &[u8] = b"runmat-exact-structural-sweep-root\0";
const ROOT_PREFIX: &[u8] = b"runmat-exact-structural-sweep-root/v1\0";
const ROOT_DOMAIN: &str = "exact-structural-sweep-root";
const ROOT_MEDIA_TYPE: &str = "application/vnd.runmat.exact-structural-sweep-root+cbor";
const ROOT_VALUE_SCHEMA: &str = "runmat.analysis.exact-structural-sweep-root";
const PUBLICATION_PREFIX: &[u8] = b"runmat-exact-structural-sweep-publication/v1\0";
const PUBLICATION_DOMAIN: &str = "exact-structural-sweep-publication";
const PUBLICATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.exact-structural-sweep-publication+cbor";
const PUBLICATION_VALUE_SCHEMA: &str = "runmat.analysis.exact-structural-sweep-publication";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(128 * 1024 * 1024, 2_000_000, 131_072, 64);

fn protocol(error: impl std::fmt::Display) -> crate::NativeExecutionError {
    crate::NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "occt-native"))]
mod conformance_tests;

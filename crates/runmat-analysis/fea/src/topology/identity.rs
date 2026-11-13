use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use super::contract::{AnalysisProjectionError, AnalysisProjectionPolicy};

const PREFIX: &[u8] = b"runmat-analysis-projection/v1\0";
const DOMAIN: &str = "analysis-projection-identity";
const LIMITS: CanonicalLimits = CanonicalLimits::new(1024, 16, 256, 8);

pub(super) fn projection_digest(
    mesh_digest: StableDigest,
    policy: AnalysisProjectionPolicy,
) -> Result<Digest, AnalysisProjectionError> {
    encode_contract(PREFIX, DOMAIN, &(mesh_digest, policy), LIMITS)
        .map(Digest::sha256)
        .map_err(|error| AnalysisProjectionError::IdentityEncoding(error.to_string()))
}

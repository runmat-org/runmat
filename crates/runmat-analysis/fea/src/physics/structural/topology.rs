use runmat_analysis_core::StructuralModel;
use runmat_canonical_codec::{encode_contract, CanonicalCodecError, CanonicalLimits};
use runmat_execution::Digest;

const EXPLICIT_TOPOLOGY_PREFIX: &[u8] = b"runmat-fea-explicit-structural-topology/v1\0";
const EXPLICIT_TOPOLOGY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 4_000_000, 4_000_000, 64);

pub(crate) fn explicit_topology_digest(
    structural: &StructuralModel,
) -> Result<Digest, CanonicalCodecError> {
    let encoded = encode_contract(
        EXPLICIT_TOPOLOGY_PREFIX,
        "explicit-structural-topology",
        structural,
        EXPLICIT_TOPOLOGY_LIMITS,
    )?;
    Ok(Digest::sha256(encoded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    #[test]
    fn topology_identity_is_stable_and_geometry_sensitive() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let structural = model.structural.as_ref().expect("explicit topology");
        let first = explicit_topology_digest(structural).expect("identity");
        assert_eq!(
            first,
            explicit_topology_digest(structural).expect("repeat identity")
        );

        let mut changed = structural.clone();
        changed.nodes[1].coordinates_m[0] += 0.25;
        assert_ne!(
            first,
            explicit_topology_digest(&changed).expect("changed identity")
        );
    }
}

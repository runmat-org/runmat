use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn signal_sampling_identities_conform() {
    assert_family(&[
        (
            &uniform::FEA_SIGNAL_SAMPLING_UNIFORM_CATALOG_ENTRY,
            &uniform::POLICY_CALL,
        ),
        (
            &adaptive::FEA_SIGNAL_SAMPLING_ADAPTIVE_CATALOG_ENTRY,
            &adaptive::POLICY_CALL,
        ),
        (
            &canonical_points::FEA_SIGNAL_SAMPLING_CANONICAL_POINTS_CATALOG_ENTRY,
            &canonical_points::POLICY_CALL,
        ),
    ]);
}

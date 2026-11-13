use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn signal_knot_identities_conform() {
    assert_family(&[
        (
            &continuous::FEA_SIGNAL_KNOT_CONTINUOUS_CATALOG_ENTRY,
            &continuous::POLICY_CALL,
        ),
        (
            &jump::FEA_SIGNAL_KNOT_JUMP_CATALOG_ENTRY,
            &jump::POLICY_CALL,
        ),
    ]);
}

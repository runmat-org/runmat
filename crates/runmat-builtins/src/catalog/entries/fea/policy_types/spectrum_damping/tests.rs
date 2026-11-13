use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spectrum_damping_identities_conform() {
    assert_family(&[
        (
            &ratio::FEA_SPECTRUM_DAMPING_RATIO_CATALOG_ENTRY,
            &ratio::POLICY_CALL,
        ),
        (
            &quality_factor::FEA_SPECTRUM_DAMPING_QUALITY_FACTOR_CATALOG_ENTRY,
            &quality_factor::POLICY_CALL,
        ),
    ]);
}

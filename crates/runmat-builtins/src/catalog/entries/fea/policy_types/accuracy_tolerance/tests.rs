use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn accuracy_tolerance_identities_conform() {
    assert_family(&[
        (
            &relative::FEA_ACCURACY_TOLERANCE_RELATIVE_CATALOG_ENTRY,
            &relative::POLICY_CALL,
        ),
        (
            &absolute::FEA_ACCURACY_TOLERANCE_ABSOLUTE_CATALOG_ENTRY,
            &absolute::POLICY_CALL,
        ),
    ]);
}

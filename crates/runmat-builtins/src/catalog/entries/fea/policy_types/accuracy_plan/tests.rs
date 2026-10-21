use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn accuracy_plan_identities_conform() {
    assert_family(&[
        (
            &new_method::FEA_ACCURACY_PLAN_NEW_CATALOG_ENTRY,
            &new_method::POLICY_CALL,
        ),
        (
            &baseline::FEA_ACCURACY_PLAN_BASELINE_CATALOG_ENTRY,
            &baseline::POLICY_CALL,
        ),
    ]);
}

use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn execution_plan_identities_conform() {
    assert_family(&[
        (
            &new_method::FEA_EXECUTION_PLAN_NEW_CATALOG_ENTRY,
            &new_method::POLICY_CALL,
        ),
        (
            &defaults::FEA_EXECUTION_PLAN_DEFAULTS_CATALOG_ENTRY,
            &defaults::POLICY_CALL,
        ),
    ]);
}

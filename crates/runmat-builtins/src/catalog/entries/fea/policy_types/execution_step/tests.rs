use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn execution_step_identities_conform() {
    assert_family(&[(
        &new_method::FEA_EXECUTION_STEP_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

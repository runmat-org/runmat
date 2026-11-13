use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn accuracy_criterion_identities_conform() {
    assert_family(&[(
        &new_method::FEA_ACCURACY_CRITERION_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn accuracy_resource_ceiling_identities_conform() {
    assert_family(&[(
        &new_method::FEA_ACCURACY_RESOURCE_CEILING_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

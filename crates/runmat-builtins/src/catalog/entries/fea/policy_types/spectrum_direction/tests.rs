use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spectrum_direction_identities_conform() {
    assert_family(&[(
        &new_method::FEA_SPECTRUM_DIRECTION_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

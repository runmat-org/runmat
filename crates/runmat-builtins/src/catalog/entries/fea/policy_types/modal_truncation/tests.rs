use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn modal_truncation_identities_conform() {
    assert_family(&[(
        &new_method::FEA_MODAL_TRUNCATION_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

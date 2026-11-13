use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn response_spectrum_identity_conforms() {
    assert_family(&[(
        &new_method::FEA_RESPONSE_SPECTRUM_NEW_CATALOG_ENTRY,
        &new_method::POLICY_CALL,
    )]);
}

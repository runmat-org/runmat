use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn rigid_response_identities_conform() {
    assert_family(&[
        (
            &gupta::FEA_RIGID_RESPONSE_GUPTA_CATALOG_ENTRY,
            &gupta::POLICY_CALL,
        ),
        (
            &lindley_yow::FEA_RIGID_RESPONSE_LINDLEY_YOW_CATALOG_ENTRY,
            &lindley_yow::POLICY_CALL,
        ),
    ]);
}

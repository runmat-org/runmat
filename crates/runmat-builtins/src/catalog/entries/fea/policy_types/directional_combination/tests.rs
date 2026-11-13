use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn directional_percentage_identity_conforms() {
    assert_family(&[(
        &percentage::FEA_DIRECTIONAL_COMBINATION_PERCENTAGE_CATALOG_ENTRY,
        &percentage::POLICY_CALL,
    )]);
}

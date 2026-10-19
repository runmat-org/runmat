use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn formulation_requirement_identities_conform() {
    assert_family(&[
        (
            &production::FEA_FORMULATION_REQUIREMENT_PRODUCTION_CATALOG_ENTRY,
            &production::POLICY_CALL,
        ),
        (
            &experimental::FEA_FORMULATION_REQUIREMENT_EXPERIMENTAL_CATALOG_ENTRY,
            &experimental::POLICY_CALL,
        ),
    ]);
}

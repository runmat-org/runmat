use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn diagnostic_kind_identities_conform() {
    assert_family(&[
        (
            &convergence::FEA_DIAGNOSTIC_KIND_CONVERGENCE_CATALOG_ENTRY,
            &convergence::POLICY_CALL,
        ),
        (
            &balance::FEA_DIAGNOSTIC_KIND_BALANCE_CATALOG_ENTRY,
            &balance::POLICY_CALL,
        ),
        (
            &constraint::FEA_DIAGNOSTIC_KIND_CONSTRAINT_CATALOG_ENTRY,
            &constraint::POLICY_CALL,
        ),
        (
            &contact::FEA_DIAGNOSTIC_KIND_CONTACT_CATALOG_ENTRY,
            &contact::POLICY_CALL,
        ),
        (
            &adaptation::FEA_DIAGNOSTIC_KIND_ADAPTATION_CATALOG_ENTRY,
            &adaptation::POLICY_CALL,
        ),
        (
            &resource::FEA_DIAGNOSTIC_KIND_RESOURCE_CATALOG_ENTRY,
            &resource::POLICY_CALL,
        ),
        (
            &execution_equivalence::FEA_DIAGNOSTIC_KIND_EXECUTION_EQUIVALENCE_CATALOG_ENTRY,
            &execution_equivalence::POLICY_CALL,
        ),
    ]);
}

use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn numerical_equivalence_identities_conform() {
    assert_family(&[
        (
            &deterministic::FEA_NUMERICAL_EQUIVALENCE_POLICY_DETERMINISTIC_CATALOG_ENTRY,
            &deterministic::POLICY_CALL,
        ),
        (
            &relative_tolerance::FEA_NUMERICAL_EQUIVALENCE_POLICY_RELATIVE_TOLERANCE_CATALOG_ENTRY,
            &relative_tolerance::POLICY_CALL,
        ),
        (
            &absolute_tolerance::FEA_NUMERICAL_EQUIVALENCE_POLICY_ABSOLUTE_TOLERANCE_CATALOG_ENTRY,
            &absolute_tolerance::POLICY_CALL,
        ),
    ]);
}

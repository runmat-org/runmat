use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn accuracy_target_identities_conform() {
    assert_family(&[
        (
            &observation::FEA_ACCURACY_TARGET_OBSERVATION_CATALOG_ENTRY,
            &observation::POLICY_CALL,
        ),
        (
            &quantity::FEA_ACCURACY_TARGET_QUANTITY_CATALOG_ENTRY,
            &quantity::POLICY_CALL,
        ),
        (
            &numerical_result::FEA_ACCURACY_TARGET_NUMERICAL_RESULT_CATALOG_ENTRY,
            &numerical_result::POLICY_CALL,
        ),
        (
            &global_equilibrium::FEA_ACCURACY_TARGET_GLOBAL_EQUILIBRIUM_CATALOG_ENTRY,
            &global_equilibrium::POLICY_CALL,
        ),
        (
            &interface_closure::FEA_ACCURACY_TARGET_INTERFACE_CLOSURE_CATALOG_ENTRY,
            &interface_closure::POLICY_CALL,
        ),
        (
            &execution_equivalence::FEA_ACCURACY_TARGET_EXECUTION_EQUIVALENCE_CATALOG_ENTRY,
            &execution_equivalence::POLICY_CALL,
        ),
    ]);
}

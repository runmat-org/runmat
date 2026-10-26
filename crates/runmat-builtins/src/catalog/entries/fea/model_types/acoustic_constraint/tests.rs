use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `prescribedPressure`, `periodic`, `phaseShiftedPeriodic`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&prescribed_pressure::ENTRY, &inference::PRESCRIBED_PRESSURE);
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
    assert_model_object_contract(
        &phase_shifted_periodic::ENTRY,
        &inference::PHASE_SHIFTED_PERIODIC,
    );
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.AcousticConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

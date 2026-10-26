use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `static`, `modal`, `harmonic`, `transient`, `nonlinear`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&static_step::ENTRY, &inference::STATIC_STEP);
    assert_model_object_contract(&modal::ENTRY, &inference::MODAL);
    assert_model_object_contract(&harmonic::ENTRY, &inference::HARMONIC);
    assert_model_object_contract(&transient::ENTRY, &inference::TRANSIENT);
    assert_model_object_contract(&nonlinear::ENTRY, &inference::NONLINEAR);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.StructuralStep."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

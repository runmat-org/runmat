use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `temperature`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&temperature::ENTRY, &inference::TEMPERATURE);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry
            .identity
            .name
            .starts_with("fea.ThermalInitialCondition."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

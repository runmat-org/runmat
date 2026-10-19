use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `perfect`, `conductance`, `resistance`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&perfect::ENTRY, &inference::PERFECT);
    assert_model_object_contract(&conductance::ENTRY, &inference::CONDUCTANCE);
    assert_model_object_contract(&resistance::ENTRY, &inference::RESISTANCE);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry
            .identity
            .name
            .starts_with("fea.ConjugateHeatTransferInterface."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

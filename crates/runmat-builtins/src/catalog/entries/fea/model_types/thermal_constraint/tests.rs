use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `prescribed`, `tie`, `periodic`, `affineJump`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&prescribed::ENTRY, &inference::PRESCRIBED);
    assert_model_object_contract(&tie::ENTRY, &inference::TIE);
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
    assert_model_object_contract(&affine_jump::ENTRY, &inference::AFFINE_JUMP);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.ThermalConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

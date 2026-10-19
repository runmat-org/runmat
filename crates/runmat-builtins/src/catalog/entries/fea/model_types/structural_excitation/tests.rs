use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `force`, `moment`, `traction`, `pressure`, `bodyForce`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&force::ENTRY, &inference::FORCE);
    assert_model_object_contract(&moment::ENTRY, &inference::MOMENT);
    assert_model_object_contract(&traction::ENTRY, &inference::TRACTION);
    assert_model_object_contract(&pressure::ENTRY, &inference::PRESSURE);
    assert_model_object_contract(&body_force::ENTRY, &inference::BODY_FORCE);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.StructuralExcitation."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

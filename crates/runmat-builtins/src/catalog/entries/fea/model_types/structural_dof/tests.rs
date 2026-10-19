use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `ux`, `uy`, `uz`, `rx`, `ry`, `rz`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&ux::ENTRY, &inference::UX);
    assert_model_object_contract(&uy::ENTRY, &inference::UY);
    assert_model_object_contract(&uz::ENTRY, &inference::UZ);
    assert_model_object_contract(&rx::ENTRY, &inference::RX);
    assert_model_object_contract(&ry::ENTRY, &inference::RY);
    assert_model_object_contract(&rz::ENTRY, &inference::RZ);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.StructuralDof."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

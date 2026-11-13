use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `fixed`, `prescribed`, `prescribedMotion`, `symmetry`, `antisymmetry`, `normalDisplacement`, `rigidCoupling`, `distributingCoupling`, `tie`, `periodic`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&fixed::ENTRY, &inference::FIXED);
    assert_model_object_contract(&prescribed::ENTRY, &inference::PRESCRIBED);
    assert_model_object_contract(&prescribed_motion::ENTRY, &inference::PRESCRIBED_MOTION);
    assert_model_object_contract(&symmetry::ENTRY, &inference::SYMMETRY);
    assert_model_object_contract(&antisymmetry::ENTRY, &inference::ANTISYMMETRY);
    assert_model_object_contract(&normal_displacement::ENTRY, &inference::NORMAL_DISPLACEMENT);
    assert_model_object_contract(&rigid_coupling::ENTRY, &inference::RIGID_COUPLING);
    assert_model_object_contract(
        &distributing_coupling::ENTRY,
        &inference::DISTRIBUTING_COUPLING,
    );
    assert_model_object_contract(&tie::ENTRY, &inference::TIE);
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.StructuralConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

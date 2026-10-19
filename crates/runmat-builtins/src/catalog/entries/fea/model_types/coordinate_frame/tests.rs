use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `globalFrame`, `material`, `element`, `boundaryNormalTangent`, `cartesian`, `cylindrical`, `spherical`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&global_frame::ENTRY, &inference::GLOBAL_FRAME);
    assert_model_object_contract(&material::ENTRY, &inference::MATERIAL);
    assert_model_object_contract(&element::ENTRY, &inference::ELEMENT);
    assert_model_object_contract(
        &boundary_normal_tangent::ENTRY,
        &inference::BOUNDARY_NORMAL_TANGENT,
    );
    assert_model_object_contract(&cartesian::ENTRY, &inference::CARTESIAN);
    assert_model_object_contract(&cylindrical::ENTRY, &inference::CYLINDRICAL);
    assert_model_object_contract(&spherical::ENTRY, &inference::SPHERICAL);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.CoordinateFrame."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

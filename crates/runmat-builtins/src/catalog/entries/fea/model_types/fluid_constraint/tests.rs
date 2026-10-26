use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `noSlipWall`, `slipWall`, `symmetry`, `prescribedVelocity`, `movingWall`, `pressureReference`, `periodic`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&no_slip_wall::ENTRY, &inference::NO_SLIP_WALL);
    assert_model_object_contract(&slip_wall::ENTRY, &inference::SLIP_WALL);
    assert_model_object_contract(&symmetry::ENTRY, &inference::SYMMETRY);
    assert_model_object_contract(&prescribed_velocity::ENTRY, &inference::PRESCRIBED_VELOCITY);
    assert_model_object_contract(&moving_wall::ENTRY, &inference::MOVING_WALL);
    assert_model_object_contract(&pressure_reference::ENTRY, &inference::PRESSURE_REFERENCE);
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.FluidConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

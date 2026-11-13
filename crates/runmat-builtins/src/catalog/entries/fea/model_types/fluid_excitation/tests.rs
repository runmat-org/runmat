use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `inletVelocity`, `massFlow`, `wallMotion`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&inlet_velocity::ENTRY, &inference::INLET_VELOCITY);
    assert_model_object_contract(&mass_flow::ENTRY, &inference::MASS_FLOW);
    assert_model_object_contract(&wall_motion::ENTRY, &inference::WALL_MOTION);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.FluidExcitation."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

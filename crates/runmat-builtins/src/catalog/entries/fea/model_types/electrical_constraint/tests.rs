use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `equipotential`, `floatingConductor`, `potentialReference`, `prescribedPotential`, `integralCurrent`, `periodic`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(&equipotential::ENTRY, &inference::EQUIPOTENTIAL);
    assert_model_object_contract(&floating_conductor::ENTRY, &inference::FLOATING_CONDUCTOR);
    assert_model_object_contract(&potential_reference::ENTRY, &inference::POTENTIAL_REFERENCE);
    assert_model_object_contract(
        &prescribed_potential::ENTRY,
        &inference::PRESCRIBED_POTENTIAL,
    );
    assert_model_object_contract(&integral_current::ENTRY, &inference::INTEGRAL_CURRENT);
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.ElectricalConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

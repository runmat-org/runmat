use super::*;
use crate::fea_contract::model_types::test_support::assert_model_object_contract;

/// Valid, unknown, wrong-arity, wrong-kind, and resident inputs for `vectorPotentialGround`, `coulombGauge`, `prescribedVectorPotential`, `periodic`, `phaseShiftedPeriodic`.
#[test]
fn constructors_enforce_their_static_argument_contracts() {
    assert_model_object_contract(
        &vector_potential_ground::ENTRY,
        &inference::VECTOR_POTENTIAL_GROUND,
    );
    assert_model_object_contract(&coulomb_gauge::ENTRY, &inference::COULOMB_GAUGE);
    assert_model_object_contract(
        &prescribed_vector_potential::ENTRY,
        &inference::PRESCRIBED_VECTOR_POTENTIAL,
    );
    assert_model_object_contract(&periodic::ENTRY, &inference::PERIODIC);
    assert_model_object_contract(
        &phase_shifted_periodic::ENTRY,
        &inference::PHASE_SHIFTED_PERIODIC,
    );
}

#[test]
fn identities_belong_to_the_family_and_are_pure_host_constructors() {
    for entry in ENTRIES {
        assert!(entry.identity.name.starts_with("fea.MagneticConstraint."));
        assert!(matches!(entry.contract.purity, BuiltinPurity::Pure));
        assert!(matches!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        ));
        let errors = crate::validate_builtin_catalog(&[*entry]);
        assert!(errors.is_empty(), "{errors:?}");
    }
}

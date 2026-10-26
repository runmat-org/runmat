use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn time_knot_signal_identities_conform() {
    assert_family(&[
        (&displacement::FEA_DISPLACEMENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &displacement::POLICY_CALL),
        (&rotation::FEA_ROTATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &rotation::POLICY_CALL),
        (&velocity::FEA_VELOCITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &velocity::POLICY_CALL),
        (&acceleration::FEA_ACCELERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &acceleration::POLICY_CALL),
        (&angular_velocity::FEA_ANGULAR_VELOCITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &angular_velocity::POLICY_CALL),
        (&angular_acceleration::FEA_ANGULAR_ACCELERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &angular_acceleration::POLICY_CALL),
        (&force::FEA_FORCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &force::POLICY_CALL),
        (&moment::FEA_MOMENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &moment::POLICY_CALL),
        (&traction::FEA_TRACTION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &traction::POLICY_CALL),
        (&pressure::FEA_PRESSURE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &pressure::POLICY_CALL),
        (&body_force::FEA_BODY_FORCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &body_force::POLICY_CALL),
        (&temperature::FEA_TEMPERATURE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &temperature::POLICY_CALL),
        (&temperature_difference::FEA_TEMPERATURE_DIFFERENCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &temperature_difference::POLICY_CALL),
        (&heat_flux::FEA_HEAT_FLUX_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &heat_flux::POLICY_CALL),
        (&heat_generation::FEA_HEAT_GENERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &heat_generation::POLICY_CALL),
        (&electric_potential::FEA_ELECTRIC_POTENTIAL_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &electric_potential::POLICY_CALL),
        (&electric_current::FEA_ELECTRIC_CURRENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &electric_current::POLICY_CALL),
        (&electric_charge::FEA_ELECTRIC_CHARGE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &electric_charge::POLICY_CALL),
        (&current_density::FEA_CURRENT_DENSITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &current_density::POLICY_CALL),
        (&charge_density::FEA_CHARGE_DENSITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &charge_density::POLICY_CALL),
        (&magnetic_vector_potential::FEA_MAGNETIC_VECTOR_POTENTIAL_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &magnetic_vector_potential::POLICY_CALL),
        (&mass_flow::FEA_MASS_FLOW_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &mass_flow::POLICY_CALL),
        (&volume_flow::FEA_VOLUME_FLOW_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY, &volume_flow::POLICY_CALL),
    ]);
}

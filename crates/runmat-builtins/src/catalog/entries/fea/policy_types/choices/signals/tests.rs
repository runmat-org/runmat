use crate::fea_contract::policy_types::conformance::assert_family;
use crate::BuiltinCatalogEntry;

const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &super::displacement_signal::constant_meters::CATALOG_ENTRY,
    &super::displacement_signal::from_time_samples::CATALOG_ENTRY,
    &super::displacement_signal::from_function::CATALOG_ENTRY,
    &super::rotation_signal::constant_radians::CATALOG_ENTRY,
    &super::rotation_signal::from_time_samples::CATALOG_ENTRY,
    &super::rotation_signal::from_function::CATALOG_ENTRY,
    &super::velocity_signal::constant_meters_per_second::CATALOG_ENTRY,
    &super::velocity_signal::from_time_samples::CATALOG_ENTRY,
    &super::velocity_signal::from_function::CATALOG_ENTRY,
    &super::acceleration_signal::constant_meters_per_second_squared::CATALOG_ENTRY,
    &super::acceleration_signal::from_time_samples::CATALOG_ENTRY,
    &super::acceleration_signal::from_function::CATALOG_ENTRY,
    &super::angular_velocity_signal::constant_radians_per_second::CATALOG_ENTRY,
    &super::angular_velocity_signal::from_time_samples::CATALOG_ENTRY,
    &super::angular_velocity_signal::from_function::CATALOG_ENTRY,
    &super::angular_acceleration_signal::constant_radians_per_second_squared::CATALOG_ENTRY,
    &super::angular_acceleration_signal::from_time_samples::CATALOG_ENTRY,
    &super::angular_acceleration_signal::from_function::CATALOG_ENTRY,
    &super::force_signal::constant_newtons::CATALOG_ENTRY,
    &super::force_signal::from_time_samples::CATALOG_ENTRY,
    &super::force_signal::from_function::CATALOG_ENTRY,
    &super::moment_signal::constant_newton_meters::CATALOG_ENTRY,
    &super::moment_signal::from_time_samples::CATALOG_ENTRY,
    &super::moment_signal::from_function::CATALOG_ENTRY,
    &super::traction_signal::constant_pascals::CATALOG_ENTRY,
    &super::traction_signal::from_time_samples::CATALOG_ENTRY,
    &super::traction_signal::from_function::CATALOG_ENTRY,
    &super::pressure_signal::constant_pascals::CATALOG_ENTRY,
    &super::pressure_signal::from_time_samples::CATALOG_ENTRY,
    &super::pressure_signal::from_function::CATALOG_ENTRY,
    &super::body_force_signal::constant_newtons_per_cubic_meter::CATALOG_ENTRY,
    &super::body_force_signal::from_time_samples::CATALOG_ENTRY,
    &super::body_force_signal::from_function::CATALOG_ENTRY,
    &super::temperature_signal::constant_kelvin::CATALOG_ENTRY,
    &super::temperature_signal::from_time_samples::CATALOG_ENTRY,
    &super::temperature_signal::from_function::CATALOG_ENTRY,
    &super::temperature_difference_signal::constant_kelvin::CATALOG_ENTRY,
    &super::temperature_difference_signal::from_time_samples::CATALOG_ENTRY,
    &super::temperature_difference_signal::from_function::CATALOG_ENTRY,
    &super::heat_flux_signal::constant_watts_per_square_meter::CATALOG_ENTRY,
    &super::heat_flux_signal::from_time_samples::CATALOG_ENTRY,
    &super::heat_flux_signal::from_function::CATALOG_ENTRY,
    &super::heat_generation_signal::constant_watts_per_cubic_meter::CATALOG_ENTRY,
    &super::heat_generation_signal::from_time_samples::CATALOG_ENTRY,
    &super::heat_generation_signal::from_function::CATALOG_ENTRY,
    &super::electric_potential_signal::constant_volts::CATALOG_ENTRY,
    &super::electric_potential_signal::from_time_samples::CATALOG_ENTRY,
    &super::electric_potential_signal::from_function::CATALOG_ENTRY,
    &super::electric_current_signal::constant_amperes::CATALOG_ENTRY,
    &super::electric_current_signal::from_time_samples::CATALOG_ENTRY,
    &super::electric_current_signal::from_function::CATALOG_ENTRY,
    &super::electric_charge_signal::constant_coulombs::CATALOG_ENTRY,
    &super::electric_charge_signal::from_time_samples::CATALOG_ENTRY,
    &super::electric_charge_signal::from_function::CATALOG_ENTRY,
    &super::current_density_signal::constant_amperes_per_square_meter::CATALOG_ENTRY,
    &super::current_density_signal::from_time_samples::CATALOG_ENTRY,
    &super::current_density_signal::from_function::CATALOG_ENTRY,
    &super::charge_density_signal::constant_coulombs_per_cubic_meter::CATALOG_ENTRY,
    &super::charge_density_signal::from_time_samples::CATALOG_ENTRY,
    &super::charge_density_signal::from_function::CATALOG_ENTRY,
    &super::magnetic_vector_potential_signal::constant_webers_per_meter::CATALOG_ENTRY,
    &super::magnetic_vector_potential_signal::from_time_samples::CATALOG_ENTRY,
    &super::magnetic_vector_potential_signal::from_function::CATALOG_ENTRY,
    &super::mass_flow_signal::constant_kilograms_per_second::CATALOG_ENTRY,
    &super::mass_flow_signal::from_time_samples::CATALOG_ENTRY,
    &super::mass_flow_signal::from_function::CATALOG_ENTRY,
    &super::volume_flow_signal::constant_cubic_meters_per_second::CATALOG_ENTRY,
    &super::volume_flow_signal::from_time_samples::CATALOG_ENTRY,
    &super::volume_flow_signal::from_function::CATALOG_ENTRY,
];

#[test]
fn constant_signal_constructors_conform() {
    assert_family(&[
        (
            &super::displacement_signal::constant_meters::CATALOG_ENTRY,
            &super::displacement_signal::constant_meters::POLICY_CALL,
        ),
        (
            &super::rotation_signal::constant_radians::CATALOG_ENTRY,
            &super::rotation_signal::constant_radians::POLICY_CALL,
        ),
        (
            &super::velocity_signal::constant_meters_per_second::CATALOG_ENTRY,
            &super::velocity_signal::constant_meters_per_second::POLICY_CALL,
        ),
        (
            &super::acceleration_signal::constant_meters_per_second_squared::CATALOG_ENTRY,
            &super::acceleration_signal::constant_meters_per_second_squared::POLICY_CALL,
        ),
        (
            &super::angular_velocity_signal::constant_radians_per_second::CATALOG_ENTRY,
            &super::angular_velocity_signal::constant_radians_per_second::POLICY_CALL,
        ),
        (
            &super::angular_acceleration_signal::constant_radians_per_second_squared::CATALOG_ENTRY,
            &super::angular_acceleration_signal::constant_radians_per_second_squared::POLICY_CALL,
        ),
        (
            &super::force_signal::constant_newtons::CATALOG_ENTRY,
            &super::force_signal::constant_newtons::POLICY_CALL,
        ),
        (
            &super::moment_signal::constant_newton_meters::CATALOG_ENTRY,
            &super::moment_signal::constant_newton_meters::POLICY_CALL,
        ),
        (
            &super::traction_signal::constant_pascals::CATALOG_ENTRY,
            &super::traction_signal::constant_pascals::POLICY_CALL,
        ),
        (
            &super::pressure_signal::constant_pascals::CATALOG_ENTRY,
            &super::pressure_signal::constant_pascals::POLICY_CALL,
        ),
        (
            &super::body_force_signal::constant_newtons_per_cubic_meter::CATALOG_ENTRY,
            &super::body_force_signal::constant_newtons_per_cubic_meter::POLICY_CALL,
        ),
        (
            &super::temperature_signal::constant_kelvin::CATALOG_ENTRY,
            &super::temperature_signal::constant_kelvin::POLICY_CALL,
        ),
        (
            &super::temperature_difference_signal::constant_kelvin::CATALOG_ENTRY,
            &super::temperature_difference_signal::constant_kelvin::POLICY_CALL,
        ),
        (
            &super::heat_flux_signal::constant_watts_per_square_meter::CATALOG_ENTRY,
            &super::heat_flux_signal::constant_watts_per_square_meter::POLICY_CALL,
        ),
        (
            &super::heat_generation_signal::constant_watts_per_cubic_meter::CATALOG_ENTRY,
            &super::heat_generation_signal::constant_watts_per_cubic_meter::POLICY_CALL,
        ),
        (
            &super::electric_potential_signal::constant_volts::CATALOG_ENTRY,
            &super::electric_potential_signal::constant_volts::POLICY_CALL,
        ),
        (
            &super::electric_current_signal::constant_amperes::CATALOG_ENTRY,
            &super::electric_current_signal::constant_amperes::POLICY_CALL,
        ),
        (
            &super::electric_charge_signal::constant_coulombs::CATALOG_ENTRY,
            &super::electric_charge_signal::constant_coulombs::POLICY_CALL,
        ),
        (
            &super::current_density_signal::constant_amperes_per_square_meter::CATALOG_ENTRY,
            &super::current_density_signal::constant_amperes_per_square_meter::POLICY_CALL,
        ),
        (
            &super::charge_density_signal::constant_coulombs_per_cubic_meter::CATALOG_ENTRY,
            &super::charge_density_signal::constant_coulombs_per_cubic_meter::POLICY_CALL,
        ),
        (
            &super::magnetic_vector_potential_signal::constant_webers_per_meter::CATALOG_ENTRY,
            &super::magnetic_vector_potential_signal::constant_webers_per_meter::POLICY_CALL,
        ),
        (
            &super::mass_flow_signal::constant_kilograms_per_second::CATALOG_ENTRY,
            &super::mass_flow_signal::constant_kilograms_per_second::POLICY_CALL,
        ),
        (
            &super::volume_flow_signal::constant_cubic_meters_per_second::CATALOG_ENTRY,
            &super::volume_flow_signal::constant_cubic_meters_per_second::POLICY_CALL,
        ),
    ]);
}

#[test]
fn family_entries_are_valid_documented_and_registered() {
    let errors = crate::catalog::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
    let mut registered = Vec::new();
    super::extend_entries(&mut registered);
    assert_eq!(registered.len(), ENTRIES.len());
    for entry in ENTRIES {
        let documentation = &entry.documentation;
        let name = entry.identity.name;
        assert_eq!(documentation.title, Some(name));
        let headings: Vec<_> = documentation
            .sections
            .iter()
            .map(|section| section.heading)
            .collect();
        assert!(
            headings.contains(&"Behavior") && headings.contains(&"Use in a study"),
            "{name}"
        );
        assert!(
            !documentation.faqs.is_empty() || !documentation.examples.is_empty(),
            "{name}"
        );
        assert!(
            !documentation.related.contains(&name),
            "{name} lists itself as related"
        );
        for example in documentation.examples {
            assert!(
                example.program.contains(name),
                "{name} example must call it"
            );
            assert!(matches!(
                example.verification,
                crate::BuiltinExampleVerification::Assertions { .. }
            ));
        }
        assert!(registered
            .iter()
            .any(|candidate| std::ptr::eq(*candidate, *entry) || candidate.identity.name == name));
    }
}

use super::RefinementIndicatorKey;

pub fn structural_static_default_refinement_indicators() -> Vec<RefinementIndicatorKey> {
    vec![
        RefinementIndicatorKey::new("structural", "stress_gradient"),
        RefinementIndicatorKey::new("structural", "strain_energy_density"),
        RefinementIndicatorKey::new("structural", "load_regions"),
        RefinementIndicatorKey::new("structural", "constraint_regions"),
    ]
}

pub fn default_refinement_indicators_for_analysis(
    profile: &str,
    run_kind: &str,
) -> Vec<RefinementIndicatorKey> {
    let profile_defaults = match profile {
        "linear_static_structural" => structural_static_default_refinement_indicators(),
        "modal_structural" => vec![
            RefinementIndicatorKey::new("modal", "modal_strain_energy"),
            RefinementIndicatorKey::new("modal", "mode_shape_curvature"),
            RefinementIndicatorKey::new("modal", "frequency_residual"),
        ],
        "transient_structural" => vec![
            RefinementIndicatorKey::new("structural", "stress_gradient"),
            RefinementIndicatorKey::new("structural", "strain_energy_density"),
            RefinementIndicatorKey::new("structural", "displacement_gradient"),
        ],
        "nonlinear_structural" => vec![
            RefinementIndicatorKey::new("structural", "stress_gradient"),
            RefinementIndicatorKey::new("structural", "strain_energy_density"),
            RefinementIndicatorKey::new("structural", "plastic_strain"),
            RefinementIndicatorKey::new("structural", "contact_pressure"),
            RefinementIndicatorKey::new("structural", "contact_gap"),
        ],
        "thermal_standalone" => vec![
            RefinementIndicatorKey::new("thermal", "temperature_gradient"),
            RefinementIndicatorKey::new("thermal", "heat_flux_gradient"),
            RefinementIndicatorKey::new("thermal", "heat_source"),
            RefinementIndicatorKey::new("thermal", "convection_regions"),
            RefinementIndicatorKey::new("thermal", "prescribed_temperature_regions"),
        ],
        "thermo_mechanical_coupled" => vec![
            RefinementIndicatorKey::new("thermo_mechanical", "thermal_gradient"),
            RefinementIndicatorKey::new("thermo_mechanical", "thermal_stress"),
            RefinementIndicatorKey::new("thermo_mechanical", "structural_von_mises"),
            RefinementIndicatorKey::new("thermo_mechanical", "strain_energy_density"),
            RefinementIndicatorKey::new("thermo_mechanical", "region_temperature_delta"),
        ],
        "electromagnetic_static" => vec![
            RefinementIndicatorKey::new("electromagnetic", "flux_density_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "electric_field_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "current_density_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "energy_density"),
            RefinementIndicatorKey::new("electromagnetic", "source_regions"),
            RefinementIndicatorKey::new("electromagnetic", "ground_regions"),
        ],
        "acoustic_harmonic" => vec![
            RefinementIndicatorKey::new("acoustic", "pressure_gradient"),
            RefinementIndicatorKey::new("acoustic", "pressure_curvature"),
            RefinementIndicatorKey::new("acoustic", "wavelength"),
            RefinementIndicatorKey::new("acoustic", "impedance_regions"),
            RefinementIndicatorKey::new("acoustic", "source_regions"),
        ],
        "cfd_steady_state" | "cfd_transient" => vec![
            RefinementIndicatorKey::new("cfd", "velocity_gradient"),
            RefinementIndicatorKey::new("cfd", "pressure_gradient"),
            RefinementIndicatorKey::new("cfd", "vorticity"),
            RefinementIndicatorKey::new("cfd", "wall_shear"),
            RefinementIndicatorKey::new("cfd", "boundary_layer"),
        ],
        "cht_coupled" => vec![
            RefinementIndicatorKey::new("cht", "interface_heat_flux_jump"),
            RefinementIndicatorKey::new("cht", "interface_temperature_jump"),
            RefinementIndicatorKey::new("cht", "solid_heat_flux_gradient"),
            RefinementIndicatorKey::new("cht", "fluid_boundary_layer"),
        ],
        "fsi_coupled" => vec![
            RefinementIndicatorKey::new("fsi", "interface_displacement_jump"),
            RefinementIndicatorKey::new("fsi", "interface_traction_jump"),
            RefinementIndicatorKey::new("fsi", "structural_stress_gradient"),
            RefinementIndicatorKey::new("fsi", "fluid_pressure_gradient"),
            RefinementIndicatorKey::new("fsi", "fluid_velocity_gradient"),
        ],
        _ => Vec::new(),
    };
    if !profile_defaults.is_empty() {
        return profile_defaults;
    }

    match run_kind {
        "linear_static" => structural_static_default_refinement_indicators(),
        "modal" => vec![
            RefinementIndicatorKey::new("modal", "modal_strain_energy"),
            RefinementIndicatorKey::new("modal", "mode_shape_curvature"),
            RefinementIndicatorKey::new("modal", "frequency_residual"),
        ],
        "transient" => vec![
            RefinementIndicatorKey::new("structural", "stress_gradient"),
            RefinementIndicatorKey::new("structural", "strain_energy_density"),
            RefinementIndicatorKey::new("structural", "displacement_gradient"),
        ],
        "nonlinear" => vec![
            RefinementIndicatorKey::new("structural", "stress_gradient"),
            RefinementIndicatorKey::new("structural", "strain_energy_density"),
            RefinementIndicatorKey::new("structural", "plastic_strain"),
            RefinementIndicatorKey::new("structural", "contact_pressure"),
            RefinementIndicatorKey::new("structural", "contact_gap"),
        ],
        "thermal" => vec![
            RefinementIndicatorKey::new("thermal", "temperature_gradient"),
            RefinementIndicatorKey::new("thermal", "heat_flux_gradient"),
            RefinementIndicatorKey::new("thermal", "heat_source"),
            RefinementIndicatorKey::new("thermal", "convection_regions"),
            RefinementIndicatorKey::new("thermal", "prescribed_temperature_regions"),
        ],
        "electromagnetic" => vec![
            RefinementIndicatorKey::new("electromagnetic", "flux_density_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "electric_field_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "current_density_gradient"),
            RefinementIndicatorKey::new("electromagnetic", "energy_density"),
            RefinementIndicatorKey::new("electromagnetic", "source_regions"),
            RefinementIndicatorKey::new("electromagnetic", "ground_regions"),
        ],
        "acoustic" => vec![
            RefinementIndicatorKey::new("acoustic", "pressure_gradient"),
            RefinementIndicatorKey::new("acoustic", "pressure_curvature"),
            RefinementIndicatorKey::new("acoustic", "wavelength"),
            RefinementIndicatorKey::new("acoustic", "impedance_regions"),
            RefinementIndicatorKey::new("acoustic", "source_regions"),
        ],
        "cfd" => vec![
            RefinementIndicatorKey::new("cfd", "velocity_gradient"),
            RefinementIndicatorKey::new("cfd", "pressure_gradient"),
            RefinementIndicatorKey::new("cfd", "vorticity"),
            RefinementIndicatorKey::new("cfd", "wall_shear"),
            RefinementIndicatorKey::new("cfd", "boundary_layer"),
        ],
        "cht" => vec![
            RefinementIndicatorKey::new("cht", "interface_heat_flux_jump"),
            RefinementIndicatorKey::new("cht", "interface_temperature_jump"),
            RefinementIndicatorKey::new("cht", "solid_heat_flux_gradient"),
            RefinementIndicatorKey::new("cht", "fluid_boundary_layer"),
        ],
        "fsi" => vec![
            RefinementIndicatorKey::new("fsi", "interface_displacement_jump"),
            RefinementIndicatorKey::new("fsi", "interface_traction_jump"),
            RefinementIndicatorKey::new("fsi", "structural_stress_gradient"),
            RefinementIndicatorKey::new("fsi", "fluid_pressure_gradient"),
            RefinementIndicatorKey::new("fsi", "fluid_velocity_gradient"),
        ],
        _ => Vec::new(),
    }
}

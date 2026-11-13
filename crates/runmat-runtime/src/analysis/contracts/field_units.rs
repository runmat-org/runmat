//! Result-facing units inferred from existing field identifiers.
pub(super) fn infer_field_unit(field_id: &str) -> Option<&'static str> {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.starts_with("structural.random_response.variance.") {
        if normalized.ends_with(".displacement") {
            return Some("m^2");
        }
        if normalized.ends_with(".stress") {
            return Some("Pa^2");
        }
        if normalized.ends_with(".shell_membrane_force")
            || normalized.ends_with(".shell_transverse_shear")
        {
            return Some("N^2");
        }
        if normalized.ends_with(".shell_bending_moment") {
            return Some("N^2*m^2");
        }
        if normalized.ends_with(".rotation") {
            return Some("rad^2");
        }
        if normalized.ends_with(".strain") {
            return Some("1");
        }
    }
    if normalized.contains("frequency_hz") || normalized.contains("frequency_response") {
        return Some("Hz");
    }
    if normalized.contains("eigenvalue") {
        return Some("rad^2/s^2");
    }
    if normalized.contains("sound_pressure_level_db") {
        return Some("dB");
    }
    if normalized.contains("temperature_gradient") {
        return Some("K/m");
    }
    if normalized.contains("temperature") || normalized.contains("temperature_jump") {
        return Some("K");
    }
    if normalized.contains("electric_potential") || normalized.ends_with("electrical.potential") {
        return Some("V");
    }
    if normalized.contains("electric_field") {
        return Some("V/m");
    }
    if normalized.contains("electric_flux_density") {
        return Some("C/m^2");
    }
    if normalized.contains("current_density") {
        return Some("A/m^2");
    }
    if normalized.contains("vector_potential") {
        return Some("Wb/m");
    }
    if normalized.contains("magnetic_flux_density") {
        return Some("T");
    }
    if normalized.contains("magnetic_field") {
        return Some("A/m");
    }
    if normalized.contains("poynting_vector")
        || normalized.ends_with("acoustic.intensity")
        || normalized.contains("boundary_heat_flux")
        || normalized.contains("interface_heat_flux")
        || normalized.ends_with(".heat_flux")
        || normalized.contains(".heat_flux.")
    {
        return Some("W/m^2");
    }
    if normalized.contains("power_loss_density")
        || normalized.contains("joule_power_density")
        || normalized.contains("joule_heat")
        || normalized.contains("heat_source")
        || normalized.contains("energy_storage_rate")
    {
        return Some("W/m^3");
    }
    if normalized.contains("energy_density") {
        return Some("J/m^3");
    }
    if normalized.contains("kinetic_energy")
        || normalized.contains("strain_energy")
        || normalized.contains("total_strain_energy")
        || normalized.contains("external_work")
        || normalized.contains("support_work")
        || normalized.contains("damping_dissipation")
        || normalized.contains("energy_balance_residual")
    {
        return Some("J");
    }
    if normalized.contains("modal_mass") {
        return Some("kg");
    }
    if normalized.contains("modal_stiffness") {
        return Some("N/m");
    }
    if normalized.contains("wall_shear_stress")
        || normalized.contains("contact_pressure")
        || normalized.contains("interface_pressure")
        || normalized.contains("fluid_pressure")
        || normalized.contains(".pressure")
        || normalized.contains("stress")
        || normalized.contains("traction")
        || normalized.contains("von_mises")
    {
        return Some("Pa");
    }
    if normalized.contains("reaction_force")
        || normalized.contains("beam_axial_force")
        || normalized.contains("beam_shear_force")
        || normalized.contains("shell_membrane_force")
        || normalized.contains("shell_transverse_shear")
    {
        return Some("N");
    }
    if normalized.contains("reaction_moment")
        || normalized.contains("beam_torsion_moment")
        || normalized.contains("beam_bending_moment")
        || normalized.contains("shell_bending_moment")
    {
        return Some("N*m");
    }
    if normalized.contains("angular_velocity") {
        return Some("rad/s");
    }
    if normalized.contains("angular_acceleration") {
        return Some("rad/s^2");
    }
    if normalized.contains("rotation") {
        return Some("rad");
    }
    if normalized.contains("contact_gap")
        || normalized.contains("displacement")
        || normalized.contains("mode_shape")
    {
        return Some("m");
    }
    if normalized.contains("velocity") {
        return Some("m/s");
    }
    if normalized.contains("acceleration") {
        return Some("m/s^2");
    }
    if normalized.contains("strain")
        || normalized.contains("residual")
        || normalized.contains("equation_scale")
        || normalized.contains("load_factor")
        || normalized.contains("coupling_iteration_count")
        || normalized.contains("reynolds_number")
        || normalized.contains("phase")
        || normalized.contains("orthogonality")
        || normalized.contains("participation_factor")
        || normalized.contains("relative_frequency_separation")
    {
        return Some("1");
    }
    None
}

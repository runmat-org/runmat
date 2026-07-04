use super::fixtures::key;
use crate::adaptive::{
    default_refinement_indicators_for_analysis, structural_static_default_refinement_indicators,
};

#[test]
fn structural_static_defaults_are_owned_by_adaptive_policy() {
    let defaults = structural_static_default_refinement_indicators();

    assert_eq!(
        defaults,
        vec![
            key("structural", "stress_gradient"),
            key("structural", "strain_energy_density"),
            key("structural", "load_regions"),
            key("structural", "constraint_regions"),
        ]
    );
}

#[test]
fn analysis_default_indicators_cover_supported_run_kinds() {
    let cases = [
        (
            "linear_static_structural",
            "linear_static",
            key("structural", "stress_gradient"),
        ),
        (
            "modal_structural",
            "modal",
            key("modal", "mode_shape_curvature"),
        ),
        (
            "transient_structural",
            "transient",
            key("structural", "displacement_gradient"),
        ),
        (
            "nonlinear_structural",
            "nonlinear",
            key("structural", "plastic_strain"),
        ),
        (
            "thermal_standalone",
            "thermal",
            key("thermal", "temperature_gradient"),
        ),
        (
            "electromagnetic_static",
            "electromagnetic",
            key("electromagnetic", "flux_density_gradient"),
        ),
        (
            "acoustic_harmonic",
            "acoustic",
            key("acoustic", "wavelength"),
        ),
        ("cfd_steady_state", "cfd", key("cfd", "boundary_layer")),
        (
            "cht_coupled",
            "cht",
            key("cht", "interface_temperature_jump"),
        ),
        ("fsi_coupled", "fsi", key("fsi", "interface_traction_jump")),
    ];

    for (profile, run_kind, expected) in cases {
        let defaults = default_refinement_indicators_for_analysis(profile, run_kind);
        assert!(
            defaults.contains(&expected),
            "{profile}/{run_kind} should include {expected:?}; got {defaults:?}"
        );
    }
}

#[test]
fn analysis_default_indicators_prefer_profile_over_run_kind() {
    let defaults =
        default_refinement_indicators_for_analysis("thermo_mechanical_coupled", "transient");

    assert!(defaults.contains(&key("thermo_mechanical", "thermal_stress")));
    assert!(!defaults.contains(&key("structural", "displacement_gradient")));
}

#[test]
fn analysis_default_indicators_fall_back_to_run_kind() {
    let defaults = default_refinement_indicators_for_analysis("custom_profile", "thermal");

    assert!(defaults.contains(&key("thermal", "temperature_gradient")));
    assert!(defaults.contains(&key("thermal", "heat_flux_gradient")));
}

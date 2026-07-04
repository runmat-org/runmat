use serde::{Deserialize, Serialize};

use crate::field::{MeshSizingField, SizingSample};

mod convergence;
mod indicator_plan;

pub use convergence::{
    evaluate_adaptive_convergence, AdaptiveConvergenceMetrics, AdaptiveConvergenceStatus,
};
pub use indicator_plan::{
    plan_refinement_indicators, RefinementIndicatorAvailability, RefinementIndicatorKey,
    RefinementIndicatorStatus, RefinementIndicatorSummary,
};

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementMarker {
    pub entity_id: String,
    pub weight: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementIndicatorSample {
    pub entity_id: String,
    pub position_m: [f64; 3],
    pub indicator_value: f64,
    pub current_size_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RefinementMarkerOptions {
    pub max_markers: usize,
    pub min_relative_value: f64,
    pub target_size_scale: f64,
}

impl Default for RefinementMarkerOptions {
    fn default() -> Self {
        Self {
            max_markers: 64,
            min_relative_value: 0.25,
            target_size_scale: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementMarkerError {
    InvalidMaxMarkers,
    InvalidMinRelativeValue,
    InvalidTargetSizeScale,
}

pub fn build_refinement_markers_from_samples(
    samples: &[RefinementIndicatorSample],
    reason: &str,
    options: RefinementMarkerOptions,
) -> Result<(Vec<RefinementMarker>, SizingFieldUpdate), RefinementMarkerError> {
    if options.max_markers == 0 {
        return Err(RefinementMarkerError::InvalidMaxMarkers);
    }
    if !options.min_relative_value.is_finite() || !(0.0..=1.0).contains(&options.min_relative_value)
    {
        return Err(RefinementMarkerError::InvalidMinRelativeValue);
    }
    if !options.target_size_scale.is_finite()
        || options.target_size_scale <= 0.0
        || options.target_size_scale >= 1.0
    {
        return Err(RefinementMarkerError::InvalidTargetSizeScale);
    }

    let mut finite_samples = samples
        .iter()
        .filter(|sample| {
            sample.indicator_value.is_finite()
                && sample.indicator_value > 0.0
                && sample.current_size_m.is_finite()
                && sample.current_size_m > 0.0
                && sample.position_m.iter().all(|value| value.is_finite())
        })
        .collect::<Vec<_>>();
    let Some(max_value) = finite_samples
        .iter()
        .map(|sample| sample.indicator_value)
        .reduce(f64::max)
    else {
        return Ok((Vec::new(), SizingFieldUpdate::default()));
    };

    finite_samples
        .retain(|sample| sample.indicator_value / max_value >= options.min_relative_value);
    finite_samples.sort_by(|left, right| {
        right
            .indicator_value
            .total_cmp(&left.indicator_value)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    finite_samples.truncate(options.max_markers);

    let markers = finite_samples
        .iter()
        .map(|sample| RefinementMarker {
            entity_id: sample.entity_id.clone(),
            weight: sample.indicator_value / max_value,
            reason: reason.to_string(),
        })
        .collect::<Vec<_>>();
    let sizing_update = SizingFieldUpdate {
        samples: finite_samples
            .into_iter()
            .map(|sample| SizingSample {
                position_m: sample.position_m,
                target_size_m: sample.current_size_m * options.target_size_scale,
                reason: Some(reason.to_string()),
            })
            .collect(),
        min_size_m: None,
        max_size_m: None,
    };

    Ok((markers, sizing_update))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SizingFieldUpdate {
    #[serde(default)]
    pub samples: Vec<SizingSample>,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
}

impl SizingFieldUpdate {
    pub fn apply_to(self, sizing: &mut MeshSizingField) {
        if let Some(min_size_m) = self.min_size_m {
            sizing.min_size_m = Some(match sizing.min_size_m {
                Some(existing) => existing.min(min_size_m),
                None => min_size_m,
            });
        }
        if let Some(max_size_m) = self.max_size_m {
            sizing.max_size_m = Some(match sizing.max_size_m {
                Some(existing) => existing.max(max_size_m),
                None => max_size_m,
            });
        }
        sizing.samples.extend(self.samples);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveIterationSummary {
    pub iteration_index: usize,
    pub node_count: usize,
    pub element_count: usize,
    pub convergence_status: AdaptiveConvergenceStatus,
    #[serde(default)]
    pub indicators: Vec<RefinementIndicatorSummary>,
    #[serde(default)]
    pub markers: Vec<RefinementMarker>,
    #[serde(default)]
    pub sizing_update: SizingFieldUpdate,
}

#[cfg(test)]
mod tests;

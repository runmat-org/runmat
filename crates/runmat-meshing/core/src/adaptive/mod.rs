use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    options::{MeshRefinementOptions, RefinementIndicatorMode, RefinementStrategy},
    size::field::{MeshSizingField, SizingSample},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveConvergenceStatus {
    NotStarted,
    Disabled,
    Pending,
    Converged,
    MaxIterationsReached,
    ElementBudgetReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdaptiveConvergenceMetrics {
    pub completed_iterations: usize,
    pub element_budget_reached: bool,
    pub previous_node_count: Option<usize>,
    pub current_node_count: Option<usize>,
    pub previous_element_count: Option<usize>,
    pub current_element_count: Option<usize>,
    pub field_change: Option<f64>,
    pub energy_change: Option<f64>,
    pub residual: Option<f64>,
}

pub fn evaluate_adaptive_convergence(
    options: &MeshRefinementOptions,
    metrics: AdaptiveConvergenceMetrics,
) -> AdaptiveConvergenceStatus {
    if matches!(
        options.strategy,
        RefinementStrategy::None | RefinementStrategy::Uniform
    ) {
        return AdaptiveConvergenceStatus::Disabled;
    }
    if metrics.element_budget_reached {
        return AdaptiveConvergenceStatus::ElementBudgetReached;
    }
    if metrics.completed_iterations >= options.max_iterations {
        return AdaptiveConvergenceStatus::MaxIterationsReached;
    }
    if metrics.completed_iterations > 0
        && matches!(
            (
                metrics.previous_node_count,
                metrics.current_node_count,
                metrics.previous_element_count,
                metrics.current_element_count,
            ),
            (Some(previous_nodes), Some(current_nodes), Some(previous_elements), Some(current_elements))
                if current_nodes <= previous_nodes && current_elements <= previous_elements
        )
    {
        return AdaptiveConvergenceStatus::Converged;
    }

    let mut considered_metric = false;
    let mut converged = true;

    if let Some(field_change) = metrics.field_change {
        considered_metric = true;
        converged &=
            field_change.is_finite() && field_change <= options.convergence.field_change_tolerance;
    }
    if let Some(energy_change) = metrics.energy_change {
        considered_metric = true;
        converged &= energy_change.is_finite()
            && energy_change <= options.convergence.energy_change_tolerance;
    }
    if let (Some(residual), Some(tolerance)) =
        (metrics.residual, options.convergence.residual_tolerance)
    {
        considered_metric = true;
        converged &= residual.is_finite() && residual <= tolerance;
    }

    if considered_metric && converged {
        AdaptiveConvergenceStatus::Converged
    } else {
        AdaptiveConvergenceStatus::Pending
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementIndicatorStatus {
    Used,
    SkippedMissingField,
    SkippedNotApplicable,
    SkippedBudget,
    SkippedQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementIndicatorSummary {
    pub namespace: String,
    pub name: String,
    pub requested_mode: RefinementIndicatorMode,
    pub status: RefinementIndicatorStatus,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RefinementIndicatorKey {
    pub namespace: String,
    pub name: String,
}

impl RefinementIndicatorKey {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementIndicatorAvailability {
    pub key: RefinementIndicatorKey,
    pub applicable: bool,
    pub field_available: bool,
}

pub fn plan_refinement_indicators(
    options: &MeshRefinementOptions,
    defaults: &[RefinementIndicatorKey],
    availability: &[RefinementIndicatorAvailability],
    element_budget_reached: bool,
    quality_blocked: bool,
) -> Vec<RefinementIndicatorSummary> {
    if matches!(
        options.strategy,
        RefinementStrategy::None | RefinementStrategy::Uniform
    ) {
        return Vec::new();
    }

    let availability_by_key = availability
        .iter()
        .map(|item| (item.key.clone(), item))
        .collect::<BTreeMap<_, _>>();
    let overrides = options
        .indicators
        .namespaces
        .iter()
        .flat_map(|(namespace, names)| {
            names.iter().map(|(name, mode)| {
                (
                    RefinementIndicatorKey::new(namespace.clone(), name.clone()),
                    *mode,
                )
            })
        })
        .collect::<BTreeMap<_, _>>();

    let mut keys = defaults.iter().cloned().collect::<BTreeSet<_>>();
    keys.extend(overrides.keys().cloned());

    keys.into_iter()
        .map(|key| {
            let requested_mode = overrides
                .get(&key)
                .copied()
                .unwrap_or(RefinementIndicatorMode::Auto);
            let (status, detail) = if matches!(requested_mode, RefinementIndicatorMode::Off) {
                (
                    RefinementIndicatorStatus::SkippedNotApplicable,
                    Some("indicator disabled by override".to_string()),
                )
            } else if element_budget_reached {
                (
                    RefinementIndicatorStatus::SkippedBudget,
                    Some("element budget reached".to_string()),
                )
            } else if quality_blocked {
                (
                    RefinementIndicatorStatus::SkippedQuality,
                    Some("mesh quality constraint blocked refinement".to_string()),
                )
            } else if let Some(available) = availability_by_key.get(&key) {
                if !available.applicable {
                    (
                        RefinementIndicatorStatus::SkippedNotApplicable,
                        Some("indicator does not apply to the active analysis".to_string()),
                    )
                } else if !available.field_available {
                    (
                        RefinementIndicatorStatus::SkippedMissingField,
                        Some("required recovered field is unavailable".to_string()),
                    )
                } else {
                    (RefinementIndicatorStatus::Used, None)
                }
            } else if matches!(requested_mode, RefinementIndicatorMode::On) {
                (
                    RefinementIndicatorStatus::SkippedMissingField,
                    Some("required recovered field is unavailable".to_string()),
                )
            } else {
                (
                    RefinementIndicatorStatus::SkippedNotApplicable,
                    Some("indicator was not selected by the active analysis".to_string()),
                )
            };

            RefinementIndicatorSummary {
                namespace: key.namespace,
                name: key.name,
                requested_mode,
                status,
                detail,
            }
        })
        .collect()
}

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

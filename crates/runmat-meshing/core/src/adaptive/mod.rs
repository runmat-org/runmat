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
mod tests {
    use super::*;
    use crate::options::{MeshRefinementOptions, RefinementIndicatorOverrides};

    fn key(namespace: &str, name: &str) -> RefinementIndicatorKey {
        RefinementIndicatorKey::new(namespace, name)
    }

    fn available(namespace: &str, name: &str) -> RefinementIndicatorAvailability {
        RefinementIndicatorAvailability {
            key: key(namespace, name),
            applicable: true,
            field_available: true,
        }
    }

    #[test]
    fn adaptive_convergence_is_disabled_for_nonadaptive_strategies() {
        let mut options = MeshRefinementOptions {
            strategy: RefinementStrategy::None,
            ..MeshRefinementOptions::default()
        };

        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    field_change: Some(0.0),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Disabled
        );

        options.strategy = RefinementStrategy::Uniform;
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    field_change: Some(0.0),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Disabled
        );
    }

    #[test]
    fn adaptive_convergence_reports_hard_stop_statuses_first() {
        let options = MeshRefinementOptions {
            strategy: RefinementStrategy::Adaptive,
            max_iterations: 2,
            ..MeshRefinementOptions::default()
        };

        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    element_budget_reached: true,
                    completed_iterations: 2,
                    field_change: Some(0.0),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::ElementBudgetReached
        );
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    completed_iterations: 2,
                    field_change: Some(0.0),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::MaxIterationsReached
        );
    }

    #[test]
    fn adaptive_convergence_requires_finite_metric_within_tolerance() {
        let options = MeshRefinementOptions {
            strategy: RefinementStrategy::Adaptive,
            ..MeshRefinementOptions::default()
        };

        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    field_change: Some(0.05),
                    energy_change: Some(0.02),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Converged
        );
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    field_change: Some(0.051),
                    energy_change: Some(0.02),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Pending
        );
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    field_change: Some(f64::NAN),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Pending
        );
        assert_eq!(
            evaluate_adaptive_convergence(&options, AdaptiveConvergenceMetrics::default()),
            AdaptiveConvergenceStatus::Pending
        );
    }

    #[test]
    fn adaptive_convergence_detects_no_topology_growth() {
        let options = MeshRefinementOptions {
            strategy: RefinementStrategy::Adaptive,
            ..MeshRefinementOptions::default()
        };

        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    completed_iterations: 1,
                    previous_node_count: Some(24),
                    current_node_count: Some(24),
                    previous_element_count: Some(48),
                    current_element_count: Some(48),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Converged
        );
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    completed_iterations: 1,
                    previous_node_count: Some(24),
                    current_node_count: Some(25),
                    previous_element_count: Some(48),
                    current_element_count: Some(49),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Pending
        );
    }

    #[test]
    fn adaptive_convergence_residual_metric_is_opt_in() {
        let mut options = MeshRefinementOptions {
            strategy: RefinementStrategy::Adaptive,
            ..MeshRefinementOptions::default()
        };

        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    residual: Some(1.0e6),
                    field_change: Some(0.0),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Converged
        );

        options.convergence.residual_tolerance = Some(1.0e-6);
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    residual: Some(1.0e-7),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Converged
        );
        assert_eq!(
            evaluate_adaptive_convergence(
                &options,
                AdaptiveConvergenceMetrics {
                    residual: Some(1.0e-5),
                    ..AdaptiveConvergenceMetrics::default()
                }
            ),
            AdaptiveConvergenceStatus::Pending
        );
    }

    #[test]
    fn sizing_field_update_merges_bounds_and_samples() {
        let mut sizing = MeshSizingField {
            global_target_size_m: Some(0.1),
            min_size_m: Some(0.02),
            max_size_m: Some(0.2),
            samples: vec![SizingSample {
                position_m: [0.0, 0.0, 0.0],
                target_size_m: 0.08,
                reason: Some("initial".to_string()),
            }],
            ..MeshSizingField::default()
        };

        SizingFieldUpdate {
            samples: vec![SizingSample {
                position_m: [1.0, 0.0, 0.0],
                target_size_m: 0.01,
                reason: Some("stress_gradient".to_string()),
            }],
            min_size_m: Some(0.01),
            max_size_m: Some(0.25),
        }
        .apply_to(&mut sizing);

        assert_eq!(sizing.global_target_size_m, Some(0.1));
        assert_eq!(sizing.min_size_m, Some(0.01));
        assert_eq!(sizing.max_size_m, Some(0.25));
        assert_eq!(sizing.samples.len(), 2);
    }

    #[test]
    fn adaptive_iteration_summary_round_trips() {
        let summary = AdaptiveIterationSummary {
            iteration_index: 1,
            node_count: 32,
            element_count: 96,
            convergence_status: AdaptiveConvergenceStatus::Pending,
            indicators: vec![RefinementIndicatorSummary {
                namespace: "structural".to_string(),
                name: "stress_gradient".to_string(),
                requested_mode: RefinementIndicatorMode::Auto,
                status: RefinementIndicatorStatus::Used,
                detail: Some("field available".to_string()),
            }],
            markers: vec![RefinementMarker {
                entity_id: "tetrahedron_1".to_string(),
                weight: 1.0,
                reason: "stress_gradient".to_string(),
            }],
            sizing_update: SizingFieldUpdate::default(),
        };

        let encoded = serde_json::to_string(&summary).expect("summary should serialize");
        let decoded: AdaptiveIterationSummary =
            serde_json::from_str(&encoded).expect("summary should deserialize");

        assert_eq!(decoded, summary);
    }

    #[test]
    fn refinement_indicator_plan_merges_defaults_and_overrides() {
        let mut options = MeshRefinementOptions::default();
        options.indicators = RefinementIndicatorOverrides {
            namespaces: BTreeMap::from([(
                "structural".to_string(),
                BTreeMap::from([
                    ("stress_gradient".to_string(), RefinementIndicatorMode::Off),
                    (
                        "strain_energy_density".to_string(),
                        RefinementIndicatorMode::On,
                    ),
                    ("plastic_strain".to_string(), RefinementIndicatorMode::On),
                ]),
            )]),
        };

        let summaries = plan_refinement_indicators(
            &options,
            &[key("structural", "stress_gradient")],
            &[
                available("structural", "stress_gradient"),
                available("structural", "strain_energy_density"),
            ],
            false,
            false,
        );

        assert_eq!(summaries.len(), 3);
        assert_eq!(
            summaries
                .iter()
                .find(|summary| summary.name == "stress_gradient")
                .expect("stress gradient summary")
                .status,
            RefinementIndicatorStatus::SkippedNotApplicable
        );
        assert_eq!(
            summaries
                .iter()
                .find(|summary| summary.name == "strain_energy_density")
                .expect("strain energy summary")
                .status,
            RefinementIndicatorStatus::Used
        );
        assert_eq!(
            summaries
                .iter()
                .find(|summary| summary.name == "plastic_strain")
                .expect("plastic strain summary")
                .status,
            RefinementIndicatorStatus::SkippedMissingField
        );
    }

    #[test]
    fn refinement_indicator_plan_reports_budget_and_quality_skips() {
        let options = MeshRefinementOptions::default();
        let defaults = [key("structural", "stress_gradient")];
        let availability = [available("structural", "stress_gradient")];

        let budget = plan_refinement_indicators(&options, &defaults, &availability, true, false);
        assert_eq!(budget[0].status, RefinementIndicatorStatus::SkippedBudget);

        let quality = plan_refinement_indicators(&options, &defaults, &availability, false, true);
        assert_eq!(quality[0].status, RefinementIndicatorStatus::SkippedQuality);
    }

    #[test]
    fn refinement_indicator_plan_distinguishes_missing_and_not_applicable() {
        let options = MeshRefinementOptions::default();
        let defaults = [
            key("structural", "stress_gradient"),
            key("structural", "contact_pressure"),
        ];
        let availability = [
            RefinementIndicatorAvailability {
                key: key("structural", "stress_gradient"),
                applicable: true,
                field_available: false,
            },
            RefinementIndicatorAvailability {
                key: key("structural", "contact_pressure"),
                applicable: false,
                field_available: true,
            },
        ];

        let summaries =
            plan_refinement_indicators(&options, &defaults, &availability, false, false);

        assert_eq!(
            summaries
                .iter()
                .find(|summary| summary.name == "stress_gradient")
                .expect("stress gradient summary")
                .status,
            RefinementIndicatorStatus::SkippedMissingField
        );
        assert_eq!(
            summaries
                .iter()
                .find(|summary| summary.name == "contact_pressure")
                .expect("contact pressure summary")
                .status,
            RefinementIndicatorStatus::SkippedNotApplicable
        );
    }

    #[test]
    fn refinement_indicator_plan_is_empty_when_refinement_is_disabled() {
        let mut options = MeshRefinementOptions::default();
        options.strategy = RefinementStrategy::None;

        assert!(plan_refinement_indicators(
            &options,
            &[key("structural", "stress_gradient")],
            &[available("structural", "stress_gradient")],
            false,
            false
        )
        .is_empty());
    }

    #[test]
    fn refinement_indicator_plan_is_empty_for_uniform_refinement() {
        let mut options = MeshRefinementOptions::default();
        options.strategy = RefinementStrategy::Uniform;
        options.indicators = RefinementIndicatorOverrides {
            namespaces: BTreeMap::from([(
                "structural".to_string(),
                BTreeMap::from([("stress_gradient".to_string(), RefinementIndicatorMode::On)]),
            )]),
        };

        assert!(plan_refinement_indicators(
            &options,
            &[key("structural", "strain_energy_density")],
            &[available("structural", "stress_gradient")],
            false,
            false,
        )
        .is_empty());
    }

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

    #[test]
    fn refinement_markers_are_deterministic_and_create_sizing_samples() {
        let samples = vec![
            RefinementIndicatorSample {
                entity_id: "tetrahedron_low".to_string(),
                position_m: [0.0, 0.0, 0.0],
                indicator_value: 0.2,
                current_size_m: 0.08,
            },
            RefinementIndicatorSample {
                entity_id: "tetrahedron_b".to_string(),
                position_m: [1.0, 0.0, 0.0],
                indicator_value: 1.0,
                current_size_m: 0.06,
            },
            RefinementIndicatorSample {
                entity_id: "tetrahedron_a".to_string(),
                position_m: [0.0, 1.0, 0.0],
                indicator_value: 1.0,
                current_size_m: 0.04,
            },
            RefinementIndicatorSample {
                entity_id: "tetrahedron_mid".to_string(),
                position_m: [0.0, 0.0, 1.0],
                indicator_value: 0.5,
                current_size_m: 0.1,
            },
        ];

        let (markers, sizing) = build_refinement_markers_from_samples(
            &samples,
            "structural.stress_gradient",
            RefinementMarkerOptions {
                max_markers: 3,
                min_relative_value: 0.5,
                target_size_scale: 0.4,
            },
        )
        .expect("marker generation should succeed");

        assert_eq!(
            markers
                .iter()
                .map(|marker| marker.entity_id.as_str())
                .collect::<Vec<_>>(),
            vec!["tetrahedron_a", "tetrahedron_b", "tetrahedron_mid"]
        );
        assert_eq!(markers[0].weight, 1.0);
        assert_eq!(markers[2].weight, 0.5);
        assert_eq!(sizing.samples.len(), 3);
        assert_eq!(sizing.samples[0].target_size_m, 0.016);
        assert_eq!(
            sizing.samples[0].reason.as_deref(),
            Some("structural.stress_gradient")
        );
    }

    #[test]
    fn refinement_markers_filter_nonfinite_and_empty_samples() {
        let samples = vec![
            RefinementIndicatorSample {
                entity_id: "nan".to_string(),
                position_m: [0.0, 0.0, 0.0],
                indicator_value: f64::NAN,
                current_size_m: 0.08,
            },
            RefinementIndicatorSample {
                entity_id: "zero".to_string(),
                position_m: [0.0, 0.0, 0.0],
                indicator_value: 0.0,
                current_size_m: 0.08,
            },
            RefinementIndicatorSample {
                entity_id: "bad_size".to_string(),
                position_m: [0.0, 0.0, 0.0],
                indicator_value: 1.0,
                current_size_m: 0.0,
            },
        ];

        let (markers, sizing) = build_refinement_markers_from_samples(
            &samples,
            "structural.stress_gradient",
            RefinementMarkerOptions::default(),
        )
        .expect("invalid samples should be filtered, not fail the batch");

        assert!(markers.is_empty());
        assert!(sizing.samples.is_empty());
    }

    #[test]
    fn refinement_marker_options_are_validated() {
        let sample = [RefinementIndicatorSample {
            entity_id: "tetrahedron".to_string(),
            position_m: [0.0, 0.0, 0.0],
            indicator_value: 1.0,
            current_size_m: 0.1,
        }];

        assert_eq!(
            build_refinement_markers_from_samples(
                &sample,
                "reason",
                RefinementMarkerOptions {
                    max_markers: 0,
                    ..RefinementMarkerOptions::default()
                }
            ),
            Err(RefinementMarkerError::InvalidMaxMarkers)
        );
        assert_eq!(
            build_refinement_markers_from_samples(
                &sample,
                "reason",
                RefinementMarkerOptions {
                    min_relative_value: 1.5,
                    ..RefinementMarkerOptions::default()
                }
            ),
            Err(RefinementMarkerError::InvalidMinRelativeValue)
        );
        assert_eq!(
            build_refinement_markers_from_samples(
                &sample,
                "reason",
                RefinementMarkerOptions {
                    target_size_scale: 1.0,
                    ..RefinementMarkerOptions::default()
                }
            ),
            Err(RefinementMarkerError::InvalidTargetSizeScale)
        );
    }
}

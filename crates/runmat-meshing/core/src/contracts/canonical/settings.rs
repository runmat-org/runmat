//! Product-facing resolution of concise meshing settings into the canonical request.

use runmat_geometry_core::GeometryTolerancePolicy;
use serde::{Deserialize, Serialize};

use super::{
    AlgorithmVersionSet, CancellationPolicy, CurveQualityTargets, ElementOrder, MeshDimension,
    MeshSizeControl, MeshingContractError, MeshingQualityTargets, MeshingRequest,
    MeshingSizeLimits, MetricCombinationRule, MetricFieldRequest, MetricTensor3,
    SurfaceQualityTargets, VolumeQualityTargets, MESHING_REQUEST_SCHEMA_VERSION,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingRequestSettings {
    pub element_order: ElementOrder,
    #[serde(default, skip_serializing_if = "MeshDimension::is_solid")]
    pub dimension: MeshDimension,
    #[serde(with = "runmat_types::wire_integer")]
    pub deterministic_seed: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub target_edge_length_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_grading_ratio: f64,
    pub limits: MeshingSizeLimits,
    pub cancellation: CancellationPolicy,
    /// Requested element layers across thin gaps (proximity sizing). `None` keeps the canonical
    /// two layers across gaps between edges and one layer across gaps between facing faces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cells_across_gap: Option<u32>,
    /// Local sizing controls on named geometry regions, ordered by control name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub size_controls: Vec<MeshSizeControl>,
}

impl Default for MeshingRequestSettings {
    fn default() -> Self {
        Self {
            element_order: ElementOrder::Tet4,
            dimension: MeshDimension::Solid,
            deterministic_seed: 0,
            target_edge_length_m: runmat_types::physical::Length::new(0.01),
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.0001),
            maximum_grading_ratio: 1.3,
            limits: MeshingSizeLimits {
                maximum_nodes: 10_000_000,
                maximum_elements: 10_000_000,
                maximum_recursion_depth: 256,
            },
            cancellation: CancellationPolicy {
                maximum_checkpoint_latency_ms: 1_000,
                maximum_work_units_between_checks: 100_000,
            },
            cells_across_gap: None,
            size_controls: Vec::new(),
        }
    }
}

impl MeshingRequestSettings {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            (
                "target edge length",
                self.target_edge_length_m.canonical_value(),
            ),
            (
                "maximum chordal deviation",
                self.maximum_chordal_deviation_m.canonical_value(),
            ),
            ("maximum grading ratio", self.maximum_grading_ratio),
        ] {
            if !value.is_finite() {
                return Err(MeshingContractError::invalid(field, "must be finite"));
            }
        }
        if self.target_edge_length_m.canonical_value() <= 0.0
            || self.maximum_chordal_deviation_m.canonical_value() <= 0.0
        {
            return Err(MeshingContractError::invalid(
                "meshing length settings",
                "target edge length and chordal deviation must be positive",
            ));
        }
        if self.maximum_grading_ratio < 1.0 {
            return Err(MeshingContractError::invalid(
                "maximum grading ratio",
                "must be at least one",
            ));
        }
        if self
            .cells_across_gap
            .is_some_and(|cells| !(1..=64).contains(&cells))
        {
            return Err(MeshingContractError::invalid(
                "cells across gap",
                "must be between 1 and 64",
            ));
        }
        super::validate_size_controls(&self.size_controls)?;
        self.limits.validate()?;
        self.cancellation.validate()
    }
}

/// Algorithm identities implemented by the production meshing host.
///
/// Worker capability admission and workload construction share this authority so an algorithm
/// upgrade cannot silently make submitted work unschedulable.
pub fn production_algorithm_versions() -> AlgorithmVersionSet {
    AlgorithmVersionSet {
        geometry: "geometry/exact-brep".into(),
        // Geometry-derived sizing is shared by curves, surfaces and volume consumers.
        curve: "curve/exact-metric-v4".into(),
        surface: "surface/exact-cdt".into(),
        plc: "plc/canonical".into(),
        tetrahedron: "tetrahedron/constrained-delaunay".into(),
        optimization: "optimization/quality-convergence".into(),
        validation: "validation/independent".into(),
    }
}

/// Resolves product settings into the one canonical request accepted by the exact DAG.
///
/// Algorithm identities are implementation facts, not user-selectable backend controls.
pub fn resolve_meshing_request(
    tolerance: GeometryTolerancePolicy,
    settings: MeshingRequestSettings,
) -> Result<MeshingRequest, MeshingContractError> {
    settings.validate()?;
    let metric = MetricTensor3::isotropic_length_m(settings.target_edge_length_m.canonical_value())
        .map_err(|error| MeshingContractError::invalid("target edge length", error.to_string()))?;
    let request = MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: settings.element_order,
        dimension: settings.dimension,
        deterministic_seed: settings.deterministic_seed,
        algorithms: production_algorithm_versions(),
        tolerance,
        metric: MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: metric,
            maximum_grading_ratio: settings.maximum_grading_ratio,
            contributions: Vec::new(),
            cells_across_gap: settings.cells_across_gap,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                    settings.maximum_chordal_deviation_m.canonical_value(),
                ),
                maximum_tangent_change_degrees: 180.0,
                minimum_metric_edge_length: 0.01,
                maximum_metric_edge_length: 1.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.01,
                minimum_metric_angle_degrees: 0.1,
                maximum_physical_aspect_ratio: 1_000.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                    settings.maximum_chordal_deviation_m.canonical_value(),
                ),
                maximum_normal_deviation_degrees: 180.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.001,
                maximum_metric_edge_length: 1.0,
            },
        },
        limits: settings.limits,
        cancellation: settings.cancellation,
        size_controls: settings.size_controls,
    };
    request.validate()?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance() -> GeometryTolerancePolicy {
        GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-12),
            model_relative_term: 1.0e-12,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        }
    }

    #[test]
    fn settings_resolve_to_a_valid_canonical_request() {
        let request = resolve_meshing_request(tolerance(), MeshingRequestSettings::default())
            .expect("default product settings should resolve");
        assert_eq!(request.element_order, ElementOrder::Tet4);
        assert_eq!(
            request.algorithms.tetrahedron,
            "tetrahedron/constrained-delaunay"
        );
        assert_eq!(request.metric.contributions, Vec::new());
        assert_eq!(request.quality.curve.maximum_metric_edge_length, 1.0);
        assert_eq!(request.quality.surface.minimum_metric_angle_degrees, 0.1);
        assert_eq!(request.quality.surface.minimum_metric_edge_length, 0.01);
        assert_eq!(
            request.quality.surface.maximum_physical_aspect_ratio,
            1_000.0
        );
        assert_eq!(request.quality.volume.maximum_metric_edge_length, 1.0);
    }

    #[test]
    fn invalid_target_length_is_rejected_before_execution() {
        let settings = MeshingRequestSettings {
            target_edge_length_m: runmat_types::physical::Length::new(0.0),
            ..MeshingRequestSettings::default()
        };
        assert!(resolve_meshing_request(tolerance(), settings).is_err());
    }

    #[test]
    fn authored_settings_reject_invalid_grading_before_geometry_resolution() {
        let settings = MeshingRequestSettings {
            maximum_grading_ratio: 0.99,
            ..MeshingRequestSettings::default()
        };
        assert!(settings.validate().is_err());
    }

    #[test]
    fn settings_contract_rejects_unknown_fields() {
        let mut value = serde_json::to_value(MeshingRequestSettings::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("backend".into(), serde_json::Value::String("legacy".into()));
        assert!(serde_json::from_value::<MeshingRequestSettings>(value).is_err());
    }
}

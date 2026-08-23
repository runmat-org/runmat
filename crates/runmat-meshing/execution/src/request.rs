//! Product-facing resolution of concise meshing settings into the canonical request.

use runmat_geometry_core::GeometryTolerancePolicy;
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CurveQualityTargets, ElementOrder,
    MeshingContractError, MeshingQualityTargets, MeshingRequest, MeshingResourceBudget,
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, SurfaceQualityTargets,
    VolumeQualityTargets, MESHING_REQUEST_SCHEMA_VERSION,
};

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingRequestSettings {
    pub element_order: ElementOrder,
    pub deterministic_seed: u64,
    pub target_edge_length_m: f64,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_grading_ratio: f64,
    pub resources: MeshingResourceBudget,
    pub cancellation: CancellationPolicy,
}

impl Default for MeshingRequestSettings {
    fn default() -> Self {
        Self {
            element_order: ElementOrder::Tet4,
            deterministic_seed: 0,
            target_edge_length_m: 0.01,
            maximum_chordal_deviation_m: 0.0001,
            maximum_grading_ratio: 1.3,
            resources: MeshingResourceBudget {
                maximum_nodes: 10_000_000,
                maximum_elements: 10_000_000,
                maximum_memory_bytes: 512 << 20,
                maximum_scratch_bytes: 512 << 20,
                maximum_wall_time_ms: 3_600_000,
                maximum_artifact_bytes: 512 << 20,
                maximum_search_work: 1_000_000_000,
                maximum_recursion_depth: 256,
                maximum_iterations: 1_000_000_000,
            },
            cancellation: CancellationPolicy {
                maximum_checkpoint_latency_ms: 1_000,
                maximum_work_units_between_checks: 100_000,
            },
        }
    }
}

/// Resolves product settings into the one canonical request accepted by the exact DAG.
///
/// Algorithm identities are implementation facts, not user-selectable backend controls.
pub fn resolve_meshing_request(
    tolerance: GeometryTolerancePolicy,
    settings: MeshingRequestSettings,
) -> Result<MeshingRequest, MeshingContractError> {
    let metric = MetricTensor3::isotropic_length_m(settings.target_edge_length_m)
        .map_err(|error| MeshingContractError::invalid("target edge length", error.to_string()))?;
    let request = MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: settings.element_order,
        deterministic_seed: settings.deterministic_seed,
        algorithms: AlgorithmVersionSet {
            geometry: "geometry/v2".into(),
            curve: "curve/v2".into(),
            surface: "surface/v2".into(),
            plc: "plc/v2".into(),
            tetrahedron: "tetrahedron/v2".into(),
            optimization: "optimization/v2".into(),
            validation: "validation/v2".into(),
        },
        tolerance,
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: metric,
            maximum_grading_ratio: settings.maximum_grading_ratio,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: settings.maximum_chordal_deviation_m,
                maximum_tangent_change_degrees: 180.0,
                minimum_metric_edge_length: 0.01,
                maximum_metric_edge_length: 10.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 0.1,
                maximum_physical_aspect_ratio: 1_000.0,
                maximum_chordal_deviation_m: settings.maximum_chordal_deviation_m,
                maximum_normal_deviation_degrees: 180.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.001,
                maximum_metric_edge_length: 10.0,
            },
        },
        resources: settings.resources,
        cancellation: settings.cancellation,
    };
    request.validate()?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance() -> GeometryTolerancePolicy {
        GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-6,
        }
    }

    #[test]
    fn settings_resolve_to_a_valid_canonical_request() {
        let request = resolve_meshing_request(tolerance(), MeshingRequestSettings::default())
            .expect("default product settings should resolve");
        assert_eq!(request.element_order, ElementOrder::Tet4);
        assert_eq!(request.algorithms.tetrahedron, "tetrahedron/v2");
        assert_eq!(request.metric.contributions, Vec::new());
    }

    #[test]
    fn invalid_target_length_is_rejected_before_execution() {
        let settings = MeshingRequestSettings {
            target_edge_length_m: 0.0,
            ..MeshingRequestSettings::default()
        };
        assert!(resolve_meshing_request(tolerance(), settings).is_err());
    }
}

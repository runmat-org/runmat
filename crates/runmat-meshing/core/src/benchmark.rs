use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    artifact::AnalysisMeshArtifact,
    evidence::{
        build_mesh_evidence_artifact, MeshCadEvidence, MeshQualityEvidence, MeshRegionEvidence,
        MeshTetRecoveryEvidence,
    },
    predicate::{tet_volume, triangle_area},
    topology::VolumeElementKind,
    validation::{volume_component_count, AnalysisMeshValidationOptions},
};

pub const MESH_BENCHMARK_SCHEMA_VERSION: &str = "mesh-benchmark/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshBenchmarkTier {
    Curve1d,
    Surface2d,
    Solid3d,
    ThinFeature,
    MultiBody,
    SizingField,
    AdaptiveRefinement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkInput {
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    #[serde(default)]
    pub timing: MeshBenchmarkTiming,
}

impl MeshBenchmarkInput {
    pub fn new(benchmark_id: impl Into<String>, tier: MeshBenchmarkTier) -> Self {
        Self {
            benchmark_id: benchmark_id.into(),
            tier,
            timing: MeshBenchmarkTiming::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshBenchmarkTiming {
    pub topology_import_ms: Option<f64>,
    pub curve_generation_ms: Option<f64>,
    pub surface_generation_ms: Option<f64>,
    pub volume_generation_ms: Option<f64>,
    pub validation_ms: Option<f64>,
    pub total_ms: Option<f64>,
    #[serde(default)]
    pub healing_warning_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    pub mesh_id: String,
    pub backend: String,
    pub algorithm: String,
    pub timing: MeshBenchmarkTiming,
    pub topology: MeshBenchmarkTopologyMetrics,
    pub cad: MeshCadEvidence,
    pub coverage: MeshBenchmarkCoverageMetrics,
    pub quality: MeshQualityEvidence,
    pub tet_recovery: MeshTetRecoveryEvidence,
    pub regions: MeshRegionEvidence,
    pub solve_readiness: MeshBenchmarkSolveReadiness,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkTopologyMetrics {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub volume_component_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkCoverageMetrics {
    pub expected_volume_m3: Option<f64>,
    pub actual_volume_m3: f64,
    pub volume_coverage_ratio: Option<f64>,
    pub expected_boundary_area_m2: Option<f64>,
    pub actual_boundary_area_m2: f64,
    pub boundary_area_ratio: Option<f64>,
    pub coverage_sample_ratio: Option<f64>,
    pub boundary_face_recovery_ratio: f64,
    pub boundary_edge_recovery_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkSolveReadiness {
    pub solve_ready: bool,
    pub validation_error_code: Option<String>,
    pub validation_error_message: Option<String>,
}

pub fn build_mesh_benchmark_report(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    input: MeshBenchmarkInput,
) -> MeshBenchmarkReport {
    let evidence = build_mesh_evidence_artifact(mesh, validation);
    let actual_volume_m3 = mesh_volume_m3(mesh);
    let actual_boundary_area_m2 = mesh_boundary_area_m2(mesh);

    MeshBenchmarkReport {
        schema_version: MESH_BENCHMARK_SCHEMA_VERSION.to_string(),
        benchmark_id: input.benchmark_id,
        tier: input.tier,
        mesh_id: mesh.mesh_id.clone(),
        backend: mesh.backend.backend.clone(),
        algorithm: mesh.backend.algorithm.clone(),
        timing: input.timing,
        topology: MeshBenchmarkTopologyMetrics {
            node_count: mesh.nodes.len(),
            volume_element_count: mesh.volume_elements.len(),
            boundary_face_count: mesh.boundary_faces.len(),
            boundary_edge_count: mesh.boundary_edges.len(),
            volume_component_count: volume_component_count(mesh),
        },
        cad: evidence.cad,
        coverage: MeshBenchmarkCoverageMetrics {
            expected_volume_m3: validation.expected_volume_m3,
            actual_volume_m3,
            volume_coverage_ratio: ratio(actual_volume_m3, validation.expected_volume_m3),
            expected_boundary_area_m2: validation.expected_boundary_area_m2,
            actual_boundary_area_m2,
            boundary_area_ratio: ratio(
                actual_boundary_area_m2,
                validation.expected_boundary_area_m2,
            ),
            coverage_sample_ratio: evidence.validation.coverage_sample_ratio,
            boundary_face_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_face_recovery_ratio,
            boundary_edge_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
        },
        quality: evidence.quality,
        tet_recovery: evidence.tet_recovery,
        regions: evidence.regions,
        solve_readiness: MeshBenchmarkSolveReadiness {
            solve_ready: evidence.validation.solve_ready,
            validation_error_code: evidence.validation.validation_error_code,
            validation_error_message: evidence.validation.validation_error_message,
        },
    }
}

fn ratio(actual: f64, expected: Option<f64>) -> Option<f64> {
    let expected = expected?;
    if !actual.is_finite() || !expected.is_finite() || expected.abs() <= f64::EPSILON {
        return None;
    }
    Some(actual / expected)
}

fn mesh_volume_m3(mesh: &AnalysisMeshArtifact) -> f64 {
    let nodes = node_coordinates(mesh);
    mesh.volume_elements
        .iter()
        .filter_map(|element| match element.kind {
            VolumeElementKind::Tet4 if element.node_ids.len() == 4 => Some(tet_volume([
                *nodes.get(&element.node_ids[0])?,
                *nodes.get(&element.node_ids[1])?,
                *nodes.get(&element.node_ids[2])?,
                *nodes.get(&element.node_ids[3])?,
            ])),
            _ => None,
        })
        .sum()
}

fn mesh_boundary_area_m2(mesh: &AnalysisMeshArtifact) -> f64 {
    let nodes = node_coordinates(mesh);
    mesh.boundary_faces
        .iter()
        .filter(|face| face.node_ids.len() == 3)
        .filter_map(|face| {
            Some(triangle_area([
                *nodes.get(&face.node_ids[0])?,
                *nodes.get(&face.node_ids[1])?,
                *nodes.get(&face.node_ids[2])?,
            ]))
        })
        .sum()
}

fn node_coordinates(mesh: &AnalysisMeshArtifact) -> BTreeMap<u32, [f64; 3]> {
    mesh.nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshNode, AnalysisVolumeElement,
            MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::{AnalysisMeshQualityReport, ElementQuality},
        sizing::MeshSizingField,
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    #[test]
    fn benchmark_report_records_solve_ready_mesh_metrics() {
        let mesh = fixture_mesh();
        let validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0 / 6.0),
            expected_boundary_area_m2: Some(0.5),
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
            ..AnalysisMeshValidationOptions::default()
        };
        let input = MeshBenchmarkInput {
            benchmark_id: "unit_tet".to_string(),
            tier: MeshBenchmarkTier::Solid3d,
            timing: MeshBenchmarkTiming {
                topology_import_ms: Some(1.0),
                volume_generation_ms: Some(2.0),
                total_ms: Some(3.0),
                ..MeshBenchmarkTiming::default()
            },
        };

        let report = build_mesh_benchmark_report(&mesh, &validation, input);

        assert_eq!(report.schema_version, MESH_BENCHMARK_SCHEMA_VERSION);
        assert_eq!(report.benchmark_id, "unit_tet");
        assert_eq!(report.tier, MeshBenchmarkTier::Solid3d);
        assert_eq!(report.topology.node_count, 4);
        assert_eq!(report.topology.volume_element_count, 1);
        assert_eq!(report.topology.volume_component_count, 1);
        assert_eq!(report.coverage.volume_coverage_ratio, Some(1.0));
        assert_eq!(report.coverage.boundary_area_ratio, Some(1.0));
        assert_eq!(report.coverage.coverage_sample_ratio, Some(1.0));
        assert_eq!(report.quality.exact_scaled_jacobian_p50, Some(0.45));
        assert!(report.solve_readiness.solve_ready);
        assert_eq!(report.solve_readiness.validation_error_code, None);
        assert_eq!(report.timing.total_ms, Some(3.0));
    }

    #[test]
    fn benchmark_report_preserves_validation_failure() {
        let mesh = fixture_mesh();
        let validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0),
            min_volume_coverage_ratio: 0.95,
            ..AnalysisMeshValidationOptions::default()
        };

        let report = build_mesh_benchmark_report(
            &mesh,
            &validation,
            MeshBenchmarkInput::new("underfilled", MeshBenchmarkTier::Solid3d),
        );

        assert!(!report.solve_readiness.solve_ready);
        assert_eq!(
            report.solve_readiness.validation_error_code.as_deref(),
            Some("volume_coverage_failed")
        );
        assert_eq!(report.coverage.volume_coverage_ratio, Some(1.0 / 6.0));
    }

    fn fixture_mesh() -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_1".to_string(),
            nodes: vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "tet_1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "solid".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "face_1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["tet_1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: vec![
                boundary_edge("edge_1", [1, 2]),
                boundary_edge("edge_2", [2, 3]),
                boundary_edge("edge_3", [1, 3]),
            ],
            quality: AnalysisMeshQualityReport {
                min_scaled_jacobian: 0.5,
                min_exact_scaled_jacobian: 0.45,
                mean_aspect_ratio: 2.0,
                max_aspect_ratio: 2.0,
                inverted_element_count: 0,
                mean_boundary_projection_error_m: 0.0,
                max_boundary_projection_error_m: 0.0,
                elements: vec![ElementQuality {
                    element_id: "tet_1".to_string(),
                    scaled_jacobian: 0.5,
                    exact_scaled_jacobian: 0.45,
                    aspect_ratio: 2.0,
                    volume_m3: 1.0 / 6.0,
                }],
            },
            sizing: MeshSizingField::default(),
            backend: MeshBackendSummary {
                backend: "production".to_string(),
                algorithm: "test".to_string(),
                tet_candidate_count: 1,
                tet_recovered_component_ratio: 1.0,
                tet_candidate_volume_ratio: 1.0,
                boundary_face_recovery_ratio: 1.0,
                boundary_edge_recovery_ratio: 1.0,
                ..MeshBackendSummary::default()
            },
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
        AnalysisMeshNode {
            node_id,
            coordinates_m,
            provenance: Vec::new(),
        }
    }

    fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
        AnalysisBoundaryEdge {
            edge_id: edge_id.to_string(),
            node_ids,
            adjacent_boundary_face_ids: vec!["face_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }
    }
}

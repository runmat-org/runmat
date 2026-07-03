use super::*;
use std::collections::BTreeMap;

use crate::{
    artifact::AnalysisMeshArtifact,
    evidence::{build_mesh_evidence_artifact, MeshEvidenceArtifact},
    predicate::{tetrahedron_volume, triangle_area},
    topology::VolumeElementKind,
    validation::{volume_component_count, AnalysisMeshValidationOptions},
};

pub fn build_mesh_benchmark_report(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    input: MeshBenchmarkInput,
) -> MeshBenchmarkReport {
    let evidence = build_mesh_evidence_artifact(mesh, validation);
    let artifacts = benchmark_artifact_metrics(mesh, &evidence);
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
        budget: benchmark_budget_metrics(mesh, validation),
        artifacts,
        topology: MeshBenchmarkTopologyMetrics {
            node_count: mesh.nodes.len(),
            volume_element_count: mesh.volume_elements.len(),
            boundary_face_count: mesh.boundary_faces.len(),
            boundary_edge_count: mesh.boundary_edges.len(),
            volume_component_count: volume_component_count(mesh),
        },
        cad: evidence.cad,
        sizing: evidence.sizing,
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
        tetrahedron_recovery: evidence.tetrahedron_recovery,
        regions: evidence.regions,
        solve_readiness: MeshBenchmarkSolveReadiness {
            solve_ready: evidence.validation.solve_ready,
            validation_error_code: evidence.validation.validation_error_code,
            validation_error_message: evidence.validation.validation_error_message,
            required_boundary_region_ids: evidence.validation.required_boundary_region_ids,
            required_material_region_ids: evidence.validation.required_material_region_ids,
            fan_fallback_component_count: evidence.validation.fan_fallback_component_count,
            unrepaired_exact_quality_total_count: evidence
                .validation
                .unrepaired_exact_quality_total_count,
            unrepaired_exact_quality_general_cavity_count: evidence
                .validation
                .unrepaired_exact_quality_general_cavity_count,
            unrepaired_exact_quality_boundary_adjacent_count: evidence
                .validation
                .unrepaired_exact_quality_boundary_adjacent_count,
            unrepaired_exact_quality_node_adjacent_count: evidence
                .validation
                .unrepaired_exact_quality_node_adjacent_count,
            unrepaired_exact_quality_interior_seed_count: evidence
                .validation
                .unrepaired_exact_quality_interior_seed_count,
            unrepaired_exact_quality_edge_star_count: evidence
                .validation
                .unrepaired_exact_quality_edge_star_count,
        },
    }
}

fn benchmark_budget_metrics(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshBenchmarkBudgetMetrics {
    let Some(max_volume_element_count) = validation.max_volume_element_count else {
        return MeshBenchmarkBudgetMetrics::default();
    };
    let volume_element_count = mesh.volume_elements.len();
    MeshBenchmarkBudgetMetrics {
        max_volume_element_count: Some(max_volume_element_count),
        volume_element_budget_used_ratio: Some(
            volume_element_count as f64 / max_volume_element_count.max(1) as f64,
        ),
        volume_element_budget_exceeded: volume_element_count > max_volume_element_count,
    }
}

fn benchmark_artifact_metrics(
    mesh: &AnalysisMeshArtifact,
    evidence: &MeshEvidenceArtifact,
) -> MeshBenchmarkArtifactMetrics {
    MeshBenchmarkArtifactMetrics {
        analysis_mesh_json_bytes: serde_json::to_vec(mesh).ok().map(|bytes| bytes.len()),
        mesh_evidence_json_bytes: serde_json::to_vec(evidence).ok().map(|bytes| bytes.len()),
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
            VolumeElementKind::Tetrahedron4 if element.node_ids.len() == 4 => {
                Some(tetrahedron_volume([
                    *nodes.get(&element.node_ids[0])?,
                    *nodes.get(&element.node_ids[1])?,
                    *nodes.get(&element.node_ids[2])?,
                    *nodes.get(&element.node_ids[3])?,
                ]))
            }
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

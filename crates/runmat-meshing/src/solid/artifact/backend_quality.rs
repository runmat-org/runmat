use std::collections::BTreeMap;

use runmat_meshing_core::quality::{
    predicate::{tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume},
    AnalysisMeshQualityReport, ElementQuality,
};
use runmat_meshing_opt::sliver::{
    classify_sliver_tetrahedra, evaluate_sliver_removal, SliverRecoveryOptions,
    SliverTetrahedronQuality,
};
use runmat_meshing_tetrahedron::generate::TetrahedronMesh;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::solid) struct BackendQualityEvidence {
    pub(super) min_exact_scaled_jacobian: f64,
    pub(super) exact_scaled_jacobian_below_threshold_count: usize,
    pub(super) exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub(super) sliver_count: usize,
    pub(super) quality_repair_target_count: usize,
    pub(super) max_aspect_ratio: f64,
    sliver_inputs: Vec<SliverTetrahedronQuality>,
}

pub(in crate::solid) fn backend_quality_evidence_from_tetrahedron_mesh(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BackendQualityEvidence {
    let coordinates_by_node_id = tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let elements = tetrahedron_mesh
        .elements
        .iter()
        .filter_map(|element| {
            let points = [
                *coordinates_by_node_id.get(&element.node_ids[0])?,
                *coordinates_by_node_id.get(&element.node_ids[1])?,
                *coordinates_by_node_id.get(&element.node_ids[2])?,
                *coordinates_by_node_id.get(&element.node_ids[3])?,
            ];
            Some(ElementQuality {
                element_id: element.element_id.id.clone(),
                scaled_jacobian: tetrahedron_scaled_jacobian(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                volume_m3: tetrahedron_volume(points),
            })
        })
        .collect::<Vec<_>>();
    backend_quality_evidence(&AnalysisMeshQualityReport {
        min_scaled_jacobian: elements
            .iter()
            .map(|element| element.scaled_jacobian)
            .fold(f64::INFINITY, f64::min)
            .min(1.0),
        min_exact_scaled_jacobian: elements
            .iter()
            .map(|element| element.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min)
            .min(1.0),
        mean_aspect_ratio: if elements.is_empty() {
            0.0
        } else {
            elements
                .iter()
                .map(|element| element.aspect_ratio)
                .sum::<f64>()
                / elements.len() as f64
        },
        max_aspect_ratio: elements
            .iter()
            .map(|element| element.aspect_ratio)
            .fold(0.0_f64, f64::max),
        inverted_element_count: elements
            .iter()
            .filter(|element| element.volume_m3 <= 0.0)
            .count(),
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OptimizationTargetEvidence {
    pub(super) target_seed_count: usize,
    pub(super) skipped_target_seed_count: usize,
    pub(super) sliver_removed_count: usize,
}

pub(super) fn optimization_target_evidence(
    initial: &BackendQualityEvidence,
    final_quality: &BackendQualityEvidence,
) -> OptimizationTargetEvidence {
    OptimizationTargetEvidence {
        target_seed_count: initial.quality_repair_target_count,
        skipped_target_seed_count: final_quality.quality_repair_target_count,
        sliver_removed_count: evaluate_sliver_removal(
            &initial.sliver_inputs,
            &final_quality.sliver_inputs,
            SliverRecoveryOptions::default(),
        )
        .ok()
        .filter(|evaluation| evaluation.accepted)
        .map_or(0, |evaluation| evaluation.removed_sliver_count),
    }
}

pub(super) fn backend_quality_evidence(
    quality: &AnalysisMeshQualityReport,
) -> BackendQualityEvidence {
    let options = SliverRecoveryOptions::default();
    let min_exact_scaled_jacobian = if quality.elements.is_empty() {
        0.0
    } else {
        quality
            .elements
            .iter()
            .map(|element| element.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min)
    };
    let exact_scaled_jacobian_below_threshold_count = quality
        .elements
        .iter()
        .filter(|element| element.exact_scaled_jacobian < options.min_exact_scaled_jacobian)
        .count();
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    for element in &quality.elements {
        *exact_scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.exact_scaled_jacobian))
            .or_default() += 1;
    }
    let sliver_inputs = quality
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| SliverTetrahedronQuality {
            tetrahedron_id: index as u32 + 1,
            aspect_ratio: element.aspect_ratio,
            exact_scaled_jacobian: element.exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
    let sliver_count =
        classify_sliver_tetrahedra(&sliver_inputs, options).map_or(0, |slivers| slivers.len());

    BackendQualityEvidence {
        min_exact_scaled_jacobian,
        exact_scaled_jacobian_below_threshold_count,
        exact_scaled_jacobian_bins,
        sliver_count,
        quality_repair_target_count: sliver_count.max(exact_scaled_jacobian_below_threshold_count),
        max_aspect_ratio: quality.max_aspect_ratio,
        sliver_inputs,
    }
}

fn scaled_jacobian_bin(value: f64) -> String {
    if !value.is_finite() {
        return "non_finite".to_string();
    }
    match value {
        value if value < 0.0 => "negative".to_string(),
        value if value < 0.15 => "0_00_to_0_15".to_string(),
        value if value < 0.35 => "0_15_to_0_35".to_string(),
        value if value < 0.65 => "0_35_to_0_65".to_string(),
        _ => "0_65_to_1_00".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{
        MeshingStage, StageEvidence, Tetrahedron4Element, TetrahedronMeshNode, TopologyEntityId,
    };

    #[test]
    fn backend_quality_evidence_from_tetrahedron_mesh_classifies_sliver_targets() {
        let evidence = backend_quality_evidence_from_tetrahedron_mesh(&tetrahedron_quality_mesh([
            0.01, 0.01, 0.001,
        ]));

        assert!(evidence.sliver_count > 0);
        assert!(evidence.quality_repair_target_count > 0);
        assert!(evidence.max_aspect_ratio > SliverRecoveryOptions::default().sliver_aspect_ratio);
        assert!(evidence.min_exact_scaled_jacobian.is_finite());
    }

    #[test]
    fn backend_quality_evidence_from_tetrahedron_mesh_reports_regular_mesh_without_targets() {
        let evidence = backend_quality_evidence_from_tetrahedron_mesh(&tetrahedron_quality_mesh([
            0.0, 0.0, 1.0,
        ]));

        assert_eq!(evidence.sliver_count, 0);
        assert_eq!(evidence.quality_repair_target_count, 0);
        assert!(evidence.min_exact_scaled_jacobian >= 0.15);
    }

    #[test]
    fn optimization_target_evidence_reports_remaining_targets_as_skipped() {
        let initial = backend_quality_evidence_from_tetrahedron_mesh(&tetrahedron_quality_mesh([
            0.01, 0.01, 0.001,
        ]));
        let final_quality =
            backend_quality_evidence_from_tetrahedron_mesh(&tetrahedron_quality_mesh([
                0.0, 0.0, 1.0,
            ]));

        let targets = optimization_target_evidence(&initial, &final_quality);

        assert!(targets.target_seed_count > 0);
        assert_eq!(targets.skipped_target_seed_count, 0);
        assert_eq!(targets.sliver_removed_count, initial.sliver_count);
    }

    #[test]
    fn optimization_target_evidence_keeps_unresolved_slivers_as_skipped_targets() {
        let initial = backend_quality_evidence_from_tetrahedron_mesh(&tetrahedron_quality_mesh([
            0.01, 0.01, 0.001,
        ]));
        let final_quality = initial.clone();

        let targets = optimization_target_evidence(&initial, &final_quality);

        assert_eq!(
            targets.target_seed_count,
            initial.quality_repair_target_count
        );
        assert_eq!(
            targets.skipped_target_seed_count,
            initial.quality_repair_target_count
        );
        assert_eq!(targets.sliver_removed_count, 0);
    }

    #[test]
    fn optimization_target_evidence_rejects_sliver_removal_when_exact_quality_regresses() {
        let initial = BackendQualityEvidence {
            min_exact_scaled_jacobian: 0.42,
            exact_scaled_jacobian_below_threshold_count: 0,
            exact_scaled_jacobian_bins: BTreeMap::new(),
            sliver_count: 1,
            quality_repair_target_count: 1,
            max_aspect_ratio: 30.0,
            sliver_inputs: vec![SliverTetrahedronQuality {
                tetrahedron_id: 1,
                aspect_ratio: 30.0,
                exact_scaled_jacobian: 0.42,
            }],
        };
        let final_quality = BackendQualityEvidence {
            min_exact_scaled_jacobian: 0.05,
            exact_scaled_jacobian_below_threshold_count: 1,
            exact_scaled_jacobian_bins: BTreeMap::new(),
            sliver_count: 0,
            quality_repair_target_count: 1,
            max_aspect_ratio: 10.0,
            sliver_inputs: vec![SliverTetrahedronQuality {
                tetrahedron_id: 1,
                aspect_ratio: 10.0,
                exact_scaled_jacobian: 0.05,
            }],
        };

        let targets = optimization_target_evidence(&initial, &final_quality);

        assert_eq!(targets.sliver_removed_count, 0);
        assert_eq!(targets.skipped_target_seed_count, 1);
    }

    fn tetrahedron_quality_mesh(apex: [f64; 3]) -> TetrahedronMesh {
        let node_ids = [entity("n0"), entity("n1"), entity("n2"), entity("n3")];
        TetrahedronMesh {
            mesh_id: "quality_fixture".to_string(),
            tetrahedron_generation_family: "test".to_string(),
            nodes: vec![
                node(node_ids[0].clone(), [0.0, 0.0, 0.0]),
                node(node_ids[1].clone(), [1.0, 0.0, 0.0]),
                node(node_ids[2].clone(), [0.0, 1.0, 0.0]),
                node(node_ids[3].clone(), apex),
            ],
            elements: vec![Tetrahedron4Element {
                element_id: entity("e0"),
                node_ids,
                material_region_id: "material".to_string(),
            }],
            boundary_faces: Vec::new(),
            recovery_complete: true,
            quality_optimized: false,
            evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
        }
    }

    fn node(node_id: TopologyEntityId, coordinates_m: [f64; 3]) -> TetrahedronMeshNode {
        TetrahedronMeshNode {
            node_id,
            coordinates_m,
        }
    }

    fn entity(id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: id.to_string(),
        }
    }
}

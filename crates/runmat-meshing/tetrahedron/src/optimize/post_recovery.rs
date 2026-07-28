use runmat_meshing_core::contracts::TetrahedronMesh;

use crate::reconnect::{
    improve_tetrahedron_mesh_with_local_flips, TetrahedronMeshLocalReconnectionOptions,
    TetrahedronMeshLocalReconnectionReport,
};

use super::{
    remove_tetrahedron_mesh_slivers, repair_tetrahedron_mesh_exact_quality,
    smooth_tetrahedron_mesh_boundary_with_projector, smooth_tetrahedron_mesh_interior,
    untangle_tetrahedron_mesh_interior, TetrahedronBoundarySmoothingProjector,
    TetrahedronMeshBoundarySmoothingOptions, TetrahedronMeshBoundarySmoothingReport,
    TetrahedronMeshExactQualityRepairOptions, TetrahedronMeshExactQualityRepairReport,
    TetrahedronMeshInteriorSmoothingOptions, TetrahedronMeshInteriorSmoothingReport,
    TetrahedronMeshSliverRemovalOptions, TetrahedronMeshSliverRemovalReport,
    TetrahedronMeshUntanglingOptions, TetrahedronMeshUntanglingReport,
};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecoveredTetrahedronMeshOptimizationOptions {
    pub local_reconnection: TetrahedronMeshLocalReconnectionOptions,
    pub untangling: TetrahedronMeshUntanglingOptions,
    pub exact_quality_repair: TetrahedronMeshExactQualityRepairOptions,
    pub sliver_removal: TetrahedronMeshSliverRemovalOptions,
    pub interior_smoothing: TetrahedronMeshInteriorSmoothingOptions,
    pub boundary_smoothing: TetrahedronMeshBoundarySmoothingOptions,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveredTetrahedronMeshOptimizationReport {
    pub local_reconnection: TetrahedronMeshLocalReconnectionReport,
    pub untangling: TetrahedronMeshUntanglingReport,
    pub exact_quality_repair: TetrahedronMeshExactQualityRepairReport,
    pub sliver_removal: TetrahedronMeshSliverRemovalReport,
    pub interior_smoothing: TetrahedronMeshInteriorSmoothingReport,
    pub boundary_smoothing: TetrahedronMeshBoundarySmoothingReport,
}

pub fn optimize_recovered_tetrahedron_mesh(
    mesh: &mut TetrahedronMesh,
    boundary_projector: &impl TetrahedronBoundarySmoothingProjector,
    options: RecoveredTetrahedronMeshOptimizationOptions,
) -> RecoveredTetrahedronMeshOptimizationReport {
    let local_reconnection =
        improve_tetrahedron_mesh_with_local_flips(mesh, options.local_reconnection);
    let untangling = untangle_tetrahedron_mesh_interior(mesh, options.untangling);
    let exact_quality_repair =
        repair_tetrahedron_mesh_exact_quality(mesh, options.exact_quality_repair);
    let sliver_removal = remove_tetrahedron_mesh_slivers(mesh, options.sliver_removal);
    let interior_smoothing = smooth_tetrahedron_mesh_interior(mesh, options.interior_smoothing);
    let boundary_smoothing = smooth_tetrahedron_mesh_boundary_with_projector(
        mesh,
        boundary_projector,
        options.boundary_smoothing,
    );

    RecoveredTetrahedronMeshOptimizationReport {
        local_reconnection,
        untangling,
        exact_quality_repair,
        sliver_removal,
        interior_smoothing,
        boundary_smoothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::{
        contracts::{
            MeshingStage, StageEvidence, Tetrahedron4Element, TetrahedronBoundaryFace,
            TetrahedronMeshNode, TopologyEntityId,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
        },
        quality::predicate::Point3,
    };
    use runmat_meshing_opt::smooth::TetrahedronSmoothingOptions;

    #[test]
    fn post_recovery_optimization_sequences_reconnection_and_smoothing() {
        let mut mesh = smoothing_fixture([0.2, 0.2, 0.02]);
        let projector = PlanarProjector;

        let report = optimize_recovered_tetrahedron_mesh(
            &mut mesh,
            &projector,
            RecoveredTetrahedronMeshOptimizationOptions {
                local_reconnection: TetrahedronMeshLocalReconnectionOptions {
                    max_attempted_reconnections: 0,
                    ..TetrahedronMeshLocalReconnectionOptions::default()
                },
                untangling: TetrahedronMeshUntanglingOptions {
                    max_attempted_seeds: 0,
                    ..TetrahedronMeshUntanglingOptions::default()
                },
                exact_quality_repair: TetrahedronMeshExactQualityRepairOptions {
                    max_attempted_seeds: 0,
                    ..TetrahedronMeshExactQualityRepairOptions::default()
                },
                sliver_removal: TetrahedronMeshSliverRemovalOptions {
                    max_attempted_elements: 0,
                    ..TetrahedronMeshSliverRemovalOptions::default()
                },
                interior_smoothing: TetrahedronMeshInteriorSmoothingOptions {
                    smoothing: TetrahedronSmoothingOptions {
                        min_volume_m3: 1.0e-18,
                        min_scaled_jacobian: 0.15,
                        min_scaled_jacobian_improvement: 1.0e-12,
                        max_aspect_ratio_growth: 10.0,
                    },
                    max_attempted_points: 4,
                    max_accepted_points: 1,
                    relaxation: 0.5,
                },
                boundary_smoothing: TetrahedronMeshBoundarySmoothingOptions {
                    max_attempted_points: 0,
                    ..TetrahedronMeshBoundarySmoothingOptions::default()
                },
            },
        );

        assert_eq!(report.local_reconnection.attempted_reconnection_count, 0);
        assert_eq!(report.untangling.attempted_seed_count, 0);
        assert_eq!(report.exact_quality_repair.attempted_seed_count, 0);
        assert_eq!(report.sliver_removal.attempted_element_count, 0);
        assert_eq!(report.interior_smoothing.attempted_point_count, 1);
        assert_eq!(report.interior_smoothing.accepted_point_count, 1);
        assert_eq!(report.boundary_smoothing.attempted_point_count, 0);
        assert!(mesh.quality_optimized);
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT],
            1
        );
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT],
            1
        );
    }

    struct PlanarProjector;

    impl TetrahedronBoundarySmoothingProjector for PlanarProjector {
        fn project_to_source_face(
            &self,
            _source_face_id: &TopologyEntityId,
            point_m: Point3,
        ) -> Option<super::super::TetrahedronBoundarySmoothingProjection> {
            Some(super::super::TetrahedronBoundarySmoothingProjection {
                point_m,
                distance_m: 0.0,
                in_bounds: true,
            })
        }
    }

    fn smoothing_fixture(interior: Point3) -> TetrahedronMesh {
        TetrahedronMesh {
            mesh_id: "post_recovery_optimization_fixture".to_string(),
            tetrahedron_generation_family: "unknown".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
                node("4", interior),
            ],
            elements: vec![
                element("0", ["4", "0", "1", "2"]),
                element("1", ["4", "0", "1", "3"]),
                element("2", ["4", "0", "2", "3"]),
                element("3", ["4", "1", "2", "3"]),
            ],
            boundary_faces: vec![
                boundary_face("boundary_0", ["0", "1", "2"]),
                boundary_face("boundary_1", ["0", "1", "3"]),
                boundary_face("boundary_2", ["0", "2", "3"]),
                boundary_face("boundary_3", ["1", "2", "3"]),
            ],
            recovery_complete: true,
            quality_optimized: false,
            evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
        }
    }

    fn element(id: &str, node_ids: [&str; 4]) -> Tetrahedron4Element {
        Tetrahedron4Element {
            element_id: entity(id),
            node_ids: node_ids.map(entity),
            material_region_id: "body".to_string(),
        }
    }

    fn boundary_face(id: &str, node_ids: [&str; 3]) -> TetrahedronBoundaryFace {
        TetrahedronBoundaryFace {
            face_id: entity(id),
            node_ids: node_ids.map(entity),
            source_face_id: entity(id),
            source_edge_ids: [None, None, None],
        }
    }

    fn node(id: &str, coordinates_m: Point3) -> TetrahedronMeshNode {
        TetrahedronMeshNode {
            node_id: entity(id),
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

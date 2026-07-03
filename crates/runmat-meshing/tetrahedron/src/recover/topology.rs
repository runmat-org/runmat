use runmat_meshing_core::contracts::{MeshingStage, TopologyEntityId};

pub(super) fn topology_face_edges(node_ids: [TopologyEntityId; 3]) -> [[TopologyEntityId; 2]; 3] {
    [
        sorted_topology_ids([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

pub(super) fn sorted_topology_ids<const N: usize>(
    mut node_ids: [TopologyEntityId; N],
) -> [TopologyEntityId; N] {
    node_ids.sort_by(|left, right| {
        topology_stage_rank(left.stage)
            .cmp(&topology_stage_rank(right.stage))
            .then_with(|| left.id.cmp(&right.id))
    });
    node_ids
}

fn topology_stage_rank(stage: MeshingStage) -> u8 {
    match stage {
        MeshingStage::CadTopology => 0,
        MeshingStage::Sizing => 1,
        MeshingStage::CurveMesh => 2,
        MeshingStage::SurfaceMesh => 3,
        MeshingStage::ProtectedBoundaryComplex => 4,
        MeshingStage::TetrahedronMesh => 5,
        MeshingStage::ConstraintRecovery => 6,
        MeshingStage::Optimization => 7,
        MeshingStage::SolveReadiness => 8,
    }
}

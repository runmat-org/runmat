use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    MeshingStage, ProtectedBoundaryComplex, Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
};

use super::{
    topology::sorted_topology_ids, TetrahedronMaterialInterfaceTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

mod boundary;
mod candidate;
mod geometry;

use boundary::{
    insert_material_partition_boundary_faces, material_partition_boundary_contract_is_satisfied,
};
use candidate::select_candidate_partition;

pub(super) struct MaterialPartitionRecovery {
    pub attempted_material_interface_count: usize,
    pub inserted_material_interface_count: usize,
    pub inserted_element_count: usize,
    pub inserted_boundary_face_count: usize,
    pub rejected_material_interface_count: usize,
    pub rolled_back_material_interface_count: usize,
    pub rolled_back_element_count: usize,
    pub rolled_back_boundary_face_count: usize,
    pub topology_candidate_count: usize,
    pub usable_candidate_count: usize,
    pub rejected_existing_candidate_count: usize,
    pub rejected_quality_candidate_count: usize,
    pub rejected_interior_candidate_set_count: usize,
    pub rejection_counts: BTreeMap<&'static str, usize>,
}

pub(super) fn recover_absent_material_interface_partitions(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> MaterialPartitionRecovery {
    let recoverable_material_interfaces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.material_interface_topology
                    == Some(TetrahedronMaterialInterfaceTopology::AbsentPartition)
        })
        .filter_map(|item| item.material_interface_id.clone())
        .collect::<BTreeSet<_>>();

    let mut recovery = MaterialPartitionRecovery {
        attempted_material_interface_count: 0,
        inserted_material_interface_count: 0,
        inserted_element_count: 0,
        inserted_boundary_face_count: 0,
        rejected_material_interface_count: 0,
        rolled_back_material_interface_count: 0,
        rolled_back_element_count: 0,
        rolled_back_boundary_face_count: 0,
        topology_candidate_count: 0,
        usable_candidate_count: 0,
        rejected_existing_candidate_count: 0,
        rejected_quality_candidate_count: 0,
        rejected_interior_candidate_set_count: 0,
        rejection_counts: BTreeMap::new(),
    };

    for material_interface_id in recoverable_material_interfaces {
        recovery.attempted_material_interface_count += 1;
        match insert_absent_material_interface_partition(
            plc,
            tetrahedron_mesh,
            &material_interface_id,
            &mut recovery,
        ) {
            Ok(inserted_partition) => {
                recovery.inserted_material_interface_count += 1;
                recovery.inserted_element_count += inserted_partition.element_count;
                recovery.inserted_boundary_face_count += inserted_partition.boundary_face_count;
            }
            Err(rejection) => {
                recovery.rejected_material_interface_count += 1;
                *recovery
                    .rejection_counts
                    .entry(rejection.evidence_key())
                    .or_default() += 1;
            }
        }
    }

    recovery
}

fn insert_absent_material_interface_partition(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    material_interface_id: &str,
    recovery: &mut MaterialPartitionRecovery,
) -> Result<InsertedMaterialPartition, MaterialPartitionRecoveryRejection> {
    let material_facets = plc
        .facets
        .iter()
        .filter(|facet| {
            facet
                .material_interface_ids
                .iter()
                .any(|facet_material_interface_id| {
                    facet_material_interface_id == material_interface_id
                })
        })
        .collect::<Vec<_>>();
    if !(3..=12).contains(&material_facets.len()) {
        return Err(MaterialPartitionRecoveryRejection::FacetCount);
    }

    let material_facet_face_keys = material_facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let rollback_mesh = tetrahedron_mesh.clone();
    let candidate_partition = select_candidate_partition(
        tetrahedron_mesh,
        &material_facets,
        &material_facet_face_keys,
        recovery,
    )?;
    let inserted_element_count = candidate_partition.elements.len();

    for (element_index, node_ids) in candidate_partition.elements.into_iter().enumerate() {
        tetrahedron_mesh.elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("material_partition:{material_interface_id}:{element_index}"),
            },
            node_ids,
            material_region_id: material_interface_id.to_string(),
        });
    }

    let inserted_boundary_face_count =
        insert_material_partition_boundary_faces(plc, &material_facets, tetrahedron_mesh);
    if !material_partition_boundary_contract_is_satisfied(plc, &material_facets, tetrahedron_mesh) {
        *tetrahedron_mesh = rollback_mesh;
        recovery.rolled_back_material_interface_count += 1;
        recovery.rolled_back_element_count += inserted_element_count;
        recovery.rolled_back_boundary_face_count += inserted_boundary_face_count;
        return Err(MaterialPartitionRecoveryRejection::PostInsertionAudit);
    }
    Ok(InsertedMaterialPartition {
        element_count: inserted_element_count,
        boundary_face_count: inserted_boundary_face_count,
    })
}

struct InsertedMaterialPartition {
    element_count: usize,
    boundary_face_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaterialPartitionRecoveryRejection {
    FacetCount,
    FacetTopology,
    ElementAlreadyExists,
    InteriorFaceTopology,
    QualityGate,
    PostInsertionAudit,
}

impl MaterialPartitionRecoveryRejection {
    fn evidence_key(self) -> &'static str {
        match self {
            Self::FacetCount => "rejected_absent_material_partition_facet_count",
            Self::FacetTopology => "rejected_absent_material_partition_facet_topology",
            Self::ElementAlreadyExists => "rejected_absent_material_partition_element_exists",
            Self::InteriorFaceTopology => {
                "rejected_absent_material_partition_interior_face_topology"
            }
            Self::QualityGate => "rejected_absent_material_partition_quality_gate",
            Self::PostInsertionAudit => "rejected_absent_material_partition_post_insertion_audit",
        }
    }
}

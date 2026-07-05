use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{
    MeshingStage, PlcFacet, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
};
use runmat_meshing_core::quality::tolerance::MeshingTolerance;
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use super::convex_polyhedron::bounds::plc_coordinates_and_bounds;
use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::material::plc_material_region_id;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use crate::protected_edges::source_edge_ids_for_boundary_face_edges;

mod partition;
mod refill;
mod shell;
use refill::{refill_nested_tetrahedron_shell_cavity, NestedTetrahedronShellRefillStrategy};
use shell::nested_tetrahedron_shell;

pub fn generate_nested_tetrahedron_shell_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_protected_boundary_complex(plc)
        .map_err(|error| TetrahedronGenerationError::InvalidProtectedBoundaryComplex { error })?;
    let (coordinates_by_id, bounds) = plc_coordinates_and_bounds(plc)?;
    let tolerance = MeshingTolerance::from_bounds(bounds[0], bounds[1]);
    let shell = nested_tetrahedron_shell(plc, &coordinates_by_id, tolerance)?;
    let target_volume_m3 = shell.outer_volume_m3 - shell.inner_volume_m3;
    if !target_volume_m3.is_finite() || target_volume_m3 <= 0.0 {
        return Err(TetrahedronGenerationError::DegenerateNestedTetrahedronShellPlc);
    }

    let refill = refill_nested_tetrahedron_shell_cavity(plc, &shell, target_volume_m3)?;

    let material_region_id = plc_material_region_id(plc);
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.extend(
        refill
            .generated_nodes
            .iter()
            .map(
                |node| -> Result<TetrahedronMeshNode, TetrahedronGenerationError> {
                    Ok(TetrahedronMeshNode {
                        node_id: mesh_node_id(&refill.cavity_id_to_node_id, node.node_id)?,
                        coordinates_m: node.coordinates_m,
                    })
                },
            )
            .collect::<Result<Vec<_>, _>>()?,
    );
    let coordinates_by_mesh_node_id = nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let elements = refill
        .refill
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| {
            Ok(Tetrahedron4Element {
                element_id: TopologyEntityId {
                    stage: MeshingStage::TetrahedronMesh,
                    id: format!("nested_tetrahedron_shell_tetrahedron_{index}"),
                },
                node_ids: [
                    mesh_node_id(&refill.cavity_id_to_node_id, tetrahedron.node_ids[0])?,
                    mesh_node_id(&refill.cavity_id_to_node_id, tetrahedron.node_ids[1])?,
                    mesh_node_id(&refill.cavity_id_to_node_id, tetrahedron.node_ids[2])?,
                    mesh_node_id(&refill.cavity_id_to_node_id, tetrahedron.node_ids[3])?,
                ],
                material_region_id: material_region_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, TetrahedronGenerationError>>()?;
    let boundary_faces = refill
        .refill
        .boundary_faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            let node_ids = [
                mesh_node_id(&refill.cavity_id_to_node_id, face.node_ids[0])?,
                mesh_node_id(&refill.cavity_id_to_node_id, face.node_ids[1])?,
                mesh_node_id(&refill.cavity_id_to_node_id, face.node_ids[2])?,
            ];
            let source_facet = source_facet_for_cavity_face(plc, face.source_face_id)?;
            Ok(TetrahedronBoundaryFace {
                face_id: TopologyEntityId {
                    stage: MeshingStage::TetrahedronMesh,
                    id: format!("nested_tetrahedron_shell_boundary_face_{index}"),
                },
                node_ids: node_ids.clone(),
                source_face_id: source_facet.source_face_id.clone(),
                source_edge_ids: source_edge_ids_for_boundary_face_edges(
                    &plc.protected_edges,
                    &coordinates_by_mesh_node_id,
                    node_ids,
                    tolerance.absolute_m,
                ),
            })
        })
        .collect::<Result<Vec<_>, TetrahedronGenerationError>>()?;

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_outer_nodes".to_string(),
        shell.outer_node_ids.len(),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_inner_nodes".to_string(),
        shell.inner_node_ids.len(),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_generated_nodes".to_string(),
        refill.generated_nodes.len(),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_refill_boundary_faces".to_string(),
        boundary_faces.len(),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_boundary_centroid_refinement_attempts".to_string(),
        usize::from(refill.boundary_centroid_refinement_attempted),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_boundary_centroid_refinement_rejected".to_string(),
        usize::from(refill.boundary_centroid_refinement_rejected),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_boundary_exact_cover_refills".to_string(),
        usize::from(refill.strategy == NestedTetrahedronShellRefillStrategy::BoundaryExactCover),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_boundary_centroid_refinement_refills".to_string(),
        usize::from(
            refill.strategy == NestedTetrahedronShellRefillStrategy::BoundaryCentroidRefinement,
        ),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_barycentric_partition_refills".to_string(),
        usize::from(refill.strategy == NestedTetrahedronShellRefillStrategy::BarycentricPartition),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_outer_facets".to_string(),
        shell.outer_facet_indices.len(),
    );
    evidence.entity_counts.insert(
        "nested_tetrahedron_shell_inner_facets".to_string(),
        shell.inner_facet_indices.len(),
    );
    record_input_plc_evidence(plc, &mut evidence);
    record_tetrahedron_material_evidence(&elements, &mut evidence);
    evidence.min_scaled_jacobian = refill
        .refill
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .reduce(f64::min);

    Ok(TetrahedronMesh {
        mesh_id: "nested_tetrahedron_shell_tetrahedron_mesh".to_string(),
        tetrahedron_generation_family: "nested_tetrahedron_shell".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn source_facet_for_cavity_face(
    plc: &ProtectedBoundaryComplex,
    source_face_id: Option<u32>,
) -> Result<&PlcFacet, TetrahedronGenerationError> {
    let Some(source_face_id) = source_face_id else {
        return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
    };
    let index = usize::try_from(source_face_id)
        .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;
    plc.facets
        .get(index)
        .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)
}

fn mesh_node_id(
    cavity_id_to_node_id: &BTreeMap<u32, TopologyEntityId>,
    cavity_id: u32,
) -> Result<TopologyEntityId, TetrahedronGenerationError> {
    cavity_id_to_node_id
        .get(&cavity_id)
        .cloned()
        .ok_or_else(|| TetrahedronGenerationError::MissingPlcNode {
            node_id: cavity_id.to_string(),
        })
}

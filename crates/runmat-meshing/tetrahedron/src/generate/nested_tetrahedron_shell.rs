use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{
    MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
};
use runmat_meshing_core::quality::tolerance::MeshingTolerance;
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use crate::cavity::constrained::{
    retriangulate_constrained_cavity_from_nodes, ConstrainedCavity, ConstrainedCavityBoundaryFace,
    ConstrainedCavityNode, ConstrainedCavityRefillOptions,
};

use super::convex_polyhedron::bounds::plc_coordinates_and_bounds;
use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::material::plc_material_region_id;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};

mod shell;
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

    let mut node_id_to_cavity_id = BTreeMap::<TopologyEntityId, u32>::new();
    let mut cavity_id_to_node_id = BTreeMap::<u32, TopologyEntityId>::new();
    let mut cavity_nodes = Vec::<ConstrainedCavityNode>::with_capacity(plc.nodes.len());
    for (index, node) in plc.nodes.iter().enumerate() {
        let cavity_id = u32::try_from(index)
            .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;
        node_id_to_cavity_id.insert(node.node_id.clone(), cavity_id);
        cavity_id_to_node_id.insert(cavity_id, node.node_id.clone());
        cavity_nodes.push(ConstrainedCavityNode {
            node_id: cavity_id,
            coordinates_m: node.coordinates_m,
        });
    }
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: plc
            .facets
            .iter()
            .enumerate()
            .map(|(index, facet)| {
                let source_face_id = u32::try_from(index).ok();
                Ok(ConstrainedCavityBoundaryFace {
                    node_ids: [
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[0])?,
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[1])?,
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[2])?,
                    ],
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id,
                    source_edge_ids: [None, None, None],
                    region_ids: facet.material_interface_ids.clone(),
                })
            })
            .collect::<Result<Vec<_>, TetrahedronGenerationError>>()?,
        protected_node_ids: Vec::new(),
        target_volume_m3,
    };
    let refill_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    };
    let refill =
        retriangulate_constrained_cavity_from_nodes(&cavity, &cavity_nodes, refill_options)
            .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?
            .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;

    let material_region_id = plc_material_region_id(plc);
    let nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let elements = refill
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
                    mesh_node_id(&cavity_id_to_node_id, tetrahedron.node_ids[0])?,
                    mesh_node_id(&cavity_id_to_node_id, tetrahedron.node_ids[1])?,
                    mesh_node_id(&cavity_id_to_node_id, tetrahedron.node_ids[2])?,
                    mesh_node_id(&cavity_id_to_node_id, tetrahedron.node_ids[3])?,
                ],
                material_region_id: material_region_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, TetrahedronGenerationError>>()?;
    let boundary_faces = plc
        .facets
        .iter()
        .map(|facet| TetrahedronBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
            source_edge_ids: super::source_edge_ids_for_face_edges(
                &plc.protected_edges,
                facet.node_ids.clone(),
            ),
        })
        .collect::<Vec<_>>();

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

fn cavity_node_id(
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    node_id: &TopologyEntityId,
) -> Result<u32, TetrahedronGenerationError> {
    node_id_to_cavity_id.get(node_id).copied().ok_or_else(|| {
        TetrahedronGenerationError::MissingPlcNode {
            node_id: node_id.id.clone(),
        }
    })
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

use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};

use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::material::plc_material_region_id;
use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};

pub fn generate_single_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;
    if plc.nodes.len() != 4 || plc.facets.len() != 4 {
        return Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc);
    }

    let coordinates_by_id = plc
        .nodes
        .iter()
        .map(|node| {
            if node
                .coordinates_m
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            {
                Err(TetrahedronGenerationError::NonFinitePlcNode {
                    node_id: node.node_id.id.clone(),
                })
            } else {
                Ok((node.node_id.clone(), node.coordinates_m))
            }
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    validate_single_tetrahedron_facets(plc)?;

    let mut node_ids = plc
        .nodes
        .iter()
        .map(|node| node.node_id.clone())
        .collect::<Vec<_>>();
    let points = [
        coordinates_by_id[&node_ids[0]],
        coordinates_by_id[&node_ids[1]],
        coordinates_by_id[&node_ids[2]],
        coordinates_by_id[&node_ids[3]],
    ];
    if tetrahedron_signed_volume(points).abs() <= f64::EPSILON {
        return Err(TetrahedronGenerationError::DegenerateSingleTetrahedronPlc);
    }
    if tetrahedron_signed_volume(points) < 0.0 {
        node_ids.swap(1, 2);
    }
    let points = [
        coordinates_by_id[&node_ids[0]],
        coordinates_by_id[&node_ids[1]],
        coordinates_by_id[&node_ids[2]],
        coordinates_by_id[&node_ids[3]],
    ];
    let min_scaled_jacobian = tetrahedron_scaled_jacobian(points);
    let material_region_id = plc_material_region_id(plc);

    let nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let elements = vec![Tetrahedron4Element {
        element_id: TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: "single_tetrahedron_0".to_string(),
        },
        node_ids: [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
        material_region_id,
    }];
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
    record_input_plc_evidence(plc, &mut evidence);
    record_tetrahedron_material_evidence(&elements, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "single_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn validate_single_tetrahedron_facets(
    plc: &ProtectedBoundaryComplex,
) -> Result<(), TetrahedronGenerationError> {
    let plc_node_ids = plc
        .nodes
        .iter()
        .map(|node| node.node_id.clone())
        .collect::<BTreeSet<_>>();
    let expected_facets = plc_node_ids
        .iter()
        .map(|omitted_node_id| {
            plc_node_ids
                .iter()
                .filter(|node_id| *node_id != omitted_node_id)
                .cloned()
                .collect::<BTreeSet<_>>()
        })
        .collect::<BTreeSet<_>>();
    let actual_facets = plc
        .facets
        .iter()
        .map(|facet| {
            let nodes = facet.node_ids.iter().cloned().collect::<BTreeSet<_>>();
            if nodes.len() != 3 || !nodes.is_subset(&plc_node_ids) {
                Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc)
            } else {
                Ok(nodes)
            }
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if actual_facets == expected_facets {
        Ok(())
    } else {
        Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc)
    }
}

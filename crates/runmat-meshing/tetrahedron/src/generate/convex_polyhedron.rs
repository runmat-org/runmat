mod bounds;
mod shape;

use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
    quality::tolerance::MeshingTolerance,
};

use super::evidence::record_input_plc_evidence;
use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use bounds::{bounds_span, plc_coordinates_and_bounds, plc_node_average};
use shape::{validate_boundary_nodes_are_hull_nodes, validate_convex_boundary_facets};

pub fn generate_convex_polyhedron_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;
    if plc.nodes.len() < 4 || plc.facets.len() < 4 {
        return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
    }

    let (coordinates_by_id, bounds) = plc_coordinates_and_bounds(plc)?;
    let tolerance = MeshingTolerance::from_bounds(bounds[0], bounds[1]);
    let interior = plc_node_average(plc)?;
    validate_convex_boundary_facets(plc, &coordinates_by_id, interior, tolerance)?;
    validate_boundary_nodes_are_hull_nodes(plc, &coordinates_by_id, bounds, tolerance)?;

    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: "convex_polyhedron_interior_0".to_string(),
    };
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.push(TetrahedronMeshNode {
        node_id: interior_id.clone(),
        coordinates_m: interior,
    });

    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(plc.facets.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    let span = bounds_span(bounds).max(1.0);
    let volume_epsilon = tolerance.volume_epsilon(span);
    for (element_index, facet) in plc.facets.iter().enumerate() {
        let mut node_ids = [
            facet.node_ids[0].clone(),
            facet.node_ids[1].clone(),
            facet.node_ids[2].clone(),
            interior_id.clone(),
        ];
        let points = [
            coordinates_by_id[&facet.node_ids[0]],
            coordinates_by_id[&facet.node_ids[1]],
            coordinates_by_id[&facet.node_ids[2]],
            interior,
        ];
        let signed_volume = tetrahedron_signed_volume(points);
        if signed_volume.abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateConvexPolyhedronPlc);
        }
        if signed_volume < 0.0 {
            node_ids.swap(1, 2);
        }
        let points = node_ids.clone().map(|node_id| {
            if node_id == interior_id {
                interior
            } else {
                coordinates_by_id[&node_id]
            }
        });
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
        elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("convex_polyhedron_tetrahedron_{element_index}"),
            },
            node_ids,
            material_region_id: facet
                .material_interface_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "solid_body".to_string()),
        });
    }

    let boundary_faces = plc
        .facets
        .iter()
        .map(|facet| TetrahedronBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
        })
        .collect::<Vec<_>>();

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("interior_nodes".to_string(), 1);
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
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "convex_polyhedron_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

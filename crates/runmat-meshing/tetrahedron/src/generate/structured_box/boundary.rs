use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{MeshingStage, ProtectedBoundaryComplex, TopologyEntityId};

use super::super::{Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError};
use super::structured_box_side_index;

pub(super) fn exterior_boundary_faces(
    elements: &[Tetrahedron4Element],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<Vec<TetrahedronBoundaryFace>, TetrahedronGenerationError> {
    let [min, max] = bounds;
    let mut plc_source_by_facet = BTreeMap::<[TopologyEntityId; 3], TopologyEntityId>::new();
    let mut plc_source_by_side = BTreeMap::<usize, TopologyEntityId>::new();
    for facet in &plc.facets {
        let coordinates = facet
            .node_ids
            .iter()
            .map(|node_id| {
                coordinates_by_id.get(node_id).copied().ok_or_else(|| {
                    TetrahedronGenerationError::MissingPlcNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
            .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
        plc_source_by_side
            .entry(side_index)
            .or_insert_with(|| facet.source_face_id.clone());
        plc_source_by_facet.insert(
            sorted_face(facet.node_ids.clone()),
            facet.source_face_id.clone(),
        );
    }

    let mut face_counts = BTreeMap::<[TopologyEntityId; 3], ([TopologyEntityId; 3], usize)>::new();
    for element in elements {
        for face in tetrahedron_faces(element.node_ids.clone()) {
            face_counts
                .entry(sorted_face(face.clone()))
                .and_modify(|(_, count)| *count += 1)
                .or_insert((face, 1));
        }
    }

    let mut boundary_faces = Vec::<TetrahedronBoundaryFace>::new();
    for (boundary_face_index, (face_key, (node_ids, count))) in face_counts.into_iter().enumerate()
    {
        if count != 1 {
            continue;
        }
        let source_face_id = match plc_source_by_facet.get(&face_key) {
            Some(source_face_id) => source_face_id.clone(),
            None => {
                let coordinates = node_ids
                    .iter()
                    .map(|node_id| coordinates_by_id[node_id])
                    .collect::<Vec<_>>();
                let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
                    .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
                plc_source_by_side
                    .get(&side_index)
                    .cloned()
                    .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?
            }
        };
        boundary_faces.push(TetrahedronBoundaryFace {
            face_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("structured_box_boundary_face_{boundary_face_index}"),
            },
            node_ids,
            source_face_id,
        });
    }

    Ok(boundary_faces)
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
    ]
}

fn sorted_face(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

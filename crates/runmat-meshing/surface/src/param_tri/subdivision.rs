use runmat_meshing_cad::{CadFaceEvaluationFrame, SourceTopologyFace};

use super::{SurfaceElement, SurfaceNode};

pub(super) fn append_centroid_subdivision(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    corner_uv: [[f64; 2]; 3],
    centroid_m: [f64; 3],
    centroid_uv: [f64; 2],
    corner_projection_error_m: f64,
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) {
    let centroid_node_id = nodes.len() as u32;
    nodes.push(SurfaceNode {
        node_id: centroid_node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: centroid_m,
    });
    let child_specs = [
        (
            [face.node_ids[0], face.node_ids[1], centroid_node_id],
            [face.edge_ids[0], face.edge_ids[1], face.edge_ids[2]],
            [corner_uv[0], corner_uv[1], centroid_uv],
        ),
        (
            [face.node_ids[1], face.node_ids[2], centroid_node_id],
            [face.edge_ids[1], face.edge_ids[2], face.edge_ids[0]],
            [corner_uv[1], corner_uv[2], centroid_uv],
        ),
        (
            [face.node_ids[2], face.node_ids[0], centroid_node_id],
            [face.edge_ids[2], face.edge_ids[0], face.edge_ids[1]],
            [corner_uv[2], corner_uv[0], centroid_uv],
        ),
    ];
    for (node_ids, source_edge_ids, parametric_node_uv) in child_specs {
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m: corner_projection_error_m,
            region_ids: face.region_ids.clone(),
            material_region_ids: face.material_region_ids.clone(),
            area_m2: face.area_m2 / 3.0,
            unit_normal: frame.unit_normal,
        });
    }
}

pub(super) fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

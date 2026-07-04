use runmat_meshing_curve::{CurveDiscretization, CurveElement, CurveNode};

pub(in crate::param_tri::tests) fn concave_trim_curve_discretization() -> CurveDiscretization {
    CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.5,
                coordinates_m: [0.5, 0.45, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 3,
                source_edge_id: 1,
                parameter: 0.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 4,
                source_edge_id: 1,
                parameter: 1.0,
                coordinates_m: [0.0, 1.0, 0.0],
            },
            CurveNode {
                node_id: 5,
                source_edge_id: 2,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 6,
                source_edge_id: 2,
                parameter: 1.0,
                coordinates_m: [0.0, 1.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.6726812023536856,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [1, 2],
                length_m: 0.6726812023536856,
            },
            CurveElement {
                element_id: 2,
                source_edge_id: 1,
                node_ids: [3, 4],
                length_m: 2.0_f64.sqrt(),
            },
            CurveElement {
                element_id: 3,
                source_edge_id: 2,
                node_ids: [5, 6],
                length_m: 1.0,
            },
        ],
    }
}

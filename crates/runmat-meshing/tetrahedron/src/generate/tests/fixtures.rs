use runmat_meshing_core::contracts::{
    MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary,
    ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
};

pub(super) fn tetrahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetrahedron".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [0.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet("0", ["0", "2", "1"]),
            facet("1", ["0", "1", "3"]),
            facet("2", ["1", "2", "3"]),
            facet("3", ["2", "0", "3"]),
        ],
        protected_edges: Vec::<PlcProtectedEdge>::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn octahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "octahedron".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 1.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [-1.0, 0.0, 0.0]),
            node("4", [0.0, -1.0, 0.0]),
            node("5", [0.0, 0.0, -1.0]),
        ],
        facets: vec![
            facet("0", ["0", "1", "2"]),
            facet("1", ["0", "2", "3"]),
            facet("2", ["0", "3", "4"]),
            facet("3", ["0", "4", "1"]),
            facet("4", ["5", "2", "1"]),
            facet("5", ["5", "3", "2"]),
            facet("6", ["5", "4", "3"]),
            facet("7", ["5", "1", "4"]),
        ],
        protected_edges: Vec::<PlcProtectedEdge>::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn octahedron_with_extra_interior_node_plc() -> ProtectedBoundaryComplex {
    let mut plc = octahedron_plc();
    plc.complex_id = "octahedron_with_extra_interior_node".to_string();
    plc.nodes.push(node("6", [0.0, 0.0, 0.0]));
    plc
}

pub(super) fn nested_tetrahedron_shells_plc() -> ProtectedBoundaryComplex {
    let mut plc = tetrahedron_plc();
    plc.complex_id = "nested_tetrahedron_shells".to_string();
    plc.nodes.extend([
        node("10", [0.2, 0.2, 0.2]),
        node("11", [0.3, 0.2, 0.2]),
        node("12", [0.2, 0.3, 0.2]),
        node("13", [0.2, 0.2, 0.3]),
    ]);
    plc.facets.extend([
        facet("10", ["10", "12", "11"]),
        facet("11", ["10", "11", "13"]),
        facet("12", ["11", "12", "13"]),
        facet("13", ["12", "10", "13"]),
    ]);
    plc
}

pub(super) fn wider_inner_nested_tetrahedron_shells_plc() -> ProtectedBoundaryComplex {
    let mut plc = tetrahedron_plc();
    plc.complex_id = "wider_inner_nested_tetrahedron_shells".to_string();
    plc.nodes.extend([
        node("10", [0.2, 0.2, 0.2]),
        node("11", [0.5, 0.2, 0.2]),
        node("12", [0.2, 0.5, 0.2]),
        node("13", [0.2, 0.2, 0.5]),
    ]);
    plc.facets.extend([
        facet("10", ["10", "12", "11"]),
        facet("11", ["10", "11", "13"]),
        facet("12", ["11", "12", "13"]),
        facet("13", ["12", "10", "13"]),
    ]);
    plc
}

pub(super) fn split_outer_edge_nested_tetrahedron_shells_plc() -> ProtectedBoundaryComplex {
    let mut plc = tetrahedron_plc();
    plc.complex_id = "split_outer_edge_nested_tetrahedron_shells".to_string();
    plc.nodes.push(node("4", [0.5, 0.0, 0.0]));
    plc.nodes.extend([
        node("10", [0.2, 0.2, 0.2]),
        node("11", [0.3, 0.2, 0.2]),
        node("12", [0.2, 0.3, 0.2]),
        node("13", [0.2, 0.2, 0.3]),
    ]);
    plc.facets = vec![
        facet_with_source("0a", ["0", "2", "4"], "0"),
        facet_with_source("0b", ["4", "2", "1"], "0"),
        facet_with_source("1a", ["0", "4", "3"], "1"),
        facet_with_source("1b", ["4", "1", "3"], "1"),
        facet("2", ["1", "2", "3"]),
        facet("3", ["2", "0", "3"]),
        facet("10", ["10", "12", "11"]),
        facet("11", ["10", "11", "13"]),
        facet("12", ["11", "12", "13"]),
        facet("13", ["12", "10", "13"]),
    ];
    plc
}

pub(super) fn split_protected_outer_edge_nested_tetrahedron_shells_plc() -> ProtectedBoundaryComplex
{
    let mut plc = split_outer_edge_nested_tetrahedron_shells_plc();
    plc.complex_id = "split_protected_outer_edge_nested_tetrahedron_shells".to_string();
    plc.protected_edges = vec![
        protected_edge("protected_edge_0", ["0", "4"], "source_edge_0"),
        protected_edge("protected_edge_1", ["4", "1"], "source_edge_0"),
    ];
    plc
}

pub(super) fn box_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "box".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [1.0, 1.0, 0.0]),
            node("3", [0.0, 1.0, 0.0]),
            node("4", [0.0, 0.0, 1.0]),
            node("5", [1.0, 0.0, 1.0]),
            node("6", [1.0, 1.0, 1.0]),
            node("7", [0.0, 1.0, 1.0]),
        ],
        facets: vec![
            facet("0", ["0", "1", "2"]),
            facet("1", ["0", "2", "3"]),
            facet("2", ["4", "6", "5"]),
            facet("3", ["4", "7", "6"]),
            facet("4", ["0", "4", "5"]),
            facet("5", ["0", "5", "1"]),
            facet("6", ["1", "5", "6"]),
            facet("7", ["1", "6", "2"]),
            facet("8", ["2", "6", "7"]),
            facet("9", ["2", "7", "3"]),
            facet("10", ["3", "7", "4"]),
            facet("11", ["3", "4", "0"]),
        ],
        protected_edges: Vec::<PlcProtectedEdge>::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn split_edge_box_plc() -> ProtectedBoundaryComplex {
    let mut plc = box_plc();
    plc.complex_id = "split_edge_box".to_string();
    plc.nodes.push(node("8", [0.5, 0.0, 0.0]));
    plc.facets = vec![
        facet("0a", ["0", "8", "2"]),
        facet("0b", ["8", "1", "2"]),
        facet("1", ["0", "2", "3"]),
        facet("2", ["4", "6", "5"]),
        facet("3", ["4", "7", "6"]),
        facet("4", ["0", "4", "5"]),
        facet("5a", ["0", "5", "8"]),
        facet("5b", ["8", "5", "1"]),
        facet("6", ["1", "5", "6"]),
        facet("7", ["1", "6", "2"]),
        facet("8", ["2", "6", "7"]),
        facet("9", ["2", "7", "3"]),
        facet("10", ["3", "7", "4"]),
        facet("11", ["3", "4", "0"]),
    ];
    plc.protected_edges = vec![
        protected_edge("protected_edge_0", ["0", "8"], "source_edge_0"),
        protected_edge("protected_edge_1", ["8", "1"], "source_edge_0"),
    ];
    plc
}

pub(super) fn protected_corner_box_plc() -> ProtectedBoundaryComplex {
    let mut plc = box_plc();
    plc.complex_id = "protected_corner_box".to_string();
    plc.protected_edges = vec![
        protected_edge("protected_edge_0", ["0", "1"], "source_edge_0"),
        protected_edge("protected_edge_1", ["1", "2"], "source_edge_1"),
        protected_edge("protected_edge_2", ["2", "3"], "source_edge_2"),
        protected_edge("protected_edge_3", ["3", "0"], "source_edge_3"),
        protected_edge("protected_edge_4", ["4", "5"], "source_edge_4"),
        protected_edge("protected_edge_5", ["5", "6"], "source_edge_5"),
        protected_edge("protected_edge_6", ["6", "7"], "source_edge_6"),
        protected_edge("protected_edge_7", ["7", "4"], "source_edge_7"),
        protected_edge("protected_edge_8", ["0", "4"], "source_edge_8"),
        protected_edge("protected_edge_9", ["1", "5"], "source_edge_9"),
        protected_edge("protected_edge_10", ["2", "6"], "source_edge_10"),
        protected_edge("protected_edge_11", ["3", "7"], "source_edge_11"),
    ];
    plc
}

pub(super) fn dented_corner_box_plc() -> ProtectedBoundaryComplex {
    let mut plc = box_plc();
    plc.complex_id = "dented_corner_box".to_string();
    plc.nodes[6].coordinates_m = [0.55, 0.55, 0.55];
    plc
}

pub(super) fn through_hole_plate_plc() -> ProtectedBoundaryComplex {
    let mut plc = ProtectedBoundaryComplex {
        complex_id: "through_hole_plate".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 1.0]),
            node("1", [3.0, 0.0, 1.0]),
            node("2", [3.0, 3.0, 1.0]),
            node("3", [0.0, 3.0, 1.0]),
            node("4", [1.0, 1.0, 1.0]),
            node("5", [2.0, 1.0, 1.0]),
            node("6", [2.0, 2.0, 1.0]),
            node("7", [1.0, 2.0, 1.0]),
            node("8", [0.0, 0.0, 0.0]),
            node("9", [3.0, 0.0, 0.0]),
            node("10", [3.0, 3.0, 0.0]),
            node("11", [0.0, 3.0, 0.0]),
            node("12", [1.0, 1.0, 0.0]),
            node("13", [2.0, 1.0, 0.0]),
            node("14", [2.0, 2.0, 0.0]),
            node("15", [1.0, 2.0, 0.0]),
        ],
        facets: through_hole_plate_facets(),
        protected_edges: Vec::<PlcProtectedEdge>::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    };
    plc.evidence
        .entity_counts
        .insert("surface_boundary_loops".to_string(), 12);
    plc.evidence
        .entity_counts
        .insert("surface_hole_loops".to_string(), 2);
    plc.evidence
        .entity_counts
        .insert("surface_boundary_nodes".to_string(), 16);
    plc.evidence
        .entity_counts
        .insert("surface_boundary_segments".to_string(), 32);
    plc
}

fn through_hole_plate_facets() -> Vec<PlcFacet> {
    [
        ("0", [0, 1, 5], "0"),
        ("1", [0, 5, 4], "0"),
        ("2", [1, 2, 6], "0"),
        ("3", [1, 6, 5], "0"),
        ("4", [2, 3, 7], "0"),
        ("5", [2, 7, 6], "0"),
        ("6", [3, 0, 4], "0"),
        ("7", [3, 4, 7], "0"),
        ("8", [8, 13, 9], "1"),
        ("9", [8, 12, 13], "1"),
        ("10", [9, 14, 10], "1"),
        ("11", [9, 13, 14], "1"),
        ("12", [10, 15, 11], "1"),
        ("13", [10, 14, 15], "1"),
        ("14", [11, 12, 8], "1"),
        ("15", [11, 15, 12], "1"),
        ("16", [0, 8, 9], "2"),
        ("17", [0, 9, 1], "2"),
        ("18", [1, 9, 10], "3"),
        ("19", [1, 10, 2], "3"),
        ("20", [2, 10, 11], "4"),
        ("21", [2, 11, 3], "4"),
        ("22", [3, 11, 8], "5"),
        ("23", [3, 8, 0], "5"),
        ("24", [4, 5, 13], "6"),
        ("25", [4, 13, 12], "6"),
        ("26", [5, 6, 14], "7"),
        ("27", [5, 14, 13], "7"),
        ("28", [6, 7, 15], "8"),
        ("29", [6, 15, 14], "8"),
        ("30", [7, 4, 12], "9"),
        ("31", [7, 12, 15], "9"),
    ]
    .into_iter()
    .map(|(facet_id, node_ids, source_face_id)| {
        facet_with_numeric_nodes(facet_id, node_ids, source_face_id)
    })
    .collect()
}

fn facet_with_numeric_nodes(id: &str, node_ids: [u32; 3], source_face_id: &str) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: node_ids
            .map(|node_id| entity(MeshingStage::ProtectedBoundaryComplex, &node_id.to_string())),
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        material_interface_ids: vec!["body".to_string()],
    }
}

fn node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        coordinates_m,
    }
}

fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
    facet_with_source(id, node_ids, id)
}

fn facet_with_source(id: &str, node_ids: [&str; 3], source_face_id: &str) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        material_interface_ids: vec!["body".to_string()],
    }
}

fn protected_edge(id: &str, node_ids: [&str; 2], source_edge_id: &str) -> PlcProtectedEdge {
    PlcProtectedEdge {
        edge_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
        ],
        source_edge_id: entity(MeshingStage::CurveMesh, source_edge_id),
        cad_curve_boundary: None,
    }
}

fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}

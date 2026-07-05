use runmat_meshing_core::contracts::{
    MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary,
    ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
};

pub(super) fn tetra_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetra".to_string(),
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
    let mut plc = tetra_plc();
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

pub(super) fn dented_corner_box_plc() -> ProtectedBoundaryComplex {
    let mut plc = box_plc();
    plc.complex_id = "dented_corner_box".to_string();
    plc.nodes[6].coordinates_m = [0.55, 0.55, 0.55];
    plc
}

fn node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        coordinates_m,
    }
}

fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, id),
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

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

fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}

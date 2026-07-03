use super::*;
use runmat_meshing_core::{
    contracts::{
        MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary,
        ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
    },
    quality::predicate::tetrahedron_signed_volume,
};

#[test]
fn generates_positive_tetrahedra_from_validated_tetra_plc() {
    let mesh = generate_initial_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("validated tetra PLC should generate an initial Tetrahedron mesh");

    assert_eq!(mesh.nodes.len(), 5);
    assert_eq!(mesh.elements.len(), 4);
    assert_eq!(mesh.boundary_faces.len(), 4);
    assert!(!mesh.recovery_complete);
    assert!(!mesh.quality_optimized);
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 4);
    assert!(mesh.evidence.min_scaled_jacobian.expect("volume evidence") > 0.0);
}

#[test]
fn rejects_unvalidated_plc_before_tetrahedron_generation() {
    let mut plc = tetra_plc();
    plc.validation.watertight = false;

    assert!(matches!(
        generate_initial_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn rejects_degenerate_plc_facet() {
    let mut plc = tetra_plc();
    plc.nodes[2].coordinates_m = plc.nodes[1].coordinates_m;

    assert!(matches!(
        generate_initial_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::DegenerateBoundaryFacet { .. })
    ));
}

#[test]
fn generates_structured_box_tetrahedra_from_validated_plc_bounds() {
    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&box_plc())
        .expect("validated box PLC should generate structured Tetrahedron mesh");

    assert_eq!(mesh.elements.len(), 6);
    assert_eq!(mesh.boundary_faces.len(), 12);
    assert_eq!(mesh.evidence.entity_counts["plc_boundary_nodes"], 8);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    for element in &mesh.elements {
        let points = element.node_ids.clone().map(|node_id| {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .expect("node exists")
                .coordinates_m
        });
        assert!(tetrahedron_signed_volume(points) > 0.0);
    }
}

#[test]
fn generates_single_tetrahedron_mesh_from_tetrahedron_plc() {
    let mesh = generate_single_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("tetrahedron PLC should generate one solver Tetrahedron4");

    assert_eq!(mesh.nodes.len(), 4);
    assert_eq!(mesh.elements.len(), 1);
    assert_eq!(mesh.boundary_faces.len(), 4);
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 1);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    let element = &mesh.elements[0];
    let points = element.node_ids.clone().map(|node_id| {
        mesh.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .expect("node exists")
            .coordinates_m
    });
    assert!(tetrahedron_signed_volume(points) > 0.0);
}

#[test]
fn solver_generation_supports_box_and_single_tetrahedron_plcs() {
    let box_mesh = generate_solver_tetrahedron_mesh_from_plc(&box_plc())
        .expect("box PLC should use structured box solver generation");
    let tetrahedron_mesh = generate_solver_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("tetrahedron PLC should use single Tetrahedron solver generation");

    assert_eq!(box_mesh.elements.len(), 6);
    assert_eq!(tetrahedron_mesh.elements.len(), 1);
}

#[test]
fn generates_convex_polyhedron_tetrahedron_mesh_from_octahedron_plc() {
    let mesh = generate_convex_polyhedron_tetrahedron_mesh_from_plc(&octahedron_plc())
        .expect("convex octahedron PLC should generate one Tetrahedron4 per boundary facet");

    assert_eq!(mesh.nodes.len(), 7);
    assert_eq!(mesh.elements.len(), 8);
    assert_eq!(mesh.boundary_faces.len(), 8);
    assert_eq!(mesh.evidence.entity_counts["interior_nodes"], 1);
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 8);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    for element in &mesh.elements {
        let points = element.node_ids.clone().map(|node_id| {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .expect("node exists")
                .coordinates_m
        });
        assert!(tetrahedron_signed_volume(points) > 0.0);
    }
}

#[test]
fn solver_generation_supports_convex_polyhedron_plcs() {
    let mesh = generate_solver_tetrahedron_mesh_from_plc(&octahedron_plc())
        .expect("convex octahedron PLC should use convex polyhedron solver generation");

    assert_eq!(mesh.mesh_id, "convex_polyhedron_tetrahedron_mesh");
    assert_eq!(mesh.elements.len(), 8);
}

#[test]
fn solver_generation_rejects_unreferenced_plc_nodes_before_shape_selection() {
    assert!(matches!(
        generate_solver_tetrahedron_mesh_from_plc(&octahedron_with_extra_interior_node_plc()),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn solver_generation_rejects_open_plc_even_when_summary_claims_ready() {
    let mut plc = tetra_plc();
    plc.facets.pop();

    assert!(matches!(
        generate_solver_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn single_tetrahedron_generation_rejects_non_tetrahedron_plc() {
    assert_eq!(
        generate_single_tetrahedron_mesh_from_plc(&box_plc()),
        Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc)
    );
}

#[test]
fn structured_box_generation_rejects_degenerate_bounds() {
    let mut plc = tetra_plc();
    for node in &mut plc.nodes {
        node.coordinates_m[2] = 0.0;
    }

    assert_eq!(
        generate_structured_box_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::DegeneratePlcBounds)
    );
}

#[test]
fn structured_box_generation_rejects_non_box_plc() {
    assert_eq!(
        generate_structured_box_tetrahedron_mesh_from_plc(&tetra_plc()),
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)
    );
}

#[test]
fn convex_polyhedron_generation_rejects_nonconvex_boundary_facet() {
    let mut plc = octahedron_plc();
    plc.nodes[1].coordinates_m = [-0.1, 0.0, 0.1];

    assert_eq!(
        generate_convex_polyhedron_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc)
    );
}

fn tetra_plc() -> ProtectedBoundaryComplex {
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

fn octahedron_plc() -> ProtectedBoundaryComplex {
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

fn octahedron_with_extra_interior_node_plc() -> ProtectedBoundaryComplex {
    let mut plc = octahedron_plc();
    plc.complex_id = "octahedron_with_extra_interior_node".to_string();
    plc.nodes.push(node("6", [0.0, 0.0, 0.0]));
    plc
}

fn box_plc() -> ProtectedBoundaryComplex {
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

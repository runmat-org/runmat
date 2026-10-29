#![cfg(feature = "occt-native")]
//! Edge and vertex names written by CAD tools on STEP topology items (`EDGE_CURVE('Girder', ...)`,
//! `VERTEX_POINT('Left base', ...)`) reach the exact model the same way face and solid names do:
//! keyed by the entity's persistent identity, outside that identity, so `select name` on an edge or
//! vertex region resolves to canonical persistent-entity IDs.
//! Fixture: `fixtures/named-edges-vertices.step` is `named-faces.step` (a 100 x 20 x 10 mm bar
//! "Bar" with end faces "Fixed end" and "Loaded end") with the edge from (0,0,0) to (0,0,10) named
//! "Fixed edge", the edge from (0,0,0) to (100,0,0) named "Bottom rail", the vertex at (0,0,0)
//! named "Origin corner" and the vertex at (100,20,10) named "Load point".

use runmat_geometry_core::{GeometryRegionRole, PersistentEntityId, PersistentEntityKind};
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    ImportedExactCad,
};

const NAMED_EDGES_VERTICES: &[u8] = include_bytes!("fixtures/named-edges-vertices.step");
/// `touching_parts_assembly.step` (plate "Plate" under block "Block") with the plate edge from
/// (-15,-15,-5) to (-15,-15,0) named "Plate corner edge", its vertex at (-15,-15,-5) named
/// "Plate corner", the block edge from (-5,-5,0) to (-5,-5,10) named "Block edge" and that edge's
/// foot vertex, which lies in the shared contact patch, named "Block foot".
const TOUCHING_NAMED_EDGES: &[u8] = include_bytes!("fixtures/touching_parts_named_edges.step");
const PORTAL_FRAME: &[u8] =
    include_bytes!("../../../../docs/simulation/examples/geometry/portal-frame.step");
const SQUARE_PLATE: &[u8] =
    include_bytes!("../../../../docs/simulation/examples/geometry/square-plate.step");

fn import(name: &str, bytes: &[u8], shared: bool) -> ImportedExactCad {
    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.shared_interfaces = shared;
    import_exact_cad(
        name,
        bytes,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap_or_else(|error| panic!("import {name}: {error}"))
}

fn one(imported: &ImportedExactCad, role: GeometryRegionRole, name: &str) -> PersistentEntityId {
    let selected = imported
        .entities_named(role, name)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(selected.len(), 1, "{name}: {selected:?}");
    selected.into_iter().next().unwrap()
}

/// Vertex position in millimetres.
fn vertex_mm(imported: &ImportedExactCad, id: &PersistentEntityId) -> [f64; 3] {
    let vertex = imported
        .topology
        .vertices
        .iter()
        .find(|vertex| &vertex.id == id)
        .unwrap_or_else(|| panic!("vertex {id:?} is not in the topology"));
    vertex
        .point_m
        .canonical_values()
        .map(|coordinate| coordinate * 1000.0)
}

fn edge_ends_mm(imported: &ImportedExactCad, id: &PersistentEntityId) -> [[f64; 3]; 2] {
    let edge = imported
        .topology
        .edges
        .iter()
        .find(|edge| &edge.id == id)
        .unwrap_or_else(|| panic!("edge {id:?} is not in the topology"));
    let mut ends = [
        vertex_mm(imported, edge.start_vertex_id.as_ref().unwrap()),
        vertex_mm(imported, edge.end_vertex_id.as_ref().unwrap()),
    ];
    ends.sort_by(|left, right| left.partial_cmp(right).unwrap());
    ends
}

fn assert_close(actual: [f64; 3], expected: [f64; 3]) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1e-6,
            "{actual:?} != {expected:?}"
        );
    }
}

// witness: FEA-MESH-GEOMETRY-001/authored_names_survive_to_run @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Authored STEP edge and vertex names select exactly the authored edges and vertices
//   by persistent identity; face and solid names are unchanged; names stay outside identity.
#[test]
fn authored_step_edge_and_vertex_names_select_their_entities() {
    {
        let imported = import("named-edges-vertices.step", NAMED_EDGES_VERTICES, false);
        let fixed_edge = one(&imported, GeometryRegionRole::Edge, "Fixed edge");
        assert_eq!(fixed_edge.kind, PersistentEntityKind::Edge);
        assert_eq!(
            edge_ends_mm(&imported, &fixed_edge),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
        );
        let bottom_rail = one(&imported, GeometryRegionRole::Edge, "Bottom rail");
        assert_eq!(
            edge_ends_mm(&imported, &bottom_rail),
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
        );
        let origin = one(&imported, GeometryRegionRole::Vertex, "Origin corner");
        assert_eq!(origin.kind, PersistentEntityKind::Vertex);
        assert_close(vertex_mm(&imported, &origin), [0.0, 0.0, 0.0]);
        let load_point = one(&imported, GeometryRegionRole::Vertex, "Load point");
        assert_close(vertex_mm(&imported, &load_point), [100.0, 20.0, 10.0]);

        // Face and body names still resolve, and a named face does not name its edges.
        one(&imported, GeometryRegionRole::Face, "Fixed end");
        one(&imported, GeometryRegionRole::Face, "Loaded end");
        one(&imported, GeometryRegionRole::Solid, "Bar");
        let named_edges = imported
            .entity_names
            .keys()
            .filter(|id| id.kind == PersistentEntityKind::Edge)
            .count();
        assert_eq!(named_edges, 2);
        // An edge name never selects a vertex, nor a vertex name an edge.
        assert!(imported
            .entities_named(GeometryRegionRole::Vertex, "Fixed edge")
            .is_err());
        assert!(imported
            .entities_named(GeometryRegionRole::Edge, "Load point")
            .is_err());
    }
    // Names are selector metadata: the persistent identity is unchanged by them.
    let named = import("named-edges-vertices.step", NAMED_EDGES_VERTICES, false);
    let unnamed = import(
        "named-faces.step",
        include_bytes!("fixtures/named-faces.step"),
        false,
    );
    assert_eq!(named.topology, unnamed.topology);
}

// witness: FEA-MESH-GEOMETRY-001/authored_names_survive_to_run @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Edge and vertex names of an assembly's parts survive part placement and the
//   shared-interface join, including a named vertex inside the shared contact patch.
#[test]
fn edge_and_vertex_names_survive_placement_and_the_shared_interface_join() {
    for shared in [false, true] {
        let imported = import(
            "touching_parts_named_edges.step",
            TOUCHING_NAMED_EDGES,
            shared,
        );
        assert_eq!(imported.topology.interfaces.len(), usize::from(shared));
        let edge = one(&imported, GeometryRegionRole::Edge, "Plate corner edge");
        let ends = edge_ends_mm(&imported, &edge);
        assert!((ends[1][2] - ends[0][2] - 5.0).abs() < 1e-6, "{ends:?}");
        let corner = one(&imported, GeometryRegionRole::Vertex, "Plate corner");
        assert!(ends.iter().any(|end| *end == vertex_mm(&imported, &corner)));
        let block_edge = one(&imported, GeometryRegionRole::Edge, "Block edge");
        let block_ends = edge_ends_mm(&imported, &block_edge);
        assert!(
            (block_ends[1][2] - block_ends[0][2] - 10.0).abs() < 1e-6,
            "{block_ends:?}"
        );
        let foot = one(&imported, GeometryRegionRole::Vertex, "Block foot");
        assert_eq!(vertex_mm(&imported, &foot), block_ends[0]);
        one(&imported, GeometryRegionRole::Solid, "Plate");
        one(&imported, GeometryRegionRole::Solid, "Block");
    }
}

// witness: FEA-MESH-GEOMETRY-001/authored_names_survive_to_run @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Selecting an edge or vertex name the source does not carry fails with a message
//   that names the requested name and lists the names the source carries for that role.
#[test]
fn an_unnamed_edge_or_vertex_selection_reports_the_carried_names() {
    let imported = import("named-edges-vertices.step", NAMED_EDGES_VERTICES, false);
    let error = imported
        .entities_named(GeometryRegionRole::Edge, "Top rail")
        .unwrap_err();
    assert_eq!(
        error,
        "no edge is named \"Top rail\"; named edges: \"Bottom rail\", \"Fixed edge\""
    );
    let error = imported
        .entities_named(GeometryRegionRole::Vertex, "Apex")
        .unwrap_err();
    assert_eq!(
        error,
        "no vertex is named \"Apex\"; named vertices: \"Load point\", \"Origin corner\""
    );
    let unnamed = import(
        "named-faces.step",
        include_bytes!("fixtures/named-faces.step"),
        false,
    );
    assert_eq!(
        unnamed
            .entities_named(GeometryRegionRole::Edge, "Fixed edge")
            .unwrap_err(),
        "no edge is named \"Fixed edge\"; the source has no named edges"
    );
}

// witness: FEA-MESH-GEOMETRY-001/authored_names_survive_to_run @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: The documentation portal-frame and square-plate sheets resolve every edge and
//   vertex name their studies select to the edge or vertex at the authored position.
#[test]
fn documentation_frame_and_plate_edge_and_vertex_names_resolve() {
    let frame = import("portal-frame.step", PORTAL_FRAME, false);
    for (name, ends) in [
        ("Ground", [[0.0, 0.0, 0.0], [4000.0, 0.0, 0.0]]),
        ("Left column", [[0.0, 0.0, 0.0], [0.0, 0.0, 3000.0]]),
        ("Right column", [[4000.0, 0.0, 0.0], [4000.0, 0.0, 3000.0]]),
        ("Girder", [[0.0, 0.0, 3000.0], [4000.0, 0.0, 3000.0]]),
    ] {
        let edge = one(&frame, GeometryRegionRole::Edge, name);
        let actual = edge_ends_mm(&frame, &edge);
        assert_close(actual[0], ends[0]);
        assert_close(actual[1], ends[1]);
    }
    for (name, point) in [
        ("Left base", [0.0, 0.0, 0.0]),
        ("Right base", [4000.0, 0.0, 0.0]),
        ("Left top", [0.0, 0.0, 3000.0]),
        ("Right top", [4000.0, 0.0, 3000.0]),
    ] {
        let vertex = one(&frame, GeometryRegionRole::Vertex, name);
        assert_close(vertex_mm(&frame, &vertex), point);
    }

    let plate = import("square-plate.step", SQUARE_PLATE, false);
    one(&plate, GeometryRegionRole::Face, "Plate");
    for name in ["South edge", "East edge", "North edge", "West edge"] {
        one(&plate, GeometryRegionRole::Edge, name);
    }
}

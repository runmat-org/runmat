#![cfg(feature = "occt-native")]
//! Multi-body STEP import: authored face and body names reach the exact model, and touching
//! bodies join into shared interfaces (one face used by two solids) when the geometry asks for it.
//! Fixtures are written by `fixtures/generate_shared_interface_fixtures.cc`.

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    ImportedExactCad,
};

const NAMED_FACES: &[u8] = include_bytes!("fixtures/named-faces.step");
const PART_IN_AIR: &[u8] = include_bytes!("fixtures/part_in_air_assembly.step");
const TWO_CAVITY_AIR: &[u8] = include_bytes!("fixtures/two_cavity_air_assembly.step");
const TOUCHING_PARTS: &[u8] = include_bytes!("fixtures/touching_parts_assembly.step");
const OVERLAPPING_PARTS: &[u8] = include_bytes!("fixtures/overlapping_parts_assembly.step");
const FUSED_COMPOUND: &[u8] = include_bytes!("fixtures/fused_compound.step");
const FREE_WIREFRAME: &[u8] = include_bytes!("fixtures/free_wireframe_part.step");

fn import(name: &str, bytes: &[u8], shared: bool) -> Result<ImportedExactCad, String> {
    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.shared_interfaces = shared;
    import_exact_cad(
        name,
        bytes,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .map_err(|error| error.to_string())
}

fn named(imported: &ImportedExactCad, kind: PersistentEntityKind, name: &str) -> Vec<PersistentEntityId> {
    imported
        .entity_names
        .iter()
        .filter(|(id, names)| id.kind == kind && names.iter().any(|candidate| candidate == name))
        .map(|(id, _)| id.clone())
        .collect()
}

fn interface_faces(imported: &ImportedExactCad) -> Vec<&PersistentEntityId> {
    imported.topology.interfaces.iter().map(|interface| &interface.face_id).collect()
}

#[test]
fn authored_face_and_body_names_reach_the_exact_model() {
    let imported = import("named-faces.step", NAMED_FACES, false).unwrap();
    assert_eq!(named(&imported, PersistentEntityKind::Face, "Fixed end").len(), 1);
    assert_eq!(named(&imported, PersistentEntityKind::Face, "Loaded end").len(), 1);
    assert_eq!(named(&imported, PersistentEntityKind::Solid, "Bar").len(), 1);
    // Names are selector metadata: renaming the file or reimporting leaves identity unchanged.
    let again = import("renamed.step", NAMED_FACES, false).unwrap();
    assert_eq!(imported.topology, again.topology);
    assert_eq!(imported.entity_names, again.entity_names);
}

#[test]
fn separate_parts_stay_independent_unless_interfaces_are_shared() {
    let independent = import("part_in_air_assembly.step", PART_IN_AIR, false).unwrap();
    assert_eq!(independent.topology.solids.len(), 2);
    assert!(independent.topology.interfaces.is_empty());
    assert_eq!(independent.topology.faces.len(), 18);

    let shared = import("part_in_air_assembly.step", PART_IN_AIR, true).unwrap();
    assert_eq!(shared.topology.solids.len(), 2);
    assert_eq!(shared.topology.regions.len(), 2);
    assert_eq!(shared.topology.faces.len(), 12);
    assert_eq!(shared.topology.interfaces.len(), 6);
    assert_eq!(shared.model.interface_count, 6);
    assert_eq!(named(&shared, PersistentEntityKind::Solid, "Part").len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Solid, "Air").len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Face, "Outer boundary").len(), 6);
    // The part's face name survives on the interface face it now shares with the air.
    let part_top = named(&shared, PersistentEntityKind::Face, "Part top");
    assert_eq!(part_top.len(), 1);
    assert!(interface_faces(&shared).contains(&&part_top[0]));
    shared.build_closure().unwrap();
}

#[test]
fn an_air_body_with_two_cavities_imports_as_one_solid() {
    let independent = import("two_cavity_air_assembly.step", TWO_CAVITY_AIR, false).unwrap();
    assert_eq!(independent.topology.solids.len(), 3);
    let air = named(&independent, PersistentEntityKind::Solid, "Air");
    assert_eq!(air.len(), 1);
    let air_solid = independent.topology.solids.iter().find(|solid| solid.id == air[0]).unwrap();
    assert_eq!(air_solid.void_shell_ids.len(), 2);

    let shared = import("two_cavity_air_assembly.step", TWO_CAVITY_AIR, true).unwrap();
    assert_eq!(shared.topology.solids.len(), 3);
    assert_eq!(shared.topology.interfaces.len(), 12);
    for name in ["Left", "Right", "Air"] {
        assert_eq!(named(&shared, PersistentEntityKind::Solid, name).len(), 1, "{name}");
    }
}

#[test]
fn partially_touching_parts_share_only_their_contact_patch() {
    let shared = import("touching_parts_assembly.step", TOUCHING_PARTS, true).unwrap();
    assert_eq!(shared.topology.solids.len(), 2);
    // The plate's top face is imprinted by the block's base; only the covered patch is shared.
    assert_eq!(shared.topology.interfaces.len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Face, "Base").len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Face, "Top load").len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Solid, "Plate").len(), 1);
    assert_eq!(named(&shared, PersistentEntityKind::Solid, "Block").len(), 1);
}

#[test]
fn interpenetrating_parts_are_rejected_by_name() {
    let error = import("overlapping_parts_assembly.step", OVERLAPPING_PARTS, true).unwrap_err();
    let message = error;
    assert!(message.contains("interpenetrates") || message.contains("same volume"), "{message}");
    assert!(message.contains("'First'") || message.contains("'Second'"), "{message}");
}

#[test]
fn a_general_fuse_compound_imports_with_its_shared_faces_and_solid_names() {
    // A STEP writer keeps each solid self-contained, so the fused cube and the cavity of its air
    // box are stored as two coincident copies of each face.
    let separate = import("fused_compound.step", FUSED_COMPOUND, false).unwrap();
    assert_eq!(separate.topology.solids.len(), 2);
    assert!(separate.topology.interfaces.is_empty());
    assert_eq!(separate.topology.faces.len(), 18);

    let shared = import("fused_compound.step", FUSED_COMPOUND, true).unwrap();
    assert_eq!(shared.topology.solids.len(), 2);
    assert_eq!(shared.topology.faces.len(), 12);
    assert_eq!(shared.topology.interfaces.len(), 6);
    for imported in [&separate, &shared] {
        assert_eq!(named(imported, PersistentEntityKind::Solid, "Part").len(), 1);
        assert_eq!(named(imported, PersistentEntityKind::Solid, "Air").len(), 1);
        assert_eq!(named(imported, PersistentEntityKind::Face, "Outer boundary").len(), 6);
    }
    shared.build_closure().unwrap();
}

#[test]
fn construction_wireframe_beside_a_solid_is_not_part_of_the_model() {
    let imported = import("free_wireframe_part.step", FREE_WIREFRAME, false).unwrap();
    assert_eq!(imported.topology.solids.len(), 1);
    assert_eq!(imported.topology.faces.len(), 6);
    assert_eq!(imported.topology.edges.len(), 12, "the free line is dropped");
    assert_eq!(imported.topology.vertices.len(), 8, "its coincident corner vertex too");
}

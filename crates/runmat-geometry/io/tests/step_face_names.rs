#![cfg(feature = "occt-native")]
//! Face names written by CAD tools on STEP representation items (`ADVANCED_FACE('Fixed end', ...)`)
//! reach the imported face regions, so studies can select faces by the names engineers gave them.
//! Fixture: `fixtures/named-faces.step`, written by `fixtures/named-faces.cc` (a 100 x 20 x 10 mm
//! bar named "Bar" with its end faces named "Fixed end" and "Loaded end").

use runmat_geometry_io::{import_geometry, GeometryImportOptions};

const NAMED_FACES: &[u8] = include_bytes!("fixtures/named-faces.step");

// witness: FEA-MESH-GEOMETRY-001/authored_names_survive_to_run @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Authored STEP face names become face region names in the imported model.
#[test]
fn authored_step_face_names_become_face_region_names() {
    let imported = import_geometry(
        "named-faces.step",
        NAMED_FACES,
        GeometryImportOptions::default(),
    )
    .unwrap();
    let regions = &imported.asset.regions;
    let is_face = |region: &runmat_geometry_core::Region| {
        region
            .cad_ownership
            .as_ref()
            .is_some_and(|owner| owner.face_id.is_some())
    };
    let face = |name: &str| regions.iter().find(|r| r.name == name && is_face(r));
    assert!(
        face("Fixed end").is_some(),
        "regions: {:?}",
        regions
            .iter()
            .map(|r| (&r.region_id, &r.name))
            .collect::<Vec<_>>()
    );
    assert!(face("Loaded end").is_some());
    // Unnamed faces keep their ordinal names, and the body keeps its product name.
    assert_eq!(regions.iter().filter(|r| is_face(r)).count(), 6);
    assert!(regions
        .iter()
        .any(|r| r.name == "Bar" && r.tag.as_deref() == Some("cad_body")));
}

const DRONE_ARM: &[u8] = include_bytes!("fixtures/drone_arm_study.step");

// The drone arm study is an assembly: its products `arm` and `motor_pod` sit under
// NEXT_ASSEMBLY_USAGE_OCCURRENCE, and the arm's `root` and `motor_seat` faces carry the
// occurrence's placement. The preview keeps those names, so studies can attach loads and supports.
#[test]
fn assembly_step_preview_keeps_face_and_product_names() {
    let imported = import_geometry(
        "drone_arm_study.step",
        DRONE_ARM,
        GeometryImportOptions::default(),
    )
    .unwrap();
    let regions = &imported.asset.regions;
    let names = || {
        regions
            .iter()
            .map(|r| (r.tag.clone(), r.name.clone()))
            .collect::<Vec<_>>()
    };
    let faces = |name: &str| {
        regions
            .iter()
            .filter(|r| r.name == name && r.tag.as_deref() == Some("occt_face"))
            .count()
    };
    assert_eq!(faces("root"), 1, "regions: {:?}", names());
    assert_eq!(faces("motor_seat"), 1, "regions: {:?}", names());
    let bodies: Vec<_> = regions
        .iter()
        .filter(|r| r.tag.as_deref() == Some("cad_body"))
        .map(|r| r.name.as_str())
        .collect();
    assert_eq!(bodies, ["arm", "motor_pod"], "regions: {:?}", names());
}

const DRONE_ESC: &[u8] = include_bytes!("fixtures/drone_esc_thermal.step");

// The drone ESC thermal study places product `esc_board_thermal` as occurrence `esc_board`. A
// body takes its product's name, as the exact import names solids, so the study's
// `select name "esc_board_thermal"` resolves in the preview too.
#[test]
fn assembly_step_preview_names_bodies_by_product() {
    let imported = import_geometry(
        "drone_esc_thermal.step",
        DRONE_ESC,
        GeometryImportOptions::default(),
    )
    .unwrap();
    let regions = &imported.asset.regions;
    let mut bodies: Vec<_> = regions
        .iter()
        .filter(|r| r.tag.as_deref() == Some("cad_body"))
        .map(|r| r.name.as_str())
        .collect();
    bodies.sort();
    assert_eq!(bodies, ["esc_board_thermal", "esc_heatsink", "thermal_pad"]);
    assert!(regions
        .iter()
        .any(|r| r.name == "fins" && r.tag.as_deref() == Some("occt_face")));
}

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

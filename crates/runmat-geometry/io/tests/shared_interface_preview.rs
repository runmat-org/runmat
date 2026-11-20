#![cfg(feature = "occt-native")]
//! The Model view of a study with `interfaces = shared` tessellates the materialized B-rep and
//! joins each display face to the exact import's source face by ordinal. Shared interfaces make
//! one face that both solids' shells use. The exact import lists that face once, so the display
//! must list it once too: otherwise every display ordinal after it is off by one and the last one
//! has no exact face ("display face has missing or duplicate exact identity evidence").
//!
//! Fixture: `fixtures/drone_arm_study.step`, the drone arm with its motor pod seated on it; the
//! pod's base and the arm's seat become one shared face.

use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{CadSemanticKind, GeometryRegionRole, LengthUnit};
use runmat_geometry_io::{
    import_exact_cad, import_materialized_geometry_preview, ExactCadImportOptions, GeometryFormat,
    GeometryImportContext, GeometryImportOptions,
};

const DRONE_ARM: &[u8] = include_bytes!("fixtures/drone_arm_study.step");

#[test]
fn shared_interface_face_previews_once_and_joins_its_exact_face() {
    let mut options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
        ..ExactCadImportOptions::default()
    };
    options.analysis.healing.shared_interfaces = true;
    let exact = import_exact_cad(
        "drone_arm_study.step",
        DRONE_ARM,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();

    // The case under test: some face is used by both solids' shells.
    let mut uses = BTreeMap::<_, usize>::new();
    for shell in &exact.topology.shells {
        for face in &shell.face_uses {
            *uses.entry(&face.entity_id).or_default() += 1;
        }
    }
    assert!(uses.values().any(|count| *count == 2), "no shared face");

    let names = exact.display_names().unwrap();
    let preview = import_materialized_geometry_preview(
        "arm_study.brep",
        &exact.representation,
        GeometryImportOptions {
            source_units: Some(LengthUnit::Millimeter),
            ..GeometryImportOptions::default()
        },
        &GeometryImportContext::new(),
        &exact.topology,
        &exact.source_face_ids,
        &names,
    )
    .unwrap();

    // Every exact face appears once in the preview, under its own persistent selection id.
    let expected: BTreeSet<_> = exact
        .source_face_ids
        .values()
        .map(|id| id.analysis_selection_id().unwrap())
        .collect();
    let faces: Vec<_> = preview
        .asset
        .regions
        .iter()
        .filter(|region| region.tag.as_deref() == Some("occt_face"))
        .collect();
    assert_eq!(faces.len(), expected.len());
    let projected: BTreeSet<_> = faces
        .iter()
        .map(|region| region.region_id.clone())
        .collect();
    assert_eq!(projected, expected);
    for region in faces {
        let ordinal = region.cad_ownership.as_ref().unwrap().face_id.unwrap();
        assert_eq!(
            exact.source_face_ids[&ordinal]
                .analysis_selection_id()
                .unwrap(),
            region.region_id
        );
    }
    // Both solids project as bodies under their part names, though the B-rep carries none.
    let mut bodies: Vec<_> = preview
        .asset
        .regions
        .iter()
        .filter(|region| region.tag.as_deref() == Some("cad_body"))
        .map(|region| region.name.as_str())
        .collect();
    bodies.sort();
    assert_eq!(bodies, ["arm", "motor_pod"]);
}

/// A materialized study's B-rep carries no CAD names, so the preview names each face and its
/// owning body from the exact import's authored names: Measure reads "Face 19 of arm".
#[test]
fn materialized_preview_faces_name_their_owning_body() {
    let options = ExactCadImportOptions {
        source_units: Some(LengthUnit::Millimeter),
        ..ExactCadImportOptions::default()
    };
    let exact = import_exact_cad(
        "drone_arm_study.step",
        DRONE_ARM,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    let names = exact.display_names().unwrap();
    let preview = import_materialized_geometry_preview(
        "arm_study.brep",
        &exact.representation,
        GeometryImportOptions {
            source_units: Some(LengthUnit::Millimeter),
            ..GeometryImportOptions::default()
        },
        &GeometryImportContext::new(),
        &exact.topology,
        &exact.source_face_ids,
        &names,
    )
    .unwrap();
    let faces: Vec<_> = preview
        .asset
        .regions
        .iter()
        .filter(|region| region.tag.as_deref() == Some("occt_face"))
        .collect();
    assert_eq!(faces.len(), exact.source_face_ids.len());
    let check_faces = exact
        .authored_names_by_role()
        .remove("face")
        .unwrap_or_default();
    let mut owners = BTreeSet::new();
    let mut named = 0;
    for region in faces {
        let ownership = region.cad_ownership.as_ref().unwrap();
        let body = ownership
            .owner_path
            .iter()
            .find(|label| label.kind == CadSemanticKind::Body)
            .unwrap_or_else(|| panic!("{} has no owning body", region.name));
        owners.insert(body.name.clone());
        match &ownership.label {
            // A named face reads its CAD name, which is also its first authored name.
            Some(label) => {
                named += 1;
                assert_eq!(label.kind, CadSemanticKind::Face);
                assert_eq!(region.name, label.name);
                assert_eq!(label.name, names[&region.region_id]);
                // The picker's name is one the check lists and one `select name` resolves to
                // this very face.
                assert!(
                    check_faces.contains(&region.name),
                    "check does not list {}",
                    region.name
                );
                let selected: Vec<String> = exact
                    .entities_named(GeometryRegionRole::Face, &region.name)
                    .unwrap()
                    .iter()
                    .map(|id| id.analysis_selection_id().unwrap())
                    .collect();
                assert!(
                    selected.contains(&region.region_id),
                    "select name \"{}\" does not resolve to the picked face",
                    region.name
                );
                assert_eq!(ownership.owner_path.last(), Some(label));
            }
            // An unnamed face keeps its numbered display name.
            None => assert!(region.name.starts_with("Face "), "{}", region.name),
        }
    }
    assert!(named > 0, "the fixture names some faces");
    assert_eq!(
        owners,
        BTreeSet::from(["arm".to_string(), "motor_pod".to_string()])
    );
}

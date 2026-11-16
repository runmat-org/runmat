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

use runmat_geometry_core::LengthUnit;
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
    // Both solids project as bodies (a materialized B-rep carries no product names; the Desktop
    // names them from the study's resolved regions).
    let bodies = preview
        .asset
        .regions
        .iter()
        .filter(|region| region.tag.as_deref() == Some("cad_body"))
        .count();
    assert_eq!(bodies, exact.topology.solids.len());
}

#![cfg(feature = "occt-native")]
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner_native::{
    prepare_exact_geometry_admission, NativeMeshingContactDefinition,
};
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
};

#[test]
fn contact_admission_uses_canonical_occurrence_ids_and_rejects_aliases() {
    let bytes = include_bytes!("../../runmat-geometry/io/tests/fixtures/independent_contact.step");
    let options = ExactCadImportOptions::default();
    let context = GeometryImportContext::new();
    let exact = import_exact_cad(
        "contact.step",
        bytes,
        GeometryFormat::Step,
        &options,
        &context,
    )
    .unwrap();
    let first = &exact.source_face_ids[&1];
    let second = &exact.source_face_ids[&6];
    let definition = NativeMeshingContactDefinition {
        side_a_face_selectors: vec![first.analysis_selection_id().unwrap()],
        side_b_face_selectors: vec![second.analysis_selection_id().unwrap()],
    };
    let admitted = prepare_exact_geometry_admission(
        "contact.step",
        bytes,
        &options,
        std::slice::from_ref(&definition),
        &context,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(admitted.topology().contacts.len(), 1);
    let contact = &admitted.topology().contacts[0];
    let actual: std::collections::BTreeSet<_> = contact
        .side_a_face_ids
        .iter()
        .chain(&contact.side_b_face_ids)
        .cloned()
        .collect();
    assert_eq!(
        actual,
        std::collections::BTreeSet::from([first.clone(), second.clone()])
    );
    for invalid in ["face_000002".to_owned(), first.source_topology_id.clone()] {
        let mut invalid_definition = definition.clone();
        invalid_definition.side_a_face_selectors = vec![invalid];
        let outcome = prepare_exact_geometry_admission(
            "contact.step",
            bytes,
            &options,
            &[invalid_definition],
            &context,
            ObjectInventoryLimits::default(),
        );
        assert!(
            outcome.is_err(),
            "contact admission must not accept an alternate identity"
        );
    }
}

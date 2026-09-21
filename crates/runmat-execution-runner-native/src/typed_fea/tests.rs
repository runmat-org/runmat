use std::fs;

use runmat_core::RunMatSession;

const CONSTRUCTED_STUDY: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 0.5 m, height = 0.25 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;

#[tokio::test]
async fn checked_constructed_study_materializes_exact_preview_geometry() {
    let fixture = tempfile::tempdir().expect("create fixture");
    let path = fixture.path().join("study.fea");
    fs::write(&path, CONSTRUCTED_STUDY).expect("write FEA source");
    let mut session = RunMatSession::with_options(false, false).expect("create session");
    let source = session
        .load_and_check_fea_source_path(&path)
        .await
        .expect("load checked FEA source");

    let materialized = super::materialize_checked_fea_study_geometry(
        &source,
        "static",
        &[],
        "typed-fea-preview-test",
    )
    .await
    .expect("materialize checked study geometry");

    assert!(materialized.output.document.is_exact());
    assert!(!materialized.kernel_representation.is_empty());
    assert_eq!(materialized.topology.faces.len(), 6);
    assert_eq!(materialized.output.source_face_ids.len(), 6);
    assert!(materialized
        .output
        .source_face_ids
        .values()
        .all(|identity| materialized
            .topology
            .faces
            .iter()
            .any(|face| face.id == *identity)));
    assert_eq!(materialized.topology.solids.len(), 1);
}

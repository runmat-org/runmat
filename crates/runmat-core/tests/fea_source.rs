#![cfg(not(target_arch = "wasm32"))]

use std::fs;

use runmat_core::RunMatSession;
use runmat_runtime::analysis::{FeaCheckStatus, FeaRootSelector};

const COMPLETE_STUDY: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
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

#[test]
fn session_checks_a_complete_fea_source() {
    let fixture = tempfile::tempdir().expect("create fixture");
    let path = fixture.path().join("study.fea");
    fs::write(&path, COMPLETE_STUDY).expect("write FEA source");
    let mut session = RunMatSession::with_options(false, false).expect("create session");

    let result = futures::executor::block_on(session.check_fea_source_path(&path))
        .expect("check FEA source");

    assert_eq!(
        result.status,
        FeaCheckStatus::Valid,
        "{:#?}",
        result.diagnostics
    );
    assert_eq!(result.root_module.as_str(), "project/study.fea");
    assert_eq!(
        result.roots,
        vec![FeaRootSelector::Study {
            id: "static".to_string(),
        }]
    );
}

#[test]
fn session_reports_an_unavailable_import_as_incomplete() {
    let fixture = tempfile::tempdir().expect("create fixture");
    let path = fixture.path().join("study.fea");
    fs::write(
        &path,
        "import material steel from \"./missing-materials.fea\"\n",
    )
    .expect("write FEA source");
    let mut session = RunMatSession::with_options(false, false).expect("create session");

    let result = futures::executor::block_on(session.check_fea_source_path(&path))
        .expect("check FEA source");

    assert_eq!(result.status, FeaCheckStatus::Incomplete);
    assert_eq!(result.root_module.as_str(), "project/study.fea");
    assert!(result.roots.is_empty());
    assert!(!result.diagnostics.is_empty());
}

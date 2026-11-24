#![cfg(not(target_arch = "wasm32"))]
//! A root-level `.fea` opened by a relative path, with no runmat.toml, has one
//! path spelling. This file holds one test because it changes the process
//! working directory.

use std::{fs, path::Path};

use runmat_core::RunMatSession;
use runmat_runtime::analysis::FeaCheckStatus;

const STUDY: &str = r#"
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
fn a_root_level_relative_source_uses_the_runtime_spelling_of_its_physical_path() {
    let fixture = tempfile::tempdir().expect("create fixture");
    fs::write(fixture.path().join("study.fea"), STUDY).expect("write FEA source");
    let previous = std::env::current_dir().expect("read working directory");
    std::env::set_current_dir(fixture.path()).expect("enter fixture");
    let mut session = RunMatSession::with_options(false, false).expect("create session");
    let loaded =
        futures::executor::block_on(session.load_and_check_fea_source_path(Path::new("study.fea")));
    std::env::set_current_dir(previous).expect("restore working directory");
    let source = loaded.expect("load and check FEA source");

    assert_eq!(source.check_result().status, FeaCheckStatus::Valid);
    assert_eq!(source.check_result().root_module.as_str(), "study.fea");
    // The runtime rebuilds the path from the source root and the logical path.
    // The checked source must carry that same spelling.
    let rebuilt = runmat_runtime::analysis::physical_source_path(
        Path::new("."),
        &source.check_result().root_module,
    )
    .expect("rebuild the physical path");
    assert_eq!(rebuilt, Path::new("./study.fea"));
    assert_eq!(source.physical_path(), rebuilt);
}

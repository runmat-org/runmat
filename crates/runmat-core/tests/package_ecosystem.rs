#![cfg(not(target_arch = "wasm32"))]

use runmat_core::RunMatSession;
use runmat_gc::gc_test_context;
use runmat_package::{build_frozen_project, FrozenProjectHandoff};
use std::collections::BTreeSet;
use std::fs;

const REPRESENTATIVE_SOURCE_COUNT: usize = 1_024;

#[test]
fn large_pure_matlab_project_freezes_analyzes_compiles_and_executes() {
    gc_test_context(|| {
        let fixture = tempfile::TempDir::new().expect("create fixture");
        let sources = fixture.path().join("src");
        fs::create_dir(&sources).expect("create source root");
        fs::write(
            fixture.path().join("runmat.toml"),
            r#"
[package]
name = "representative-ecosystem-toolbox"
version = "1.0.0"

[sources]
roots = ["src"]
"#,
        )
        .expect("write manifest");

        for index in 0..REPRESENTATIVE_SOURCE_COUNT {
            fs::write(
                sources.join(format!("ecosystem_fn_{index:04}.m")),
                format!(
                    "function value = ecosystem_fn_{index:04}(input)\nvalue = input + 2;\nend\n"
                ),
            )
            .expect("write representative source");
        }

        let frozen = build_frozen_project(&fixture.path().join("runmat.toml"), BTreeSet::new())
            .expect("freeze representative project");
        assert_eq!(frozen.all_sources().count(), REPRESENTATIVE_SOURCE_COUNT);
        let expected_revision = frozen.revision();

        let handoff = FrozenProjectHandoff::new(frozen);
        let mut session =
            RunMatSession::with_options(false, false).expect("create interpreter session");
        assert_eq!(
            session
                .install_project_handoff(handoff)
                .expect("install frozen project"),
            expected_revision
        );

        let result = runmat_core::execute_text_request_for_testing(
            &mut session,
            "result = ecosystem_fn_1023(40); assert(result == 42);",
        )
        .expect("analyze, compile, and execute source from large frozen graph");
        assert!(
            result.error.is_none(),
            "representative ecosystem execution failed: {:?}",
            result.error
        );
    });
}

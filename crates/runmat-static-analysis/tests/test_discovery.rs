use runmat_parser::CompatMode;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};

fn digest(value: &str) -> String {
    runmat_execution::Digest::sha256(value).to_string()
}

fn environment() -> runmat_execution::ProgramEnvironment {
    runmat_execution::ProgramEnvironment::new(
        1,
        1,
        runmat_execution::Digest::sha256(b"runtime"),
        runmat_execution::Digest::sha256(b"catalog"),
        "matlab",
    )
    .unwrap()
}

fn source(path: &str, content: &str) -> SavedRunSource {
    SavedRunSource {
        owner_identity: "path:workspace".into(),
        relative_path: path.into(),
        content: content.into(),
    }
}

#[test]
fn canonical_function_factory_discovers_without_execution() {
    let snapshot = FrozenTestRunSnapshot::freeze(
        digest("graph"),
        "sha256:base-sources",
        environment(),
        digest("config"),
        vec![source(
            "tests/solverTest.m",
            "function tests = solverTest()\n tests = functiontests(localfunctions);\nend\nfunction testFast(testCase)\nend\nfunction helper()\nend\n",
        )],
        Vec::new(),
    )
    .unwrap();

    let discovery =
        runmat_static_analysis::testing::discover_frozen_tests(&snapshot, CompatMode::Matlab);
    assert!(
        discovery.diagnostics.is_empty(),
        "{:?}",
        discovery.diagnostics
    );
    assert_eq!(discovery.suites.len(), 1);
    assert_eq!(discovery.suites[0].tests.len(), 1);
    assert_eq!(
        discovery.suites[0].tests[0].display_name,
        "solverTest/testFast"
    );
}

#[test]
fn discovery_diagnostics_retain_package_aware_source_locations() {
    let snapshot = FrozenTestRunSnapshot::freeze(
        digest("graph"),
        "sha256:base-sources",
        environment(),
        digest("config"),
        vec![source("tests/BrokenTest.m", "function testBad(\n")],
        Vec::new(),
    )
    .unwrap();

    let discovery =
        runmat_static_analysis::testing::discover_frozen_tests(&snapshot, CompatMode::Matlab);
    assert!(discovery.suites.is_empty());
    assert_eq!(discovery.diagnostics.len(), 1);
    let diagnostic = &discovery.diagnostics[0];
    assert_eq!(diagnostic.code, "RunMat:TestDiscovery:ParseError");
    let source = diagnostic.source.as_ref().unwrap();
    assert_eq!(source.owner_identity, "path:workspace");
    assert_eq!(source.relative_path, "tests/BrokenTest.m");
    assert!(source.span.start_line >= 1);
}

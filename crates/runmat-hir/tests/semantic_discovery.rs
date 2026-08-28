use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
use runmat_hir::testing::{
    discover_tests, discover_tests_with_materialization, SemanticDiscoveryInput, SemanticTestSource,
};
use runmat_hir::{lower, LoweringContext};
use runmat_test::discovery::{MaterializationResponse, MaterializedValue};

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
    .with_domain_contribution(
        DomainContribution::new("runmat.test.config", Digest::sha256(b"test-config")).unwrap(),
    )
    .unwrap()
}

fn lower_source(source: &str) -> runmat_hir::HirAssembly {
    let program = runmat_parser::parse(source).expect("parse");
    lower(&program, &LoweringContext::empty())
        .expect("lower")
        .assembly
}

#[test]
fn discovers_script_sections_and_function_tests_without_scanning_lines() {
    let script_text = "shared = 1;\n%% adds values\nassert(shared + 1 == 2)\n%% rejects bad input\nassert(1 == 1)\n";
    let function_text = "function tests = solverTest()\n tests = 1;\nend\nfunction testFast(testCase)\nend\nfunction helper()\nend\nfunction slowTest(testCase)\nend\n";
    let script_assembly = lower_source(script_text);
    let function_assembly = lower_source(function_text);
    let sources = [
        SemanticTestSource {
            owner_identity: "standalone:project",
            relative_source_identity: "tests/test_script.m",
            source_text: script_text,
            assembly: &script_assembly,
        },
        SemanticTestSource {
            owner_identity: "standalone:project",
            relative_source_identity: "tests/solverTest.m",
            source_text: function_text,
            assembly: &function_assembly,
        },
    ];
    let discovery = discover_tests(&SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &sources,
    });

    assert_eq!(discovery.suites.len(), 2);
    let script = discovery
        .suites
        .iter()
        .find(|suite| suite.display_name == "test_script")
        .unwrap();
    assert_eq!(
        script
            .tests
            .iter()
            .map(|test| test.display_name.as_str())
            .collect::<Vec<_>>(),
        ["test_script/adds values", "test_script/rejects bad input"]
    );
    assert_eq!(script.tests[0].procedure.source.span.start_line, 3);

    let functions = discovery
        .suites
        .iter()
        .find(|suite| suite.display_name == "solverTest")
        .unwrap();
    assert_eq!(functions.tests.len(), 2);
    assert!(functions
        .tests
        .iter()
        .all(|test| !test.display_name.ends_with("/helper")));
    assert!(discovery.pending_materialization.is_empty());
    discovery.into_plan("native").expect("fully static plan");
}

#[test]
fn discovers_inherited_class_tests_fixtures_tags_and_static_parameters() {
    let base_text = "classdef BaseTest < matlab.unittest.TestCase\n  properties(TestParameter)\n    Mode = {'base-a','base-b'}\n  end\n  methods(TestMethodSetup)\n    function prepare(obj)\n    end\n  end\n  methods(Test)\n    function testInherited(obj)\n    end\n  end\nend";
    let derived_text = "classdef (TestTags={'fast'}) DerivedTest < BaseTest\n  properties(TestParameter)\n    Mode = {'a','b'}\n  end\n  methods(Test)\n    function testLocal(obj)\n    end\n  end\n  methods(TestMethodSetup)\n    function prepare(obj)\n    end\n  end\n  methods(TestMethodTeardown)\n    function cleanup(obj)\n    end\n  end\nend";
    let base_assembly = lower_source(base_text);
    let derived_assembly = lower_source(derived_text);
    let sources = [
        SemanticTestSource {
            owner_identity: "registry:acme/solver@1.0.0#sha256:tree",
            relative_source_identity: "tests/BaseTest.m",
            source_text: base_text,
            assembly: &base_assembly,
        },
        SemanticTestSource {
            owner_identity: "registry:acme/solver@1.0.0#sha256:tree",
            relative_source_identity: "tests/DerivedTest.m",
            source_text: derived_text,
            assembly: &derived_assembly,
        },
    ];
    let discovery = discover_tests(&SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &sources,
    });
    let derived = discovery
        .suites
        .iter()
        .find(|suite| suite.display_name == "DerivedTest")
        .unwrap();

    assert_eq!(derived.tests.len(), 4);
    assert_eq!(derived.fixtures.len(), 2);
    assert!(derived
        .tests
        .iter()
        .all(|test| test.tags == ["fast"] && test.parameters.len() == 1));
    assert_eq!(
        derived
            .tests
            .iter()
            .filter(|test| test.display_name.contains("testInherited"))
            .count(),
        2
    );
    assert!(derived
        .tests
        .iter()
        .filter(|test| test.display_name.contains("testInherited"))
        .all(|test| test.procedure.source.relative_path == "tests/BaseTest.m"));
    assert!(discovery.pending_materialization.is_empty());
    let selected = discovery
        .clone()
        .select(&runmat_test::descriptor::TestSelector {
            names: vec!["testLocal".into()],
            tags: vec!["fast".into()],
            source_prefixes: vec!["tests/".into()],
            excluded_tags: Vec::new(),
        });
    assert_eq!(selected.suites.len(), 1);
    let selected_derived = selected
        .suites
        .iter()
        .find(|suite| suite.display_name == "DerivedTest")
        .unwrap();
    assert_eq!(selected_derived.tests.len(), 2);

    let encoded = serde_json::to_vec(&discovery).unwrap();
    let reversed_sources = [
        SemanticTestSource {
            owner_identity: sources[1].owner_identity,
            relative_source_identity: sources[1].relative_source_identity,
            source_text: sources[1].source_text,
            assembly: sources[1].assembly,
        },
        SemanticTestSource {
            owner_identity: sources[0].owner_identity,
            relative_source_identity: sources[0].relative_source_identity,
            source_text: sources[0].source_text,
            assembly: sources[0].assembly,
        },
    ];
    let reversed = discover_tests(&SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &reversed_sources,
    });
    assert_eq!(encoded, serde_json::to_vec(&reversed).unwrap());
}

#[test]
fn unresolved_test_superclasses_are_reported_at_the_class_source() {
    let source_text = "classdef BrokenTest < MissingBase\n methods(Test)\n  function testValue(obj)\n  end\n end\nend";
    let assembly = lower_source(source_text);
    let sources = [SemanticTestSource {
        owner_identity: "standalone:broken",
        relative_source_identity: "tests/BrokenTest.m",
        source_text,
        assembly: &assembly,
    }];
    let discovery = discover_tests(&SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &sources,
    });

    assert!(discovery.suites.is_empty());
    assert_eq!(discovery.diagnostics.len(), 1);
    assert_eq!(
        discovery.diagnostics[0].code,
        "RunMat:TestDiscovery:UnresolvedSuperclass"
    );
    assert_eq!(
        discovery.diagnostics[0]
            .source
            .as_ref()
            .unwrap()
            .relative_path,
        "tests/BrokenTest.m"
    );
}

#[test]
fn dynamic_fixture_metadata_is_pending_and_blocks_plan_creation() {
    let source_text = "classdef (SharedTestFixtures={buildFixture}) DynamicTest < matlab.unittest.TestCase\n  methods(Test)\n    function testValue(obj)\n    end\n  end\nend";
    let assembly = lower_source(source_text);
    let sources = [SemanticTestSource {
        owner_identity: "standalone:dynamic",
        relative_source_identity: "tests/DynamicTest.m",
        source_text,
        assembly: &assembly,
    }];
    let discovery = discover_tests(&SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &sources,
    });
    assert_eq!(discovery.pending_materialization.len(), 1);
    assert_eq!(
        discovery.pending_materialization[0].expression,
        "{buildFixture}"
    );
    assert!(discovery.into_plan("browser").is_err());
}

#[test]
fn bounded_materialization_resolves_dynamic_parameters_and_shared_fixtures() {
    let source_text = "classdef (SharedTestFixtures={buildFixture}) DynamicTest < matlab.unittest.TestCase\n  properties(TestParameter)\n    Mode = 1 + 2\n  end\n  methods(Test)\n    function testValue(obj)\n    end\n  end\nend";
    let assembly = lower_source(source_text);
    let sources = [SemanticTestSource {
        owner_identity: "standalone:dynamic",
        relative_source_identity: "tests/DynamicTest.m",
        source_text,
        assembly: &assembly,
    }];
    let input = SemanticDiscoveryInput {
        program_revision: revision(),
        sources: &sources,
    };
    let pending = discover_tests(&input);
    assert_eq!(pending.pending_materialization.len(), 2);
    let responses = pending
        .pending_materialization
        .iter()
        .map(|request| MaterializationResponse {
            request_id: request.id.clone(),
            program_revision: request.program_revision.clone(),
            values: if request.semantic_path.contains("shared-fixtures") {
                vec![MaterializedValue {
                    name: "WorkingFolderFixture".into(),
                    normalized_identity: "working-folder".into(),
                    value: serde_json::json!({"kind": "working-folder"}),
                }]
            } else {
                vec![
                    MaterializedValue {
                        name: "small".into(),
                        normalized_identity: "1".into(),
                        value: serde_json::json!(1),
                    },
                    MaterializedValue {
                        name: "large".into(),
                        normalized_identity: "2".into(),
                        value: serde_json::json!(2),
                    },
                ]
            },
            diagnostics: Vec::new(),
        })
        .collect::<Vec<_>>();

    let resolved = discover_tests_with_materialization(&input, &responses).unwrap();
    assert!(resolved.pending_materialization.is_empty());
    assert_eq!(resolved.suites[0].tests.len(), 2);
    assert_eq!(resolved.suites[0].fixtures.len(), 1);
    resolved.into_plan("browser").unwrap();
}

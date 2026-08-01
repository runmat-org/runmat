use runmat_test::context::{TestCommand, TestExecutionContext};
use runmat_test::descriptor::{
    FixtureScope, ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan,
};
use runmat_test::event::{RedactionPolicy, SequencedEventSink};
use runmat_test::executor::{ExecutionFailure, ExecutionRequest, ExecutionResponse, TestExecutor};
use runmat_test::identity::{TestId, TestIdentityInput};
use runmat_test::lifecycle::{
    ExecutionPhase, FixtureScopeKey, LifecycleCase, LifecycleEngine, LifecycleStep, NeverCancelled,
    QualificationKind,
};
use runmat_test::plan::ProgramRevision;
use runmat_test::result::{Diagnostic, DiagnosticSeverity};
use sha2::{Digest, Sha256};

fn canonical_fixture() {
    let revision = ProgramRevision {
        graph_digest: "sha256:graph".into(),
        source_digest: "sha256:source".into(),
        semantic_schema: 1,
        compiler_schema: 1,
        test_config_digest: "sha256:test".into(),
    };
    let id = TestId::derive(&TestIdentityInput {
        owner_identity: "registry:acme/tool@1.0.0#sha256:tree",
        relative_source_identity: "tests/test_solver.m",
        semantic_scheme: "class-method",
        semantic_item_path: "TestSolver/testConverges",
        parameter_identity: "method=fast",
        fixture_identity: "TestSolver",
    });
    assert_eq!(
        revision.canonical_identity(),
        "sha256:graph|sha256:source|1|1|sha256:test"
    );
    assert_eq!(
        id.as_str(),
        "v1:sha256:d787f219e0ba29ebd4aac4cdafc1ff5ede0afac7aaa5152928e3b1afa429e389"
    );

    let run_id = runmat_test::identity::RunId::derive(
        &revision.canonical_identity(),
        "cross-host-conformance",
    );
    let test_scope = FixtureScopeKey {
        scope: FixtureScope::Test,
        identity: id.as_str().into(),
    };
    let case = LifecycleCase {
        context: TestExecutionContext {
            run_id: run_id.clone(),
            test_id: id,
            attempt: 1,
            random_seed: 42,
        },
        setups: vec![LifecycleStep {
            scope: test_scope.clone(),
            procedure: procedure("setup"),
        }],
        body: procedure("body"),
        declared_teardowns: vec![LifecycleStep {
            scope: test_scope,
            procedure: procedure("declared-teardown"),
        }],
    };
    let mut executor = ConformanceExecutor;
    let mut events = Vec::new();
    let mut sink = SequencedEventSink::new(run_id, &mut events);
    let outcome = LifecycleEngine::new(RedactionPolicy::new(["secret".into()], 1_024)).execute(
        &case,
        &mut executor,
        &NeverCancelled,
        &mut sink,
    );
    let bytes = serde_json::to_vec(&(outcome, events)).unwrap();
    assert_eq!(
        format!("sha256:{:x}", Sha256::digest(bytes)),
        "sha256:91a4e29b04c5c863773423aa3b0858da77580628e5af9614c5653b469f12ef50"
    );
}

struct ConformanceExecutor;

impl TestExecutor for ConformanceExecutor {
    fn execute(
        &mut self,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResponse, ExecutionFailure> {
        let commands = match request.procedure.semantic_path.as_str() {
            "setup" => vec![TestCommand::AddTeardown {
                scope: FixtureScopeKey {
                    scope: FixtureScope::Test,
                    identity: request.context.test_id.as_str().into(),
                },
                procedure: procedure("dynamic-teardown"),
            }],
            "body" => vec![TestCommand::Qualify {
                qualification: QualificationKind::VerificationFailed,
                diagnostic: Diagnostic {
                    identifier: "runmat:test:VerificationFailed".into(),
                    message: "expected true".into(),
                    severity: DiagnosticSeverity::Error,
                    phase: ExecutionPhase::TestBody,
                    source: None,
                    details: Vec::new(),
                },
            }],
            _ => Vec::new(),
        };
        Ok(ExecutionResponse {
            commands,
            output: if request.phase == ExecutionPhase::TestBody {
                "captured secret".into()
            } else {
                String::new()
            },
        })
    }
}

fn procedure(name: &str) -> ProcedureDescriptor {
    ProcedureDescriptor {
        semantic_path: name.into(),
        display_name: name.into(),
        kind: ProcedureKind::Function,
        source: SourceDescriptor {
            owner_identity: "registry:acme/tool@1.0.0#sha256:tree".into(),
            relative_path: "tests/test_solver.m".into(),
            semantic_path: name.into(),
            span: SourceSpan {
                start_byte: 0,
                end_byte: 1,
                start_line: 1,
                start_column: 1,
                end_line: 1,
                end_column: 2,
            },
        },
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_canonical_fixture() {
    canonical_fixture();
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::canonical_fixture;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_canonical_fixture() {
        canonical_fixture();
    }
}

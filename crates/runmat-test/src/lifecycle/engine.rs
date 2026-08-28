use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::context::{TestCommand, TestExecutionContext};
use crate::descriptor::{FixtureScope, ProcedureDescriptor};
use crate::event::{EventSink, RedactionPolicy, SequencedEventSink, TestEventPayload};
use crate::executor::{ExecutionFault, ExecutionRequest, ExecutionResponse, TestExecutor};
use crate::result::{Artifact, AttemptResult, Diagnostic, DiagnosticSeverity};

use super::{
    state::LifecycleState, CancellationProbe, ExecutionPhase, FixtureScopeKey, LifecycleOutcome,
    LifecycleStep, RegisteredTeardown,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LifecycleCase {
    pub context: TestExecutionContext,
    #[serde(default)]
    pub setups: Vec<LifecycleStep>,
    pub body: ProcedureDescriptor,
    #[serde(default)]
    pub declared_teardowns: Vec<LifecycleStep>,
}

pub struct LifecycleEngine {
    redaction: RedactionPolicy,
}

impl LifecycleEngine {
    pub fn new(redaction: RedactionPolicy) -> Self {
        Self { redaction }
    }

    pub async fn execute<E, S, C>(
        &self,
        case: &LifecycleCase,
        executor: &mut E,
        cancellation: &C,
        events: &mut SequencedEventSink<'_, S>,
    ) -> LifecycleOutcome
    where
        E: TestExecutor,
        S: EventSink,
        C: CancellationProbe,
    {
        let mut state = LifecycleState::default();
        let mut diagnostics = Vec::new();
        let mut artifacts = Vec::new();
        let mut output = String::new();
        let mut dynamic_teardowns = Vec::new();
        let mut next_teardown_order = 0;
        let mut active_scopes = vec![FixtureScopeKey {
            scope: FixtureScope::Test,
            identity: case.context.test_id.as_str().to_owned(),
        }];
        let mut active_keys = BTreeSet::from([active_scopes[0].clone()]);
        let mut executed = Vec::new();

        events.emit(TestEventPayload::TestStarted {
            test_id: case.context.test_id.clone(),
            attempt: case.context.attempt,
        });

        for step in &case.setups {
            if cancellation.is_cancelled() {
                self.record_fault(
                    case,
                    setup_phase(step.scope.scope),
                    ExecutionFault::Cancelled("cancelled before setup invocation".into()),
                    &mut state,
                    &mut diagnostics,
                    events,
                );
                break;
            }
            if active_keys.insert(step.scope.clone()) {
                active_scopes.push(step.scope.clone());
            }
            self.invoke(
                case,
                setup_phase(step.scope.scope),
                &step.scope,
                &step.procedure,
                executor,
                events,
                &mut state,
                &mut diagnostics,
                &mut artifacts,
                &mut output,
                &mut dynamic_teardowns,
                &mut next_teardown_order,
                &mut executed,
            )
            .await;
            if state.abort_test {
                break;
            }
        }

        if !state.abort_test {
            if cancellation.is_cancelled() {
                self.record_fault(
                    case,
                    ExecutionPhase::TestBody,
                    ExecutionFault::Cancelled("cancelled before test body".into()),
                    &mut state,
                    &mut diagnostics,
                    events,
                );
            } else {
                self.invoke(
                    case,
                    ExecutionPhase::TestBody,
                    &active_scopes[0],
                    &case.body,
                    executor,
                    events,
                    &mut state,
                    &mut diagnostics,
                    &mut artifacts,
                    &mut output,
                    &mut dynamic_teardowns,
                    &mut next_teardown_order,
                    &mut executed,
                )
                .await;
            }
        }

        active_scopes.sort_by_key(|scope| scope.scope);
        for scope in active_scopes.iter().rev() {
            while let Some(index) = dynamic_teardowns
                .iter()
                .enumerate()
                .filter(|(_, teardown)| teardown.scope == *scope)
                .max_by_key(|(_, teardown)| teardown.registration_order)
                .map(|(index, _)| index)
            {
                let teardown = dynamic_teardowns.remove(index);
                self.invoke(
                    case,
                    ExecutionPhase::DynamicTeardown,
                    &teardown.scope,
                    &teardown.procedure,
                    executor,
                    events,
                    &mut state,
                    &mut diagnostics,
                    &mut artifacts,
                    &mut output,
                    &mut dynamic_teardowns,
                    &mut next_teardown_order,
                    &mut executed,
                )
                .await;
            }
            for teardown in case
                .declared_teardowns
                .iter()
                .filter(|teardown| teardown.scope == *scope)
            {
                self.invoke(
                    case,
                    teardown_phase(scope.scope),
                    &teardown.scope,
                    &teardown.procedure,
                    executor,
                    events,
                    &mut state,
                    &mut diagnostics,
                    &mut artifacts,
                    &mut output,
                    &mut dynamic_teardowns,
                    &mut next_teardown_order,
                    &mut executed,
                )
                .await;
            }
        }

        let attempt = AttemptResult {
            test_id: case.context.test_id.clone(),
            attempt: case.context.attempt,
            state: state.result,
            diagnostics,
            artifacts,
            output,
            abort_run: state.abort_run,
        };
        events.emit(TestEventPayload::TestFinished {
            result: attempt.clone(),
        });
        LifecycleOutcome {
            attempt,
            executed_procedures: executed,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn invoke<E: TestExecutor, S: EventSink>(
        &self,
        case: &LifecycleCase,
        phase: ExecutionPhase,
        scope: &FixtureScopeKey,
        procedure: &ProcedureDescriptor,
        executor: &mut E,
        events: &mut SequencedEventSink<'_, S>,
        state: &mut LifecycleState,
        diagnostics: &mut Vec<Diagnostic>,
        artifacts: &mut Vec<Artifact>,
        output: &mut String,
        dynamic_teardowns: &mut Vec<RegisteredTeardown>,
        next_teardown_order: &mut u64,
        executed: &mut Vec<String>,
    ) -> bool {
        let procedure_identity = procedure.semantic_path.clone();
        events.emit(TestEventPayload::PhaseStarted {
            test_id: case.context.test_id.clone(),
            attempt: case.context.attempt,
            phase,
            procedure: procedure_identity.clone(),
        });
        executed.push(procedure_identity.clone());
        let request = ExecutionRequest {
            context: case.context.clone(),
            phase,
            scope: scope.clone(),
            procedure: procedure.clone(),
        };
        let response = executor.execute(&request).await;
        let completed = match response {
            Ok(response) => {
                self.apply_response(
                    case,
                    phase,
                    response,
                    events,
                    state,
                    diagnostics,
                    artifacts,
                    output,
                    dynamic_teardowns,
                    next_teardown_order,
                );
                true
            }
            Err(failure) => {
                self.apply_response(
                    case,
                    phase,
                    failure.partial,
                    events,
                    state,
                    diagnostics,
                    artifacts,
                    output,
                    dynamic_teardowns,
                    next_teardown_order,
                );
                self.record_fault(case, phase, failure.fault, state, diagnostics, events);
                false
            }
        };
        events.emit(TestEventPayload::PhaseFinished {
            test_id: case.context.test_id.clone(),
            attempt: case.context.attempt,
            phase,
            procedure: procedure_identity,
        });
        completed
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_response<S: EventSink>(
        &self,
        case: &LifecycleCase,
        phase: ExecutionPhase,
        response: ExecutionResponse,
        events: &mut SequencedEventSink<'_, S>,
        state: &mut LifecycleState,
        diagnostics: &mut Vec<Diagnostic>,
        artifacts: &mut Vec<Artifact>,
        output: &mut String,
        dynamic_teardowns: &mut Vec<RegisteredTeardown>,
        next_teardown_order: &mut u64,
    ) {
        if !response.output.is_empty() {
            let remaining = self.redaction.max_text_bytes.saturating_sub(output.len());
            let redacted = self
                .redaction
                .redact_with_limit(&response.output, remaining);
            output.push_str(&redacted.text);
            events.emit(TestEventPayload::Output {
                test_id: case.context.test_id.clone(),
                attempt: case.context.attempt,
                text: redacted.text,
                truncated: redacted.truncated,
            });
        }
        for command in response.commands {
            match command {
                TestCommand::AddTeardown { scope, procedure } => {
                    if phase.is_teardown() {
                        self.record_fault(
                            case,
                            phase,
                            ExecutionFault::Uncaught(
                                "addTeardown cannot be registered from a teardown phase".into(),
                            ),
                            state,
                            diagnostics,
                            events,
                        );
                    } else {
                        dynamic_teardowns.push(RegisteredTeardown {
                            scope,
                            procedure,
                            registration_order: *next_teardown_order,
                        });
                        *next_teardown_order += 1;
                    }
                }
                TestCommand::Qualify {
                    qualification,
                    mut diagnostic,
                } => {
                    diagnostic.phase = phase;
                    state.apply_qualification(qualification);
                    diagnostics.push(diagnostic.clone());
                    events.emit(TestEventPayload::Qualification {
                        test_id: case.context.test_id.clone(),
                        attempt: case.context.attempt,
                        kind: qualification,
                        diagnostic,
                    });
                }
                TestCommand::RecordDiagnostic { mut diagnostic } => {
                    diagnostic.phase = phase;
                    diagnostics.push(diagnostic.clone());
                    events.emit(TestEventPayload::Diagnostic {
                        test_id: case.context.test_id.clone(),
                        attempt: case.context.attempt,
                        diagnostic,
                    });
                }
                TestCommand::AttachArtifact { artifact } => {
                    artifacts.push(artifact.clone());
                    events.emit(TestEventPayload::Artifact {
                        test_id: case.context.test_id.clone(),
                        attempt: case.context.attempt,
                        artifact,
                    });
                }
            }
        }
    }

    fn record_fault<S: EventSink>(
        &self,
        case: &LifecycleCase,
        phase: ExecutionPhase,
        fault: ExecutionFault,
        state: &mut LifecycleState,
        diagnostics: &mut Vec<Diagnostic>,
        events: &mut SequencedEventSink<'_, S>,
    ) {
        let (identifier, message) = match &fault {
            ExecutionFault::Uncaught(message) => ("runmat:test:UncaughtError", message),
            ExecutionFault::TimedOut(message) => ("runmat:test:Timeout", message),
            ExecutionFault::Cancelled(message) => ("runmat:test:Cancelled", message),
            ExecutionFault::WorkerCrashed(message) => ("runmat:test:WorkerCrash", message),
        };
        state.apply_fault(&fault);
        let redacted = self.redaction.redact(message);
        let diagnostic = Diagnostic {
            identifier: identifier.into(),
            message: redacted.text,
            severity: if matches!(fault, ExecutionFault::Cancelled(_)) {
                DiagnosticSeverity::Information
            } else {
                DiagnosticSeverity::Error
            },
            phase,
            source: None,
            details: Vec::new(),
        };
        diagnostics.push(diagnostic.clone());
        events.emit(TestEventPayload::Diagnostic {
            test_id: case.context.test_id.clone(),
            attempt: case.context.attempt,
            diagnostic,
        });
    }
}

fn setup_phase(scope: FixtureScope) -> ExecutionPhase {
    match scope {
        FixtureScope::Run => ExecutionPhase::RunSetup,
        FixtureScope::Suite => ExecutionPhase::SuiteSetup,
        FixtureScope::Class => ExecutionPhase::ClassSetup,
        FixtureScope::Test => ExecutionPhase::TestSetup,
    }
}

fn teardown_phase(scope: FixtureScope) -> ExecutionPhase {
    match scope {
        FixtureScope::Run => ExecutionPhase::RunTeardown,
        FixtureScope::Suite => ExecutionPhase::SuiteTeardown,
        FixtureScope::Class => ExecutionPhase::ClassTeardown,
        FixtureScope::Test => ExecutionPhase::TestTeardown,
    }
}

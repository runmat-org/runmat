use std::collections::{BTreeMap, HashMap};

use runmat_runtime::testing::{
    install_test_context_in, ActiveTestContext, RuntimeTeardownInvocation,
};
use runmat_test::context::TestCommand;
use runmat_test::descriptor::{ParameterDescriptor, ProcedureDescriptor, ProcedureKind};
use runmat_test::executor::{
    ExecutionFailure, ExecutionFault, ExecutionFuture, ExecutionRequest, ExecutionResponse,
    TestExecutor,
};
use runmat_test::identity::TestId;
use runmat_test::protocol::ProtocolLimits;

use crate::{ExecutableUnit, InvocationControl, ProcedureInvocation, RunError, RunMatSession};

use super::source_catalog::TestSourceCatalog;

#[derive(Clone)]
struct RegisteredRuntimeTeardown {
    unit: ExecutableUnit,
    invocation: RuntimeTeardownInvocation,
}

/// Core's adapter from the portable lifecycle protocol to exact RunMat
/// executable units. It owns no lifecycle policy: it only compiles, caches,
/// invokes, and translates runtime side effects into protocol responses.
pub struct CoreTestExecutor<'a> {
    session: &'a mut RunMatSession,
    catalog: TestSourceCatalog,
    parameters: BTreeMap<TestId, Vec<ParameterDescriptor>>,
    units: BTreeMap<String, ExecutableUnit>,
    runtime_teardowns: HashMap<String, RegisteredRuntimeTeardown>,
    control: InvocationControl,
    limits: ProtocolLimits,
}

impl<'a> CoreTestExecutor<'a> {
    pub(super) fn new(
        session: &'a mut RunMatSession,
        catalog: TestSourceCatalog,
        parameters: BTreeMap<TestId, Vec<ParameterDescriptor>>,
        control: InvocationControl,
    ) -> Self {
        Self {
            session,
            catalog,
            parameters,
            units: BTreeMap::new(),
            runtime_teardowns: HashMap::new(),
            control,
            limits: ProtocolLimits::default(),
        }
    }

    pub(super) fn coverage_fragments(
        &self,
        counts: &BTreeMap<u64, u64>,
    ) -> Vec<runmat_test::coverage::CoverageFragment> {
        let program_revision = self.catalog.revision().canonical_identity();
        self.units
            .values()
            .map(|unit| {
                let unit_counts = unit
                    .coverage_plan()
                    .sites()
                    .iter()
                    .filter_map(|site| {
                        counts
                            .get(&site.counter_key)
                            .copied()
                            .map(|count| (site.counter_key, count))
                    })
                    .collect();
                unit.coverage_plan()
                    .fragment(program_revision.clone(), unit_counts)
            })
            .collect()
    }

    async fn execute_request(
        &mut self,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResponse, ExecutionFailure> {
        if let Some(teardown) = self
            .runtime_teardowns
            .remove(&request.procedure.semantic_path)
        {
            return self.execute_runtime_teardown(request, teardown).await;
        }

        let unit = self.unit_for(&request.procedure).await?;
        let parameters = self
            .parameters
            .get(&request.context.test_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let invocation = super::procedure::invocation_for(
            self.session.runtime_context(),
            &unit,
            &request.procedure,
            parameters,
        )
        .map_err(|error| run_error_failure(RunError::Runtime(error)))?;
        self.execute_invocation(request, unit, invocation).await
    }

    async fn unit_for(
        &mut self,
        procedure: &ProcedureDescriptor,
    ) -> Result<ExecutableUnit, ExecutionFailure> {
        let span_only = procedure.kind == ProcedureKind::ScriptSection;
        let key = if span_only {
            format!(
                "{}:{}:{}-{}",
                procedure.source.owner_identity,
                procedure.source.relative_path,
                procedure.source.span.start_byte,
                procedure.source.span.end_byte
            )
        } else {
            format!(
                "{}:{}",
                procedure.source.owner_identity, procedure.source.relative_path
            )
        };
        if let Some(unit) = self.units.get(&key) {
            return Ok(unit.clone());
        }
        let source = self
            .catalog
            .executable_source(&procedure.source)
            .map_err(uncaught)?;
        let unit = self
            .session
            .compile_test_executable_unit(
                source,
                &procedure.source,
                self.catalog.snapshot(),
                span_only,
            )
            .await
            .map_err(run_error_failure)?;
        self.units.insert(key, unit.clone());
        Ok(unit)
    }

    async fn execute_invocation(
        &mut self,
        request: &ExecutionRequest,
        unit: ExecutableUnit,
        invocation: ProcedureInvocation,
    ) -> Result<ExecutionResponse, ExecutionFailure> {
        let runtime = self.session.runtime_context().clone();
        let guard = install_test_context_in(
            &runtime,
            ActiveTestContext {
                execution: request.context.clone(),
                phase: request.phase,
                scope: request.scope.clone(),
            },
            self.limits,
        );
        let handle = guard.handle();
        reset_console_output(&runtime);
        let result = self
            .session
            .invoke_executable(&unit, invocation, &self.control)
            .await;
        let commands = handle.take_commands();
        for invocation in handle.take_runtime_teardowns() {
            self.runtime_teardowns.insert(
                invocation.semantic_path.clone(),
                RegisteredRuntimeTeardown {
                    unit: unit.clone(),
                    invocation,
                },
            );
        }
        drop(guard);
        let output = take_console_output(&runtime);
        let response = ExecutionResponse { commands, output };
        match result {
            Ok(_) => Ok(response),
            Err(error) if qualification_already_recorded(&response.commands, &error) => {
                Ok(response)
            }
            Err(error) => Err(run_error_failure_with_partial(error, response)),
        }
    }

    async fn execute_runtime_teardown(
        &mut self,
        request: &ExecutionRequest,
        teardown: RegisteredRuntimeTeardown,
    ) -> Result<ExecutionResponse, ExecutionFailure> {
        let runtime = self
            .session
            .execution_runtime_context(teardown.unit.revision().program_revision.clone());
        let runtime = teardown
            .unit
            .bind_runtime_sources(runtime)
            .map_err(uncaught)?;
        let bytecode = teardown.unit.runtime_bytecode(&runtime).map_err(uncaught)?;
        let selected = crate::callable_selection::select(
            &runtime,
            &bytecode.function_registry,
            teardown.invocation.callback.clone(),
        )
        .map_err(|error| uncaught(error.to_string()))?;
        // This is a routing hint, not execution admission. The executable
        // branch selects and authenticates the original target again after
        // its actual Admission and source binding. Other callable families
        // retain the canonical runtime callback dispatch instead.
        if crate::executable::callback::owned_function(
            &teardown.unit,
            &bytecode.function_registry,
            selected.selected_target(),
        )
        .is_some()
        {
            drop(selected);
            return self
                .execute_invocation(
                    request,
                    teardown.unit,
                    ProcedureInvocation::retained_callable(
                        teardown.invocation.callback,
                        teardown.invocation.arguments,
                    ),
                )
                .await;
        }
        reset_console_output(&runtime);
        match runtime
            .scope(selected.invoke_retained(Some(&runtime), teardown.invocation.arguments, 0))
            .await
        {
            Ok(_) => Ok(ExecutionResponse {
                commands: Vec::new(),
                output: take_console_output(&runtime),
            }),
            Err(error) => Err(ExecutionFailure {
                fault: ExecutionFault::Uncaught(error.to_string()),
                partial: ExecutionResponse {
                    commands: Vec::new(),
                    output: take_console_output(&runtime),
                },
            }),
        }
    }
}

impl TestExecutor for CoreTestExecutor<'_> {
    fn execute<'a>(&'a mut self, request: &'a ExecutionRequest) -> ExecutionFuture<'a> {
        Box::pin(async move { self.execute_request(request).await })
    }
}

fn reset_console_output(runtime: &runmat_runtime::context::RuntimeContext) {
    let _context = runtime.enter();
    runmat_runtime::console::reset_thread_buffer();
}

fn take_console_output(runtime: &runmat_runtime::context::RuntimeContext) -> String {
    let _context = runtime.enter();
    runmat_runtime::console::take_thread_buffer()
        .into_iter()
        .filter(|entry| entry.stream != runmat_runtime::console::ConsoleStream::ClearScreen)
        .fold(String::new(), |mut output, entry| {
            output.push_str(entry.text.as_str());
            output
        })
}

fn qualification_already_recorded(commands: &[TestCommand], error: &RunError) -> bool {
    let identifier = match error {
        RunError::Runtime(error) => error.identifier(),
        _ => None,
    };
    identifier.is_some_and(|identifier| {
        identifier.starts_with("RunMat:Test:")
            && commands
                .iter()
                .any(|command| matches!(command, TestCommand::Qualify { .. }))
    })
}

fn uncaught(message: String) -> ExecutionFailure {
    ExecutionFault::Uncaught(message).into()
}

fn run_error_failure(error: RunError) -> ExecutionFailure {
    run_error_failure_with_partial(error, ExecutionResponse::default())
}

fn run_error_failure_with_partial(error: RunError, partial: ExecutionResponse) -> ExecutionFailure {
    let fault = match &error {
        RunError::Runtime(error)
            if error
                .identifier()
                .is_some_and(|id| id.contains("Cancelled") || id.contains("Interrupt")) =>
        {
            ExecutionFault::Cancelled(error.to_string())
        }
        RunError::Runtime(error)
            if error
                .identifier()
                .is_some_and(|id| id.contains("Deadline") || id.contains("Timeout")) =>
        {
            ExecutionFault::TimedOut(error.to_string())
        }
        _ => ExecutionFault::Uncaught(error.to_string()),
    };
    ExecutionFailure { fault, partial }
}

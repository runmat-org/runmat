use std::collections::{BTreeMap, HashMap};

use runmat_runtime::testing::{
    install_test_context_in, ActiveTestContext, RuntimeTeardownInvocation,
};
use runmat_test::context::TestCommand;
use runmat_test::descriptor::{
    FixtureScope, ParameterDescriptor, ProcedureDescriptor, ProcedureKind,
};
use runmat_test::executor::{
    ExecutionFailure, ExecutionFault, ExecutionFuture, ExecutionRequest, ExecutionResponse,
    TestExecutor,
};
use runmat_test::identity::TestId;
use runmat_test::lifecycle::{ExecutionPhase, FixtureScopeKey};
use runmat_test::protocol::ProtocolLimits;
use runmat_value::Value;

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
        let invocation = super::procedure::invocation_for(&unit, &request.procedure, parameters)
            .map_err(uncaught)?;
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
            .executable_source(&procedure.source, span_only)
            .map_err(uncaught)?;
        let unit = self
            .session
            .compile_executable_unit(source, Some(self.catalog.revision()))
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
                scope: scope_for(request),
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
        let name = callback_name(&teardown.invocation.callback)
            .ok_or_else(|| uncaught("runtime teardown callback is not callable".to_string()))?;
        if teardown
            .unit
            .procedure_names()
            .iter()
            .any(|item| item == &name)
        {
            return self
                .execute_invocation(
                    request,
                    teardown.unit,
                    ProcedureInvocation::function(name, teardown.invocation.arguments),
                )
                .await;
        }
        let runtime = self.session.runtime_context().clone();
        reset_console_output(&runtime);
        match runtime
            .scope(runmat_runtime::call_builtin_async(
                &name,
                &teardown.invocation.arguments,
            ))
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

fn scope_for(request: &ExecutionRequest) -> FixtureScopeKey {
    let scope = match request.phase {
        ExecutionPhase::RunSetup | ExecutionPhase::RunTeardown => FixtureScope::Run,
        ExecutionPhase::SuiteSetup | ExecutionPhase::SuiteTeardown => FixtureScope::Suite,
        ExecutionPhase::ClassSetup | ExecutionPhase::ClassTeardown => FixtureScope::Class,
        ExecutionPhase::TestSetup
        | ExecutionPhase::TestBody
        | ExecutionPhase::DynamicTeardown
        | ExecutionPhase::TestTeardown => FixtureScope::Test,
    };
    FixtureScopeKey {
        scope,
        identity: if scope == FixtureScope::Test {
            request.context.test_id.as_str().to_owned()
        } else {
            request.procedure.semantic_path.clone()
        },
    }
}

fn callback_name(value: &Value) -> Option<String> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.clone()),
        Value::BoundFunctionHandle { name, .. } => Some(name.clone()),
        Value::Closure(closure) => Some(closure.function_name.clone()),
        _ => None,
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
        .map(|entry| entry.text)
        .collect()
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

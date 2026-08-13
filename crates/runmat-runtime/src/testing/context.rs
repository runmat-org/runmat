use std::cell::RefCell;
use std::rc::Rc;

use runmat_test::context::{TestCommand, TestExecutionContext};
use runmat_test::descriptor::{ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan};
use runmat_test::lifecycle::{ExecutionPhase, FixtureScopeKey};
use runmat_test::protocol::ProtocolLimits;

#[derive(Clone, Debug)]
pub struct ActiveTestContext {
    pub execution: TestExecutionContext,
    pub phase: ExecutionPhase,
    pub scope: FixtureScopeKey,
}

#[derive(Clone, Debug)]
pub struct RuntimeTeardownInvocation {
    pub semantic_path: String,
    pub callback: runmat_value::Value,
    pub arguments: Vec<runmat_value::Value>,
}

#[derive(Debug)]
pub(crate) struct ContextState {
    active: ActiveTestContext,
    limits: ProtocolLimits,
    commands: Vec<TestCommand>,
    runtime_teardowns: Vec<RuntimeTeardownInvocation>,
}

#[derive(Clone, Debug)]
pub struct TestContextHandle {
    state: Rc<RefCell<ContextState>>,
}

impl TestContextHandle {
    pub fn active(&self) -> ActiveTestContext {
        self.state.borrow().active.clone()
    }

    pub fn commands(&self) -> Vec<TestCommand> {
        self.state.borrow().commands.clone()
    }

    pub fn take_commands(&self) -> Vec<TestCommand> {
        std::mem::take(&mut self.state.borrow_mut().commands)
    }

    pub fn runtime_teardowns(&self) -> Vec<RuntimeTeardownInvocation> {
        self.state.borrow().runtime_teardowns.clone()
    }

    pub fn take_runtime_teardowns(&self) -> Vec<RuntimeTeardownInvocation> {
        std::mem::take(&mut self.state.borrow_mut().runtime_teardowns)
    }
}

#[derive(Debug)]
pub struct TestContextGuard {
    state: Rc<RefCell<ContextState>>,
    context: Option<Rc<crate::context::RuntimeContextState>>,
}

impl TestContextGuard {
    pub fn handle(&self) -> TestContextHandle {
        TestContextHandle {
            state: Rc::clone(&self.state),
        }
    }
}

impl Drop for TestContextGuard {
    fn drop(&mut self) {
        if let Some(context) = &self.context {
            let popped = context.test_contexts.borrow_mut().pop();
            debug_assert!(popped
                .as_ref()
                .is_some_and(|state| Rc::ptr_eq(state, &self.state)));
        } else {
            TEST_CONTEXT_STACK.with(|stack| {
                let popped = stack.borrow_mut().pop();
                debug_assert!(popped
                    .as_ref()
                    .is_some_and(|state| Rc::ptr_eq(state, &self.state)));
            });
        }
    }
}

thread_local! {
    static TEST_CONTEXT_STACK: RefCell<Vec<Rc<RefCell<ContextState>>>> =
        const { RefCell::new(Vec::new()) };
}

pub fn install_test_context(active: ActiveTestContext, limits: ProtocolLimits) -> TestContextGuard {
    let state = Rc::new(RefCell::new(ContextState {
        active,
        limits,
        commands: Vec::new(),
        runtime_teardowns: Vec::new(),
    }));
    let context = crate::context::legacy::active().map(|context| Rc::clone(context.state()));
    if let Some(context) = &context {
        context.test_contexts.borrow_mut().push(Rc::clone(&state));
    } else {
        TEST_CONTEXT_STACK.with(|stack| stack.borrow_mut().push(Rc::clone(&state)));
    }
    TestContextGuard { state, context }
}

fn current_test_context_state() -> Option<Rc<RefCell<ContextState>>> {
    if let Some(context) = crate::context::legacy::active() {
        return context.state().test_contexts.borrow().last().cloned();
    }
    TEST_CONTEXT_STACK.with(|stack| stack.borrow().last().cloned())
}

pub fn record_runtime_teardown(
    callback: runmat_value::Value,
    arguments: Vec<runmat_value::Value>,
) -> Result<(), &'static str> {
    {
        let state = current_test_context_state()
            .ok_or("addTeardown requires an active test lifecycle context")?;
        let mut state = state.borrow_mut();
        if state.active.phase.is_teardown() {
            return Err("addTeardown cannot be registered from a teardown phase");
        }
        if state.commands.len() >= state.limits.max_commands_per_invocation as usize {
            return Err("testing command limit exceeded");
        }
        let callback_name = match &callback {
            runmat_value::Value::FunctionHandle(name)
            | runmat_value::Value::ExternalFunctionHandle(name)
            | runmat_value::Value::MethodFunctionHandle(name) => name.clone(),
            runmat_value::Value::BoundFunctionHandle { name, .. } => name.clone(),
            runmat_value::Value::Closure(closure) => closure.function_name.clone(),
            _ => return Err("addTeardown callback must be a function handle"),
        };
        let semantic_path = format!(
            "runtime-teardown:{}:{}:{}",
            state.active.execution.test_id.as_str(),
            state.active.execution.attempt,
            state.runtime_teardowns.len()
        );
        let source = crate::source_context::current_source_info();
        let relative_path = source
            .as_ref()
            .map(|source| {
                source
                    .fullpath_name
                    .as_deref()
                    .unwrap_or(source.name.as_ref())
                    .to_string()
            })
            .unwrap_or_default();
        let procedure = ProcedureDescriptor {
            semantic_path: semantic_path.clone(),
            display_name: callback_name,
            kind: ProcedureKind::Teardown,
            source: SourceDescriptor {
                owner_identity: "runtime:test-context".into(),
                semantic_path: semantic_path.clone(),
                relative_path,
                span: SourceSpan {
                    start_byte: 0,
                    end_byte: 0,
                    start_line: 1,
                    start_column: 1,
                    end_line: 1,
                    end_column: 1,
                },
            },
        };
        let scope = state.active.scope.clone();
        state
            .commands
            .push(TestCommand::AddTeardown { scope, procedure });
        state.runtime_teardowns.push(RuntimeTeardownInvocation {
            semantic_path,
            callback,
            arguments,
        });
        Ok(())
    }
}

pub fn active_test_context() -> Option<ActiveTestContext> {
    current_test_context_state().map(|state| state.borrow().active.clone())
}

pub fn record_test_command(command: TestCommand) -> Result<(), &'static str> {
    {
        let state = current_test_context_state()
            .ok_or("testing command requires an active test lifecycle context")?;
        let mut state = state.borrow_mut();
        if state.commands.len() >= state.limits.max_commands_per_invocation as usize {
            return Err("testing command limit exceeded");
        }
        state.commands.push(command);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use runmat_test::descriptor::FixtureScope;
    use runmat_test::identity::{RunId, TestId};
    use runmat_test::lifecycle::QualificationKind;
    use runmat_test::result::{Diagnostic, DiagnosticSeverity};

    use super::*;

    #[test]
    fn scoped_context_records_bounded_domain_commands() {
        let guard = install_test_context(
            ActiveTestContext {
                execution: TestExecutionContext {
                    run_id: RunId::derive("revision", "run"),
                    test_id: TestId::derive(&runmat_test::identity::TestIdentityInput {
                        owner_identity: "owner",
                        relative_source_identity: "test.m",
                        semantic_scheme: "function",
                        semantic_item_path: "test",
                        parameter_identity: "",
                        fixture_identity: "",
                    }),
                    attempt: 1,
                    random_seed: 7,
                },
                phase: ExecutionPhase::TestBody,
                scope: FixtureScopeKey {
                    scope: FixtureScope::Test,
                    identity: "test".into(),
                },
            },
            ProtocolLimits::default(),
        );
        record_test_command(TestCommand::Qualify {
            qualification: QualificationKind::VerificationFailed,
            diagnostic: Diagnostic {
                identifier: "RunMat:VerificationFailed".into(),
                message: "nope".into(),
                severity: DiagnosticSeverity::Error,
                phase: ExecutionPhase::TestBody,
                source: None,
                details: Vec::new(),
            },
        })
        .unwrap();
        assert_eq!(guard.handle().commands().len(), 1);
        drop(guard);
        assert!(active_test_context().is_none());
    }
}

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub type RunSuiteFuture = Pin<Box<dyn Future<Output = BuiltinResult<Value>>>>;
pub type RunSuiteService = dyn Fn(Option<RuntimeContext>, Value, Vec<Value>) -> RunSuiteFuture;
pub type RunTestsService = dyn Fn(Option<RuntimeContext>, Vec<Value>) -> RunSuiteFuture;
pub type DiscoverTestsService = dyn Fn(Option<RuntimeContext>, Vec<Value>) -> RunSuiteFuture;

/// Runtime-facing ports supplied by the active Core execution composition.
///
/// MATLAB builtins may project arguments and results, but all lifecycle and
/// scheduling authority remains behind these services.
#[derive(Clone)]
pub struct RuntimeTestServices {
    run_suite: Rc<RunSuiteService>,
    run_tests: Rc<RunTestsService>,
    discover_tests: Rc<DiscoverTestsService>,
}

impl std::fmt::Debug for RuntimeTestServices {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeTestServices")
            .finish_non_exhaustive()
    }
}

impl RuntimeTestServices {
    pub fn new(
        run_suite: impl Fn(Option<RuntimeContext>, Value, Vec<Value>) -> RunSuiteFuture + 'static,
        run_tests: impl Fn(Option<RuntimeContext>, Vec<Value>) -> RunSuiteFuture + 'static,
        discover_tests: impl Fn(Option<RuntimeContext>, Vec<Value>) -> RunSuiteFuture + 'static,
    ) -> Self {
        Self {
            run_suite: Rc::new(run_suite),
            run_tests: Rc::new(run_tests),
            discover_tests: Rc::new(discover_tests),
        }
    }
}

thread_local! {
    static TEST_SERVICES_STACK: RefCell<Vec<RuntimeTestServices>> =
        const { RefCell::new(Vec::new()) };
}

pub struct TestServicesGuard {
    context: Option<Rc<crate::context::RuntimeContextState>>,
}

impl Drop for TestServicesGuard {
    fn drop(&mut self) {
        if let Some(context) = &self.context {
            let popped = context.test_services.borrow_mut().pop();
            debug_assert!(popped.is_some());
        } else {
            TEST_SERVICES_STACK.with(|stack| {
                let popped = stack.borrow_mut().pop();
                debug_assert!(popped.is_some());
            });
        }
    }
}

pub fn install_test_services(services: RuntimeTestServices) -> TestServicesGuard {
    let context = crate::context::legacy::active().map(|context| Rc::clone(context.state()));
    if let Some(context) = &context {
        context.test_services.borrow_mut().push(services);
    } else {
        TEST_SERVICES_STACK.with(|stack| stack.borrow_mut().push(services));
    }
    TestServicesGuard { context }
}

fn current_test_services() -> Option<(RuntimeTestServices, Option<RuntimeContext>)> {
    if let Some(context) = crate::context::legacy::active() {
        let service = { context.state().test_services.borrow().last().cloned() };
        return service.map(|service| (service, Some(context)));
    }
    TEST_SERVICES_STACK.with(|stack| {
        stack
            .borrow()
            .last()
            .cloned()
            .map(|service| (service, None))
    })
}

pub async fn run_test_suite(suite: Value, plugins: Vec<Value>) -> BuiltinResult<Value> {
    let service = current_test_services();
    let Some((service, caller)) = service else {
        return Err(crate::build_runtime_error(
            "TestRunner.run requires an active Core test executor",
        )
        .with_identifier("RunMat:Testing:RequiresExecutor")
        .with_builtin("matlab.unittest.TestRunner.run")
        .build());
    };
    (service.run_suite)(caller, suite, plugins).await
}

pub async fn run_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let service = current_test_services();
    let Some((service, caller)) = service else {
        return Err(
            crate::build_runtime_error("runtests requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("runtests")
                .build(),
        );
    };
    (service.run_tests)(caller, args).await
}

pub async fn discover_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let service = current_test_services();
    let Some((service, caller)) = service else {
        return Err(
            crate::build_runtime_error("testsuite requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("testsuite")
                .build(),
        );
    };
    (service.discover_tests)(caller, args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_installation_is_scoped_and_async_safe() {
        let _guard = install_test_services(RuntimeTestServices::new(
            |_caller, suite, _plugins| Box::pin(async move { Ok(suite) }),
            |_caller, args| {
                Box::pin(async move {
                    let count = args.len();
                    Ok(Value::Cell(
                        runmat_value::CellArray::new(args, 1, count).unwrap(),
                    ))
                })
            },
            |_caller, args| {
                Box::pin(async move {
                    Ok(args
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| Value::output_list(Vec::new())))
                })
            },
        ));
        let suite = Value::String("suite".into());
        assert_eq!(
            futures::executor::block_on(run_test_suite(suite.clone(), Vec::new())).unwrap(),
            suite
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_service_receives_selected_callers_ancestry_at_admission() {
        use crate::execution::RuntimeExecutionService;
        use crate::foreign::MexCallbackAncestry;

        let parent = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_mex_callback_ancestry(MexCallbackAncestry::from_wire(2).unwrap());
        let unrelated = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let service = RuntimeTestServices::new(
            |_caller, suite, _plugins| Box::pin(async move { Ok(suite) }),
            |caller, _args| {
                let depth = caller
                    .expect("the caller is selected at admission")
                    .mex_callback_ancestry()
                    .wire();
                Box::pin(async move { Ok(Value::Num(f64::from(depth))) })
            },
            |_caller, _args| Box::pin(async move { Ok(Value::Num(0.0)) }),
        );
        let observed = futures::executor::block_on(parent.scope(async {
            let _guard = install_test_services(service);
            run_tests(Vec::new()).await.unwrap()
        }));
        assert_eq!(observed, Value::Num(2.0));
        assert_eq!(parent.mex_callback_ancestry().wire(), 2);
        assert_eq!(unrelated.mex_callback_ancestry().wire(), 0);

        let child = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .try_inherit_foreign_callback_lineage_for_child(&parent)
            .expect("the child has independent Runtime owners");
        assert_eq!(child.mex_callback_ancestry().wire(), 2);
        assert!(!Rc::ptr_eq(parent.state(), child.state()));
        assert!(parent
            .clone()
            .try_inherit_foreign_callback_lineage_for_child(&parent)
            .is_none());
    }
}

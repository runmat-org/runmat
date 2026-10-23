use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_value::{OutputListStorage, Value};

use crate::context::RuntimeContext;
use crate::execution::{rooted_continuation, RootedValue, RootedValueLease};
use crate::BuiltinResult;

pub type RunSuiteFuture = Pin<Box<dyn Future<Output = BuiltinResult<Value>>>>;
pub type RunSuiteService =
    dyn Fn(Option<RuntimeContext>, Value, OutputListStorage) -> RunSuiteFuture;
pub type RunTestsService =
    dyn Fn(Option<RuntimeContext>, crate::call::BuiltinArguments) -> RunSuiteFuture;
pub type DiscoverTestsService =
    dyn Fn(Option<RuntimeContext>, crate::call::BuiltinArguments) -> RunSuiteFuture;

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
        run_suite: impl Fn(Option<RuntimeContext>, Value, OutputListStorage) -> RunSuiteFuture + 'static,
        run_tests: impl Fn(Option<RuntimeContext>, crate::call::BuiltinArguments) -> RunSuiteFuture
            + 'static,
        discover_tests: impl Fn(Option<RuntimeContext>, crate::call::BuiltinArguments) -> RunSuiteFuture
            + 'static,
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

/// The service future owns suite and plugin backing. Field order retires it
/// before unregistering the input GC roots on completion or cancellation.
struct SuiteServiceOwner {
    future: RunSuiteFuture,
    _inputs_root: RootedValueLease,
}

/// If continuation admission fails before the service is constructed, retire
/// the input values and their backing before unregistering their GC roots.
struct SuiteInputs {
    suite: Value,
    plugins: OutputListStorage,
    root: RootedValueLease,
}

pub(crate) fn run_test_suite_in_context(
    runtime: Option<RuntimeContext>,
    suite: Value,
    plugins: OutputListStorage,
) -> BuiltinResult<rooted_continuation::RuntimeContinuationFuture<'static, RootedValue>> {
    let service = match runtime.as_ref() {
        Some(runtime) => runtime.state().test_services.borrow().last().cloned(),
        None => TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned()),
    };
    let Some(service) = service else {
        return Err(crate::build_runtime_error(
            "TestRunner.run requires an active Core test executor",
        )
        .with_identifier("RunMat:Testing:RequiresExecutor")
        .with_builtin("matlab.unittest.TestRunner.run")
        .build());
    };
    let root = match runtime.as_ref() {
        Some(runtime) => RootedValueLease::for_value_slices_in_context(
            &[std::slice::from_ref(&suite), plugins.as_slice()],
            runtime,
        )?,
        None => RootedValueLease::for_values(std::iter::once(&suite).chain(plugins.iter()))
            .map_err(|error| error.into_runtime_error())?,
    };
    let inputs = SuiteInputs {
        suite,
        plugins,
        root,
    };
    Ok(rooted_continuation::admit_rooted_future_with(
        runtime,
        "matlab.unittest.TestRunner.run",
        crate::builtin::builtin_control_error,
        move |caller| {
            let SuiteInputs {
                suite,
                plugins,
                root,
            } = inputs;
            let owner = SuiteServiceOwner {
                future: (service.run_suite)(caller, suite, plugins),
                _inputs_root: root,
            };
            async move {
                // Capture the complete owner, not just its future field: Rust
                // 2021 disjoint capture would otherwise drop the input lease
                // before this continuation is first polled.
                let mut owner = owner;
                let value = owner.future.as_mut().await?;
                let rooted = RootedValue::new(value).map_err(|error| error.into_runtime_error())?;
                drop(owner);
                Ok(rooted)
            }
        },
    ))
}

pub async fn run_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let caller = crate::context::legacy::active();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, caller.as_ref())?;
    run_tests_in_context(caller, args).await
}

pub async fn run_tests_in_context(
    caller: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let service = match caller.as_ref() {
        Some(runtime) => runtime.state().test_services.borrow().last().cloned(),
        None => TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned()),
    };
    let Some(service) = service else {
        return Err(
            crate::build_runtime_error("runtests requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("runtests")
                .build(),
        );
    };
    let future = (service.run_tests)(caller.clone(), args);
    match caller {
        Some(caller) => caller.scope(future).await,
        None => future.await,
    }
}

pub async fn discover_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let caller = crate::context::legacy::active();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, caller.as_ref())?;
    discover_tests_in_context(caller, args).await
}

pub async fn discover_tests_in_context(
    caller: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let service = match caller.as_ref() {
        Some(runtime) => runtime.state().test_services.borrow().last().cloned(),
        None => TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned()),
    };
    let Some(service) = service else {
        return Err(
            crate::build_runtime_error("testsuite requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("testsuite")
                .build(),
        );
    };
    let future = (service.discover_tests)(caller.clone(), args);
    match caller {
        Some(caller) => caller.scope(future).await,
        None => future.await,
    }
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
                        runmat_value::CellArray::new(args.into_iter().collect(), 1, count).unwrap(),
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
            futures::executor::block_on(
                run_test_suite_in_context(
                    None,
                    suite.clone(),
                    OutputListStorage::unbounded(Vec::new()),
                )
                .unwrap()
            )
            .unwrap()
            .value(),
            &suite
        );

        runmat_gc::gc_test_context(|| {
            let input = runmat_gc::gc_allocate_rooted(Value::Num(17.0)).unwrap();
            let target = input.handle();
            let queued = run_test_suite_in_context(
                None,
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "QueuedSuite".into(),
                    target,
                    valid: true,
                }),
                OutputListStorage::empty(),
            )
            .unwrap();
            drop(input);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());
            // The service has never been polled: its captured Value is not
            // itself a GC root. The admitted request must own that lease.
            drop(queued);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
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

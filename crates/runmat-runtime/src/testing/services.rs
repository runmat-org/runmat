use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_builtins::Value;

use crate::BuiltinResult;

pub type RunSuiteFuture = Pin<Box<dyn Future<Output = BuiltinResult<Value>>>>;
pub type RunSuiteService = dyn Fn(Value) -> RunSuiteFuture;
pub type RunTestsService = dyn Fn(Vec<Value>) -> RunSuiteFuture;
pub type DiscoverTestsService = dyn Fn(Vec<Value>) -> RunSuiteFuture;

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
        run_suite: impl Fn(Value) -> RunSuiteFuture + 'static,
        run_tests: impl Fn(Vec<Value>) -> RunSuiteFuture + 'static,
        discover_tests: impl Fn(Vec<Value>) -> RunSuiteFuture + 'static,
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

pub struct TestServicesGuard;

impl Drop for TestServicesGuard {
    fn drop(&mut self) {
        TEST_SERVICES_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            debug_assert!(popped.is_some());
        });
    }
}

pub fn install_test_services(services: RuntimeTestServices) -> TestServicesGuard {
    TEST_SERVICES_STACK.with(|stack| stack.borrow_mut().push(services));
    TestServicesGuard
}

pub async fn run_test_suite(suite: Value) -> BuiltinResult<Value> {
    let service = TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned());
    let Some(service) = service else {
        return Err(crate::build_runtime_error(
            "TestRunner.run requires an active Core test executor",
        )
        .with_identifier("RunMat:Testing:RequiresExecutor")
        .with_builtin("matlab.unittest.TestRunner.run")
        .build());
    };
    (service.run_suite)(suite).await
}

pub async fn run_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let service = TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned());
    let Some(service) = service else {
        return Err(
            crate::build_runtime_error("runtests requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("runtests")
                .build(),
        );
    };
    (service.run_tests)(args).await
}

pub async fn discover_tests(args: Vec<Value>) -> BuiltinResult<Value> {
    let service = TEST_SERVICES_STACK.with(|stack| stack.borrow().last().cloned());
    let Some(service) = service else {
        return Err(
            crate::build_runtime_error("testsuite requires an active Core test executor")
                .with_identifier("RunMat:Testing:RequiresExecutor")
                .with_builtin("testsuite")
                .build(),
        );
    };
    (service.discover_tests)(args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_installation_is_scoped_and_async_safe() {
        let _guard = install_test_services(RuntimeTestServices::new(
            |suite| Box::pin(async move { Ok(suite) }),
            |args| {
                Box::pin(async move {
                    let count = args.len();
                    Ok(Value::Cell(
                        runmat_builtins::CellArray::new(args, 1, count).unwrap(),
                    ))
                })
            },
            |args| {
                Box::pin(async move {
                    Ok(args
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| Value::OutputList(Vec::new())))
                })
            },
        ));
        let suite = Value::String("suite".into());
        assert_eq!(
            futures::executor::block_on(run_test_suite(suite.clone())).unwrap(),
            suite
        );
    }
}

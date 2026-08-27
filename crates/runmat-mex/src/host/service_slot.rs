use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::{
    MexAsyncHostServices, MexAsyncOperation, MexAsyncResult, MexBoundaryHostServices,
    MexDiagnostic, MexEngineCompletion, MxArray,
};

pub trait ConcurrentMexBoundaryHostServices:
    MexBoundaryHostServices + MexAsyncHostServices + Send + Sync
{
}

impl<T> ConcurrentMexBoundaryHostServices for T where
    T: MexBoundaryHostServices + MexAsyncHostServices + Send + Sync
{
}

thread_local! {
    static LOCAL_SERVICES: RefCell<BTreeMap<u64, Rc<dyn MexBoundaryHostServices>>> =
        RefCell::new(BTreeMap::new());
}

static NEXT_LOCAL_SERVICE: AtomicU64 = AtomicU64::new(1);

pub(crate) struct LocalBoundaryServiceGuard {
    key: u64,
}

impl LocalBoundaryServiceGuard {
    pub(super) fn register(
        services: Rc<dyn MexBoundaryHostServices>,
    ) -> (Self, BoundaryServiceSlot) {
        let key = NEXT_LOCAL_SERVICE.fetch_add(1, Ordering::Relaxed);
        LOCAL_SERVICES.with(|registry| {
            registry.borrow_mut().insert(key, services);
        });
        (Self { key }, BoundaryServiceSlot::Local(key))
    }
}

impl Drop for LocalBoundaryServiceGuard {
    fn drop(&mut self) {
        LOCAL_SERVICES.with(|registry| {
            registry.borrow_mut().remove(&self.key);
        });
    }
}

#[derive(Clone)]
pub(crate) enum BoundaryServiceSlot {
    Unavailable,
    Local(u64),
    #[cfg(not(target_family = "wasm"))]
    Concurrent(Arc<dyn ConcurrentMexBoundaryHostServices>),
}

impl BoundaryServiceSlot {
    #[cfg(not(target_family = "wasm"))]
    pub(super) fn concurrent(services: Arc<dyn ConcurrentMexBoundaryHostServices>) -> Self {
        Self::Concurrent(services)
    }

    pub(super) fn is_concurrent(&self) -> bool {
        #[cfg(not(target_family = "wasm"))]
        if matches!(self, Self::Concurrent(_)) {
            return true;
        }
        false
    }

    pub(super) fn is_available(&self) -> bool {
        !matches!(self, Self::Unavailable)
    }

    fn with<T>(
        &self,
        operation: impl FnOnce(&dyn MexBoundaryHostServices) -> Result<T, MexDiagnostic>,
    ) -> Result<T, MexDiagnostic> {
        match self {
            Self::Unavailable => Err(unavailable()),
            #[cfg(not(target_family = "wasm"))]
            Self::Concurrent(services) => operation(services.as_ref()),
            Self::Local(key) => LOCAL_SERVICES.with(|registry| {
                let registry = registry.borrow();
                let services = registry.get(key).ok_or_else(wrong_thread)?;
                operation(services.as_ref())
            }),
        }
    }
}

impl MexBoundaryHostServices for BoundaryServiceSlot {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        self.with(|services| services.eval(command))
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        self.with(|services| services.call(function, arguments, requested_outputs))
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.with(|services| services.get_variable(workspace, name))
    }

    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic> {
        self.with(|services| services.put_variable(workspace, name, value))
    }

    fn get_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
    ) -> Result<MxArray, MexDiagnostic> {
        self.with(|services| services.get_object_property(object, index, name))
    }

    fn set_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        self.with(|services| services.set_object_property(object, index, name, value))
    }
}

impl MexAsyncHostServices for BoundaryServiceSlot {
    fn submit(
        &self,
        operation: MexAsyncOperation,
    ) -> Result<Arc<dyn MexAsyncResult>, MexDiagnostic> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Concurrent(services) => services.submit(operation),
            Self::Local(_) => {
                let completion = self.with(|services| {
                    Ok(services.execute_async(operation, Arc::new(AtomicBool::new(false))))
                })?;
                Ok(Arc::new(CompletedAsyncResult(completion)))
            }
            Self::Unavailable => Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:AsyncHostUnavailable".into()),
                message: "asynchronous MEX engine operations require an active host".into(),
            }),
        }
    }
}

struct CompletedAsyncResult(MexEngineCompletion<Vec<MxArray>>);

impl MexAsyncResult for CompletedAsyncResult {
    fn cancel(&self, _allow_interrupt: bool) -> bool {
        false
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn wait(&self, _timeout: Option<Duration>) -> bool {
        true
    }

    fn result(&self) -> MexEngineCompletion<Vec<MxArray>> {
        self.0.clone()
    }
}

fn unavailable() -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:HostServiceUnavailable".into()),
        message: "MEX host callbacks require an active invocation".into(),
    }
}

fn wrong_thread() -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:ThreadAffinity".into()),
        message: "this synchronous MEX host service belongs to another thread".into(),
    }
}

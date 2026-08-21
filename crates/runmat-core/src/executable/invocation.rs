use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_value::Value;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ExecutableBackendPolicy {
    #[default]
    Established,
    ForcedGenericNative,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcedureTarget {
    Entrypoint,
    Function(String),
}

#[derive(Clone, Debug)]
pub struct ProcedureInvocation {
    pub target: ProcedureTarget,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

impl ProcedureInvocation {
    pub fn entrypoint() -> Self {
        Self {
            target: ProcedureTarget::Entrypoint,
            arguments: Vec::new(),
            requested_outputs: 0,
        }
    }

    pub fn function(name: impl Into<String>, arguments: Vec<Value>) -> Self {
        Self {
            target: ProcedureTarget::Function(name.into()),
            arguments,
            requested_outputs: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct InvocationControl {
    cancellation: Option<Arc<AtomicBool>>,
    deadline_unix_ms: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    backend: ExecutableBackendPolicy,
}

impl InvocationControl {
    pub fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    pub fn with_deadline_unix_ms(mut self, deadline_unix_ms: u64) -> Self {
        self.deadline_unix_ms = Some(deadline_unix_ms);
        self
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn force_generic_native(mut self) -> Self {
        self.backend = ExecutableBackendPolicy::ForcedGenericNative;
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn backend(&self) -> ExecutableBackendPolicy {
        self.backend
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    pub(crate) fn deadline_elapsed(&self) -> bool {
        self.deadline_unix_ms
            .is_some_and(|deadline| runmat_time::unix_timestamp_ms() >= deadline as u128)
    }
}

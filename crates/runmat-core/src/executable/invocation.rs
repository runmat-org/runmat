use runmat_execution::cancellation::CancellationToken;

use runmat_value::Value;

mod arguments;
pub use arguments::ProcedureArguments;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ExecutableBackendPolicy {
    #[default]
    Established,
    ForcedGenericNative,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProcedureTarget {
    Entrypoint,
    Function(String),
    /// Exact compiler identity; display names are not callable authority.
    SemanticFunction(runmat_types::ProgramFunctionId),
    /// Select an original callable only against this immutable unit's exact
    /// runtime definition owners. Arguments remain a separate ingress owner.
    RetainedCallable(runmat_runtime::execution::RootedValue),
}

#[derive(Clone, Debug)]
pub struct ProcedureInvocation {
    pub target: ProcedureTarget,
    pub arguments: ProcedureArguments,
    pub requested_outputs: usize,
}

impl ProcedureInvocation {
    pub fn entrypoint() -> Self {
        Self {
            target: ProcedureTarget::Entrypoint,
            arguments: Vec::new().into(),
            requested_outputs: 0,
        }
    }

    pub fn function(name: impl Into<String>, arguments: Vec<Value>) -> Self {
        Self {
            target: ProcedureTarget::Function(name.into()),
            arguments: arguments.into(),
            requested_outputs: 0,
        }
    }

    pub fn retained_function(
        name: impl Into<String>,
        arguments: runmat_runtime::call::BuiltinArguments,
    ) -> Self {
        Self {
            target: ProcedureTarget::Function(name.into()),
            arguments: ProcedureArguments::Retained(arguments),
            requested_outputs: 0,
        }
    }

    pub fn retained_callable(
        callback: runmat_runtime::execution::RootedValue,
        arguments: runmat_runtime::call::BuiltinArguments,
    ) -> Self {
        Self {
            target: ProcedureTarget::RetainedCallable(callback),
            arguments: ProcedureArguments::Retained(arguments),
            requested_outputs: 0,
        }
    }

    pub(crate) fn admit(
        self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<AdmittedProcedureInvocation, runmat_runtime::RuntimeError> {
        let arguments = self.arguments.admit(runtime)?;
        let arguments = if matches!(&self.target, ProcedureTarget::RetainedCallable(_)) {
            arguments
        } else {
            arguments.into_supplied_arguments(Some(runtime))?
        };
        Ok(AdmittedProcedureInvocation {
            target: self.target,
            arguments,
            requested_outputs: self.requested_outputs,
        })
    }
}

/// This live backend carrier has one root authority, never a second Vec view.
pub(crate) struct AdmittedProcedureInvocation {
    pub target: ProcedureTarget,
    pub arguments: runmat_runtime::call::BuiltinArguments,
    pub requested_outputs: usize,
}

#[derive(Clone, Debug, Default)]
pub struct InvocationControl {
    cancellation: Option<CancellationToken>,
    deadline_unix_ms: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    backend: ExecutableBackendPolicy,
}

impl InvocationControl {
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
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

    /// One live control token reaches both admission and running safepoints.
    /// A token's existing deadline can only be tightened by this invocation.
    pub(crate) fn cancellation(&self) -> Option<CancellationToken> {
        if self.cancellation.is_none() && self.deadline_unix_ms.is_none() {
            return None;
        }
        Some(
            self.cancellation
                .clone()
                .unwrap_or_default()
                .with_deadline(self.deadline_unix_ms),
        )
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

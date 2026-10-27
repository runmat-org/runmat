use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::value::{
    GraphClassDependency, GraphClassMethodDependency, GraphInvocationRoots, PayloadGraph,
    ValueLimits,
};

use crate::execution::value_codec::ValueCodecError;
#[path = "deferred_program/capture.rs"]
mod capture;
pub use capture::{DeferredSelectedCodeCapture, RestoredCallableGraph};
#[path = "deferred_program/reentry.rs"]
mod reentry;
pub use reentry::AdmittedCallableReentry;

/// Exact accepted-task result authority. The implementation retains admitted
/// selected products and source/definition leases, never a strong execution
/// context. A fetch supplies its explicit issuing RuntimeContext and receives
/// a freshly rooted result; no ambient or latest-source lookup is permitted.
pub trait GraphResultReceipt {
    /// Opaque local downcast seam for comparing two selected admissions after
    /// a durable idempotency retry returns the same JobId. Not wire authority.
    fn as_any(&self) -> &dyn std::any::Any;

    /// True only for the same issuing session and exact ordered selected
    /// product/source/definition owners. Equal V3 bytes under a fresh
    /// issuance are not interchangeable result receipts.
    fn same_selected_admission(&self, other: &dyn GraphResultReceipt) -> bool;

    /// Verify the explicit fetch context against the issuing session without
    /// consuming the result. Services call this again for cached graph values.
    fn check_issuing_context(
        &self,
        runtime: &crate::context::RuntimeContext,
    ) -> Result<(), super::ExecutionServiceError>;

    fn decode_in_context(
        &self,
        runtime: &crate::context::RuntimeContext,
        graph: &PayloadGraph,
    ) -> Result<crate::execution::RootedValueSequence, super::ExecutionServiceError>;

    /// Localize a structured V8 worker failure against this accepted task's
    /// exact admitted source receipt. The failure is borrowed so a foreign
    /// fetch cannot consume the issuing session's stored response.
    fn decode_failure_in_context(
        &self,
        runtime: &crate::context::RuntimeContext,
        failure: &runmat_execution::ProgramRuntimeFailure,
    ) -> Result<crate::RuntimeError, super::ExecutionServiceError>;
}

/// One graph and its exact admitted executable dependencies selected at spawn.
/// The worker must admit every V3 unit before reconstructing graph values.
pub struct DeferredGraphCapture {
    pub graph: PayloadGraph,
    /// Minimum selected-value schema proved by the captured code and roots.
    pub required_schema: runmat_execution::value::GraphRequiredSchema,
    pub roots: GraphInvocationRoots,
    pub primary_unit: Arc<[u8]>,
    /// Exact ordered compact policy refs from primary and retained issued
    /// receipts. `None` is a truly pure unit, not an archive-presence guess.
    pub selected_foreign_policies: Vec<Option<runmat_execution::ForeignPolicyRef>>,
    pub retained_units: Vec<Arc<[u8]>>,
    pub class_dependencies: Vec<GraphClassDependency>,
    pub class_method_dependencies: Vec<GraphClassMethodDependency>,
    pub static_dependencies: Vec<runmat_execution::value::GraphStaticDependency>,
    /// Ordered checked foreign archives for primary and retained V3 units.
    /// None is permitted only for a unit with no issued policy or interop.
    pub selected_bundle_archives: Vec<Option<Arc<[u8]>>>,
    pub receipt: Rc<dyn GraphResultReceipt>,
}

/// VM-issued, execution-port-free snapshot authority. Implementations keep a
/// weak selected-generation owner at Future creation; they inspect Values and
/// select their exact admitted origin/lease products only after spawn reserves
/// the task. No execution service or live RuntimeContext is retained.
pub trait DeferredGraphSnapshot {
    /// Revision of this immutable selected primary, not the spawning caller.
    /// Restored products without a retained revision report no observation.
    fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        None
    }

    fn select_retained_worker_callable(
        &self,
        _runtime: &crate::context::RuntimeContext,
        _selected: crate::call::descriptor::RepeatableCallable,
    ) -> Result<crate::execution::SelectedWorkerCallable, crate::RuntimeError> {
        Err(reentry::unavailable(
            "selected program cannot admit this retained worker callable",
        ))
    }

    fn capture_callables(
        &self,
        _selected: &[crate::call::descriptor::CallableDescriptor],
        _limits: ValueLimits,
        _workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<DeferredSelectedCodeCapture, ValueCodecError> {
        Err(ValueCodecError::unsupported(
            "$",
            "selected program cannot archive callable captures",
        ))
    }

    fn restore_callables(
        &self,
        _runtime: &crate::context::RuntimeContext,
        _code: runmat_execution::value::PortableSelectedCode,
    ) -> Result<RestoredCallableGraph, crate::RuntimeError> {
        Err(reentry::unavailable(
            "selected program cannot restore callable captures",
        ))
    }
    /// Only a restored admitted producer returns a reentry receipt. The
    /// ordinary selected program keeps its existing call route (None).
    fn retained_callable_reentry(
        &self,
        _runtime: &crate::context::RuntimeContext,
        _selected: &crate::call::descriptor::CallableDescriptor,
    ) -> Result<Option<AdmittedCallableReentry>, crate::RuntimeError> {
        Ok(None)
    }
    /// Re-enter through this producer's exact admitted graph. There is no
    /// callback-override setter and no saved Runtime or invoker closure.
    fn invoke_retained_reentry(
        self: Rc<Self>,
        _runtime: crate::context::RuntimeContext,
        _receipt: AdmittedCallableReentry,
        _selected: crate::call::descriptor::CallableDescriptor<crate::call::BuiltinArguments>,
    ) -> Result<crate::user_functions::BoundFunctionFuture, crate::RuntimeError> {
        Err(reentry::unavailable(
            "selected program has no retained graph invocation",
        ))
    }
    /// Select a local repeated callback against the same original immutable
    /// program owner, without acquiring a pool, checking worker eligibility
    /// or constructing a worker archive. This is NOT graph restore authority.
    fn select_repeatable_callable(
        &self,
        _runtime: &crate::context::RuntimeContext,
        _target: crate::execution::RootedValue,
    ) -> Result<crate::call::descriptor::RepeatableCallable, crate::RuntimeError> {
        let mut error = crate::runtime_error::semantic_error(
            "RunMat:Execution:SelectedCallableUnavailable",
            "selected program has no local callback admission",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(error)
    }
    fn select_callable(
        &self,
        _runtime: &crate::context::RuntimeContext,
        _target: crate::execution::RootedValue,
    ) -> Result<crate::execution::SelectedWorkerCallable, crate::RuntimeError> {
        Err(crate::runtime_error::semantic_error(
            "RunMat:Execution:SelectedCallableUnavailable",
            "selected program has no worker callback admission",
        ))
    }

    fn snapshot(
        &self,
        invocation: &super::DeferredInvocation,
        limits: ValueLimits,
        workspace: Option<&super::super::value_codec::ValidationWorkspaceOwner>,
    ) -> Result<DeferredGraphCapture, ValueCodecError>;
}

#[derive(Clone)]
pub struct DeferredGraphProgram {
    snapshot: Rc<dyn DeferredGraphSnapshot>,
    workspace: Option<super::super::value_codec::ValidationWorkspaceOwner>,
}

impl DeferredGraphProgram {
    pub fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        self.snapshot.program_revision()
    }

    pub fn select_retained_worker_callable(
        &self,
        runtime: &crate::context::RuntimeContext,
        selected: crate::call::descriptor::RepeatableCallable,
    ) -> Result<crate::execution::SelectedWorkerCallable, crate::RuntimeError> {
        self.snapshot
            .select_retained_worker_callable(runtime, selected)
    }

    pub fn restore_compiled_builtin(
        runtime: &crate::context::RuntimeContext,
        builtin: runmat_types::BuiltinId,
    ) -> Result<crate::call::descriptor::RepeatableCallable, crate::RuntimeError> {
        crate::call::descriptor::RepeatableCallable::from_compiled_builtin(runtime, builtin)
    }

    pub fn capture_callables(
        &self,
        selected: &[crate::call::descriptor::CallableDescriptor],
        limits: ValueLimits,
    ) -> Result<DeferredSelectedCodeCapture, ValueCodecError> {
        self.snapshot
            .capture_callables(selected, limits, self.workspace.as_ref())
    }

    pub fn restore_callables(
        &self,
        runtime: &crate::context::RuntimeContext,
        code: runmat_execution::value::PortableSelectedCode,
    ) -> Result<RestoredCallableGraph, crate::RuntimeError> {
        self.snapshot.restore_callables(runtime, code)
    }
    pub fn new(snapshot: Rc<dyn DeferredGraphSnapshot>) -> Self {
        Self {
            snapshot,
            workspace: None,
        }
    }

    pub fn with_validation_workspace(
        mut self,
        owner: super::super::value_codec::ValidationWorkspaceOwner,
    ) -> Self {
        self.workspace = Some(owner);
        self
    }

    pub fn select_callable(
        &self,
        runtime: &crate::context::RuntimeContext,
        target: crate::execution::RootedValue,
    ) -> Result<crate::execution::SelectedWorkerCallable, crate::RuntimeError> {
        self.snapshot.select_callable(runtime, target)
    }
    pub fn select_repeatable_callable(
        &self,
        runtime: &crate::context::RuntimeContext,
        target: crate::execution::RootedValue,
    ) -> Result<crate::call::descriptor::RepeatableCallable, crate::RuntimeError> {
        let selected = self.snapshot.select_repeatable_callable(runtime, target)?;
        match self
            .snapshot
            .retained_callable_reentry(runtime, selected.selected_descriptor())?
        {
            Some(receipt) => {
                receipt.validate(runtime, &selected.selected_descriptor().target)?;
                Ok(selected.with_admitted_graph_reentry(self.clone(), receipt))
            }
            None => Ok(selected),
        }
    }
    pub(crate) fn invoke_retained_reentry(
        &self,
        runtime: &crate::context::RuntimeContext,
        receipt: &AdmittedCallableReentry,
        selected: crate::call::descriptor::CallableDescriptor<crate::call::BuiltinArguments>,
    ) -> Result<crate::user_functions::BoundFunctionFuture, crate::RuntimeError> {
        receipt.validate(runtime, &selected.target)?;
        let runtime = runtime
            .clone()
            .with_cancellation_token(runtime.cancellation());
        Rc::clone(&self.snapshot).invoke_retained_reentry(runtime, receipt.clone(), selected)
    }

    pub fn snapshot(
        &self,
        invocation: &super::DeferredInvocation,
        limits: ValueLimits,
    ) -> Result<DeferredGraphCapture, ValueCodecError> {
        self.snapshot
            .snapshot(invocation, limits, self.workspace.as_ref())
    }
}

impl std::fmt::Debug for DeferredGraphProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DeferredGraphProgram { selected: true, .. }")
    }
}

impl PartialEq for DeferredGraphProgram {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.snapshot, &other.snapshot)
            && match (&self.workspace, &other.workspace) {
                (Some(a), Some(b)) => a.same_owner(b),
                (None, None) => true,
                _ => false,
            }
    }
}

/// One selected program authority for a deferred call. A graph route cannot
/// accidentally fall back to a raw bytecode or region capture.
#[derive(Clone, Debug, PartialEq)]
pub enum DeferredProgram {
    Local,
    InterpreterV2(Arc<[u8]>),
    RegionSourcesV1(Arc<[u8]>),
    SelectedGraph(DeferredGraphProgram),
}

impl DeferredProgram {
    /// Observational revision follows the actual immutable program owner.
    /// A selected graph must never inherit a different caller's revision.
    pub fn program_revision<'a>(
        &'a self,
        caller: Option<&'a runmat_execution::ProgramRevision>,
    ) -> Option<&'a runmat_execution::ProgramRevision> {
        match self {
            Self::SelectedGraph(selected) => selected.program_revision(),
            Self::Local | Self::InterpreterV2(_) | Self::RegionSourcesV1(_) => caller,
        }
    }
}

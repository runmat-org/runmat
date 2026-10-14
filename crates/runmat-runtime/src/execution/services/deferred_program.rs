use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::value::{
    GraphClassDependency, GraphClassMethodDependency, GraphInvocationRoots, PayloadGraph,
    ValueLimits,
};

use crate::execution::value_codec::ValueCodecError;

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
    fn snapshot(
        &self,
        invocation: &super::DeferredInvocation,
        limits: ValueLimits,
    ) -> Result<DeferredGraphCapture, ValueCodecError>;
}

#[derive(Clone)]
pub struct DeferredGraphProgram(Rc<dyn DeferredGraphSnapshot>);

impl DeferredGraphProgram {
    pub fn new(snapshot: Rc<dyn DeferredGraphSnapshot>) -> Self {
        Self(snapshot)
    }

    pub fn snapshot(
        &self,
        invocation: &super::DeferredInvocation,
        limits: ValueLimits,
    ) -> Result<DeferredGraphCapture, ValueCodecError> {
        self.0.snapshot(invocation, limits)
    }
}

impl std::fmt::Debug for DeferredGraphProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DeferredGraphProgram { selected: true, .. }")
    }
}

impl PartialEq for DeferredGraphProgram {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
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

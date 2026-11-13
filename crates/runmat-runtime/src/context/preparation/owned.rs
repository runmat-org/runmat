use crate::execution::RootedValueLease;
use crate::RuntimeError;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_value::Value;

/// A completed prepared body is not yet a completed builtin call. Only the
/// dispatcher may deliver this one-shot observation after compatibility and
/// workspace reconciliation. Dropping it abandons feedback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedCallStatus {
    Succeeded,
    CompletedFailure,
}

pub trait PreparedCallCompletion {
    fn binding(&self) -> BuiltinBindingIdentity;
    fn status(&self) -> PreparedCallStatus;
    fn publish(self: Box<Self>, outer_started: runmat_time::Instant, elapsed_ns: u64);
}

/// Backend-owned feedback and temporary ownership. Implementations must retain
/// borrowed input protection and adopt returned aliases before releasing temps.
/// No timing assumptions are imposed by the runtime.
pub trait BuiltinPreparationObservation {
    /// Actual selected physical body, not a metadata snapshot. Only domains
    /// which supplied its admitted instruction may consume this delivery.
    fn take_compound(&mut self) -> Option<super::AdmittedBuiltinCompound> {
        None
    }
    /// Transfer the authenticated exact numerical result owner.
    fn take_host_numerical_output(&mut self) -> Option<super::PreparedHostNumericalOutput> {
        None
    }
    /// A completed private preparation effect may forbid retry of the body.
    /// This can only strengthen to Never; it never requests a new replay.
    fn gpu_gather_retry_policy(&self) -> crate::GpuGatherRetry {
        crate::GpuGatherRetry::Legacy
    }
    /// Optional provider-owned operation scope. None deliberately masks an outer
    /// scope; opaque bodies must not borrow another candidate's completion proof.
    fn provider_recorder(
        &self,
    ) -> Option<runmat_accelerate_api::observation::ProviderOperationRecorder> {
        None
    }
    fn begin_execution(&mut self) {}
    fn finish(self: Box<Self>, result: Result<&Value, &RuntimeError>) -> Result<(), RuntimeError>;
    fn abort(self: Box<Self>);
}

/// This carrier deliberately has no Clone or into-values escape hatch: a pending
/// observation cannot accidentally outlive or detach from its execution scope.
/// Its operands are consumed only by private admitted runtime bodies, never by
/// an embedding caller's arbitrary callback that could publish temporary aliases.
pub struct PreparedBuiltinOperands {
    operands: Vec<Value>,
    observation: Option<Box<dyn BuiltinPreparationObservation>>,
    _operands_root: RootedValueLease,
}

impl PreparedBuiltinOperands {
    pub(super) fn take_compound(&mut self) -> Option<super::AdmittedBuiltinCompound> {
        self.observation
            .as_mut()
            .and_then(|observation| observation.take_compound())
    }
    pub(super) fn take_host_numerical_output(
        &mut self,
    ) -> Option<super::PreparedHostNumericalOutput> {
        self.observation
            .as_mut()
            .and_then(|observation| observation.take_host_numerical_output())
    }
    pub(super) fn forbids_gpu_gather_retry(&self) -> bool {
        self.observation.as_ref().is_some_and(|observation| {
            observation.gpu_gather_retry_policy() == crate::GpuGatherRetry::Never
        })
    }
    pub fn new(
        operands: Vec<Value>,
        observation: Option<Box<dyn BuiltinPreparationObservation>>,
    ) -> Result<Self, RuntimeError> {
        let no_replay = observation.as_ref().is_some_and(|observation| {
            observation.gpu_gather_retry_policy() == crate::GpuGatherRetry::Never
        });
        let operands_root = match RootedValueLease::for_values(operands.iter()) {
            Ok(root) => root,
            Err(error) => {
                if let Some(observation) = observation {
                    observation.abort();
                }
                let mut error = error.into_runtime_error();
                if no_replay {
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                }
                return Err(error);
            }
        };
        Ok(Self {
            operands,
            observation,
            _operands_root: operands_root,
        })
    }

    pub fn passthrough(operands: Vec<Value>) -> Result<Self, RuntimeError> {
        Self::new(operands, None)
    }

    pub(crate) fn operands(&self) -> &[Value] {
        &self.operands
    }

    pub(super) fn take_operands(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.operands)
    }

    pub(super) fn begin_execution(&mut self) {
        if let Some(observation) = &mut self.observation {
            observation.begin_execution();
        }
    }

    pub(super) fn provider_recorder(
        &self,
    ) -> Option<runmat_accelerate_api::observation::ProviderOperationRecorder> {
        self.observation
            .as_ref()
            .and_then(|observation| observation.provider_recorder())
    }

    pub(super) fn finish(
        &mut self,
        result: Result<&Value, &RuntimeError>,
    ) -> Result<(), RuntimeError> {
        if let Some(observation) = self.observation.take() {
            observation.finish(result)?;
        }
        Ok(())
    }
}

impl Drop for PreparedBuiltinOperands {
    fn drop(&mut self) {
        if let Some(observation) = self.observation.take() {
            observation.abort();
        }
    }
}

use crate::RuntimeError;
use runmat_value::Value;

/// Backend-owned feedback and temporary ownership. Implementations must retain
/// borrowed input protection and adopt returned aliases before releasing temps.
/// No timing assumptions are imposed by the runtime.
pub trait BuiltinPreparationObservation {
    fn begin_execution(&mut self) {}
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>);
    fn abort(self: Box<Self>);
}

/// This carrier deliberately has no Clone or into-values escape hatch: a pending
/// observation cannot accidentally outlive or detach from its execution scope.
pub struct PreparedBuiltinOperands {
    operands: Vec<Value>,
    observation: Option<Box<dyn BuiltinPreparationObservation>>,
}

impl PreparedBuiltinOperands {
    pub fn new(
        operands: Vec<Value>,
        observation: Option<Box<dyn BuiltinPreparationObservation>>,
    ) -> Self {
        Self {
            operands,
            observation,
        }
    }

    pub fn passthrough(operands: Vec<Value>) -> Self {
        Self::new(operands, None)
    }

    pub fn operands(&self) -> &[Value] {
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

    pub(super) fn finish(&mut self, result: &Result<Value, RuntimeError>) {
        if let Some(observation) = self.observation.take() {
            observation.finish(result);
        }
    }
}

impl Drop for PreparedBuiltinOperands {
    fn drop(&mut self) {
        if let Some(observation) = self.observation.take() {
            observation.abort();
        }
    }
}

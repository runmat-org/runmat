//! One selected function's execution authority and required capabilities.

#[derive(Clone, Debug)]
pub struct SelectedGraphFunction {
    pub binding: runmat_value::BoundFunctionBinding,
    pub capabilities: runmat_types::CapabilitySet,
    pub spawn_safety: runmat_types::SpawnSafetyFact,
}

pub type SelectedGraphFunctionResolver = dyn Fn(
    runmat_types::FunctionId,
    &runmat_types::CallableIdentity,
) -> Option<SelectedGraphFunction>;

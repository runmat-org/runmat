//! Provider-local physical POWER service. Language parsing and host fallback
//! belong to callers; this service never replays numerical work on the host.
mod execute;
mod compound;
mod evaluation;
mod prepare;
mod probe;
mod projection;
mod scalar_projection;
mod symbolic;
mod validate;

pub use compound::{PowerPairCompound, PowerPairExecuted, PowerPairExecutionEvidence};
pub use evaluation::{PowerEvaluationPhysicalProposal, PowerEvaluationRecipe};
pub use execute::{
    execute_resident_scalar_power, execute_resident_scalar_power_admitted,
    execute_resident_scalar_power_projected, execute_resident_scalar_power_projected_admitted,
    ExecutedResidentScalarPower,
};
pub use prepare::{
    prepare_resident_scalar_power, prepare_resident_scalar_power_against_plan, PowerPhysicalRecipe,
    PowerPhysicalRoute, PowerScalarConstruction, PreparedResidentScalarPower,
};
pub use probe::{
    PowerProbeDomain, PowerProbeObserved, PowerProbePhysicalProposal, PowerProbeRecipe,
};
pub use projection::{PowerPairPhysicalProjection, PowerPairProjection};
pub use scalar_projection::{
    PowerScalarExecutionEvidence, PowerTypedScalarProjection, PowerTypedScalarResourceBounds,
};
pub use symbolic::{
    exact_single_base_request, plan_resident_scalar_power, plan_scalar_power_for_dense_output,
    PowerSymbolicPhysicalWork, PowerSymbolicPlan,
};
pub use validate::PowerServiceError;

#[cfg(test)]
mod tests;

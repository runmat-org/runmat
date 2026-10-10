//! Provider-local physical POWER service. Language parsing and host fallback
//! belong to callers; this service never replays numerical work on the host.
mod execute;
mod prepare;
mod symbolic;
mod validate;

pub use execute::{execute_resident_scalar_power, ExecutedResidentScalarPower};
pub use prepare::{
    prepare_resident_scalar_power, prepare_resident_scalar_power_against_plan, PowerPhysicalRecipe,
    PowerPhysicalRoute, PowerScalarConstruction, PreparedResidentScalarPower,
};
pub use symbolic::{plan_resident_scalar_power, PowerSymbolicPhysicalWork, PowerSymbolicPlan};
pub use validate::PowerServiceError;

#[cfg(test)]
mod tests;

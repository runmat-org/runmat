//! Physical contact foundation. Surface selection does not implement enforcement or equilibrium.
mod surfaces;
pub use surfaces::{CanonicalContactSurfaces, ContactSurfaceError};

mod penalty;
pub use penalty::{
    ContactPenaltyPolicy, ContactPointError, ContactPointFrame, ContactPointMode,
    ContactPointResponse, ContactPointState, PenaltyContactLaw,
};

mod normal;
pub use normal::{
    NormalContactContribution, NormalContactError, NormalContactPoint, NormalContactTangentAction,
};

mod history;
pub use history::{
    ContactConfigurationPoint, ContactConfigurationState, ContactHistoryError,
    ContactMaterialPointHistory, ContactPointBinding, ContactPointTangentAction, ContactPointTrial,
    ContactPreparedPoint, ContactReferenceInventory,
};

mod support;
pub use support::{ContactSupportBudget, ContactSupportError, ContactTangentSupport};

mod operator;
pub use operator::{
    CanonicalContactOperator, ContactAssemblyDiagnostics, ContactAssemblyTrial,
    ContactInterfacePolicy, ContactOperatorError, ContactOperatorPolicy, ContactOperatorResources,
    ContactPreparationDemand,
};

#[cfg(test)]
pub(crate) use operator::tests::Fixture as ContactOperatorTestFixture;

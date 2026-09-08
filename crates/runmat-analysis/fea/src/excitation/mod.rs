//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod model;
mod realize;

#[cfg(test)]
pub(crate) use model::ComplexScalar;
pub(crate) use model::{
    ComplexVector3, RealizedExcitation, RealizedExcitationKind, RealizedExcitationTarget,
    RealizedExcitations,
};
pub(crate) use realize::realize_model_excitations;

#[cfg(test)]
mod tests;

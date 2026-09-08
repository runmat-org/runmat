//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod action;
mod activation;
mod model;
mod realize;
mod sample;
mod target;

#[cfg(test)]
pub(crate) use model::{ComplexScalar, RealizedExcitation};
pub(crate) use model::{
    ComplexVector3, RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
};
pub(crate) use realize::realize_model_excitations;

#[cfg(test)]
mod tests;

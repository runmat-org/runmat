//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod model;
mod realize;

pub(crate) use model::{
    ComplexVector3, RealizedExcitation, RealizedExcitationKind, RealizedExcitations,
};
pub(crate) use realize::realize_model_excitations;

#[cfg(test)]
mod tests;

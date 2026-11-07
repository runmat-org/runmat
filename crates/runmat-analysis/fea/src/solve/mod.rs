pub mod accelerator;
mod affine;
pub(crate) use affine::{
    affine_recovery_work, affine_scatter_work, affine_transpose_work, expand_linear as affine_expand_linear,
    solve_affine_constraint_system_with_kind_owned, AffineOwnedSolveOptions,
};
pub mod backend;
pub mod buckling;
pub mod certificate;
pub(crate) mod convergence;
pub mod dynamics;
pub mod fixed_point;
pub mod general;
pub(crate) mod krylov;
pub mod linear;
pub mod modal;
pub mod nonlinear;
pub mod preconditioner;
pub(crate) mod sparse_direct;
mod reaction;
#[cfg(test)]
pub(crate) use reaction::{recover_structural_reaction, recover_structural_reaction_scoped};
pub(crate) mod shared_sparse;
pub use shared_sparse::with_solver_plan;
mod static_balance;
pub use static_balance::StructuralStaticBalanceEvidence;
pub(crate) use static_balance::{
    recover_structural_static_balance, recover_structural_static_balance_with_capacity,
    StructuralStaticBalance,
};
pub mod transient;
pub mod uniqueness;

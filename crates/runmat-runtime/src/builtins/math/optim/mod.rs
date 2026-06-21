//! Optimization and nonlinear equation solving builtins.

pub(crate) mod brent;
pub(crate) mod common;
pub mod fminbnd;
pub mod fsolve;
pub mod fzero;
pub mod integral;
pub(crate) mod least_squares;
pub mod linprog;
pub mod lsqcurvefit;
pub mod optimoptions;
pub mod optimset;
pub mod quad;
pub(crate) mod type_resolvers;

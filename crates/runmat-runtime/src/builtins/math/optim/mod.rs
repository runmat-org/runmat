//! Optimization and nonlinear equation solving builtins.

pub(crate) mod brent;
pub(crate) mod common;
pub mod fminbnd;
pub mod fsolve;
pub mod fzero;
pub mod integral;
pub mod optimoptions;
pub mod optimset;
pub(crate) mod type_resolvers;

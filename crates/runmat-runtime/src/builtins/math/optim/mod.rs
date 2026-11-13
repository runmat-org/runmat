//! Optimization and nonlinear equation solving builtins.

pub(crate) mod brent;
pub(crate) mod common;
pub mod coneprog;
pub mod fgoalattain;
pub mod fminbnd;
pub mod fmincon;
pub mod fminimax;
pub mod fminsearch;
pub mod fminunc;
pub mod fseminf;
pub mod fsolve;
pub mod fzero;
pub mod ga;
pub mod integer_constraint;
pub mod integral;
pub mod intlinprog;
pub(crate) mod least_squares;
pub mod linprog;
pub mod lsqcurvefit;
pub mod lsqlin;
pub mod lsqnonlin;
pub mod lsqnonneg;
pub mod milp_observers;
pub mod optimget;
pub(crate) mod optimmodel;
pub mod optimoptions;
pub mod optimset;
pub mod optimvar;
pub mod optimwarmstart;
pub mod particleswarm;
pub mod patternsearch;
pub mod quad;
pub mod quadprog;
pub(crate) mod quadrature;
#[cfg(test)]
mod scalar_provider_tests;
pub mod simulannealbnd;
pub(crate) mod surrogateopt;
#[cfg(test)]
pub(crate) mod type_resolvers;

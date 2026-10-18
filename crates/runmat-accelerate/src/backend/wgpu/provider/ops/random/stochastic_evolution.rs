//! Deterministic, request-bound stochastic-evolution step execution.
//!
//! Runtime owns random-stream order and supplies each noise tile. This module
//! only executes the already-certified real-floating arithmetic.

#[path = "stochastic_evolution/execution.rs"]
mod execution;
#[path = "stochastic_evolution/feasibility.rs"]
mod feasibility;
#[path = "stochastic_evolution/submission.rs"]
mod submission;

#[cfg(test)]
#[path = "stochastic_evolution/tests.rs"]
mod tests;

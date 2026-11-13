//! Typed search selection, bounded random proposals and original evaluation.
mod latin;
mod custom;
mod merit;
mod nelder;
mod genetic;
pub(super) use genetic::run as run_genetic;
pub(super) mod rbf;
mod run;
mod spec;
mod window;
mod original_bounds;
pub(super) use original_bounds::Bounds;
#[cfg(test)]
mod tests;
pub(super) use custom::run as run_custom;
pub(super) use nelder::run as run_nelder;
pub(super) use run::run;
pub(super) use spec::metadata;
pub(super) use spec::syntax;
pub(super) use spec::{parse, prepare, Search};
pub(super) use window::Window;

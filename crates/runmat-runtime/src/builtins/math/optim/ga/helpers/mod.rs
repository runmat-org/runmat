//! Public predefined functions are real registered callables. Their selected
//! function-handle identities use normal semantic invocation, not name casts.
mod admission;
pub(crate) mod selection;
pub(crate) mod scaling;
pub(crate) mod creation;
pub(crate) mod crossover;
mod mutation_controls;
pub(crate) mod mutation;
pub(crate) mod plot;
#[cfg(feature = "plot-core")]
mod plot_state;
#[cfg(feature = "plot-core")]
mod plot_data;
#[cfg(feature = "plot-core")]
mod plot_metrics;
#[cfg(feature = "plot-core")]
mod plot_stopping;

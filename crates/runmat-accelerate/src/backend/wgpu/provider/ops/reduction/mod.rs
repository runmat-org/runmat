use super::*;

mod autotune;
mod typed_recipe;
mod typed_selection;
mod core;
pub(crate) use core::{AnyStageObserved, FrozenAnyPipeline, FrozenAnySnapshot};
mod extrema_geometry;
#[cfg(test)]
mod extrema_tests;
mod float_nd_sum;
#[cfg(test)]
mod global_extrema_tests;
mod global_pass;
mod global_resources;
mod median;
mod nd;
mod mean_dispatch;
pub(super) mod phase_context;
mod typed_extrema;
mod typed_std;

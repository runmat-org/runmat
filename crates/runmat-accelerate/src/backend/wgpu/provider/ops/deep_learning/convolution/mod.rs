//! Provider-owned learned 1-D cross-correlation. The runtime supplies an
//! already-lowered request; this module authenticates and executes its
//! physical buffers without reinterpreting constructor options.

use super::*;

mod admission;
mod adjoint;
mod dispatch;
mod geometry;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod interval_tests;

use geometry::{next_chunk, ConvParams, Geometry};

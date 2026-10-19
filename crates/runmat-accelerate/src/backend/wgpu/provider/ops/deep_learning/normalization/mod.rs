use super::*;

mod admission;
mod adjoint;
mod dispatch;
mod geometry;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod adjoint_tests;

use geometry::{geometry, next_chunk, shape_count, NormParams};

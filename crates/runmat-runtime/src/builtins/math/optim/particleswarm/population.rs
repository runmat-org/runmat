//! Particle-domain scratch uses the shared typed original allocation boundary.
pub(super) use super::common::population::Population;
use super::ScalarCooperation;
use crate::BuiltinResult;
pub(super) fn capacity<T>(n: usize, cooperation: &ScalarCooperation) -> BuiltinResult<Vec<T>> {
    super::POLICY.capacity(n, cooperation.runtime())
}

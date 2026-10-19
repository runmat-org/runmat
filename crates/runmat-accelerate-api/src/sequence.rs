//! Exact-class real row-sequence request shared by sequence constructors.

use crate::ProviderPrecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderSequenceKind {
    Linear,
    Log10,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderSequenceRequest {
    pub kind: ProviderSequenceKind,
    pub precision: ProviderPrecision,
    /// Linear endpoints or logarithmic exponents, in that order.
    pub start: f64,
    pub stop: f64,
    pub count: usize,
    /// Exact output endpoints, already selected under the builtin's class rule.
    pub first: f64,
    pub last: f64,
}

impl ProviderSequenceRequest {
    pub fn linear(start: f64, stop: f64, count: usize, precision: ProviderPrecision) -> Self {
        Self {
            kind: ProviderSequenceKind::Linear,
            precision,
            start,
            stop,
            count,
            first: start,
            last: stop,
        }
    }

    pub fn log10(
        start: f64,
        stop: f64,
        count: usize,
        precision: ProviderPrecision,
        first: f64,
        last: f64,
    ) -> Self {
        Self {
            kind: ProviderSequenceKind::Log10,
            precision,
            start,
            stop,
            count,
            first,
            last,
        }
    }
}

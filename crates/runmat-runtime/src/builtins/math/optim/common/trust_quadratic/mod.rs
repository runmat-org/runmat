//! Shared cooperative quadratic trust-ball kernels. Numerical candidates do
//! not select an optimizer, input grammar, callback contract or stop policy.
use super::ScalarCooperation;
use crate::{BuiltinResult, RuntimeError};
use num_complex::Complex64;
pub(crate) mod bounds;
pub(crate) mod cg;
pub(crate) mod factorization;
pub(crate) mod residual;
pub(crate) mod model;
pub(crate) mod pattern;
pub(crate) enum Materialization {
    Unavailable,
    Complete,
    Undefined,
}

type ProductPreflight =
    fn(&runmat_value::Value, usize, Option<&crate::context::RuntimeContext>) -> BuiltinResult<()>;

#[derive(Clone, Copy)]
pub(crate) struct ValueSourcePolicy {
    pub numeric: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub resident: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub product_preflight: Option<ProductPreflight>,
}
#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identity: runmat_builtins::BuiltinCatalogIdentity,
    pub value_source: Option<ValueSourcePolicy>,
    pub ownership_identifier: &'static str,
    pub invalid: fn(String) -> RuntimeError,
    pub allocation: fn(runmat_value::ExactHostNumericAllocationError) -> RuntimeError,
}
impl Policy {
    fn value_source(self) -> BuiltinResult<ValueSourcePolicy> {
        self.value_source.ok_or_else(|| {
            self.invalid("this numerical operator has no Value/callback ingress authority")
        })
    }
    fn invalid(self, detail: impl Into<String>) -> RuntimeError {
        (self.invalid)(detail.into())
    }
    fn buffer<T>(self, count: usize) -> BuiltinResult<Vec<T>> {
        runmat_value::exact_capacity_vec(count).map_err(self.allocation)
    }
    async fn zeros(
        self,
        count: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut result = self.buffer(count)?;
        while result.len() < count {
            let end = result.len().saturating_add(1024).min(count);
            let units = end - result.len();
            result.resize(end, 0.0);
            cooperation.completed(units as u64).await?;
        }
        Ok(result)
    }
    async fn filled<T: Clone>(
        self,
        count: usize,
        value: T,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<T>> {
        let mut result = self.buffer(count)?;
        while result.len() < count {
            let end = result.len().saturating_add(1024).min(count);
            let units = end - result.len();
            result.resize(end, value.clone());
            cooperation.completed(units as u64).await?;
        }
        Ok(result)
    }
    fn admission(self, mut error: RuntimeError) -> RuntimeError {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    }
}

/// Borrowed numerical product session: the caller retains the actual model
/// generation and original callback/Hinfo authority throughout every await.
pub(crate) trait Operator {
    fn policy(&self) -> Policy;
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>>;
    async fn copy_band(
        &self,
        _n: usize,
        _bandwidth: usize,
        _output: &mut [f64],
        _cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Materialization> {
        Ok(Materialization::Unavailable)
    }
    async fn copy_matrix(
        &mut self,
        _n: usize,
        _output: &mut Vec<Complex64>,
        _cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Materialization> {
        Ok(Materialization::Unavailable)
    }
    /// Equality-projected consumers keep PCG search vectors in the admissible
    /// subspace after preconditioning. Unconstrained consumers are identity.
    async fn project(
        &mut self,
        _values: &mut [f64],
        _cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        Ok(Some(()))
    }
}
fn norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |length, &value| length.hypot(value))
}
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

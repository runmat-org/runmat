//! Borrowed measurement and observation seams for one owned simplex.
use super::*;
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub identity: BuiltinCatalogIdentity,
    pub invalid: &'static BuiltinErrorDescriptor,
    pub allocation_identifier: &'static str,
    pub retry: crate::GpuGatherRetry,
}
impl Policy {
    pub fn invalid(self, detail: impl std::fmt::Display) -> RuntimeError {
        let mut error = crate::build_runtime_error(format!("{}: {detail}", self.identity.name))
            .with_builtin(self.identity.name);
        if let Some(identifier) = self.invalid.identifier {
            error = error.with_identifier(identifier);
        }
        error.with_gpu_gather_retry(self.retry).build()
    }
    pub fn buffer<T>(self, count: usize) -> BuiltinResult<Vec<T>> {
        runmat_value::exact_capacity_vec(count).map_err(|source| {
            crate::build_runtime_error(format!(
                "{}: workspace geometry or allocation failed",
                self.identity.name
            ))
            .with_builtin(self.identity.name)
            .with_identifier(self.allocation_identifier)
            .with_gpu_gather_retry(self.retry)
            .with_source(source)
            .build()
        })
    }
    pub fn sort_failure(self, source: std::collections::TryReserveError) -> RuntimeError {
        crate::build_runtime_error(format!(
            "{}: simplex sort allocation failed",
            self.identity.name
        ))
        .with_builtin(self.identity.name)
        .with_identifier(self.allocation_identifier)
        .with_gpu_gather_retry(self.retry)
        .with_source(source)
        .build()
    }
}
pub(in crate::builtins::math::optim) struct Point {
    pub x: Vec<f64>,
    pub f: f64,
}
pub(in crate::builtins::math::optim) enum Initial {
    Evaluate(Vec<f64>),
    Known(Point),
}
pub(in crate::builtins::math::optim) struct Controls {
    pub tol_x: f64,
    pub tol_fun: f64,
    pub max_iter: usize,
}
pub(in crate::builtins::math::optim) enum Measurement {
    Defined(f64),
    Undefined,
    Stopped(OracleStop),
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum OracleStop {
    Evaluations,
    Time,
}
pub(in crate::builtins::math::optim) trait Oracle {
    fn cooperation(&mut self) -> &mut ScalarCooperation;
    fn calls(&self) -> usize;
    fn remaining(&self) -> usize;
    async fn measure(&mut self, x: &[f64]) -> BuiltinResult<Measurement>;
}
pub(in crate::builtins::math::optim) trait Events {
    async fn notify(
        &mut self,
        phase: OutputState,
        point: &Point,
        iterations: usize,
        calls: usize,
        procedure: Procedure,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Procedure {
    Initial,
    Simplex,
    Reflect,
    Expand,
    ContractOutside,
    ContractInside,
    Shrink,
}
impl Procedure {
    pub fn name(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Simplex => "initial simplex",
            Self::Reflect => "reflect",
            Self::Expand => "expand",
            Self::ContractOutside => "contract outside",
            Self::ContractInside => "contract inside",
            Self::Shrink => "shrink",
        }
    }
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Reason {
    Converged,
    Iterations,
    Evaluations,
    Time,
    Observer,
}
impl Reason {
    pub fn flag(self) -> f64 {
        match self {
            Self::Converged => 1.0,
            Self::Observer => -1.0,
            _ => 0.0,
        }
    }
}
pub(in crate::builtins::math::optim) struct Outcome {
    pub point: Point,
    pub reason: Reason,
    pub iterations: usize,
    pub func_count: usize,
}

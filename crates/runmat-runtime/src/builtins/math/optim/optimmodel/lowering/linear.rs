//! One geometry owner for proposal, exact forward and reverse row routing.
use super::operations::Operation;
use super::*;
use crate::builtins::array::shape::diag;
use crate::builtins::math::{
    linalg::ops::trace,
    reduction::{cumsum, mean, sum},
};
pub(super) enum Linear {
    Sum(sum::ReductionGeometry),
    Mean(mean::AffineGeometry),
    Trace(trace::TraceGeometry),
    Diagonal(diag::DiagonalGeometry),
    Scan(cumsum::ScanGeometry),
}
pub(super) enum Rows {
    One(Option<usize>),
    Scan(cumsum::ScanRows),
}
impl Iterator for Rows {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::One(value) => value.take(),
            Self::Scan(value) => value.next(),
        }
    }
}
impl Linear {
    pub(super) async fn prepare(
        operation: Operation,
        shape: &[usize],
        arguments: &[Value],
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        c.completed(0).await?;
        let runtime = c.runtime().cloned();
        crate::context::RuntimeScopeFactory::new(runtime, move || async move {
            let arguments = crate::call::FunctionArgumentView::from_slice(arguments);
            let result = match operation {
                Operation::Sum => {
                    Self::Sum(sum::reduction_geometry(shape, arguments, c.runtime()).await?)
                }
                Operation::Mean => {
                    let (runtime, budget) = c.runtime_budget();
                    Self::Mean(mean::affine_geometry(shape, arguments, runtime, budget).await?)
                }
                Operation::Trace => {
                    if !arguments.is_empty() {
                        return Err(policy.invalid("trace source has trailing arguments"));
                    }
                    Self::Trace(trace::trace_geometry(shape)?)
                }
                Operation::Diagonal => {
                    let Some(geometry) =
                        diag::diagonal_geometry(shape, arguments, c.runtime()).await?
                    else {
                        return Ok(None);
                    };
                    Self::Diagonal(geometry)
                }
                Operation::Cumsum => Self::Scan(cumsum::scan_geometry(
                    runmat_builtins::CumulativeKind::Sum,
                    shape,
                    arguments,
                )?),
                _ => return Ok(None),
            };
            Ok(Some(result))
        })
        .construct()
        .await
    }
    pub(super) fn outputs(&mut self, source: usize, policy: Policy) -> BuiltinResult<Rows> {
        Ok(match self {
            Self::Sum(value) => Rows::One(value.project(source)),
            Self::Mean(value) => Rows::One(value.reduction.project(source)),
            Self::Trace(value) => Rows::One(value.project(source)),
            Self::Diagonal(value) => Rows::One(value.project(source)),
            Self::Scan(value) => Rows::Scan(
                value
                    .outputs(source)
                    .ok_or_else(|| policy.invalid("scan source row exceeds admitted geometry"))?,
            ),
        })
    }
    pub(super) fn weight(&self, source: usize) -> f64 {
        match self {
            Self::Mean(value) => value.weight(source),
            _ => 1.,
        }
    }
}

mod normalization;
pub(super) use normalization::Normalization;

impl Linear {
    pub(super) fn uniform_result(&self, value: f64) -> Option<f64> {
        match self {
            Self::Sum(plan) => Some(value * plan.factor() as f64),
            Self::Mean(_) => Some(value),
            Self::Trace(plan) => Some(value * plan.factor() as f64),
            Self::Diagonal(plan) if plan.copies_uniform() => Some(value),
            _ => None,
        }
    }
}

impl Linear {
    pub(super) fn validate(
        &self,
        input: &[usize],
        output: &[usize],
        policy: Policy,
    ) -> BuiltinResult<()> {
        let diagonal;
        let actual = match self {
            Self::Sum(value) => value.output_shape(),
            Self::Mean(value) => value.reduction.output_shape(),
            Self::Trace(_) => &[1, 1],
            Self::Diagonal(value) => {
                diagonal = value.output_shape();
                &diagonal
            }
            Self::Scan(_) => input,
        };
        if crate::builtins::common::shape::array_shapes_equal(actual, output) {
            Ok(())
        } else {
            Err(policy.invalid("linear geometry differs from the original expression node"))
        }
    }
}

mod exact;
pub(super) use exact::denominators as exact_denominators;

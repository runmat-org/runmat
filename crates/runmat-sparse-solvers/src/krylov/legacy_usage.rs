//! Admitted reference kernels for the synchronous MINRES and BiCGStab recurrences.

use super::{
    usage::{ArithmeticDomain, ScalarWork, VectorWork},
    KrylovOps,
};
use crate::{error::SolverResult, par, scalar::Scalar};

fn domain<T: Scalar>() -> ArithmeticDomain {
    if T::IS_COMPLEX {
        ArithmeticDomain::Complex
    } else {
        ArithmeticDomain::Real
    }
}

pub(super) async fn scalar<T: Scalar, O: KrylovOps<T>, R>(
    ops: &mut O,
    work: ScalarWork,
    run: impl FnOnce() -> R,
) -> SolverResult<R> {
    ops.measured(VectorWork::Scalar(work), run).await
}

pub(super) fn arithmetic_domain<T: Scalar>() -> ArithmeticDomain {
    domain::<T>()
}

pub(super) async fn magnitude<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    value: T,
) -> SolverResult<f64> {
    scalar::<T, _, _>(
        ops,
        if T::IS_COMPLEX && value.im() != 0.0 {
            ScalarWork::Hypot
        } else {
            ScalarWork::Zero
        },
        || {
            if value.im() == 0.0 {
                value.re().abs()
            } else {
                value.abs()
            }
        },
    )
    .await
}

pub(super) async fn dot<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    left: &[T],
    right: &[T],
) -> SolverResult<T> {
    ops.measured(
        VectorWork::Dot {
            elements: left.len(),
            domain: domain::<T>(),
        },
        || par::dot(left, right),
    )
    .await
}

pub(super) async fn norm<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    values: &[T],
) -> SolverResult<f64> {
    ops.measured(
        VectorWork::Norm {
            elements: values.len(),
            domain: domain::<T>(),
        },
        || par::norm2(values),
    )
    .await
}

pub(super) async fn sub<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    left: &[T],
    right: &[T],
    target: &mut [T],
) -> SolverResult<()> {
    ops.measured(
        VectorWork::Residual {
            elements: target.len(),
            domain: domain::<T>(),
        },
        || par::sub_into(left, right, target),
    )
    .await
}

pub(super) async fn axpy<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    scale: T,
    source: &[T],
    target: &mut [T],
) -> SolverResult<()> {
    ops.measured(
        VectorWork::Axpy {
            elements: target.len(),
            domain: domain::<T>(),
        },
        || par::axpy(scale, source, target),
    )
    .await
}

pub(super) async fn real_axpy<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    scale: f64,
    source: &[T],
    target: &mut [T],
) -> SolverResult<()> {
    ops.measured(
        VectorWork::RealAxpy {
            elements: target.len(),
            domain: domain::<T>(),
        },
        || par::axpy(T::from_f64(scale), source, target),
    )
    .await
}

pub(super) async fn xpby<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    source: &[T],
    scale: T,
    target: &mut [T],
) -> SolverResult<()> {
    ops.measured(
        VectorWork::Axpy {
            elements: target.len(),
            domain: domain::<T>(),
        },
        || par::xpby(source, scale, target),
    )
    .await
}

pub(super) async fn real_scale<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    source: &[T],
    scale: f64,
    target: &mut [T],
) -> SolverResult<()> {
    ops.measured(
        VectorWork::RealScale {
            elements: target.len(),
            domain: domain::<T>(),
        },
        || {
            for (target, source) in target.iter_mut().zip(source) {
                *target = source.scale(scale);
            }
        },
    )
    .await
}

pub(super) struct MinresDirectionCoefficients {
    pub(super) oldeps: f64,
    pub(super) delta: f64,
    pub(super) denom: f64,
}

pub(super) async fn minres_direction<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    v: &[T],
    w1: &[T],
    w2: &[T],
    coefficients: MinresDirectionCoefficients,
    w: &mut [T],
) -> SolverResult<()> {
    let MinresDirectionCoefficients {
        oldeps,
        delta,
        denom,
    } = coefficients;
    ops.measured(
        VectorWork::MinresDirection {
            elements: w.len(),
            domain: domain::<T>(),
        },
        || {
            for i in 0..w.len() {
                w[i] = (v[i] - w1[i].scale(oldeps) - w2[i].scale(delta)).scale(denom);
            }
        },
    )
    .await
}

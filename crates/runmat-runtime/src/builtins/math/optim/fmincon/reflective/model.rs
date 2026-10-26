//! Original objective Hessian/Hinfo generation and its canonical product policy.
use super::super::{invalid, numeric, IDENTITY};
use crate::builtins::math::optim::common::{trust_quadratic, ScalarCooperation};
use crate::execution::RootedValue;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::Value;
pub(super) use trust_quadratic::model::{Model, Products};

fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    crate::build_runtime_error("fmincon: reflective model allocation failed")
        .with_builtin("fmincon")
        .with_identifier("RunMat:fmincon:AllocationFailed")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
fn numeric_policy() -> numeric::Policy {
    numeric::Policy {
        owner: IDENTITY,
        invalid: None,
        numeric: Some(&runmat_builtins::FMINCON_NUMERIC_EXTENSION),
        single: Some(&runmat_builtins::FMINCON_SINGLE_EXTENSION),
        resident: Some(&runmat_builtins::FMINCON_RESIDENT_EXTENSION),
        sparse: None,
    }
}
fn product_preflight(
    value: &Value,
    n: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if numeric::preflight(
        value,
        "Hessian product",
        numeric::Geometry::Matrix(n, 1),
        &numeric_policy(),
        runtime,
    )? {
        Ok(())
    } else {
        Err(invalid(
            "HessianMultiplyFcn must return a finite real n-by-1 product",
        ))
    }
}
pub(super) fn policy() -> trust_quadratic::Policy {
    trust_quadratic::Policy {
        identity: IDENTITY,
        ownership_identifier: "RunMat:fmincon:HessianOwnership",
        value_source: Some(trust_quadratic::ValueSourcePolicy {
            numeric: &runmat_builtins::FMINCON_NUMERIC_EXTENSION,
            resident: &runmat_builtins::FMINCON_RESIDENT_EXTENSION,
            product_preflight: Some(product_preflight),
        }),
        invalid,
        allocation,
    }
}
pub(super) async fn matrix(
    value: RootedValue,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Model> {
    if !numeric::preflight(
        value.value(),
        "objective Hessian",
        numeric::Geometry::Matrix(n, n),
        &numeric_policy(),
        cooperation.runtime(),
    )? {
        return Err(invalid(
            "objective Hessian must be a real symmetric n-by-n matrix",
        ));
    }
    Model::matrix(policy(), value, n, cooperation).await
}

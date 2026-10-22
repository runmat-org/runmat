//! Managed sparse incomplete LU factorization.

use crate::builtins::math::linalg::ops::sparse_factor::{
    IncompleteLuKind, IncompleteLuOptions, IncompleteLuPart, ManagedIncompleteLu,
    ManagedIncompleteLuError, ManagedSparseScalar, ModifiedIncompleteLu, SparsePublication,
    SparsePublicationError,
};
use crate::context::{
    host_allocation::{HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, ILU_ERROR_INVALID_ARGUMENT, ILU_ERROR_INVALID_INPUT,
    ILU_ERROR_NUMERICAL, ILU_ERROR_RUNTIME,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ExactHostSparseClass, NumericDType, SparseTensor, Value};

const NAME: &str = "ilu";

mod output;
use output::evaluate;
#[cfg(test)]
use output::evaluate_typed;

mod options;
use options::{parse_options, Options};

#[runtime_builtin(
    name = "ilu",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::ilu"
)]
async fn ilu_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count().unwrap_or(1);
    if outputs > 3 {
        return Err(error(&ILU_ERROR_INVALID_ARGUMENT, None));
    }
    let Value::SparseTensor(sparse) = value else {
        return Err(error(&ILU_ERROR_INVALID_INPUT, None));
    };
    let options = parse_options(rest.view()).await?;
    let runtime = runtime
        .or_else(crate::context::legacy::active)
        .ok_or_else(|| error(&ILU_ERROR_RUNTIME, Some("ilu: runtime context unavailable")))?;
    let mut published = evaluate(sparse, options, outputs.max(1), &runtime).await?;
    if crate::output_count::current_output_count().is_none() {
        return Ok(published.values[0]
            .take()
            .expect("implicit ILU output is present"));
    }
    let retained = super::iterative::output::retained_list_bytes(outputs).map_err(|source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(&runtime),
            "RunMat:ilu:Runtime",
            "ilu: output list allocation",
            source,
        )
    })?;
    let receipt = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:ilu:Runtime",
                "ilu: output list allocation",
                source,
            )
        })?,
    )
    .map_err(|source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(&runtime),
            "RunMat:ilu:Runtime",
            "ilu: output list allocation",
            source,
        )
    })?;
    if let Some(receipt) = receipt {
        let permit = receipt.output_permit(&runtime).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:ilu:Runtime",
                "ilu: output list allocation",
                source,
            )
        })?;
        let values = [
            published.values[0].take(),
            published.values[1].take(),
            published.values[2].take(),
            None,
            None,
        ];
        return super::iterative::output::initialize_list(outputs, values, &permit).map_err(
            |source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(&runtime),
                    "RunMat:ilu:Runtime",
                    "ilu: output list allocation",
                    source.into(),
                )
            },
        );
    }
    Ok(Value::output_list(
        published.values[..outputs]
            .iter_mut()
            .map(|value| value.take().expect("selected ILU output is present"))
            .collect::<Vec<_>>(),
    ))
}

fn error(d: &'static BuiltinErrorDescriptor, detail: Option<&str>) -> RuntimeError {
    let mut b = build_runtime_error(detail.unwrap_or(d.message)).with_builtin(NAME);
    if let Some(id) = d.identifier {
        b = b.with_identifier(id);
    }
    b.build()
}

#[cfg(test)]
mod tests;

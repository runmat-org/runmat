//! Final GMRES result publication under the same claimed list/value owner
//! used by PCG. Scratch remains live until this result has been installed.

use runmat_value::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, ExactHostTensorAllocation,
    NumericDType, Tensor, Value,
};

use super::super::iterative::output::{initialize_list, retained_list_bytes};
use super::super::iterative::value::IterativeValue;
use super::recurrence::GmresOutcome;
use super::workspace::GmresGeometry;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

pub(super) fn quote<T: IterativeValue>(
    geometry: GmresGeometry,
    requested: Option<usize>,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let scratch = geometry.scratch_bytes(T::DTYPE, T::LANES)?;
    let mut output = if requested == Some(0) {
        0
    } else {
        T::output_bytes(geometry.n)?
    };
    if requested.is_some_and(|count| count >= 4) {
        output = output
            .checked_add(HostAllocationPlan::retained_tensor_bytes(
                NumericDType::F64,
                &[1, 2],
            )?)
            .ok_or(HostAllocationError::Overflow)?;
    }
    if requested.is_some_and(|count| count >= 5) {
        output = output
            .checked_add(HostAllocationPlan::retained_tensor_bytes_for_dimensions(
                T::DTYPE,
                geometry.history_elements,
                2,
            )?)
            .ok_or(HostAllocationError::Overflow)?;
    }
    if let Some(count @ 2..=5) = requested {
        output = output
            .checked_add(retained_list_bytes(count)?)
            .ok_or(HostAllocationError::Overflow)?;
    }
    HostAllocationPlan::new(scratch, output)
}

fn admitted_iteration(
    outcome: GmresOutcome,
    permit: &dyn ExactHostAllocationPermit,
) -> Result<Value, ExactHostNumericAllocationError> {
    let mut output = ExactHostTensorAllocation::new(NumericDType::F64, &[1, 2], permit)?;
    output
        .as_f64_slice_mut()
        .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?
        .copy_from_slice(&[outcome.outer as f64, outcome.inner as f64]);
    output.into_tensor().map(Value::Tensor)
}

fn unbounded_iteration(outcome: GmresOutcome) -> BuiltinResult<Value> {
    Tensor::new(vec![outcome.outer as f64, outcome.inner as f64], vec![1, 2])
        .map(Value::Tensor)
        .map_err(|_| invalid("unable to retain iteration pair"))
}

pub(super) async fn publish_admitted<T: IterativeValue>(
    receipt: &HostAllocationReceipt,
    context: &RuntimeContext,
    solution: &[T],
    history: &[f64],
    outcome: GmresOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    receipt
        .checkpoint(context)
        .map_err(|error| admission(context, error))?;
    if requested == Some(0) {
        report(outcome, requested)?;
        receipt
            .checkpoint(context)
            .map_err(|error| admission(context, error))?;
        return Ok(Value::output_list(Vec::new()));
    }
    let x =
        super::super::iterative::copy::admitted::<T>(receipt, context, solution, "gmres").await?;
    let residuals = if requested.is_some_and(|count| count >= 5) {
        Some(
            super::super::iterative::output::admitted_history(
                receipt,
                context,
                T::DTYPE,
                &history[..outcome.residual_count],
                "gmres",
            )
            .await?,
        )
    } else {
        None
    };
    let result = receipt
        .with_output_allocation(context, |permit| {
            let iteration = requested
                .is_some_and(|count| count >= 4)
                .then(|| admitted_iteration(outcome, permit))
                .transpose()?;
            match requested {
                None | Some(1) => Ok(x),
                Some(0) => Ok(Value::output_list(Vec::new())),
                Some(count @ 2..=5) => initialize_list(
                    count,
                    [
                        Some(x),
                        Some(Value::Num(f64::from(outcome.flag))),
                        Some(Value::Num(outcome.relres)),
                        iteration,
                        residuals,
                    ],
                    permit,
                ),
                _ => Err(ExactHostNumericAllocationError::UnsupportedClass),
            }
        })
        .map_err(|error| admission(context, error))?;
    report(outcome, requested)?;
    receipt
        .checkpoint(context)
        .map_err(|error| admission(context, error))?;
    Ok(result)
}

pub(super) async fn publish_unbounded<T: IterativeValue>(
    context: Option<&RuntimeContext>,
    solution: &[T],
    history: &[f64],
    outcome: GmresOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
    if requested == Some(0) {
        report(outcome, requested)?;
        super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
        return Ok(Value::output_list(Vec::new()));
    }
    let x = super::super::iterative::copy::unbounded::<T>(context, solution, "gmres").await?;
    let result = match requested {
        None | Some(1) => Ok(x),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(count @ 2..=5) => {
            let mut output = Vec::with_capacity(count);
            output.push(x);
            output.push(Value::Num(f64::from(outcome.flag)));
            if count >= 3 {
                output.push(Value::Num(outcome.relres));
            }
            if count >= 4 {
                output.push(unbounded_iteration(outcome)?);
            }
            if count >= 5 {
                output.push(
                    super::super::iterative::output::unbounded_history(
                        context,
                        T::DTYPE,
                        &history[..outcome.residual_count],
                        "gmres",
                    )
                    .await?,
                );
            }
            Ok(Value::output_list(output))
        }
        _ => Err(invalid("gmres has at most five outputs")),
    }?;
    super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
    report(outcome, requested)?;
    super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
    Ok(result)
}

fn invalid(message: &str) -> RuntimeError {
    super::request::invalid(message)
}

fn report(outcome: GmresOutcome, requested: Option<usize>) -> BuiltinResult<()> {
    if requested.is_some_and(|count| count >= 2) {
        return Ok(());
    }
    let message = match outcome.flag {
        0 => "gmres converged to the requested tolerance.",
        1 => "gmres stopped: the iteration limit was reached.",
        2 => "gmres stopped: the preconditioner inverse action failed numerically.",
        3 => "gmres stopped: consecutive iterates stagnated.",
        _ => "gmres stopped: numerical breakdown.",
    };
    crate::console::record_console_line(crate::console::ConsoleStream::Stdout, message)
}

fn admission(context: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    super::execution::admission(Some(context), error)
}

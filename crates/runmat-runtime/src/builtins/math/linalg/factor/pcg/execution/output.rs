use super::super::super::iterative::output::{initialize_list, retained_list_bytes};
use super::{
    invalid, map_operator_in_context, BuiltinResult, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, NumericDType, PcgOutcome, PcgValue, RuntimeContext, SparseOperatorError,
    Value,
};

fn admission(context: Option<&RuntimeContext>, error: HostAllocationError) -> super::RuntimeError {
    map_operator_in_context(SparseOperatorError::Admission(error), context)
}

fn response(
    x: Value,
    residuals: Option<Value>,
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> Value {
    match requested {
        None | Some(1) => x,
        Some(0) => Value::output_list(Vec::new()),
        Some(count) => {
            let mut output = Vec::with_capacity(count);
            output.push(x);
            if count > 1 {
                output.push(Value::Num(f64::from(outcome.flag)));
            }
            if count > 2 {
                output.push(Value::Num(outcome.relres));
            }
            if count > 3 {
                output.push(Value::Num(outcome.iterations as f64));
            }
            if count > 4 {
                output.push(residuals.expect("five outputs prepare residual history"));
            }
            Value::output_list(output)
        }
    }
}

pub(super) struct PcgQuote {
    pub(super) main_count: usize,
    pub(super) residual_count: usize,
    pub(super) plan: HostAllocationPlan,
}

pub(super) fn quote<T: PcgValue>(
    n: usize,
    maximum_iterations: usize,
    vector_count: usize,
    requested: Option<usize>,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<PcgQuote> {
    if !matches!(vector_count, 5 | 6) {
        return Err(invalid("invalid PCG workspace vector count"));
    }
    let geometry = runmat_sparse_solvers::krylov::pcg::PcgGeometry::checked(
        n,
        maximum_iterations,
        vector_count == 6,
    )
    .ok_or_else(|| invalid("workspace dimensions overflow or are empty"))?;
    let main_count = geometry
        .native_elements
        .checked_mul(T::LANES)
        .ok_or_else(|| invalid("workspace dimensions overflow"))?;
    let residual_count = geometry.history_elements;
    let scratch = HostAllocationPlan::numeric_workspace_bytes(T::DTYPE, main_count)
        .and_then(|bytes| {
            bytes
                .checked_add(HostAllocationPlan::numeric_workspace_bytes(
                    NumericDType::F64,
                    residual_count,
                )?)
                .ok_or(HostAllocationError::Overflow)
        })
        .map_err(|error| admission(context, error))?;
    let mut output = T::output_bytes(n).map_err(|error| admission(context, error))?;
    if requested.is_some_and(|count| count >= 5) {
        output = output
            .checked_add(
                HostAllocationPlan::retained_tensor_bytes_for_dimensions(
                    T::DTYPE,
                    residual_count,
                    2,
                )
                .map_err(|error| admission(context, error))?,
            )
            .ok_or_else(|| admission(context, HostAllocationError::Overflow))?;
    }
    if let Some(count @ 2..=5) = requested {
        output = output
            .checked_add(retained_list_bytes(count).map_err(|error| admission(context, error))?)
            .ok_or_else(|| admission(context, HostAllocationError::Overflow))?;
    }
    Ok(PcgQuote {
        main_count,
        residual_count,
        plan: HostAllocationPlan::new(scratch, output)
            .map_err(|error| admission(context, error))?,
    })
}

pub(super) async fn publish_admitted<T: PcgValue>(
    receipt: &HostAllocationReceipt,
    context: &RuntimeContext,
    x: &[T],
    norms: &[f64],
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    receipt
        .checkpoint(context)
        .map_err(|error| admission(Some(context), error))?;
    if requested == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    let x = super::super::super::iterative::copy::admitted::<T>(receipt, context, x, "pcg").await?;
    let history = if requested.is_some_and(|count| count >= 5) {
        Some(
            super::super::super::iterative::output::admitted_history(
                receipt,
                context,
                T::DTYPE,
                &norms[..outcome.residual_count],
                "pcg",
            )
            .await?,
        )
    } else {
        None
    };
    let result = receipt
        .with_output_allocation(context, |permit| {
            if let Some(count @ 2..=5) = requested {
                initialize_list(
                    count,
                    [
                        Some(x),
                        Some(Value::Num(f64::from(outcome.flag))),
                        Some(Value::Num(outcome.relres)),
                        Some(Value::Num(outcome.iterations as f64)),
                        history,
                    ],
                    permit,
                )
            } else {
                Ok(response(x, history, outcome, requested))
            }
        })
        .map_err(|error| admission(Some(context), error))?;
    Ok(result)
}

pub(super) async fn publish_unbounded<T: PcgValue>(
    context: Option<&RuntimeContext>,
    mut main: Vec<T>,
    n: usize,
    norms: &[f64],
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    super::super::super::iterative::copy::unbounded_checkpoint(context, "pcg")?;
    if requested == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    main.truncate(n);
    let x = T::unbounded_output(main).map_err(invalid)?;
    let history = if requested.is_some_and(|count| count >= 5) {
        Some(
            super::super::super::iterative::output::unbounded_history(
                context,
                T::DTYPE,
                &norms[..outcome.residual_count],
                "pcg",
            )
            .await?,
        )
    } else {
        None
    };
    super::super::super::iterative::copy::unbounded_checkpoint(context, "pcg")?;
    Ok(response(x, history, outcome, requested))
}

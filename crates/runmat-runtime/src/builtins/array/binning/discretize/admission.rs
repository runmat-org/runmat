//! Representation and option admission precedes every provider transfer.
use super::{edges, error, labels::Labels, numeric};
use crate::{builtins::common::shape::effective_rank, BuiltinResult};
use runmat_value::Value;

pub(super) fn validate(
    x: &Value,
    edge_input: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 2) {
        return Err(error::invalid("discretize: too many output arguments"));
    }
    real_numeric(x, "X")?;
    real_numeric(edge_input, "edges")?;
    vector(edge_input, "edges or scalar bin count")?;
    let (labels, _) = edges::tail_arguments(rest)?;
    let label_count = if let Some(labels) = labels {
        vector(labels, "replacement values")?;
        match labels {
            Value::GpuTensor(handle) => {
                real_numeric(labels, "replacement values")?;
                if runmat_accelerate_api::handle_is_logical(handle) {
                    return Err(error::invalid(
                        "discretize: replacement values must be real numeric or text",
                    ));
                }
                Some(
                    handle
                        .shape()
                        .iter()
                        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                        .ok_or_else(|| error::invalid("discretize: replacement shape overflows"))?,
                )
            }
            _ => Some(Labels::validate_len(labels)?),
        }
    } else {
        None
    };
    // A resident scalar's contents are unknown until its supported download.
    // Shape alone still proves when a second output cannot be requested.
    let shape = shape(edge_input);
    if outputs == Some(2) && shape.iter().any(|extent| *extent != 1) {
        return Err(error::invalid(
            "discretize: the second edge output requires a scalar bin count",
        ));
    }
    if !matches!(edge_input, Value::GpuTensor(_)) {
        let bins = match edges::bin_count(edge_input)? {
            Some(count) => count,
            None => {
                let values = numeric::values(edge_input, "edges")?;
                edges::validate_edges(&values)?;
                values.len() - 1
            }
        };
        if let Some(length) = label_count {
            if length != bins {
                return Err(error::invalid(
                    "discretize: number of replacement values must match number of bins",
                ));
            }
        }
    }
    Ok(())
}

fn real_numeric(value: &Value, role: &str) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => true,
        Value::SparseTensor(value) => !value.is_complex(),
        Value::GpuTensor(handle) => handle.descriptor().storage.is_some_and(|storage| {
            matches!(storage, runmat_accelerate_api::GpuTensorStorage::Real)
        }),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(error::invalid(format!(
            "discretize: {role} must be real numeric or logical"
        )))
    }
}

pub(super) fn shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(value) => value.shape().to_vec(),
        Value::LogicalArray(value) => value.shape.to_vec(),
        Value::SparseTensor(value) => vec![value.rows, value.cols],
        Value::GpuTensor(value) => value.shape().to_vec(),
        Value::StringArray(value) => value.shape().to_vec(),
        Value::CharArray(value) => value.shape().to_vec(),
        Value::Cell(value) => value.shape().to_vec(),
        _ => vec![1, 1],
    }
}

pub(super) fn vector(value: &Value, role: &str) -> BuiltinResult<()> {
    let shape = shape(value);
    if effective_rank(&shape) <= 2
        && shape.iter().take(2).filter(|extent| **extent != 1).count() <= 1
    {
        Ok(())
    } else {
        Err(error::invalid(format!(
            "discretize: {role} must be a vector"
        )))
    }
}

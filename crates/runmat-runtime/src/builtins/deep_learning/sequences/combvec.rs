use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{builtins::common::tensor, gather_if_needed_async, BuiltinResult};

use super::super::{deep_learning_error, tensor_value, MAX_COMBVEC_COLUMNS, MAX_PAD_ELEMENTS};

#[runtime_builtin(
    name = "combvec",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::sequences"
)]
pub(in crate::builtins::deep_learning) async fn combvec_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::super::errors::admit_single_output("combvec", &runmat_builtins::SEQUENCE_INVALID_INPUT)?;
    if args.is_empty() {
        return Err(deep_learning_error(
            "combvec",
            "combvec: expected at least one numeric vector",
        ));
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::COMBVEC_INTEGER_EXTENSION,
            "combvec",
        )?;
    }
    // Parse host arguments and check resident shapes before the first provider
    // operation. Host-invalid later inputs must not download an earlier input.
    let mut parsed = Vec::with_capacity(args.len());
    let mut known_rows = 0usize;
    let mut known_columns = 1usize;
    for (index, arg) in args.iter().enumerate() {
        let input = match arg {
            Value::GpuTensor(handle) => {
                if !matches!(handle.shape(), [columns] if *columns > 0)
                    && !matches!(handle.shape(), [rows, columns] if *rows > 0 && *columns > 0)
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_storage(handle)
                        != runmat_accelerate_api::GpuTensorStorage::Real
                {
                    return Err(deep_learning_error(
                        "combvec",
                        format!(
                            "combvec: input {} must be a nonempty real numeric 2-D matrix",
                            index + 1
                        ),
                    ));
                }
                None
            }
            _ => Some(CombvecInput::from_value(arg, index + 1)?),
        };
        let (rows, columns) = match &input {
            Some(matrix) => (matrix.rows, matrix.cols),
            None => {
                let Value::GpuTensor(handle) = arg else {
                    unreachable!()
                };
                match handle.shape() {
                    [columns] => (1, *columns),
                    [rows, columns] => (*rows, *columns),
                    _ => unreachable!("resident matrix shape was admitted above"),
                }
            }
        };
        known_rows = known_rows.checked_add(rows).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output row count overflows usize")
        })?;
        known_columns = known_columns.checked_mul(columns).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output column count overflows usize")
        })?;
        if known_columns > MAX_COMBVEC_COLUMNS
            || known_rows
                .checked_mul(known_columns)
                .is_none_or(|total| total > MAX_PAD_ELEMENTS)
        {
            return Err(deep_learning_error(
                "combvec",
                "combvec: output exceeds supported capacity",
            ));
        }
        parsed.push(input);
    }
    let mut matrices = Vec::with_capacity(args.len());
    let mut total_cols = 1usize;
    let mut total_rows = 0usize;
    for (idx, (arg, host)) in args.iter().zip(parsed).enumerate() {
        let matrix = match host {
            Some(host) => host,
            None => CombvecInput::from_value(&gather_if_needed_async(arg).await?, idx + 1)?,
        };
        total_cols = total_cols.checked_mul(matrix.cols).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output column count overflows usize")
        })?;
        if total_cols > MAX_COMBVEC_COLUMNS {
            return Err(deep_learning_error(
                "combvec",
                format!("combvec: output exceeds {MAX_COMBVEC_COLUMNS} columns"),
            ));
        }
        total_rows = total_rows.checked_add(matrix.rows).ok_or_else(|| {
            deep_learning_error("combvec", "combvec: output row count overflows usize")
        })?;
        matrices.push(matrix);
    }
    let total_elements = total_rows.checked_mul(total_cols).ok_or_else(|| {
        deep_learning_error("combvec", "combvec: output element count overflows usize")
    })?;
    if total_elements > MAX_PAD_ELEMENTS {
        return Err(deep_learning_error(
            "combvec",
            format!("combvec: output exceeds {MAX_PAD_ELEMENTS} elements"),
        ));
    }
    let mut data = vec![0.0; total_elements];
    for col in 0..total_cols {
        let mut divisor = 1usize;
        let mut row_offset = 0usize;
        for matrix in &matrices {
            let source_col = (col / divisor) % matrix.cols;
            let source_start = source_col * matrix.rows;
            let target_start = row_offset + col * total_rows;
            data[target_start..target_start + matrix.rows]
                .copy_from_slice(&matrix.data[source_start..source_start + matrix.rows]);
            row_offset += matrix.rows;
            divisor *= matrix.cols;
        }
    }
    tensor_value(data, vec![total_rows, total_cols], "combvec")
}

struct CombvecInput {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl CombvecInput {
    fn from_value(value: &Value, index: usize) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) if n.is_finite() => Ok(Self {
                data: vec![*n],
                rows: 1,
                cols: 1,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![i.to_f64()],
                rows: 1,
                cols: 1,
            }),
            Value::Tensor(t) if t.shape.len() <= 2 => {
                if t.rows == 0 || t.cols == 0 {
                    return Err(deep_learning_error(
                        "combvec",
                        format!("combvec: input {index} must have at least one column"),
                    ));
                }
                let data = tensor::tensor_values_f64(t);
                for value in &data {
                    if !value.is_finite() {
                        return Err(deep_learning_error(
                            "combvec",
                            format!("combvec: input {index} must contain finite values"),
                        ));
                    }
                }
                Ok(Self {
                    data,
                    rows: t.rows,
                    cols: t.cols,
                })
            }
            other => Err(deep_learning_error(
                "combvec",
                format!("combvec: input {index} must be a finite numeric scalar or 2-D matrix, got {other:?}"),
            )),
        }
    }
}

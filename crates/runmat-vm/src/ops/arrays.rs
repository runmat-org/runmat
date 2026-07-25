use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, IntValue, IntegerStorage, LogicalArray, Tensor, Value};
use runmat_runtime::RuntimeError;
use std::future::Future;

pub fn pack_to_row(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    let mut tmp: Vec<Value> = Vec::with_capacity(count);
    for _ in 0..count {
        tmp.push(
            stack
                .pop()
                .ok_or(mex("StackUnderflow", "stack underflow"))?,
        );
    }
    tmp.reverse();
    let tens = pack_numeric_values(tmp, vec![1, count], "PackToRow")?;
    stack.push(Value::Tensor(tens));
    Ok(())
}

pub fn pack_to_col(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    let mut tmp: Vec<Value> = Vec::with_capacity(count);
    for _ in 0..count {
        tmp.push(
            stack
                .pop()
                .ok_or(mex("StackUnderflow", "stack underflow"))?,
        );
    }
    tmp.reverse();
    let tens = pack_numeric_values(tmp, vec![count, 1], "PackToCol")?;
    stack.push(Value::Tensor(tens));
    Ok(())
}

pub fn create_matrix(stack: &mut Vec<Value>, rows: usize, cols: usize) -> Result<(), RuntimeError> {
    let total_elements = rows * cols;
    let mut row_major = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        row_major.push(
            stack
                .pop()
                .ok_or(mex("StackUnderflow", "stack underflow"))?,
        );
    }
    row_major.reverse();
    if total_elements == 0 {
        let matrix = Tensor::new_2d(Vec::new(), rows, cols)
            .map_err(|e| format!("Matrix creation error: {e}"))?;
        stack.push(Value::Tensor(matrix));
    } else if row_major.iter().all(|v| matches!(v, Value::Bool(_))) {
        let mut data = vec![0u8; total_elements];
        for r in 0..rows {
            for c in 0..cols {
                let Value::Bool(value) = row_major[r * cols + c] else {
                    unreachable!()
                };
                data[r + c * rows] = if value { 1 } else { 0 };
            }
        }
        let matrix = LogicalArray::new(data, vec![rows, cols])
            .map_err(|e| format!("Logical matrix creation error: {e}"))?;
        stack.push(Value::LogicalArray(matrix));
    } else if row_major.iter().any(|v| matches!(v, Value::Complex(_, _))) {
        let mut data = vec![(0.0, 0.0); total_elements];
        for r in 0..rows {
            for c in 0..cols {
                data[r + c * rows] = scalar_to_complex(&row_major[r * cols + c])?;
            }
        }
        let matrix = ComplexTensor::new_2d(data, rows, cols)
            .map_err(|e| format!("Complex matrix creation error: {e}"))?;
        stack.push(Value::ComplexTensor(matrix));
    } else if let Some(target) = leftmost_integer_target(&row_major) {
        let zero = target.cast_f64_assignment(0.0);
        let mut values = vec![zero; total_elements];
        for r in 0..rows {
            for c in 0..cols {
                values[r + c * rows] = scalar_to_integer(&row_major[r * cols + c], &target)?;
            }
        }
        let storage = target
            .from_same_class_values(values)
            .map_err(|e| format!("Integer matrix creation error: {e}"))?;
        let matrix = Tensor::new_integer(storage, vec![rows, cols])
            .map_err(|e| format!("Integer matrix creation error: {e}"))?;
        stack.push(Value::Tensor(matrix));
    } else {
        let mut data = vec![0.0; total_elements];
        for r in 0..rows {
            for c in 0..cols {
                data[r + c * rows] = scalar_to_real(&row_major[r * cols + c])?;
            }
        }
        let matrix =
            Tensor::new_2d(data, rows, cols).map_err(|e| format!("Matrix creation error: {e}"))?;
        stack.push(Value::Tensor(matrix));
    }
    Ok(())
}

fn pack_numeric_values(
    values: Vec<Value>,
    shape: Vec<usize>,
    context: &str,
) -> Result<Tensor, RuntimeError> {
    if let Some(target) = leftmost_integer_target(&values) {
        let converted = values
            .iter()
            .map(|value| scalar_to_integer(value, &target))
            .collect::<Result<Vec<_>, _>>()?;
        let storage = target
            .from_same_class_values(converted)
            .map_err(|e| format!("{context}: {e}"))?;
        return Tensor::new_integer(storage, shape).map_err(|e| format!("{context}: {e}").into());
    }

    let vals = values
        .iter()
        .map(scalar_to_real)
        .collect::<Result<Vec<_>, _>>()?;
    Tensor::new(vals, shape).map_err(|e| format!("{context}: {e}").into())
}

fn leftmost_integer_target(values: &[Value]) -> Option<IntegerStorage> {
    values.iter().find_map(|value| match value {
        Value::Int(value) => Some(IntegerStorage::from_scalar(value.clone()).zeros_like(0)),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor
            .integer_storage()
            .map(|storage| storage.zeros_like(0)),
        _ => None,
    })
}

fn scalar_to_integer(value: &Value, target: &IntegerStorage) -> Result<IntValue, RuntimeError> {
    if let Some(exact) = scalar_integer_value(value) {
        return Ok(target.cast_exact_assignment(&exact));
    }
    Ok(target.cast_f64_assignment(scalar_to_real(value)?))
}

fn scalar_integer_value(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn scalar_to_complex(value: &Value) -> Result<(f64, f64), RuntimeError> {
    match value {
        Value::Complex(re, im) => Ok((*re, *im)),
        _ => Ok((scalar_to_real(value)?, 0.0)),
    }
}

fn scalar_to_real(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Ok(value.try_into()?),
    }
}

pub async fn create_matrix_dynamic<F, Fut>(
    stack: &mut Vec<Value>,
    num_rows: usize,
    mut create_from_values: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Vec<Vec<Value>>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    let mut row_lengths = Vec::new();
    for _ in 0..num_rows {
        let row_len: f64 = (&stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?)
            .try_into()?;
        row_lengths.push(row_len as usize);
    }
    row_lengths.reverse();
    let mut rows_data = Vec::new();
    for &row_len in row_lengths.iter().rev() {
        let mut row_values = Vec::new();
        for _ in 0..row_len {
            row_values.push(
                stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?,
            );
        }
        row_values.reverse();
        rows_data.push(row_values);
    }
    rows_data.reverse();
    let result = create_from_values(rows_data).await?;
    stack.push(result);
    Ok(())
}

pub async fn create_range<F, Fut>(
    stack: &mut Vec<Value>,
    has_step: bool,
    mut call_colon: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    if has_step {
        let end = stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?;
        let step = stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?;
        let start = stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?;
        stack.push(call_colon(vec![start, step, end]).await?);
    } else {
        let end = stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?;
        let start = stack
            .pop()
            .ok_or(mex("StackUnderflow", "stack underflow"))?;
        stack.push(call_colon(vec![start, end]).await?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_to_row_preserves_leftmost_integer_class_exactly() {
        let mut stack = vec![
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Num(3.5),
            Value::Int(IntValue::U8(9)),
        ];

        pack_to_row(&mut stack, 3).expect("pack row");

        let Value::Tensor(output) = stack.pop().expect("output") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 4, 9]))
        );
    }

    #[test]
    fn pack_to_col_preserves_leftmost_integer_class_exactly() {
        let mut stack = vec![
            Value::Int(IntValue::I8(12)),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Num(-200.0),
        ];

        pack_to_col(&mut stack, 3).expect("pack col");

        let Value::Tensor(output) = stack.pop().expect("output") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![12, i8::MAX, i8::MIN]))
        );
    }

    #[test]
    fn create_matrix_preserves_exact_integer_storage_column_major() {
        let scalar = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63]), vec![1, 1])
            .expect("scalar integer tensor");
        let mut stack = vec![
            Value::Tensor(scalar),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Num(7.4),
            Value::Int(IntValue::U16(11)),
        ];

        create_matrix(&mut stack, 2, 2).expect("create matrix");

        let Value::Tensor(output) = stack.pop().expect("output") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, 7, u64::MAX, 11]))
        );
    }
}

pub fn unpack(stack: &mut Vec<Value>, out_count: usize) -> Result<(), RuntimeError> {
    let value = stack
        .pop()
        .ok_or(mex("StackUnderflow", "stack underflow"))?;
    match value {
        Value::OutputList(values) => {
            if values.len() < out_count {
                let message = format!(
                    "Requested {out_count} outputs but call produced {} output value(s)",
                    values.len()
                );
                return Err(mex("TooManyOutputs", &message));
            }
            for v in values.into_iter().take(out_count) {
                stack.push(v);
            }
        }
        other => {
            if out_count > 1 {
                let message = format!(
                    "Requested {out_count} outputs but call produced a single output value"
                );
                return Err(mex("TooManyOutputs", &message));
            }
            stack.push(other);
        }
    }
    Ok(())
}

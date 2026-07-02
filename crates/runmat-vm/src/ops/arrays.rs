use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, LogicalArray, SymbolicArray, SymbolicExpr, Tensor, Value};
use runmat_runtime::RuntimeError;
use std::future::Future;

pub fn pack_to_row(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    let mut vals: Vec<f64> = Vec::with_capacity(count);
    let mut tmp: Vec<Value> = Vec::with_capacity(count);
    for _ in 0..count {
        tmp.push(
            stack
                .pop()
                .ok_or(mex("StackUnderflow", "stack underflow"))?,
        );
    }
    tmp.reverse();
    for v in tmp {
        let n: f64 = (&v).try_into()?;
        vals.push(n);
    }
    let tens = Tensor::new(vals, vec![1, count]).map_err(|e| format!("PackToRow: {e}"))?;
    stack.push(Value::Tensor(tens));
    Ok(())
}

pub fn pack_to_col(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    let mut vals: Vec<f64> = Vec::with_capacity(count);
    let mut tmp: Vec<Value> = Vec::with_capacity(count);
    for _ in 0..count {
        tmp.push(
            stack
                .pop()
                .ok_or(mex("StackUnderflow", "stack underflow"))?,
        );
    }
    tmp.reverse();
    for v in tmp {
        let n: f64 = (&v).try_into()?;
        vals.push(n);
    }
    let tens = Tensor::new(vals, vec![count, 1]).map_err(|e| format!("PackToCol: {e}"))?;
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
    } else if row_major
        .iter()
        .any(|v| matches!(v, Value::Symbolic(_) | Value::SymbolicArray(_)))
    {
        let mut data = vec![SymbolicExpr::constant(0.0); total_elements];
        for r in 0..rows {
            for c in 0..cols {
                data[r + c * rows] = scalar_to_symbolic(&row_major[r * cols + c])?;
            }
        }
        let matrix = SymbolicArray::new_2d(data, rows, cols)
            .map_err(|e| format!("Symbolic matrix creation error: {e}"))?;
        stack.push(Value::SymbolicArray(matrix));
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

fn scalar_to_symbolic(value: &Value) -> Result<SymbolicExpr, RuntimeError> {
    match value {
        Value::Symbolic(expr) => Ok(expr.clone()),
        Value::SymbolicArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::Num(n) => Ok(SymbolicExpr::constant(*n)),
        Value::Int(i) => Ok(SymbolicExpr::constant(i.to_f64())),
        Value::Bool(flag) => Ok(SymbolicExpr::constant(if *flag { 1.0 } else { 0.0 })),
        _ => Err(RuntimeError::new(format!(
            "cannot convert {value:?} to symbolic scalar"
        ))),
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

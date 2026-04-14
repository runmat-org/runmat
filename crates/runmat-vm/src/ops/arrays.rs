use crate::interpreter::errors::mex;
use runmat_builtins::{Tensor, Value};
use runmat_runtime::RuntimeError;
use std::future::Future;

pub fn pack_to_row(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    let mut vals: Vec<f64> = Vec::with_capacity(count);
    let mut tmp: Vec<Value> = Vec::with_capacity(count);
    for _ in 0..count {
        tmp.push(stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?);
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
        tmp.push(stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?);
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
        let val: f64 = (&stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?).try_into()?;
        row_major.push(val);
    }
    row_major.reverse();
    let mut data = vec![0.0; total_elements];
    for r in 0..rows {
        for c in 0..cols {
            data[r + c * rows] = row_major[r * cols + c];
        }
    }
    let matrix = Tensor::new_2d(data, rows, cols).map_err(|e| format!("Matrix creation error: {e}"))?;
    stack.push(Value::Tensor(matrix));
    Ok(())
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
        let row_len: f64 = (&stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?).try_into()?;
        row_lengths.push(row_len as usize);
    }
    row_lengths.reverse();
    let mut rows_data = Vec::new();
    for &row_len in row_lengths.iter().rev() {
        let mut row_values = Vec::new();
        for _ in 0..row_len {
            row_values.push(stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?);
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
        let end = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
        let step = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
        let start = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
        stack.push(call_colon(vec![start, step, end]).await?);
    } else {
        let end = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
        let start = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
        stack.push(call_colon(vec![start, end]).await?);
    }
    Ok(())
}

pub fn unpack(stack: &mut Vec<Value>, out_count: usize) -> Result<(), RuntimeError> {
    let value = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
    match value {
        Value::OutputList(values) => {
            for i in 0..out_count {
                if let Some(v) = values.get(i) {
                    stack.push(v.clone());
                } else {
                    stack.push(Value::Num(0.0));
                }
            }
        }
        other => {
            stack.push(other);
            for _ in 1..out_count {
                stack.push(Value::Num(0.0));
            }
        }
    }
    Ok(())
}

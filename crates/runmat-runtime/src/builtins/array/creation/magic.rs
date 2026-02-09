//! MATLAB-compatible `magic` builtin.

use runmat_builtins::{ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::tensor;

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("magic").build()
}

fn magic_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() != 1 {
        return Type::Unknown;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    Type::tensor()
}

#[runtime_builtin(
    name = "magic",
    category = "array/creation",
    summary = "Generate an n-by-n magic square.",
    keywords = "magic,magic square,array",
    accel = "array_construct",
    type_resolver(magic_type),
    builtin_path = "crate::builtins::array::creation::magic"
)]
async fn magic_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let n = parse_order(args).await?;
    let tensor = magic_tensor(n)?;
    Ok(tensor::tensor_into_value(tensor))
}

async fn parse_order(args: Vec<Value>) -> crate::BuiltinResult<usize> {
    if args.len() != 1 {
        return Err(builtin_error("magic: requires exactly one input argument"));
    }
    let value = &args[0];
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| builtin_error(format!("magic: {err}")))?
    else {
        return Err(builtin_error("magic: input must be a numeric scalar"));
    };
    if !raw.is_finite() {
        return Err(builtin_error("magic: dimension must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(builtin_error("magic: dimension must be an integer"));
    }
    if rounded < 0.0 {
        return Err(builtin_error("magic: dimension must be non-negative"));
    }
    let n = rounded as usize;
    if n == 2 {
        return Err(builtin_error("magic: magic squares of order 2 do not exist"));
    }
    Ok(n)
}

fn magic_tensor(n: usize) -> Result<Tensor, crate::RuntimeError> {
    if n == 0 {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| builtin_error(format!("magic: {err}")));
    }

    let data = if n % 2 == 1 {
        magic_odd(n)
    } else if n % 4 == 0 {
        magic_doubly_even(n)
    } else {
        magic_singly_even(n)
    }
    .map_err(builtin_error)?;

    let data: Vec<f64> = data.into_iter().map(|v| v as f64).collect();
    Tensor::new(data, vec![n, n]).map_err(|err| builtin_error(format!("magic: {err}")))
}

fn magic_odd(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let mut data = vec![0usize; size];
    let mut row = 0usize;
    let mut col = n / 2;
    for val in 1..=size {
        data[idx(row, col, n)] = val;
        let next_row = (row + n - 1) % n;
        let next_col = (col + 1) % n;
        if data[idx(next_row, next_col, n)] != 0 {
            row = (row + 1) % n;
        } else {
            row = next_row;
            col = next_col;
        }
    }
    Ok(data)
}

fn magic_doubly_even(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let mut data = vec![0usize; size];

    for row in 0..n {
        for col in 0..n {
            let value = row * n + col + 1;
            let row_mod = row % 4;
            let col_mod = col % 4;
            let keep = row_mod == col_mod || row_mod + col_mod == 3;
            let final_value = if keep { value } else { size + 1 - value };
            data[idx(row, col, n)] = final_value;
        }
    }

    Ok(data)
}

fn magic_singly_even(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let m = n / 2;
    let m_sq = m
        .checked_mul(m)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let base = magic_odd(m)?;
    let mut data = vec![0usize; size];

    for row in 0..m {
        for col in 0..m {
            let value = base[idx(row, col, m)];
            data[idx(row, col, n)] = value;
            data[idx(row, col + m, n)] = value + 2 * m_sq;
            data[idx(row + m, col, n)] = value + 3 * m_sq;
            data[idx(row + m, col + m, n)] = value + m_sq;
        }
    }

    let k = (m - 1) / 2;

    for row in 0..m {
        if row == k {
            continue;
        }
        for col in 0..k {
            swap_cells(&mut data, n, row, col, row + m, col);
        }
    }

    if k > 0 {
        let start = m - k + 1;
        for row in 0..m {
            for col in start..m {
                swap_cells(&mut data, n, row, col + m, row + m, col + m);
            }
        }
    }

    swap_cells(&mut data, n, k, k, k + m, k);

    Ok(data)
}

fn idx(row: usize, col: usize, n: usize) -> usize {
    row + col * n
}

fn swap_cells(data: &mut [usize], n: usize, row_a: usize, col_a: usize, row_b: usize, col_b: usize) {
    let idx_a = idx(row_a, col_a, n);
    let idx_b = idx(row_b, col_b, n);
    data.swap(idx_a, idx_b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn magic_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::magic_builtin(args))
    }

    #[test]
    fn magic_rejects_two() {
        let err = magic_builtin(vec![Value::Num(2.0)]).unwrap_err();
        assert!(err.to_string().contains("order 2"));
    }

    #[test]
    fn magic_zero_is_empty() {
        let value = magic_builtin(vec![Value::Num(0.0)]).expect("magic");
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![0, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn magic_three_matches_matlab() {
        let value = magic_builtin(vec![Value::Num(3.0)]).expect("magic");
        let tensor = match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![3, 3]);
        let expected = vec![8.0, 3.0, 4.0, 1.0, 5.0, 9.0, 6.0, 7.0, 2.0];
        assert_eq!(tensor.data, expected);
    }

    #[test]
    fn magic_four_matches_matlab() {
        let value = magic_builtin(vec![Value::Num(4.0)]).expect("magic");
        let tensor = match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![4, 4]);
        let expected = vec![
            16.0, 5.0, 9.0, 4.0, 2.0, 11.0, 7.0, 14.0, 3.0, 10.0, 6.0, 15.0, 13.0, 8.0,
            12.0, 1.0,
        ];
        assert_eq!(tensor.data, expected);
    }

    #[test]
    fn magic_six_matches_matlab() {
        let value = magic_builtin(vec![Value::Num(6.0)]).expect("magic");
        let tensor = match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![6, 6]);
        let expected = vec![
            35.0, 3.0, 31.0, 8.0, 30.0, 4.0, 1.0, 32.0, 9.0, 28.0, 5.0, 36.0, 6.0,
            7.0, 2.0, 33.0, 34.0, 29.0, 26.0, 21.0, 22.0, 17.0, 12.0, 13.0, 19.0, 23.0,
            27.0, 10.0, 14.0, 18.0, 24.0, 25.0, 20.0, 15.0, 16.0, 11.0,
        ];
        assert_eq!(tensor.data, expected);
    }
}

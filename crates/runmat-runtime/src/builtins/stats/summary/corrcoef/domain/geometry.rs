use super::*;

pub(super) struct Matrix {
    pub(super) data: Vec<f64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl Matrix {
    pub(super) fn from_single(tensor: Tensor) -> Self {
        let (rows, cols) = single_geometry(tensor.shape(), tensor.len());
        Self {
            data: centered_tensor_values(&tensor, rows, cols),
            rows,
            cols,
        }
    }

    pub(super) fn from_pair(left: Tensor, right: Tensor) -> Self {
        let rows = left.len();
        if rows == 1 {
            return Self {
                data: centered_scalar_pair(&left, &right),
                rows: 2,
                cols: 1,
            };
        }
        let mut data = centered_tensor_values(&left, rows, 1);
        data.extend(centered_tensor_values(&right, rows, 1));
        Self {
            data,
            rows,
            cols: 2,
        }
    }

    #[inline]
    pub(super) fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }
}

fn centered_scalar_pair(left: &Tensor, right: &Tensor) -> Vec<f64> {
    match (left.integer_storage(), right.integer_storage()) {
        (Some(left), Some(right)) => {
            let left = left
                .value_at(0)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(0);
            let right = right
                .value_at(0)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(left);
            vec![0.0, (right - left) as f64]
        }
        _ => vec![left.materialize_f64()[0], right.materialize_f64()[0]],
    }
}

fn single_geometry(shape: &[usize], len: usize) -> (usize, usize) {
    let (rows, cols) = match shape {
        [] => (1, 1),
        [count] => (1, *count),
        [rows, cols, ..] => (*rows, *cols),
    };
    if rows == 1 || cols == 1 {
        (len, 1)
    } else {
        (rows, cols)
    }
}

fn centered_tensor_values(tensor: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor.materialize_f64();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let anchor = storage
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let value = storage
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(anchor);
            values.push((value - anchor) as f64);
        }
    }
    values
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

#[derive(Debug, Clone)]
pub(super) struct ComplexMatrix {
    pub(super) data: Vec<Complex64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl ComplexMatrix {
    pub(super) fn from_single(tensor: ComplexTensor) -> Self {
        let (rows, cols) = single_geometry(tensor.shape(), tensor.len());
        Self {
            data: centered_complex_values(&tensor, rows, cols),
            rows,
            cols,
        }
    }

    pub(super) fn from_pair(left: NumericInput, right: NumericInput) -> BuiltinResult<Self> {
        let rows = left.len();
        if rows == 1 {
            return Ok(Self {
                data: centered_complex_scalar_pair(&left, &right),
                rows: 2,
                cols: 1,
            });
        }
        let mut data = centered_numeric_values(left, rows)?;
        data.extend(centered_numeric_values(right, rows)?);
        Ok(Self {
            data,
            rows,
            cols: 2,
        })
    }

    #[inline]
    pub(super) fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }
}

fn centered_complex_scalar_pair(left: &NumericInput, right: &NumericInput) -> Vec<Complex64> {
    let left = exact_complex_scalar(left);
    let right = exact_complex_scalar(right);
    match (left, right) {
        (
            ExactComplexScalar::Integer(left_real, left_imag),
            ExactComplexScalar::Integer(right_real, right_imag),
        ) => vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(
                (right_real - left_real) as f64,
                (right_imag - left_imag) as f64,
            ),
        ],
        (left, right) => vec![left.materialize(), right.materialize()],
    }
}

enum ExactComplexScalar {
    Integer(i128, i128),
    Floating(Complex64),
}

impl ExactComplexScalar {
    fn materialize(self) -> Complex64 {
        match self {
            Self::Integer(real, imag) => Complex64::new(real as f64, imag as f64),
            Self::Floating(value) => value,
        }
    }
}

fn exact_complex_scalar(input: &NumericInput) -> ExactComplexScalar {
    match input {
        NumericInput::Real(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                ExactComplexScalar::Integer(value, 0)
            } else {
                ExactComplexScalar::Floating(Complex64::new(tensor.materialize_f64()[0], 0.0))
            }
        }
        NumericInput::Complex(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let real = storage
                    .real()
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                let imag = storage
                    .imag()
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                ExactComplexScalar::Integer(real, imag)
            } else {
                let (real, imag) = tensor.materialize_f64()[0];
                ExactComplexScalar::Floating(Complex64::new(real, imag))
            }
        }
    }
}

fn centered_numeric_values(input: NumericInput, rows: usize) -> BuiltinResult<Vec<Complex64>> {
    Ok(match input {
        NumericInput::Real(tensor) => centered_tensor_values(&tensor, rows, 1)
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect(),
        NumericInput::Complex(tensor) => centered_complex_values(&tensor, rows, 1),
    })
}

fn centered_complex_values(tensor: &ComplexTensor, rows: usize, cols: usize) -> Vec<Complex64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor
            .materialize_f64()
            .into_iter()
            .map(|(real, imag)| Complex64::new(real, imag))
            .collect();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let real_anchor = storage
            .real()
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        let imag_anchor = storage
            .imag()
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let real = storage
                .real()
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(real_anchor);
            let imag = storage
                .imag()
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(imag_anchor);
            values.push(Complex64::new(
                (real - real_anchor) as f64,
                (imag - imag_anchor) as f64,
            ));
        }
    }
    values
}

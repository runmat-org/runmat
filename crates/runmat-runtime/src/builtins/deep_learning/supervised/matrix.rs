use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(super) struct Matrix {
    pub(super) data: Vec<f64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl Matrix {
    pub(super) fn from_tensor(
        tensor: Tensor,
        function: &'static str,
        label: &str,
    ) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must be a 2-D numeric matrix"),
            ));
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let data = tensor::tensor_into_values_f64(tensor);
        if data.iter().any(|value| !value.is_finite()) {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must contain finite numeric values"),
            ));
        }
        Ok(Self { data, rows, cols })
    }

    pub(super) fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub(super) fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    pub(super) fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row + col * self.rows] = value;
    }

    pub(super) fn select_rows(
        &self,
        rows: &[usize],
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let mut out = vec![0.0; rows.len() * self.cols];
        for col in 0..self.cols {
            for (out_row, &row) in rows.iter().enumerate() {
                if row >= self.rows {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: internal mini-batch row index out of range"),
                    ));
                }
                out[out_row + col * rows.len()] = self.get(row, col);
            }
        }
        Ok(Self {
            data: out,
            rows: rows.len(),
            cols: self.cols,
        })
    }

    pub(super) fn into_tensor(self, function: &'static str) -> BuiltinResult<Tensor> {
        Tensor::new(self.data, vec![self.rows, self.cols])
            .map_err(|err| deep_learning_error(function, err))
    }
}

pub(super) fn numeric_matrix(
    value: Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Matrix> {
    let tensor = crate::builtins::common::tensor::value_into_tensor_for(function, value)
        .map_err(|err| deep_learning_error(function, format!("{function}: {err}")))?;
    Matrix::from_tensor(tensor, function, label)
}

/// Convert admitted host data once without cloning its typed backing storage.
/// Resident values are handled by the caller after host-decidable controls.
pub(super) fn numeric_matrix_ref(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Matrix> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape.len() > 2 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must be a 2-D numeric matrix"),
                ));
            }
            let data = tensor.materialize_f64();
            if data.iter().any(|value| !value.is_finite()) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must contain finite numeric values"),
                ));
            }
            Ok(Matrix {
                data,
                rows: tensor.rows,
                cols: tensor.cols,
            })
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must be a 2-D numeric matrix"),
                ));
            }
            let (rows, cols) = match logical.shape.as_slice() {
                [rows, cols] => (*rows, *cols),
                [cols] => (1, *cols),
                [] => (0, 0),
                _ => unreachable!("rank was checked above"),
            };
            Ok(Matrix {
                data: logical
                    .data
                    .iter()
                    .map(|&value| if value != 0 { 1.0 } else { 0.0 })
                    .collect(),
                rows,
                cols,
            })
        }
        Value::Num(number) => numeric_matrix(Value::Num(*number), function, label),
        Value::Int(number) => numeric_matrix(Value::Int(number.clone()), function, label),
        Value::Bool(boolean) => numeric_matrix(Value::Bool(*boolean), function, label),
        _ => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be numeric or logical values"),
        )),
    }
}

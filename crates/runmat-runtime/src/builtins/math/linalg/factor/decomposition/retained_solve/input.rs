//! Borrowed logical RHS view. A right solve changes indices, not storage.
use super::*;

pub(super) struct Input<'a> {
    pub value: &'a Value,
    pub rows: usize,
    pub columns: usize,
    pub right: bool,
    pub scalar_operator: bool,
    pub scale: (f64, f64),
}
impl Input<'_> {
    pub fn parts(&self, index: usize) -> Option<(f64, f64)> {
        let (real, imag) = self.raw_parts(index)?;
        let scale = if self.right {
            complex_conj(self.scale)
        } else {
            self.scale
        };
        if scale.1 == 0.0 {
            Some((real / scale.0, imag / scale.0))
        } else {
            Some(runmat_value::divide_complex_f64((real, imag), scale))
        }
    }

    /// The retained scaled-factor route must not divide before its native
    /// solve. This view retains the existing right-hand adjoint geometry.
    pub fn raw_parts(&self, index: usize) -> Option<(f64, f64)> {
        let index = if self.right && !self.scalar_operator {
            let row = index % self.columns;
            let column = index / self.columns;
            column + row * self.rows
        } else {
            index
        };
        let pair = if let Value::SparseTensor(t) = self.value {
            let row = index % self.rows;
            let column = index / self.rows;
            t.row_indices[t.col_ptrs[column]..t.col_ptrs[column + 1]]
                .binary_search(&row)
                .ok()
                .map(|i| native::coefficient(self.value, t.col_ptrs[column] + i))
                .transpose()
                .ok()?
                .unwrap_or((
                    runmat_value::NumericScalar::F64(0.0),
                    runmat_value::NumericScalar::F64(0.0),
                ))
        } else {
            native::coefficient(self.value, index).ok()?
        };
        let (real, mut imag) = (
            native::checked_double(pair.0).ok()?,
            native::checked_double(pair.1).ok()?,
        );
        if self.right {
            imag = -imag;
        }
        Some((real, imag))
    }
    pub fn output_index(&self, index: usize, output_rows: usize) -> usize {
        if self.right && !self.scalar_operator {
            let row = index % output_rows;
            let column = index / output_rows;
            column + row * self.rows
        } else {
            index
        }
    }
}
pub(super) fn dtype(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(t) => t.numeric_dtype(),
        Value::ComplexTensor(t) => t.numeric_dtype(),
        Value::SparseTensor(t) => t.numeric_dtype().unwrap_or(NumericDType::F64),
        _ => NumericDType::F64,
    }
}
pub(super) fn complex(value: &Value) -> bool {
    matches!(value, Value::ComplexTensor(_))
        || matches!(value,Value::SparseTensor(t) if t.as_complex_f32_slice().is_some()||t.as_complex_f64_slice().is_some())
}
pub(super) async fn validate(value: &Value, runtime: &RuntimeContext) -> BuiltinResult<()> {
    // Topology and physical matrix geometry are validated cooperatively before
    // the indexed view can inspect sparse pointers. Implicit sparse zeros are
    // not expanded merely to prove finite/checked input values.
    native::inspect_original(value, false, runtime).await?;
    native::check_factor_recipe(value, None, runtime).await
}

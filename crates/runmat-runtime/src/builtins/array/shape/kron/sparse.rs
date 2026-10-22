//! CSC Kronecker product. Only stored/nonzero factor pairs are multiplied;
//! neither input nor the output acquires a dense workspace.
use super::*;
use runmat_value::{ComplexElement, ComplexStorage, NumericScalar};

#[derive(Clone)]
enum Element {
    Real(f64),
    Complex(f64, f64),
    Integer(IntValue),
}

impl Element {
    fn zero(&self) -> bool {
        match self {
            Self::Real(value) => *value == 0.0,
            Self::Complex(real, imag) => *real == 0.0 && *imag == 0.0,
            Self::Integer(value) => value.is_zero(),
        }
    }
    fn parts(&self) -> (f64, f64) {
        match self {
            Self::Real(real) => (*real, 0.0),
            Self::Complex(real, imag) => (*real, *imag),
            Self::Integer(value) => (value.to_f64(), 0.0),
        }
    }
}

fn error(detail: impl std::fmt::Display) -> RuntimeError {
    kron_error_with_message(format!("kron: {detail}"), &KRON_ERROR_INTERNAL)
}

fn shape(value: &Value) -> crate::BuiltinResult<(usize, usize)> {
    let dims = match value {
        Value::SparseTensor(sparse) => return Ok((sparse.rows, sparse.cols)),
        Value::Tensor(tensor) => tensor.shape(),
        Value::ComplexTensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => &array.shape,
        Value::CharArray(array) => return Ok((array.rows, array.cols)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => return Ok((1, 1)),
        _ => return Err(kron_error(&KRON_ERROR_UNSUPPORTED_INPUT)),
    };
    if dims.len() != 2 {
        return Err(kron_error_with_message(
            "kron: sparse Kronecker output requires two-dimensional operands",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        ));
    }
    Ok((dims[0], dims[1]))
}

fn numeric(value: NumericScalar) -> Element {
    match value {
        NumericScalar::F32(value) => Element::Real(f64::from(value)),
        NumericScalar::F64(value) => Element::Real(value),
        value => Element::Integer(value.into_int_value().expect("integer numeric scalar")),
    }
}

fn element(value: &Value, index: usize) -> crate::BuiltinResult<Element> {
    let element = match value {
        Value::SparseTensor(sparse) => {
            if let Some(value) = sparse.numeric_value_at(index) {
                numeric(value)
            } else if let Some(ComplexElement(real, imag)) = sparse.complex_value_at(index) {
                Element::Complex(real, imag)
            } else {
                return Err(error("sparse entry has no authoritative value"));
            }
        }
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                Element::Integer(
                    storage
                        .value_at(index)
                        .ok_or_else(|| error("integer index"))?,
                )
            } else if let Some(values) = tensor.as_f32_slice() {
                Element::Real(f64::from(values[index]))
            } else {
                Element::Real(tensor.as_f64_slice().ok_or_else(|| error("real storage"))?[index])
            }
        }
        Value::ComplexTensor(tensor) => match tensor.complex_storage() {
            ComplexStorage::F64(values) => {
                let ComplexElement(real, imag) = values[index];
                Element::Complex(real, imag)
            }
            ComplexStorage::F32(values) => {
                let ComplexElement(real, imag) = values[index];
                Element::Complex(f64::from(real), f64::from(imag))
            }
            ComplexStorage::Integer(_) => {
                return Err(error("complex integer arithmetic is unsupported"))
            }
        },
        Value::LogicalArray(array) => Element::Real(if array.data[index] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(array) => Element::Real(f64::from(array.data()[index].raw())),
        Value::Num(value) => Element::Real(*value),
        Value::Int(value) => Element::Integer(value.clone()),
        Value::Bool(value) => Element::Real(if *value { 1.0 } else { 0.0 }),
        Value::Complex(real, imag) => Element::Complex(*real, *imag),
        _ => return Err(kron_error(&KRON_ERROR_UNSUPPORTED_INPUT)),
    };
    Ok(element)
}

fn column(value: &Value, col: usize, rows: usize) -> crate::BuiltinResult<Vec<(usize, Element)>> {
    let mut entries = Vec::new();
    if let Value::SparseTensor(sparse) = value {
        let range = sparse.col_ptrs[col]..sparse.col_ptrs[col + 1];
        entries.try_reserve_exact(range.len()).map_err(error)?;
        for index in range {
            let entry = element(value, index)?;
            if !entry.zero() {
                entries.push((sparse.row_indices[index], entry));
            }
        }
    } else {
        entries.try_reserve_exact(rows).map_err(error)?;
        let offset = col
            .checked_mul(rows)
            .ok_or_else(|| error("column offset overflow"))?;
        for row in 0..rows {
            let entry = element(value, offset + row)?;
            if !entry.zero() {
                entries.push((row, entry));
            }
        }
    }
    Ok(entries)
}

enum Values {
    F64(Vec<f64>),
    F32(Vec<f32>),
    C64(Vec<(f64, f64)>),
    C32(Vec<(f32, f32)>),
    Integer(IntegerStorage),
}

impl Values {
    fn reserve(&mut self, additional: usize) -> crate::BuiltinResult<()> {
        match self {
            Self::F64(values) => values.try_reserve(additional),
            Self::F32(values) => values.try_reserve(additional),
            Self::C64(values) => values.try_reserve(additional),
            Self::C32(values) => values.try_reserve(additional),
            Self::Integer(values) => values.try_reserve(additional),
        }
        .map_err(error)
    }

    fn push_nonzero(&mut self, left: &Element, right: &Element) -> crate::BuiltinResult<bool> {
        match self {
            Self::Integer(values) => {
                let operand = |value: &Element| match value {
                    Element::Integer(value) => Value::Int(value.clone()),
                    Element::Real(value) => Value::Num(*value),
                    Element::Complex(_, _) => unreachable!("complex integer admission"),
                };
                let product = integer_binary_scalar(
                    &operand(left),
                    &operand(right),
                    IntegerBinaryOp::Multiply,
                    BUILTIN_NAME,
                )
                .map_err(|detail| kron_error_with_message(detail, &KRON_ERROR_UNSUPPORTED_INPUT))?;
                if product.is_zero() {
                    return Ok(false);
                }
                values.push_same_class(product).map_err(error)?;
            }
            Self::F64(values) => {
                let product = left.parts().0 * right.parts().0;
                if product == 0.0 {
                    return Ok(false);
                }
                values.push(product);
            }
            Self::F32(values) => {
                let product = (left.parts().0 as f32) * (right.parts().0 as f32);
                if product == 0.0 {
                    return Ok(false);
                }
                values.push(product);
            }
            Self::C64(values) => {
                let product = kron_complex_product(left.parts(), right.parts(), false);
                if product.0 == 0.0 && product.1 == 0.0 {
                    return Ok(false);
                }
                values.push(product);
            }
            Self::C32(values) => {
                let (real, imag) = kron_complex_product(left.parts(), right.parts(), true);
                let product = (real as f32, imag as f32);
                if product.0 == 0.0 && product.1 == 0.0 {
                    return Ok(false);
                }
                values.push(product);
            }
        }
        Ok(true)
    }

    fn finish(
        self,
        rows: usize,
        cols: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
    ) -> crate::BuiltinResult<Value> {
        let result = match self {
            Self::F64(values) => SparseTensor::new(rows, cols, pointers, indices, values),
            Self::F32(values) => SparseTensor::new_f32(rows, cols, pointers, indices, values),
            Self::C64(values) => SparseTensor::new_complex(rows, cols, pointers, indices, values),
            Self::C32(values) => {
                SparseTensor::new_complex_f32(rows, cols, pointers, indices, values)
            }
            Self::Integer(values) => {
                SparseTensor::new_integer(rows, cols, pointers, indices, values)
            }
        }
        .map_err(error)?;
        Ok(Value::SparseTensor(result))
    }
}

fn output_values(left: &Value, right: &Value) -> crate::BuiltinResult<Values> {
    let integer_values = [left, right].into_iter().find_map(|value| match value {
        Value::SparseTensor(sparse) => sparse
            .integer_storage()
            .map(|storage| storage.zeros_like(0)),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.zeros_like(0)),
        Value::Int(value) => Some(IntegerStorage::from_scalar(value.clone()).zeros_like(0)),
        _ => None,
    });
    if let Some(values) = integer_values {
        return Ok(Values::Integer(values));
    }
    let complex = contains_complex(left) || contains_complex(right);
    let single = kron_uses_single(left, right);
    Ok(match (complex, single) {
        (false, false) => Values::F64(Vec::new()),
        (false, true) => Values::F32(Vec::new()),
        (true, false) => Values::C64(Vec::new()),
        (true, true) => Values::C32(Vec::new()),
    })
}

pub(super) fn kron(left: &Value, right: &Value) -> crate::BuiltinResult<Value> {
    let (left_rows, left_cols) = shape(left)?;
    let (right_rows, right_cols) = shape(right)?;
    let rows = left_rows
        .checked_mul(right_rows)
        .ok_or_else(|| error("row extent overflow"))?;
    let cols = left_cols
        .checked_mul(right_cols)
        .ok_or_else(|| error("column extent overflow"))?;
    rows.checked_mul(cols)
        .ok_or_else(|| error("output extent overflow"))?;
    let mut values = output_values(left, right)?;
    let mut pointers = Vec::new();
    pointers
        .try_reserve_exact(
            cols.checked_add(1)
                .ok_or_else(|| error("column pointer overflow"))?,
        )
        .map_err(error)?;
    pointers.push(0);
    let mut indices = Vec::new();
    for left_col in 0..left_cols {
        let left_entries = column(left, left_col, left_rows)?;
        for right_col in 0..right_cols {
            let right_entries = column(right, right_col, right_rows)?;
            let additional = left_entries
                .len()
                .checked_mul(right_entries.len())
                .ok_or_else(|| error("nonzero product overflow"))?;
            indices.try_reserve(additional).map_err(error)?;
            values.reserve(additional)?;
            for (left_row, left_value) in &left_entries {
                let row_base = left_row
                    .checked_mul(right_rows)
                    .ok_or_else(|| error("row index overflow"))?;
                for (right_row, right_value) in &right_entries {
                    if values.push_nonzero(left_value, right_value)? {
                        indices.push(row_base + right_row);
                    }
                }
            }
            pointers.push(indices.len());
        }
    }
    values.finish(rows, cols, pointers, indices)
}

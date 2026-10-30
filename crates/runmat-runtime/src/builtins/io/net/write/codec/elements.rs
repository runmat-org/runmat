use super::{write_error_with_message, write_flow, BuiltinResult, Value, WriteElement};
use super::{WRITE_ERROR_INTERNAL, WRITE_ERROR_INVALID_DATA};
use runmat_value::{SparseTensor, Tensor};

pub(super) fn numeric_elements(
    value: &Value,
) -> BuiltinResult<Box<dyn Iterator<Item = WriteElement> + '_>> {
    let elements: Box<dyn Iterator<Item = WriteElement> + '_> = match value {
        Value::Tensor(tensor) => {
            Box::new((0..tensor.len()).map(move |index| tensor_element(tensor, index)))
        }
        Value::SparseTensor(sparse) => {
            let total = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                write_error_with_message(
                    "write: sparse matrix dimensions overflow",
                    &WRITE_ERROR_INTERNAL,
                )
            })?;
            if total > 10_000_000 {
                return Err(write_error_with_message(
                    format!("write: cannot densify sparse tensor {}x{} ({} elements exceeds safe threshold)", sparse.rows, sparse.cols, total),
                    &WRITE_ERROR_INTERNAL,
                ));
            }
            if sparse.is_complex() {
                return Err(write_flow(
                    &WRITE_ERROR_INVALID_DATA,
                    "write: complex data is not supported",
                ));
            }
            Box::new(SparseElements {
                sparse,
                index: 0,
                stored: 0,
                total,
            })
        }
        Value::Num(number) => Box::new(std::iter::once(WriteElement::Floating(*number))),
        Value::Int(integer) => Box::new(std::iter::once(WriteElement::Integer(integer.clone()))),
        Value::Bool(boolean) => Box::new(std::iter::once(WriteElement::Floating(if *boolean {
            1.0
        } else {
            0.0
        }))),
        Value::LogicalArray(array) => Box::new(
            array
                .data
                .iter()
                .map(|byte| WriteElement::Floating(if *byte != 0 { 1.0 } else { 0.0 })),
        ),
        Value::CharArray(array) => Box::new(
            array
                .data()
                .iter()
                .map(|ch| WriteElement::Floating((u32::from(*ch) & 0xff) as f64)),
        ),
        Value::String(runmat_value::StringElement::Text(text)) => Box::new(
            text.chars()
                .map(|ch| WriteElement::Floating(u32::from(ch) as f64)),
        ),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Box::new(
                array.data()[0]
                    .as_str()
                    .expect("checked nonmissing scalar string")
                    .chars()
                    .map(|ch| WriteElement::Floating(u32::from(ch) as f64)),
            )
        }
        Value::StringArray(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: string array input must be scalar",
            ))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: complex data is not supported",
            ))
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: symbolic data is not supported",
            ))
        }
        Value::GpuTensor(_) => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: GPU tensor should have been gathered before encoding",
            ))
        }
        _ => {
            return Err(write_flow(
                &WRITE_ERROR_INVALID_DATA,
                "write: unsupported input type",
            ))
        }
    };
    Ok(elements)
}

fn tensor_element(tensor: &Tensor, index: usize) -> WriteElement {
    let value = tensor
        .numeric_value_at(index)
        .expect("index within authoritative numeric storage");
    value.into_int_value().map_or_else(
        || WriteElement::Floating(value.materialize_f64()),
        WriteElement::Integer,
    )
}

struct SparseElements<'a> {
    sparse: &'a SparseTensor,
    index: usize,
    stored: usize,
    total: usize,
}

impl Iterator for SparseElements<'_> {
    type Item = WriteElement;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.total {
            return None;
        }
        let col = self.index / self.sparse.rows;
        let row = self.index % self.sparse.rows;
        self.index += 1;
        let end = self.sparse.col_ptrs[col + 1];
        if self.stored < end && self.sparse.row_indices[self.stored] == row {
            let value = self
                .sparse
                .numeric_value_at(self.stored)
                .expect("validated sparse stored value");
            self.stored += 1;
            Some(value.into_int_value().map_or_else(
                || WriteElement::Floating(value.materialize_f64()),
                WriteElement::Integer,
            ))
        } else {
            Some(WriteElement::Floating(0.0))
        }
    }
}

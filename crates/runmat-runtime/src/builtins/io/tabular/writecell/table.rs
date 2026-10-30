use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::builtins::io::tabular) enum CellValue {
    Empty,
    Number(f64),
    Single(f32),
    Integer(IntValue),
    Boolean(bool),
    Text(String),
}

#[derive(Debug, Clone)]
pub(in crate::builtins::io::tabular) struct CellTable {
    pub(in crate::builtins::io::tabular) rows: usize,
    pub(in crate::builtins::io::tabular) cols: usize,
    data: Vec<CellValue>,
}

impl CellTable {
    pub(in crate::builtins::io::tabular) fn from_cells(
        rows: usize,
        cols: usize,
        data: Vec<CellValue>,
    ) -> BuiltinResult<Self> {
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            writecell_error_with(&WRITECELL_ERROR_DATA, "writecell: cell table size overflow")
        })?;
        if data.len() != expected {
            return Err(writecell_error_with(
                &WRITECELL_ERROR_DATA,
                format!(
                    "writecell: cell table has {} values for {rows}-by-{cols} shape",
                    data.len()
                ),
            ));
        }
        Ok(Self { rows, cols, data })
    }

    pub(in crate::builtins::io::tabular) async fn from_value(
        runtime: Option<&crate::context::RuntimeContext>,
        cooperation: &mut TabularCooperation<'_>,
        value: Value,
    ) -> BuiltinResult<Self> {
        let cell = match value {
            Value::Cell(cell) => cell,
            other => {
                return Err(writecell_error_with(
                    &WRITECELL_ERROR_DATA,
                    format!("writecell: input must be a cell array, got {other:?}"),
                ));
            }
        };
        ensure_cell_shape(&cell)?;

        let mut data = Vec::with_capacity(cell.data().len());
        for row in 0..cell.rows {
            for col in 0..cell.cols {
                let value = cell.get(row, col).map_err(|message| {
                    writecell_error_with(&WRITECELL_ERROR_DATA, format!("writecell: {message}"))
                })?;
                let gathered = await_io(runtime, BUILTIN_NAME, gather_if_needed_async(&value))
                    .await?
                    .map_err(map_control_flow)?;
                cooperation.completed(1).await?;
                data.push(cell_value_from_value(gathered)?);
            }
        }
        Ok(Self {
            rows: cell.rows,
            cols: cell.cols,
            data,
        })
    }

    pub(in crate::builtins::io::tabular) fn get(&self, row: usize, col: usize) -> &CellValue {
        &self.data[row * self.cols + col]
    }
}

pub(super) fn ensure_cell_shape(cell: &CellArray) -> BuiltinResult<()> {
    if cell.shape.len() <= 2 || cell.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(writecell_error_with(
        &WRITECELL_ERROR_DATA_SHAPE,
        "writecell: input cell array must be 2-D",
    ))
}

pub(super) fn cell_value_from_value(value: Value) -> BuiltinResult<CellValue> {
    match value {
        Value::Num(n) => Ok(CellValue::Number(n)),
        Value::Int(i) => Ok(CellValue::Integer(i)),
        Value::Bool(b) => Ok(CellValue::Boolean(b)),
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(CellValue::Text(s.into_string_unbounded()))
        }
        value @ (Value::CharArray(_) | Value::StringArray(_))
            if crate::builtins::common::validation::value_is_text_scalar(&value) =>
        {
            Ok(CellValue::Text(
                crate::builtins::common::validation::value_text_scalar_string(&value)
                    .expect("admitted scalar text"),
            ))
        }
        Value::StringArray(sa) if sa.data().is_empty() => Ok(CellValue::Empty),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            scalar_tensor_cell_value(&tensor)
        }
        Value::Tensor(tensor) if tensor::tensor_element_len(&tensor) == 0 => Ok(CellValue::Empty),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(CellValue::Boolean(logical.data[0] != 0))
        }
        Value::LogicalArray(logical) if logical.data.is_empty() => Ok(CellValue::Empty),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            "writecell: complex values are not supported; split real and imaginary parts first",
        )),
        Value::Cell(_) => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            "writecell: nested cell arrays are not supported",
        )),
        other => Err(writecell_error_with(
            &WRITECELL_ERROR_DATA,
            format!("writecell: unsupported cell value {other:?}"),
        )),
    }
}

pub(in crate::builtins::io::tabular) fn scalar_tensor_cell_value(
    tensor: &runmat_value::Tensor,
) -> BuiltinResult<CellValue> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage.value_at(0).expect("one-element integer storage");
        Ok(CellValue::Integer(value))
    } else {
        Ok(
            match tensor
                .numeric_value_at(0)
                .expect("one-element numeric storage")
            {
                runmat_value::NumericScalar::F32(value) => CellValue::Single(value),
                value => CellValue::Number(value.materialize_f64()),
            },
        )
    }
}

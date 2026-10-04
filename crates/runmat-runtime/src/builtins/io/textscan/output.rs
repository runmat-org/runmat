use super::*;

pub(super) fn build_output(
    columns: Vec<ColumnData>,
    options: &TextscanOptions,
) -> BuiltinResult<Value> {
    let values = if options.collect_output {
        collect_output(columns)?
    } else {
        columns
            .into_iter()
            .map(column_to_value)
            .collect::<BuiltinResult<Vec<_>>>()?
    };
    let len = values.len();
    CellArray::new(values, 1, len)
        .map(Value::Cell)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

fn collect_output(columns: Vec<ColumnData>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < columns.len() {
        if matches!(columns[idx].kind(), OutputKind::Numeric(_)) {
            let start = idx;
            let kind = columns[idx].kind();
            while idx < columns.len() && columns[idx].kind() == kind {
                idx += 1;
            }
            out.push(numeric_group_to_value(&columns[start..idx])?);
        } else {
            out.push(column_to_value(columns[idx].clone())?);
            idx += 1;
        }
    }
    Ok(out)
}

fn column_to_value(column: ColumnData) -> BuiltinResult<Value> {
    match column {
        ColumnData::Numeric { dtype, values } => {
            let rows = values.len();
            tensor_from_numeric_scalars(dtype, values, vec![rows, 1])
        }
        ColumnData::Text(values) => cell_string_column(&values),
    }
}

fn numeric_group_to_value(columns: &[ColumnData]) -> BuiltinResult<Value> {
    let rows = columns.first().map(ColumnData::len).unwrap_or(0);
    let cols = columns.len();
    let Some(OutputKind::Numeric(dtype)) = columns.first().map(ColumnData::kind) else {
        unreachable!("numeric group has no numeric dtype");
    };
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        let ColumnData::Numeric {
            dtype: column_dtype,
            values,
        } = column
        else {
            unreachable!("numeric group contains text column");
        };
        debug_assert_eq!(*column_dtype, dtype);
        if values.len() != rows {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: collected numeric columns have inconsistent lengths",
            ));
        }
        data.extend_from_slice(values);
    }
    tensor_from_numeric_scalars(dtype, data, vec![rows, cols])
}

fn tensor_from_numeric_scalars(
    dtype: NumericDType,
    values: Vec<NumericScalar>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let mut storage = NumericStorage::zeros(dtype, values.len());
    for (index, value) in values.into_iter().enumerate() {
        storage.set_value(index, value).map_err(|err| {
            textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}"))
        })?;
    }
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

fn cell_string_column(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        values.len(),
        1,
    )
    .map(Value::Cell)
    .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

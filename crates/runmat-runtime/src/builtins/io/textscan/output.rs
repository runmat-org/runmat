use super::*;

pub(super) async fn build_output(
    columns: Vec<ColumnData>,
    options: &TextscanOptions,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Value> {
    let values = if options.collect_output {
        collect_output(columns, cooperation).await?
    } else {
        let mut values = Vec::with_capacity(columns.len());
        for column in columns {
            values.push(column_to_value(column, cooperation).await?);
        }
        values
    };
    cooperation.completed(0).await?;
    let len = values.len();
    CellArray::new(values, 1, len)
        .map(Value::Cell)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

async fn collect_output(
    columns: Vec<ColumnData>,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::new();
    let mut columns = columns.into_iter().peekable();
    while let Some(column) = columns.next() {
        match column.kind() {
            OutputKind::Numeric(kind) => {
                let mut group = vec![column];
                while columns
                    .peek()
                    .is_some_and(|next| next.kind() == OutputKind::Numeric(kind))
                {
                    group.push(columns.next().expect("peeked column"));
                }
                out.push(numeric_group_to_value(group, cooperation).await?);
            }
            OutputKind::Text => out.push(column_to_value(column, cooperation).await?),
        }
        cooperation.completed(1).await?;
    }
    Ok(out)
}

async fn column_to_value(
    column: ColumnData,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Value> {
    match column {
        ColumnData::Numeric { dtype, values } => {
            let rows = values.len();
            tensor_from_numeric_scalars(dtype, values, vec![rows, 1], cooperation).await
        }
        ColumnData::Text(values) => cell_string_column(values, cooperation).await,
    }
}

async fn numeric_group_to_value(
    columns: Vec<ColumnData>,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Value> {
    let rows = columns.first().map(ColumnData::len).unwrap_or(0);
    let cols = columns.len();
    let Some(OutputKind::Numeric(dtype)) = columns.first().map(ColumnData::kind) else {
        unreachable!("numeric group has no numeric dtype");
    };
    let count = rows.checked_mul(cols).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            "textscan: collected output shape overflows",
        )
    })?;
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|error| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_PARSE,
            "textscan: unable to reserve collected output",
            error,
        )
    })?;
    for column in columns {
        let ColumnData::Numeric {
            dtype: column_dtype,
            values,
        } = column
        else {
            unreachable!("numeric group contains text column");
        };
        debug_assert_eq!(column_dtype, dtype);
        if values.len() != rows {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: collected numeric columns have inconsistent lengths",
            ));
        }
        data.extend(values);
        cooperation.completed(rows).await?;
    }
    tensor_from_numeric_scalars(dtype, data, vec![rows, cols], cooperation).await
}

async fn tensor_from_numeric_scalars(
    dtype: NumericDType,
    values: Vec<NumericScalar>,
    shape: Vec<usize>,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Value> {
    let mut storage = NumericStorage::zeros(dtype, values.len());
    let mut work = 0usize;
    for (index, value) in values.into_iter().enumerate() {
        storage.set_value(index, value).map_err(|err| {
            textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}"))
        })?;
        work += 1;
        if work == 4096 {
            cooperation.completed(work).await?;
            work = 0;
        }
    }
    cooperation.completed(work).await?;
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

async fn cell_string_column(
    values: Vec<String>,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<Value> {
    let rows = values.len();
    let mut cells = Vec::with_capacity(rows);
    for value in values {
        cells.push(Value::string(value));
        if cells.len() % 4096 == 0 {
            cooperation.completed(4096).await?;
        }
    }
    cooperation.completed(rows % 4096).await?;
    CellArray::new(cells, rows, 1)
        .map(Value::Cell)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

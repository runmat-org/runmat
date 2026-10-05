use super::*;

#[derive(Debug, Clone)]
pub(super) struct XlsTable {
    rows: usize,
    cols: usize,
    cells: Vec<CellValue>,
}

impl XlsTable {
    pub(super) async fn preflight_host_known(
        value: &Value,
        cooperation: &mut TabularCooperation<'_>,
    ) -> BuiltinResult<()> {
        match value {
            Value::GpuTensor(_) => Ok(()),
            Value::Cell(cell) => {
                ensure_shape_2d(&cell.shape, "cell array")?;
                checked_cell_count(cell.rows, cell.cols, "cell array")?;
                for (index, member) in cell.data.iter().enumerate() {
                    if !matches!(member, Value::GpuTensor(_)) {
                        admit_cell_value(member)?;
                    }
                    if (index + 1) % 1024 == 0 {
                        cooperation.completed(1024).await?;
                    }
                }
                cooperation.completed(cell.data.len() % 1024).await
            }
            Value::Tensor(tensor) => {
                ensure_shape_2d(&tensor.shape, "numeric array")?;
                checked_cell_count(tensor.rows(), tensor.cols(), "numeric array")?;
                Ok(())
            }
            Value::SparseTensor(sparse) => {
                checked_cell_count(sparse.rows, sparse.cols, "sparse array")?;
                Ok(())
            }
            Value::LogicalArray(logical) => {
                ensure_shape_2d(&logical.shape, "logical array")?;
                let (rows, cols) = shape_rows_cols(&logical.shape);
                checked_cell_count(rows, cols, "logical array")?;
                Ok(())
            }
            Value::StringArray(strings) => {
                ensure_shape_2d(&strings.shape, "string array")?;
                checked_cell_count(strings.rows(), strings.cols(), "string array")?;
                Ok(())
            }
            Value::CharArray(chars) => {
                if chars.rows > 1 && chars.cols > 0 {
                    checked_cell_count(chars.rows, 1, "character array")?;
                }
                Ok(())
            }
            Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: complex values are not supported; split real and imaginary parts first",
            )),
            other => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                format!("xlswrite: unsupported data value {other:?}"),
            )),
        }
    }

    pub(super) async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Cell(cell) => Self::from_cell_array(cell).await,
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::SparseTensor(sparse) => Self::from_sparse(sparse),
            Value::LogicalArray(logical) => Self::from_logical(logical),
            Value::StringArray(strings) => Self::from_string_array(strings),
            Value::String(text) => Ok(Self::single(CellValue::Text(text))),
            Value::CharArray(chars) => Self::from_char_array(chars),
            Value::Num(value) => Ok(Self::single(CellValue::Number(value))),
            Value::Int(value) => Ok(Self::single(CellValue::Integer(value))),
            Value::Bool(value) => Ok(Self::single(CellValue::Boolean(value))),
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: complex values are not supported; split real and imaginary parts first",
            )),
            other => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                format!("xlswrite: unsupported data value {other:?}"),
            )),
        }
    }

    fn single(value: CellValue) -> Self {
        Self {
            rows: 1,
            cols: 1,
            cells: vec![value],
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            cells: Vec::new(),
        }
    }

    pub(super) fn from_calamine_range(
        range: &calamine::Range<SpreadsheetData>,
    ) -> BuiltinResult<Self> {
        let Some((start_row, start_col)) = range.start() else {
            return Ok(Self::empty());
        };
        let Some((end_row, end_col)) = range.end() else {
            return Ok(Self::empty());
        };
        let rows = usize::try_from(end_row + 1).map_err(|_| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: existing sheet row count is too large",
            )
        })?;
        let cols = usize::try_from(end_col + 1).map_err(|_| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: existing sheet column count is too large",
            )
        })?;
        checked_cell_count(rows, cols, "existing worksheet")?;
        let mut cells = vec![CellValue::Empty; rows * cols];
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                let row_usize = row as usize;
                let col_usize = col as usize;
                cells[row_usize * cols + col_usize] = spreadsheet_cell_to_cell_value(
                    range
                        .get_value((row, col))
                        .unwrap_or(&SpreadsheetData::Empty),
                );
            }
        }
        Ok(Self { rows, cols, cells })
    }

    pub(super) fn overlay(&mut self, start: RangeStart, incoming: &XlsTable) -> BuiltinResult<()> {
        let required_rows = start.row.checked_add(incoming.rows).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
        })?;
        let required_cols = start.col.checked_add(incoming.cols).ok_or_else(|| {
            xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
        })?;
        checked_cell_count(required_rows, required_cols, "worksheet")?;
        if required_rows > self.rows || required_cols > self.cols {
            self.resize(required_rows.max(self.rows), required_cols.max(self.cols))?;
        }
        for row in 0..incoming.rows {
            for col in 0..incoming.cols {
                let dest = (start.row + row) * self.cols + start.col + col;
                self.cells[dest] = incoming.cells[row * incoming.cols + col].clone();
            }
        }
        Ok(())
    }

    fn resize(&mut self, rows: usize, cols: usize) -> BuiltinResult<()> {
        checked_cell_count(rows, cols, "worksheet")?;
        let mut resized = vec![CellValue::Empty; rows * cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                resized[row * cols + col] = self.cells[row * self.cols + col].clone();
            }
        }
        self.rows = rows;
        self.cols = cols;
        self.cells = resized;
        Ok(())
    }

    async fn from_cell_array(cell: CellArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&cell.shape, "cell array")?;
        checked_cell_count(cell.rows, cell.cols, "cell array")?;
        let mut cells = Vec::with_capacity(cell.data.len());
        for row in 0..cell.rows {
            for col in 0..cell.cols {
                let value = cell.get(row, col).map_err(|message| {
                    xlswrite_error_with(&XLSWRITE_ERROR_DATA, format!("xlswrite: {message}"))
                })?;
                let gathered = gather_if_needed_async(&value)
                    .await
                    .map_err(map_control_flow)?;
                cells.push(cell_value_from_scalar(gathered)?);
            }
        }
        Ok(Self {
            rows: cell.rows,
            cols: cell.cols,
            cells,
        })
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        ensure_shape_2d(&tensor.shape, "numeric array")?;
        let rows = tensor.rows();
        let cols = tensor.cols();
        checked_cell_count(rows, cols, "numeric array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let value = tensor
                    .numeric_value_at(row + col * rows)
                    .expect("index within authoritative numeric storage");
                if let Some(value) = value.into_int_value() {
                    cells.push(CellValue::Integer(value));
                } else {
                    cells.push(CellValue::Number(value.materialize_f64()));
                }
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_sparse(sparse: SparseTensor) -> BuiltinResult<Self> {
        let rows = sparse.rows;
        let cols = sparse.cols;
        checked_cell_count(rows, cols, "sparse array")?;
        let mut cells = if sparse.is_logical() {
            vec![CellValue::Boolean(false); rows * cols]
        } else {
            vec![CellValue::Number(0.0); rows * cols]
        };
        if sparse.is_logical() {
            for col in 0..cols {
                for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                    let row = sparse.row_indices[entry];
                    cells[row * cols + col] = CellValue::Boolean(true);
                }
            }
        } else if let Some(storage) = sparse.integer_storage() {
            for col in 0..cols {
                let start = sparse.col_ptrs[col];
                let end = sparse.col_ptrs[col + 1];
                for entry in start..end {
                    let row = sparse.row_indices[entry];
                    let value = storage.value_at(entry).ok_or_else(|| {
                        xlswrite_error_with(
                            &XLSWRITE_ERROR_DATA,
                            format!("xlswrite: sparse integer value buffer missing entry {entry}"),
                        )
                    })?;
                    cells[row * cols + col] = CellValue::Integer(value);
                }
            }
        } else {
            for col in 0..cols {
                let start = sparse.col_ptrs[col];
                let end = sparse.col_ptrs[col + 1];
                for entry in start..end {
                    let row = sparse.row_indices[entry];
                    cells[row * cols + col] = CellValue::Number(
                        sparse
                            .numeric_value_at(entry)
                            .expect("sparse storage index is valid")
                            .materialize_f64(),
                    );
                }
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_logical(logical: LogicalArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&logical.shape, "logical array")?;
        let (rows, cols) = shape_rows_cols(&logical.shape);
        checked_cell_count(rows, cols, "logical array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Boolean(logical.data[row + col * rows] != 0));
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_string_array(strings: StringArray) -> BuiltinResult<Self> {
        ensure_shape_2d(&strings.shape, "string array")?;
        let rows = strings.rows();
        let cols = strings.cols();
        checked_cell_count(rows, cols, "string array")?;
        let mut cells = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Text(strings.data[row + col * rows].clone()));
            }
        }
        Ok(Self { rows, cols, cells })
    }

    fn from_char_array(chars: CharArray) -> BuiltinResult<Self> {
        if chars.rows == 0 || chars.cols == 0 {
            return Ok(Self {
                rows: 0,
                cols: 0,
                cells: Vec::new(),
            });
        }
        if chars.rows == 1 {
            return Ok(Self::single(CellValue::Text(chars.data.iter().collect())));
        }
        checked_cell_count(chars.rows, 1, "character array")?;
        let mut cells = Vec::with_capacity(chars.rows);
        for row in 0..chars.rows {
            let mut text = String::with_capacity(chars.cols);
            for col in 0..chars.cols {
                text.push(chars.data[row * chars.cols + col]);
            }
            cells.push(CellValue::Text(text));
        }
        Ok(Self {
            rows: chars.rows,
            cols: 1,
            cells,
        })
    }

    pub(super) fn into_cell_table(self) -> BuiltinResult<CellTable> {
        CellTable::from_cells(self.rows, self.cols, self.cells).map_err(map_writecell_error)
    }
}

fn cell_value_from_scalar(value: Value) -> BuiltinResult<CellValue> {
    admit_cell_value(&value)?;
    match value {
        Value::Num(n) => Ok(CellValue::Number(n)),
        Value::Int(i) => Ok(CellValue::Integer(i)),
        Value::Bool(b) => Ok(CellValue::Boolean(b)),
        Value::String(s) => Ok(CellValue::Text(s)),
        Value::CharArray(ca) if ca.rows == 1 => Ok(CellValue::Text(ca.data.iter().collect())),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(CellValue::Text(sa.data[0].clone())),
        Value::StringArray(sa) if sa.data.is_empty() => Ok(CellValue::Empty),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            writecell::scalar_tensor_cell_value(&tensor)
        }
        Value::Tensor(tensor) if tensor::tensor_element_len(&tensor) == 0 => Ok(CellValue::Empty),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(CellValue::Boolean(logical.data[0] != 0))
        }
        Value::LogicalArray(logical) if logical.data.is_empty() => Ok(CellValue::Empty),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: complex cell values are not supported",
        )),
        Value::Cell(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: nested cell arrays are not supported",
        )),
        other => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: unsupported cell value {other:?}"),
        )),
    }
}

fn admit_cell_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Ok(()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(()),
        Value::StringArray(strings) if strings.data.len() <= 1 => Ok(()),
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(tensor) || tensor::tensor_element_len(tensor) == 0 =>
        {
            Ok(())
        }
        Value::LogicalArray(logical) if logical.data.len() <= 1 => Ok(()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: complex cell values are not supported",
        )),
        Value::Cell(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: nested cell arrays are not supported",
        )),
        other => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: unsupported cell value {other:?}"),
        )),
    }
}

fn ensure_shape_2d(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 || shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} must be 2-D"),
        ))
    }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn spreadsheet_cell_to_cell_value(cell: &SpreadsheetData) -> CellValue {
    match cell {
        SpreadsheetData::Empty => CellValue::Empty,
        SpreadsheetData::Int(value) => CellValue::Number(*value as f64),
        SpreadsheetData::Float(value) => CellValue::Number(*value),
        SpreadsheetData::String(value) => CellValue::Text(value.clone()),
        SpreadsheetData::Bool(value) => CellValue::Boolean(*value),
        SpreadsheetData::DateTime(value) => CellValue::Number(value.as_f64()),
        SpreadsheetData::DateTimeIso(value) | SpreadsheetData::DurationIso(value) => {
            CellValue::Text(value.clone())
        }
        SpreadsheetData::Error(value) => CellValue::Text(value.to_string()),
    }
}

fn checked_cell_count(rows: usize, cols: usize, context: &str) -> BuiltinResult<usize> {
    if rows > MAX_EXCEL_ROW_INDEX + 1 || cols > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} exceeds Excel worksheet limits"),
        ));
    }
    let cells = rows.checked_mul(cols).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} cell count overflows supported bounds"),
        )
    })?;
    if cells > MAX_XLSWRITE_CELLS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!(
                "xlswrite: {context} exceeds maximum supported size of {MAX_XLSWRITE_CELLS} cells"
            ),
        ));
    }
    Ok(cells)
}

pub(super) fn validate_write_bounds(table: &XlsTable, start: RangeStart) -> BuiltinResult<()> {
    let end_row = start.row.checked_add(table.rows).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
    })?;
    let end_col = start.col.checked_add(table.cols).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
    })?;
    if end_row > MAX_EXCEL_ROW_INDEX + 1 || end_col > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range exceeds Excel worksheet limits",
        ));
    }
    Ok(())
}

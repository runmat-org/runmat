use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensor {
    pub rows: usize,
    pub cols: usize,
    /// Column pointers into `row_indices` and the numeric value storage; length is `cols + 1`.
    pub col_ptrs: Vec<usize>,
    /// Zero-based row indices, sorted within each column.
    pub row_indices: Vec<usize>,
    storage: SparseValueStorage,
}

#[derive(Debug, Clone, PartialEq)]
enum SparseValueStorage {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer(IntegerStorage),
    Logical,
}

impl fmt::Display for SparseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}x{} {} sparse matrix with {} nonzero entries",
            self.rows,
            self.cols,
            self.class_name(),
            self.nnz()
        )?;
        if self.nnz() == 0 {
            return Ok(());
        }
        for col in 0..self.cols {
            for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                let row = self.row_indices[idx];
                let value = match &self.storage {
                    SparseValueStorage::F64(values) => format_number(values[idx]),
                    SparseValueStorage::F32(values) => format_number(f64::from(values[idx])),
                    SparseValueStorage::Integer(storage) => storage
                        .value_at(idx)
                        .expect("validated sparse storage")
                        .decimal_string(),
                    SparseValueStorage::Logical => "1".to_string(),
                };
                writeln!(f, "  ({},{})  {}", row + 1, col + 1, value)?;
            }
        }
        Ok(())
    }
}

type SparseCscParts<T> = (Vec<usize>, Vec<usize>, Vec<T>);

impl SparseTensor {
    pub fn new(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::F64(values),
        })
    }

    /// Constructs a sparse matrix backed by native single-precision values.
    pub fn new_f32(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::F32(values),
        })
    }

    /// Constructs a sparse matrix backed by an exact integer value buffer.
    pub fn new_integer(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        integer_data: IntegerStorage,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, integer_data.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Integer(integer_data),
        })
    }

    /// Constructs a sparse logical matrix whose CSC pattern is the complete
    /// authoritative set of true elements.
    pub fn new_logical(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, row_indices.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Logical,
        })
    }

    fn validate_structure(
        rows: usize,
        cols: usize,
        col_ptrs: &[usize],
        row_indices: &[usize],
        values_len: usize,
    ) -> Result<(), String> {
        if col_ptrs.len() != cols.saturating_add(1) {
            return Err(format!(
                "SparseTensor col_ptrs length {} doesn't match cols {}",
                col_ptrs.len(),
                cols
            ));
        }
        if row_indices.len() != values_len {
            return Err(format!(
                "SparseTensor row index length {} doesn't match value length {}",
                row_indices.len(),
                values_len
            ));
        }
        if col_ptrs.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err("SparseTensor col_ptrs must start at 0".to_string());
        }
        if col_ptrs.last().copied().unwrap_or(usize::MAX) != values_len {
            return Err("SparseTensor final col_ptr must equal nnz".to_string());
        }
        for window in col_ptrs.windows(2) {
            if window[0] > window[1] {
                return Err("SparseTensor col_ptrs must be nondecreasing".to_string());
            }
        }
        for col in 0..cols {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];
            let mut prev: Option<usize> = None;
            for &row in &row_indices[start..end] {
                if row >= rows {
                    return Err(format!("SparseTensor row index {row} exceeds rows {rows}"));
                }
                if prev.is_some_and(|p| p >= row) {
                    return Err("SparseTensor row indices must be sorted and unique".to_string());
                }
                prev = Some(row);
            }
        }
        Ok(())
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            storage: SparseValueStorage::F64(Vec::new()),
        }
    }

    /// Creates an all-zero sparse matrix retaining the `single` class.
    pub fn zeros_f32(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            storage: SparseValueStorage::F32(Vec::new()),
        }
    }

    /// Creates an all-false sparse logical matrix.
    pub fn zeros_logical(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            storage: SparseValueStorage::Logical,
        }
    }

    /// Creates an all-zero sparse matrix retaining an exact integer class.
    pub fn zeros_with_integer_storage(rows: usize, cols: usize, storage: &IntegerStorage) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            storage: SparseValueStorage::Integer(storage.zeros_like(0)),
        }
    }

    /// Creates a typed sparse matrix using the class of `prototype`.
    pub fn new_integer_like(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<IntValue>,
        prototype: &IntegerStorage,
    ) -> Result<Self, String> {
        Self::new_integer(
            rows,
            cols,
            col_ptrs,
            row_indices,
            prototype.from_same_class_values(values)?,
        )
    }

    pub fn nnz(&self) -> usize {
        match &self.storage {
            SparseValueStorage::F64(values) => values.len(),
            SparseValueStorage::F32(values) => values.len(),
            SparseValueStorage::Integer(storage) => storage.len(),
            SparseValueStorage::Logical => self.row_indices.len(),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }

    pub fn to_dense(&self) -> Result<Tensor, String> {
        let len = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor dense dimensions overflow usize".to_string())?;
        match &self.storage {
            SparseValueStorage::F64(values) => {
                let mut data = Vec::new();
                data.try_reserve_exact(len)
                    .map_err(|err| format!("SparseTensor dense allocation failed: {err}"))?;
                data.resize(len, 0.0);
                for col in 0..self.cols {
                    for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        data[self.row_indices[idx] + col * self.rows] = values[idx];
                    }
                }
                Tensor::new(data, self.shape())
            }
            SparseValueStorage::F32(values) => {
                let mut data = Vec::new();
                data.try_reserve_exact(len)
                    .map_err(|err| format!("SparseTensor dense allocation failed: {err}"))?;
                data.resize(len, 0.0);
                for col in 0..self.cols {
                    for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        data[self.row_indices[idx] + col * self.rows] = values[idx];
                    }
                }
                Tensor::from_f32(data, self.shape())
            }
            SparseValueStorage::Integer(integer_data) => {
                let mut data = integer_data.zeros_like(len);
                for col in 0..self.cols {
                    for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        let row = self.row_indices[idx];
                        let value = integer_data.value_at(idx).ok_or_else(|| {
                            "SparseTensor integer storage is inconsistent".to_string()
                        })?;
                        data.set_value(row + col * self.rows, value)?;
                    }
                }
                Tensor::new_integer(data, self.shape())
            }
            SparseValueStorage::Logical => {
                Err("SparseTensor logical storage requires to_dense_logical".to_string())
            }
        }
    }

    pub fn to_dense_logical(&self) -> Result<LogicalArray, String> {
        if !self.is_logical() {
            return Err("SparseTensor numeric storage requires to_dense".to_string());
        }
        let len = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor dense dimensions overflow usize".to_string())?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|err| format!("SparseTensor dense allocation failed: {err}"))?;
        data.resize(len, 0);
        for col in 0..self.cols {
            for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                data[self.row_indices[idx] + col * self.rows] = 1;
            }
        }
        LogicalArray::new(data, self.shape())
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| {
                let index = start + offset;
                match &self.storage {
                    SparseValueStorage::F64(values) => values[index],
                    SparseValueStorage::F32(values) => f64::from(values[index]),
                    SparseValueStorage::Integer(storage) => storage
                        .value_at(index)
                        .expect("validated sparse storage index")
                        .to_f64(),
                    SparseValueStorage::Logical => 1.0,
                }
            })
    }

    pub fn logical_at(&self, row: usize, col: usize) -> Option<bool> {
        if !self.is_logical() || row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        Some(self.row_indices[start..end].binary_search(&row).is_ok())
    }

    /// Returns an exact stored integer value when this sparse matrix is typed.
    pub fn integer_at(&self, row: usize, col: usize) -> Option<IntValue> {
        let integer_data = self.integer_storage()?;
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .and_then(|offset| integer_data.value_at(start + offset))
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        match &self.storage {
            SparseValueStorage::Integer(storage) => Some(storage),
            SparseValueStorage::F64(_)
            | SparseValueStorage::F32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    /// Borrows stored nonzero values when this sparse matrix is double.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match &self.storage {
            SparseValueStorage::F64(values) => Some(values),
            SparseValueStorage::F32(_)
            | SparseValueStorage::Integer(_)
            | SparseValueStorage::Logical => None,
        }
    }

    /// Borrows stored nonzero values when this sparse matrix is single.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            SparseValueStorage::F32(values) => Some(values),
            SparseValueStorage::F64(_)
            | SparseValueStorage::Integer(_)
            | SparseValueStorage::Logical => None,
        }
    }

    pub fn is_logical(&self) -> bool {
        matches!(self.storage, SparseValueStorage::Logical)
    }

    pub fn value_byte_size(&self) -> usize {
        match &self.storage {
            SparseValueStorage::Logical => 0,
            _ => self
                .numeric_dtype()
                .expect("non-logical sparse storage has a numeric dtype")
                .byte_size(),
        }
    }

    /// Explicitly materializes stored nonzero values in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_f64(&self) -> Vec<f64> {
        match &self.storage {
            SparseValueStorage::F64(values) => values.clone(),
            SparseValueStorage::F32(values) => values.iter().copied().map(f64::from).collect(),
            SparseValueStorage::Integer(storage) => storage.to_f64_vec(),
            SparseValueStorage::Logical => vec![1.0; self.nnz()],
        }
    }

    /// Reads one stored nonzero value without routing integers through floating point.
    pub fn numeric_value_at(&self, index: usize) -> Option<NumericScalar> {
        match &self.storage {
            SparseValueStorage::F64(values) => values.get(index).copied().map(NumericScalar::F64),
            SparseValueStorage::F32(values) => values.get(index).copied().map(NumericScalar::F32),
            SparseValueStorage::Integer(storage) => {
                storage.value_at(index).map(NumericScalar::from)
            }
            SparseValueStorage::Logical => (index < self.nnz()).then_some(NumericScalar::F64(1.0)),
        }
    }

    pub fn numeric_dtype(&self) -> Option<NumericDType> {
        match &self.storage {
            SparseValueStorage::F64(_) => Some(NumericDType::F64),
            SparseValueStorage::F32(_) => Some(NumericDType::F32),
            SparseValueStorage::Integer(storage) => Some(storage.numeric_dtype()),
            SparseValueStorage::Logical => None,
        }
    }

    fn merged_linear_updates<T: Clone>(
        &self,
        updates: &[(usize, T)],
        mut stored_value: impl FnMut(usize) -> Result<T, String>,
        is_zero: impl Fn(&T) -> bool,
    ) -> Result<SparseCscParts<T>, String> {
        let total = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor assignment dimensions overflow usize".to_string())?;
        let mut latest = BTreeMap::new();
        for (index, value) in updates {
            if *index >= total {
                return Err(format!(
                    "SparseTensor assignment linear index {} exceeds {} elements",
                    index, total
                ));
            }
            latest.insert(*index, value.clone());
        }

        let capacity = self
            .nnz()
            .checked_add(latest.len())
            .ok_or_else(|| "SparseTensor assignment nnz overflow".to_string())?;
        let mut col_ptrs = Vec::with_capacity(self.cols.saturating_add(1));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        row_indices
            .try_reserve_exact(capacity)
            .map_err(|error| format!("SparseTensor assignment allocation failed: {error}"))?;
        values
            .try_reserve_exact(capacity)
            .map_err(|error| format!("SparseTensor assignment allocation failed: {error}"))?;
        col_ptrs.push(0);

        for col in 0..self.cols {
            let column_start = col * self.rows;
            let column_end = column_start + self.rows;
            let mut stored = self.col_ptrs[col];
            let stored_end = self.col_ptrs[col + 1];
            for (&linear, value) in latest.range(column_start..column_end) {
                let row = linear - column_start;
                while stored < stored_end && self.row_indices[stored] < row {
                    row_indices.push(self.row_indices[stored]);
                    values.push(stored_value(stored)?);
                    stored += 1;
                }
                if stored < stored_end && self.row_indices[stored] == row {
                    stored += 1;
                }
                if !is_zero(value) {
                    row_indices.push(row);
                    values.push(value.clone());
                }
            }
            while stored < stored_end {
                row_indices.push(self.row_indices[stored]);
                values.push(stored_value(stored)?);
                stored += 1;
            }
            col_ptrs.push(values.len());
        }
        Ok((col_ptrs, row_indices, values))
    }

    /// Applies floating updates in one CSC merge. Repeated indices use the
    /// final assignment value and zeros are elided without densifying.
    pub fn with_updated_linear_values(&self, updates: &[(usize, f64)]) -> Result<Self, String> {
        let stored_values = self.as_f64_slice().ok_or_else(|| {
            "cannot assign floating sparse value to typed integer storage".to_string()
        })?;
        let (col_ptrs, row_indices, values) = self.merged_linear_updates(
            updates,
            |index| {
                stored_values
                    .get(index)
                    .copied()
                    .ok_or_else(|| "SparseTensor double storage is inconsistent".to_string())
            },
            |value| *value == 0.0,
        )?;
        Self::new(self.rows, self.cols, col_ptrs, row_indices, values)
    }

    /// Applies native single-precision updates in one CSC merge.
    pub fn with_updated_f32_linear_values(&self, updates: &[(usize, f32)]) -> Result<Self, String> {
        let stored_values = self
            .as_f32_slice()
            .ok_or_else(|| "cannot assign single sparse value to non-single storage".to_string())?;
        let (col_ptrs, row_indices, values) = self.merged_linear_updates(
            updates,
            |index| {
                stored_values
                    .get(index)
                    .copied()
                    .ok_or_else(|| "SparseTensor single storage is inconsistent".to_string())
            },
            |value| *value == 0.0,
        )?;
        Self::new_f32(self.rows, self.cols, col_ptrs, row_indices, values)
    }

    /// Applies exact integer updates in one CSC merge. Values must already be
    /// in this sparse matrix's class; coercion belongs to the VM layer.
    pub fn with_updated_integer_linear_values(
        &self,
        updates: &[(usize, IntValue)],
    ) -> Result<Self, String> {
        let storage = self
            .integer_storage()
            .ok_or_else(|| "cannot assign integer sparse value to floating storage".to_string())?;
        let (col_ptrs, row_indices, values) = self.merged_linear_updates(
            updates,
            |index| {
                storage
                    .value_at(index)
                    .ok_or_else(|| "SparseTensor integer storage is inconsistent".to_string())
            },
            IntValue::is_zero,
        )?;
        Self::new_integer_like(self.rows, self.cols, col_ptrs, row_indices, values, storage)
    }

    pub fn with_updated_logical_linear_values(
        &self,
        updates: &[(usize, bool)],
    ) -> Result<Self, String> {
        if !self.is_logical() {
            return Err("cannot assign logical sparse value to numeric storage".to_string());
        }
        let (col_ptrs, row_indices, _) =
            self.merged_linear_updates(updates, |_| Ok(true), |value| !*value)?;
        Self::new_logical(self.rows, self.cols, col_ptrs, row_indices)
    }

    pub fn with_updated_value(&self, row: usize, col: usize, value: f64) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_linear_values(&[(index, value)])
    }

    pub fn with_updated_f32_value(
        &self,
        row: usize,
        col: usize,
        value: f32,
    ) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_f32_linear_values(&[(index, value)])
    }

    pub fn with_updated_integer_value(
        &self,
        row: usize,
        col: usize,
        value: IntValue,
    ) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_integer_linear_values(&[(index, value)])
    }

    pub fn with_updated_logical_value(
        &self,
        row: usize,
        col: usize,
        value: bool,
    ) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_logical_linear_values(&[(index, value)])
    }

    /// Expands sparse dimensions without materializing implicit zero entries.
    pub fn with_expanded_shape(&self, rows: usize, cols: usize) -> Result<Self, String> {
        if rows < self.rows || cols < self.cols {
            return Err(format!(
                "SparseTensor cannot shrink shape ({}, {}) to ({rows}, {cols})",
                self.rows, self.cols
            ));
        }
        let mut col_ptrs = self.col_ptrs.clone();
        col_ptrs.resize(
            cols.checked_add(1)
                .ok_or_else(|| "SparseTensor expanded column count overflow".to_string())?,
            self.nnz(),
        );
        match &self.storage {
            SparseValueStorage::F64(values) => Self::new(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                values.clone(),
            ),
            SparseValueStorage::F32(values) => Self::new_f32(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                values.clone(),
            ),
            SparseValueStorage::Integer(storage) => Self::new_integer(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                storage.clone(),
            ),
            SparseValueStorage::Logical => {
                Self::new_logical(rows, cols, col_ptrs, self.row_indices.clone())
            }
        }
    }

    fn checked_assignment_linear_index(&self, row: usize, col: usize) -> Result<usize, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "SparseTensor assignment index ({}, {}) exceeds shape ({}, {})",
                row, col, self.rows, self.cols
            ));
        }
        col.checked_mul(self.rows)
            .and_then(|base| base.checked_add(row))
            .ok_or_else(|| "SparseTensor assignment linear index overflow".to_string())
    }

    fn checked_deletion_indices(
        indices: &[usize],
        bound: usize,
        axis: &str,
    ) -> Result<Vec<usize>, String> {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(format!(
                    "SparseTensor {axis} deletion indices must be unique"
                ));
            }
        }
        if sorted.iter().any(|&index| index >= bound) {
            return Err(format!(
                "SparseTensor {axis} deletion index exceeds dimension"
            ));
        }
        Ok(sorted)
    }

    fn rebuilt_csc<T: Clone>(
        &self,
        source_columns: &[usize],
        mut map_row: impl FnMut(usize) -> Option<usize>,
        mut stored_value: impl FnMut(usize) -> Result<T, String>,
    ) -> Result<SparseCscParts<T>, String> {
        let mut col_ptrs = Vec::new();
        col_ptrs
            .try_reserve_exact(
                source_columns
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| "SparseTensor deletion column count overflow".to_string())?,
            )
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        row_indices
            .try_reserve_exact(self.nnz())
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        values
            .try_reserve_exact(self.nnz())
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        col_ptrs.push(0);
        for &source_column in source_columns {
            let start = self.col_ptrs[source_column];
            let end = self.col_ptrs[source_column + 1];
            for index in start..end {
                if let Some(row) = map_row(self.row_indices[index]) {
                    row_indices.push(row);
                    values.push(stored_value(index)?);
                }
            }
            col_ptrs.push(values.len());
        }
        Ok((col_ptrs, row_indices, values))
    }

    /// Deletes complete sparse matrix rows without materializing dense storage.
    pub fn with_deleted_rows(&self, rows: &[usize]) -> Result<Self, String> {
        let rows = Self::checked_deletion_indices(rows, self.rows, "row")?;
        let source_columns = (0..self.cols).collect::<Vec<_>>();
        let output_rows = self
            .rows
            .checked_sub(rows.len())
            .ok_or_else(|| "SparseTensor deletion row count underflow".to_string())?;
        let map_row = |row| match rows.binary_search(&row) {
            Ok(_) => None,
            Err(removed_before) => Some(row - removed_before),
        };
        match &self.storage {
            SparseValueStorage::F64(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.get(index).copied().ok_or_else(|| {
                            "SparseTensor double storage is inconsistent".to_string()
                        })
                    })?;
                Self::new(output_rows, self.cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::F32(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.get(index).copied().ok_or_else(|| {
                            "SparseTensor single storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_f32(output_rows, self.cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::Integer(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.value_at(index).ok_or_else(|| {
                            "SparseTensor integer storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_integer_like(
                    output_rows,
                    self.cols,
                    col_ptrs,
                    row_indices,
                    values,
                    storage,
                )
            }
            SparseValueStorage::Logical => {
                let (col_ptrs, row_indices, _) =
                    self.rebuilt_csc(&source_columns, map_row, |_| Ok(true))?;
                Self::new_logical(output_rows, self.cols, col_ptrs, row_indices)
            }
        }
    }

    /// Deletes complete sparse matrix columns without materializing dense storage.
    pub fn with_deleted_columns(&self, columns: &[usize]) -> Result<Self, String> {
        let columns = Self::checked_deletion_indices(columns, self.cols, "column")?;
        let source_columns = (0..self.cols)
            .filter(|column| columns.binary_search(column).is_err())
            .collect::<Vec<_>>();
        match &self.storage {
            SparseValueStorage::F64(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.get(index).copied().ok_or_else(|| {
                            "SparseTensor double storage is inconsistent".to_string()
                        })
                    })?;
                Self::new(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    values,
                )
            }
            SparseValueStorage::F32(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.get(index).copied().ok_or_else(|| {
                            "SparseTensor single storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_f32(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    values,
                )
            }
            SparseValueStorage::Integer(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.value_at(index).ok_or_else(|| {
                            "SparseTensor integer storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_integer_like(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    values,
                    storage,
                )
            }
            SparseValueStorage::Logical => {
                let (col_ptrs, row_indices, _) =
                    self.rebuilt_csc(&source_columns, Some, |_| Ok(true))?;
                Self::new_logical(self.rows, source_columns.len(), col_ptrs, row_indices)
            }
        }
    }

    pub fn class_name(&self) -> &'static str {
        self.numeric_dtype()
            .map(NumericDType::class_name)
            .unwrap_or("logical")
    }
}

#[cfg(test)]
mod sparse_tensor_tests {
    use super::*;

    #[test]
    fn typed_sparse_scalar_updates_preserve_exact_values_and_zero_elision() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("sparse");
        let inserted = sparse
            .with_updated_integer_value(1, 1, IntValue::U64(9_223_372_036_854_775_808))
            .expect("insert");
        assert_eq!(inserted.col_ptrs, vec![0, 1, 2]);
        assert_eq!(inserted.row_indices, vec![0, 1]);
        assert_eq!(
            inserted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                u64::MAX,
                9_223_372_036_854_775_808
            ]))
        );

        let removed = inserted
            .with_updated_integer_value(0, 0, IntValue::U64(0))
            .expect("remove");
        assert_eq!(removed.col_ptrs, vec![0, 0, 1]);
        assert_eq!(removed.row_indices, vec![1]);
        assert_eq!(
            removed.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808]))
        );
    }

    #[test]
    fn floating_sparse_scalar_updates_keep_csc_order_and_elide_zero() {
        let sparse =
            SparseTensor::new(3, 1, vec![0, 2], vec![0, 2], vec![1.0, 3.0]).expect("sparse");
        let inserted = sparse.with_updated_value(1, 0, 2.0).expect("insert");
        assert_eq!(inserted.row_indices, vec![0, 1, 2]);
        assert_eq!(inserted.as_f64_slice(), Some(&[1.0, 2.0, 3.0][..]));

        let removed = inserted.with_updated_value(1, 0, 0.0).expect("remove");
        assert_eq!(removed.row_indices, vec![0, 2]);
        assert_eq!(removed.as_f64_slice(), Some(&[1.0, 3.0][..]));
    }

    #[test]
    fn single_sparse_storage_survives_dense_and_structural_paths() {
        let sparse = SparseTensor::new_f32(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.25, 3.5, 2.0, 4.0, 5.75],
        )
        .expect("single sparse");
        assert_eq!(sparse.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(sparse.class_name(), "single");
        assert_eq!(sparse.numeric_value_at(1), Some(NumericScalar::F32(3.5)));
        assert_eq!(
            sparse.as_f32_slice(),
            Some(&[1.25, 3.5, 2.0, 4.0, 5.75][..])
        );
        assert!(sparse.as_f64_slice().is_none());

        let dense = sparse.to_dense().expect("dense single");
        assert_eq!(dense.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            dense.as_f32_slice(),
            Some(&[1.25, 0.0, 3.5, 0.0, 2.0, 0.0, 4.0, 0.0, 5.75][..])
        );

        let updated = sparse
            .with_updated_f32_value(2, 0, 0.0)
            .expect("remove single");
        assert_eq!(updated.row_indices, vec![0, 1, 0, 2]);
        assert_eq!(updated.as_f32_slice(), Some(&[1.25, 2.0, 4.0, 5.75][..]));

        let expanded = sparse.with_expanded_shape(4, 4).expect("expand single");
        assert_eq!(expanded.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(expanded.col_ptrs, vec![0, 2, 3, 5, 5]);
        assert_eq!(expanded.as_f32_slice(), sparse.as_f32_slice());

        let rows = sparse.with_deleted_rows(&[1]).expect("delete row");
        assert_eq!(rows.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(rows.shape(), vec![2, 3]);
        assert_eq!(rows.as_f32_slice(), Some(&[1.25, 3.5, 4.0, 5.75][..]));

        let columns = sparse.with_deleted_columns(&[1]).expect("delete column");
        assert_eq!(columns.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(columns.shape(), vec![3, 2]);
        assert_eq!(columns.as_f32_slice(), Some(&[1.25, 3.5, 4.0, 5.75][..]));
    }

    #[test]
    fn sparse_structural_deletion_preserves_csc_and_exact_integer_values() {
        let sparse = SparseTensor::new_integer(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            IntegerStorage::U64(vec![1, u64::MAX, 9_223_372_036_854_775_808, 4, 5]),
        )
        .expect("sparse");

        let without_middle_row = sparse.with_deleted_rows(&[1]).expect("delete row");
        assert_eq!(without_middle_row.shape(), vec![2, 3]);
        assert_eq!(without_middle_row.col_ptrs, vec![0, 2, 2, 4]);
        assert_eq!(without_middle_row.row_indices, vec![0, 1, 0, 1]);
        assert_eq!(
            without_middle_row.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX, 4, 5]))
        );

        let without_outer_columns = sparse
            .with_deleted_columns(&[0, 2])
            .expect("delete columns");
        assert_eq!(without_outer_columns.shape(), vec![3, 1]);
        assert_eq!(without_outer_columns.col_ptrs, vec![0, 1]);
        assert_eq!(without_outer_columns.row_indices, vec![1]);
        assert_eq!(
            without_outer_columns.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808]))
        );

        assert!(sparse.with_deleted_rows(&[1, 1]).is_err());
        assert!(sparse.with_deleted_columns(&[3]).is_err());
    }

    #[test]
    fn logical_sparse_pattern_is_authoritative_across_core_structural_paths() {
        let sparse =
            SparseTensor::new_logical(3, 3, vec![0, 2, 3, 4], vec![0, 2, 1, 2]).expect("logical");
        assert!(sparse.is_logical());
        assert_eq!(sparse.numeric_dtype(), None);
        assert_eq!(sparse.class_name(), "logical");
        assert_eq!(sparse.nnz(), 4);
        assert_eq!(sparse.logical_at(2, 0), Some(true));
        assert_eq!(sparse.logical_at(0, 1), Some(false));
        assert_eq!(
            sparse.to_dense_logical().expect("dense").data,
            vec![1, 0, 1, 0, 1, 0, 0, 0, 1]
        );

        let updated = sparse
            .with_updated_logical_value(1, 0, true)
            .expect("insert true")
            .with_updated_logical_value(2, 0, false)
            .expect("remove false");
        assert_eq!(updated.row_indices, vec![0, 1, 1, 2]);

        let expanded = updated.with_expanded_shape(4, 4).expect("expand");
        assert!(expanded.is_logical());
        assert_eq!(expanded.col_ptrs, vec![0, 2, 3, 4, 4]);

        let rows = sparse.with_deleted_rows(&[1]).expect("delete row");
        assert!(rows.is_logical());
        assert_eq!(rows.shape(), vec![2, 3]);
        assert_eq!(rows.row_indices, vec![0, 1, 1]);

        let columns = sparse.with_deleted_columns(&[1]).expect("delete column");
        assert!(columns.is_logical());
        assert_eq!(columns.shape(), vec![3, 2]);
        assert_eq!(columns.row_indices, vec![0, 2, 2]);
    }

    #[test]
    fn sparse_expansion_preserves_csc_and_integer_storage() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
        )
        .expect("sparse");
        let expanded = sparse.with_expanded_shape(4, 4).expect("expand");
        assert_eq!(expanded.shape(), vec![4, 4]);
        assert_eq!(expanded.col_ptrs, vec![0, 1, 2, 2, 2]);
        assert_eq!(expanded.row_indices, vec![1, 0]);
        assert_eq!(expanded.integer_storage(), sparse.integer_storage());
        assert!(expanded.with_expanded_shape(1, 4).is_err());
    }

    #[test]
    fn sparse_display_reports_exact_integer_class_and_values() {
        let sparse = SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");
        let text = sparse.to_string();

        assert!(text.contains("2x1 uint64 sparse matrix with 1 nonzero entries"));
        assert!(text.contains("18446744073709551615"));
        assert!(!text.contains("18446744073709552000"));
    }

    #[test]
    fn sparse_compatibility_reads_derive_from_authoritative_integer_storage() {
        let unsigned = SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");
        assert_eq!(unsigned.get(1, 0), Some(u64::MAX as f64));
        let dense = unsigned.to_dense().expect("dense uint64 sparse");
        assert_eq!(
            dense.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, u64::MAX]))
        );

        let signed = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        )
        .expect("int64 sparse");
        assert_eq!(signed.get(0, 0), Some(i64::MIN as f64));
        assert_eq!(signed.get(1, 1), Some(i64::MAX as f64));
        let text = signed.to_string();
        assert!(text.contains("-9223372036854775808"));
        assert!(text.contains("9223372036854775807"));
    }

    #[test]
    fn to_dense_rejects_overflowing_dimensions() {
        let sparse = SparseTensor {
            rows: usize::MAX,
            cols: 2,
            col_ptrs: vec![0, 0, 0],
            row_indices: Vec::new(),
            storage: SparseValueStorage::F64(Vec::new()),
        };

        let err = sparse.to_dense().unwrap_err();
        assert!(err.contains("overflow"));
    }
}

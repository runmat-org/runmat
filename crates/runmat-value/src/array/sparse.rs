use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensor {
    pub rows: usize,
    pub cols: usize,
    /// Column pointers into `row_indices` and the numeric value storage; length is `cols + 1`.
    pub col_ptrs: HostIndexBuffer,
    /// Zero-based row indices, sorted within each column.
    pub row_indices: HostIndexBuffer,
    storage: SparseValueStorage,
}

#[derive(Debug, Clone, PartialEq)]
enum SparseValueStorage {
    Numeric(HostNumericBuffer),
    ComplexF64(HostComplexBuffer<f64>),
    ComplexF32(HostComplexBuffer<f32>),
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
                    SparseValueStorage::Numeric(storage) => format_sparse_scalar(
                        storage.value_at(idx).expect("validated sparse storage"),
                    ),
                    SparseValueStorage::ComplexF64(storage) => {
                        let ComplexElement(real, imaginary) = storage[idx];
                        Value::Complex(real, imaginary).to_string()
                    }
                    SparseValueStorage::ComplexF32(storage) => {
                        let ComplexElement(real, imaginary) = storage[idx];
                        Value::Complex(f64::from(real), f64::from(imaginary)).to_string()
                    }
                    SparseValueStorage::Logical => "1".to_string(),
                };
                writeln!(f, "  ({},{})  {}", row + 1, col + 1, value)?;
            }
        }
        Ok(())
    }
}

type SparseCscParts<T> = (Vec<usize>, Vec<usize>, Vec<T>);

fn format_sparse_scalar(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => format_number(value),
        NumericScalar::F32(value) => format_number(f64::from(value)),
        value => value
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .decimal_string(),
    }
}

fn sparse_scalar_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        value => value
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .to_f64(),
    }
}

impl SparseTensor {
    pub fn new(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        let col_ptrs = col_ptrs.into();
        let row_indices = row_indices.into();
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F64(values),
            )),
        })
    }

    /// Constructs a sparse matrix backed by native single-precision values.
    pub fn new_f32(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        values: Vec<f32>,
    ) -> Result<Self, String> {
        let col_ptrs = col_ptrs.into();
        let row_indices = row_indices.into();
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F32(values),
            )),
        })
    }

    /// Constructs a sparse matrix backed by an exact integer value buffer.
    pub fn new_integer(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        integer_data: IntegerStorage,
    ) -> Result<Self, String> {
        let col_ptrs = col_ptrs.into();
        let row_indices = row_indices.into();
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, integer_data.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::from_integer_storage(integer_data),
            )),
        })
    }

    /// Constructs a double-precision complex sparse matrix using the canonical
    /// interleaved host representation.
    pub fn new_complex(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        values: Vec<(f64, f64)>,
    ) -> Result<Self, String> {
        Self::from_host_complex_buffers(
            rows,
            cols,
            col_ptrs.into(),
            row_indices.into(),
            values.into(),
        )
    }

    /// Constructs a sparse logical matrix whose CSC pattern is the complete
    /// authoritative set of true elements.
    pub fn new_logical(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
    ) -> Result<Self, String> {
        let col_ptrs = col_ptrs.into();
        let row_indices = row_indices.into();
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, row_indices.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Logical,
        })
    }

    pub fn from_host_numeric_buffers(
        rows: usize,
        cols: usize,
        col_ptrs: HostIndexBuffer,
        row_indices: HostIndexBuffer,
        values: HostNumericBuffer,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::Numeric(values),
        })
    }

    /// Constructs a complex sparse matrix while retaining the supplied
    /// pointer-stable value and index owners.
    pub fn from_host_complex_buffers(
        rows: usize,
        cols: usize,
        col_ptrs: HostIndexBuffer,
        row_indices: HostIndexBuffer,
        values: HostComplexBuffer<f64>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::ComplexF64(values),
        })
    }

    /// Constructs a single-precision complex sparse matrix while retaining
    /// the canonical interleaved component class.
    pub fn new_complex_f32(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        values: Vec<(f32, f32)>,
    ) -> Result<Self, String> {
        Self::from_host_complex_f32_buffers(
            rows,
            cols,
            col_ptrs.into(),
            row_indices.into(),
            values.into(),
        )
    }

    pub fn from_host_complex_f32_buffers(
        rows: usize,
        cols: usize,
        col_ptrs: HostIndexBuffer,
        row_indices: HostIndexBuffer,
        values: HostComplexBuffer<f32>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage: SparseValueStorage::ComplexF32(values),
        })
    }

    /// Constructs a complex sparse matrix with this matrix's component class.
    /// Values use the common computation representation at the API boundary
    /// and are narrowed only when the prototype is complex `single`.
    pub fn new_complex_like(
        &self,
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
        values: Vec<(f64, f64)>,
    ) -> Result<Self, String> {
        match &self.storage {
            SparseValueStorage::ComplexF64(_) => {
                Self::new_complex(rows, cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::ComplexF32(_) => Self::new_complex_f32(
                rows,
                cols,
                col_ptrs,
                row_indices,
                values
                    .into_iter()
                    .map(|(real, imaginary)| (real as f32, imaginary as f32))
                    .collect(),
            ),
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => {
                Err("complex sparse construction requires a complex prototype".to_string())
            }
        }
    }

    pub fn from_host_logical_pattern(
        rows: usize,
        cols: usize,
        col_ptrs: HostIndexBuffer,
        row_indices: HostIndexBuffer,
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

    pub fn numeric_host_buffer(&self) -> Option<&HostNumericBuffer> {
        match &self.storage {
            SparseValueStorage::Numeric(values) => Some(values),
            SparseValueStorage::ComplexF64(_)
            | SparseValueStorage::ComplexF32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    pub fn complex_host_buffer(&self) -> Option<&HostComplexBuffer<f64>> {
        match &self.storage {
            SparseValueStorage::ComplexF64(values) => Some(values),
            SparseValueStorage::Numeric(_)
            | SparseValueStorage::ComplexF32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    pub fn complex_f32_host_buffer(&self) -> Option<&HostComplexBuffer<f32>> {
        match &self.storage {
            SparseValueStorage::ComplexF32(values) => Some(values),
            SparseValueStorage::Numeric(_)
            | SparseValueStorage::ComplexF64(_)
            | SparseValueStorage::Logical => None,
        }
    }

    fn from_numeric_scalars(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        dtype: NumericDType,
        values: Vec<NumericScalar>,
    ) -> Result<Self, String> {
        let mut storage = NumericStorage::zeros(dtype, values.len());
        for (index, value) in values.into_iter().enumerate() {
            storage.set_value(index, value)?;
        }
        Self::from_host_numeric_buffers(
            rows,
            cols,
            col_ptrs.into(),
            row_indices.into(),
            HostNumericBuffer::from_numeric_storage(storage),
        )
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
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F64(Vec::new()),
            )),
        }
    }

    /// Creates an all-zero sparse matrix retaining the `single` class.
    pub fn zeros_f32(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F32(Vec::new()),
            )),
        }
    }

    /// Creates an all-zero double-precision complex sparse matrix.
    pub fn zeros_complex(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::ComplexF64(Vec::<ComplexElement<f64>>::new().into()),
        }
    }

    /// Creates an all-zero complex sparse matrix retaining the `single` class.
    pub fn zeros_complex_f32(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::ComplexF32(Vec::<ComplexElement<f32>>::new().into()),
        }
    }

    /// Creates an empty sparse matrix with the same value representation as
    /// this matrix. Only the shape changes.
    pub fn zeros_like(&self, rows: usize, cols: usize) -> Self {
        match &self.storage {
            SparseValueStorage::Numeric(values) => Self {
                rows,
                cols,
                col_ptrs: vec![0; cols.saturating_add(1)].into(),
                row_indices: Vec::new().into(),
                storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                    NumericStorage::zeros(values.numeric_dtype(), 0),
                )),
            },
            SparseValueStorage::ComplexF64(_) => Self::zeros_complex(rows, cols),
            SparseValueStorage::ComplexF32(_) => Self::zeros_complex_f32(rows, cols),
            SparseValueStorage::Logical => Self::zeros_logical(rows, cols),
        }
    }

    /// Creates an all-false sparse logical matrix.
    pub fn zeros_logical(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::Logical,
        }
    }

    /// Creates an all-zero sparse matrix retaining an exact integer class.
    pub fn zeros_with_integer_storage(rows: usize, cols: usize, storage: &IntegerStorage) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::from_integer_storage(storage.zeros_like(0)),
            )),
        }
    }

    /// Creates a typed sparse matrix using the class of `prototype`.
    pub fn new_integer_like(
        rows: usize,
        cols: usize,
        col_ptrs: impl Into<HostIndexBuffer>,
        row_indices: impl Into<HostIndexBuffer>,
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
            SparseValueStorage::Numeric(values) => values.len(),
            SparseValueStorage::ComplexF64(values) => values.len(),
            SparseValueStorage::ComplexF32(values) => values.len(),
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
            SparseValueStorage::Numeric(values) => {
                let mut data = NumericStorage::zeros(values.numeric_dtype(), len);
                for col in 0..self.cols {
                    for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        let row = self.row_indices[idx];
                        let value = values.value_at(idx).ok_or_else(|| {
                            "SparseTensor numeric storage is inconsistent".to_string()
                        })?;
                        data.set_value(row + col * self.rows, value)?;
                    }
                }
                Tensor::from_numeric_storage(data, self.shape())
            }
            SparseValueStorage::Logical => {
                Err("SparseTensor logical storage requires to_dense_logical".to_string())
            }
            SparseValueStorage::ComplexF64(_) => {
                Err("SparseTensor complex storage requires to_dense_complex".to_string())
            }
            SparseValueStorage::ComplexF32(_) => {
                Err("SparseTensor complex storage requires to_dense_complex".to_string())
            }
        }
    }

    pub fn to_dense_complex(&self) -> Result<ComplexTensor, String> {
        let len = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor dense dimensions overflow usize".to_string())?;
        match &self.storage {
            SparseValueStorage::ComplexF64(values) => {
                let mut data = vec![ComplexElement(0.0, 0.0); len];
                for col in 0..self.cols {
                    for index in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        data[self.row_indices[index] + col * self.rows] = values[index];
                    }
                }
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(HostComplexBuffer::from_elements(data)),
                    self.shape(),
                )
            }
            SparseValueStorage::ComplexF32(values) => {
                let mut data = vec![ComplexElement(0.0_f32, 0.0_f32); len];
                for col in 0..self.cols {
                    for index in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                        data[self.row_indices[index] + col * self.rows] = values[index];
                    }
                }
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(HostComplexBuffer::from_elements(data)),
                    self.shape(),
                )
            }
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => Err(
                "SparseTensor real or logical storage requires its matching dense conversion"
                    .to_string(),
            ),
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

    /// Reads a real or logical sparse element. Complex storage must be read
    /// through [`Self::complex_at`] so the imaginary component cannot be lost.
    pub fn real_at(&self, row: usize, col: usize) -> Result<Option<f64>, String> {
        if self.is_complex() {
            return Err("SparseTensor contains complex storage; use complex_at".to_string());
        }
        if row >= self.rows || col >= self.cols {
            return Ok(None);
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        Ok(self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| {
                let index = start + offset;
                match &self.storage {
                    SparseValueStorage::Numeric(storage) => sparse_scalar_f64(
                        storage
                            .value_at(index)
                            .expect("validated sparse storage index"),
                    ),
                    SparseValueStorage::ComplexF64(_) | SparseValueStorage::ComplexF32(_) => {
                        unreachable!("complex storage rejected")
                    }
                    SparseValueStorage::Logical => 1.0,
                }
            }))
    }

    /// Reads a sparse element after the caller has established that the
    /// matrix is real or logical.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.real_at(row, col)
            .expect("SparseTensor::get requires real or logical storage")
    }

    /// Returns a stored complex value. An in-bounds implicit zero is not a
    /// stored CSC entry and therefore returns `None`, matching `get`.
    pub fn complex_at(&self, row: usize, col: usize) -> Option<(f64, f64)> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        let index = start + self.row_indices[start..end].binary_search(&row).ok()?;
        match &self.storage {
            SparseValueStorage::ComplexF64(values) => Some(values[index].into()),
            SparseValueStorage::ComplexF32(values) => {
                let ComplexElement(real, imaginary) = values[index];
                Some((f64::from(real), f64::from(imaginary)))
            }
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => None,
        }
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
            SparseValueStorage::Numeric(storage) => storage.integer_storage(),
            SparseValueStorage::ComplexF64(_)
            | SparseValueStorage::ComplexF32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    /// Borrows stored nonzero values when this sparse matrix is double.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match &self.storage {
            SparseValueStorage::Numeric(values) => values.as_f64_slice(),
            SparseValueStorage::ComplexF64(_)
            | SparseValueStorage::ComplexF32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    /// Borrows stored nonzero values when this sparse matrix is single.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            SparseValueStorage::Numeric(values) => values.as_f32_slice(),
            SparseValueStorage::ComplexF64(_)
            | SparseValueStorage::ComplexF32(_)
            | SparseValueStorage::Logical => None,
        }
    }

    /// Borrows stored nonzero values when this sparse matrix is complex double.
    pub fn as_complex_f64_slice(&self) -> Option<&[ComplexElement<f64>]> {
        self.complex_host_buffer().map(|values| &values[..])
    }

    pub fn as_complex_f32_slice(&self) -> Option<&[ComplexElement<f32>]> {
        self.complex_f32_host_buffer().map(|values| &values[..])
    }

    pub fn is_complex(&self) -> bool {
        matches!(
            self.storage,
            SparseValueStorage::ComplexF64(_) | SparseValueStorage::ComplexF32(_)
        )
    }

    pub fn is_logical(&self) -> bool {
        matches!(self.storage, SparseValueStorage::Logical)
    }

    pub fn value_byte_size(&self) -> usize {
        match &self.storage {
            SparseValueStorage::Logical => 0,
            SparseValueStorage::ComplexF64(_) => std::mem::size_of::<ComplexElement<f64>>(),
            SparseValueStorage::ComplexF32(_) => std::mem::size_of::<ComplexElement<f32>>(),
            SparseValueStorage::Numeric(values) => values.numeric_dtype().byte_size(),
        }
    }

    /// Explicitly materializes stored nonzero values in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_real_f64(&self) -> Result<Vec<f64>, String> {
        match &self.storage {
            SparseValueStorage::Numeric(values) => Ok(values.materialize_f64()),
            SparseValueStorage::ComplexF64(_) | SparseValueStorage::ComplexF32(_) => Err(
                "SparseTensor contains complex storage; use materialize_complex_f64".to_string(),
            ),
            SparseValueStorage::Logical => Ok(vec![1.0; self.nnz()]),
        }
    }

    /// Materializes stored values after the caller has established that the
    /// matrix is real or logical.
    pub fn materialize_f64(&self) -> Vec<f64> {
        self.materialize_real_f64()
            .expect("SparseTensor::materialize_f64 requires real or logical storage")
    }

    pub fn materialize_complex_f64(&self) -> Result<Vec<(f64, f64)>, String> {
        match &self.storage {
            SparseValueStorage::ComplexF64(values) => {
                Ok(values.iter().copied().map(Into::into).collect())
            }
            SparseValueStorage::ComplexF32(values) => Ok(values
                .iter()
                .map(|value| {
                    let ComplexElement(real, imaginary) = *value;
                    (f64::from(real), f64::from(imaginary))
                })
                .collect()),
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => {
                Err("SparseTensor does not contain complex storage".to_string())
            }
        }
    }

    /// Reads one stored nonzero value without routing integers through floating point.
    pub fn numeric_value_at(&self, index: usize) -> Option<NumericScalar> {
        match &self.storage {
            SparseValueStorage::Numeric(values) => values.value_at(index),
            SparseValueStorage::ComplexF64(_) | SparseValueStorage::ComplexF32(_) => None,
            SparseValueStorage::Logical => (index < self.nnz()).then_some(NumericScalar::F64(1.0)),
        }
    }

    pub fn complex_value_at(&self, index: usize) -> Option<ComplexElement<f64>> {
        match &self.storage {
            SparseValueStorage::ComplexF64(values) => values.get(index).copied(),
            SparseValueStorage::ComplexF32(values) => values.get(index).map(|value| {
                let ComplexElement(real, imaginary) = *value;
                ComplexElement(f64::from(real), f64::from(imaginary))
            }),
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => None,
        }
    }

    pub fn numeric_dtype(&self) -> Option<NumericDType> {
        match &self.storage {
            SparseValueStorage::Numeric(storage) => Some(storage.numeric_dtype()),
            SparseValueStorage::ComplexF64(_) => Some(NumericDType::F64),
            SparseValueStorage::ComplexF32(_) => Some(NumericDType::F32),
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

    /// Applies complex updates in one CSC merge while retaining the matrix's
    /// component class. A value is elided only when both components are zero.
    pub fn with_updated_complex_linear_values(
        &self,
        updates: &[(usize, (f64, f64))],
    ) -> Result<Self, String> {
        match &self.storage {
            SparseValueStorage::ComplexF64(stored_values) => {
                let (col_ptrs, row_indices, values) = self.merged_linear_updates(
                    updates,
                    |index| {
                        stored_values
                            .get(index)
                            .copied()
                            .map(Into::into)
                            .ok_or_else(|| {
                                "SparseTensor complex storage is inconsistent".to_string()
                            })
                    },
                    |(real, imaginary)| *real == 0.0 && *imaginary == 0.0,
                )?;
                Self::new_complex(self.rows, self.cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::ComplexF32(stored_values) => {
                let updates = updates
                    .iter()
                    .map(|(index, (real, imaginary))| (*index, (*real as f32, *imaginary as f32)))
                    .collect::<Vec<_>>();
                let (col_ptrs, row_indices, values) = self.merged_linear_updates(
                    &updates,
                    |index| {
                        stored_values
                            .get(index)
                            .copied()
                            .map(Into::into)
                            .ok_or_else(|| {
                                "SparseTensor complex-single storage is inconsistent".to_string()
                            })
                    },
                    |(real, imaginary)| *real == 0.0 && *imaginary == 0.0,
                )?;
                Self::new_complex_f32(self.rows, self.cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::Numeric(_) | SparseValueStorage::Logical => {
                Err("cannot assign complex sparse value to real storage".to_string())
            }
        }
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

    pub fn with_updated_complex_value(
        &self,
        row: usize,
        col: usize,
        value: (f64, f64),
    ) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_complex_linear_values(&[(index, value)])
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
            SparseValueStorage::Numeric(values) => Self::from_host_numeric_buffers(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                values.clone(),
            ),
            SparseValueStorage::ComplexF64(values) => Self::from_host_complex_buffers(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                values.clone(),
            ),
            SparseValueStorage::ComplexF32(values) => Self::from_host_complex_f32_buffers(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                values.clone(),
            ),
            SparseValueStorage::Logical => {
                Self::from_host_logical_pattern(rows, cols, col_ptrs, self.row_indices.clone())
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
            SparseValueStorage::Numeric(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.value_at(index).ok_or_else(|| {
                            "SparseTensor numeric storage is inconsistent".to_string()
                        })
                    })?;
                Self::from_numeric_scalars(
                    output_rows,
                    self.cols,
                    col_ptrs,
                    row_indices,
                    storage.numeric_dtype(),
                    values,
                )
            }
            SparseValueStorage::ComplexF64(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.get(index).copied().map(Into::into).ok_or_else(|| {
                            "SparseTensor complex storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_complex(output_rows, self.cols, col_ptrs, row_indices, values)
            }
            SparseValueStorage::ComplexF32(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, map_row, |index| {
                        storage.get(index).copied().map(Into::into).ok_or_else(|| {
                            "SparseTensor complex single storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_complex_f32(output_rows, self.cols, col_ptrs, row_indices, values)
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
            SparseValueStorage::Numeric(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.value_at(index).ok_or_else(|| {
                            "SparseTensor numeric storage is inconsistent".to_string()
                        })
                    })?;
                Self::from_numeric_scalars(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    storage.numeric_dtype(),
                    values,
                )
            }
            SparseValueStorage::ComplexF64(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.get(index).copied().map(Into::into).ok_or_else(|| {
                            "SparseTensor complex storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_complex(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    values,
                )
            }
            SparseValueStorage::ComplexF32(storage) => {
                let (col_ptrs, row_indices, values) =
                    self.rebuilt_csc(&source_columns, Some, |index| {
                        storage.get(index).copied().map(Into::into).ok_or_else(|| {
                            "SparseTensor complex single storage is inconsistent".to_string()
                        })
                    })?;
                Self::new_complex_f32(
                    self.rows,
                    source_columns.len(),
                    col_ptrs,
                    row_indices,
                    values,
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
        match &self.storage {
            SparseValueStorage::Numeric(values) => values.numeric_dtype().class_name(),
            SparseValueStorage::ComplexF64(_) => "double",
            SparseValueStorage::ComplexF32(_) => "single",
            SparseValueStorage::Logical => "logical",
        }
    }
}

#[cfg(test)]
mod sparse_tensor_tests {
    use super::*;

    #[test]
    fn sparse_complex_single_retains_class_shape_and_native_components() {
        let sparse = SparseTensor::new_complex_f32(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(1.25, -2.5), (3.5, 4.75)],
        )
        .expect("complex single sparse value");

        assert_eq!(sparse.class_name(), "single");
        assert_eq!(sparse.numeric_dtype(), Some(NumericDType::F32));
        assert!(sparse.is_complex());
        assert_eq!(
            sparse.as_complex_f32_slice(),
            Some([ComplexElement(1.25, -2.5), ComplexElement(3.5, 4.75)].as_slice())
        );
        assert_eq!(sparse.complex_at(1, 1), Some((3.5, 4.75)));
        let dense = sparse.to_dense_complex().expect("dense conversion");
        assert_eq!(dense.numeric_dtype(), NumericDType::F32);

        let updated = sparse
            .with_updated_complex_value(1, 0, (6.25, -7.5))
            .expect("complex single update");
        assert_eq!(updated.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(updated.complex_at(1, 0), Some((6.25, -7.5)));

        let empty = sparse.zeros_like(3, 1);
        assert!(empty.is_complex());
        assert_eq!(empty.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(empty.shape(), vec![3, 1]);
    }

    #[test]
    fn sparse_clone_and_shape_expansion_retain_unchanged_host_allocations() {
        let sparse = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 4.0]).unwrap();
        let clone = sparse.clone();
        assert!(sparse.col_ptrs.shares_allocation_with(&clone.col_ptrs));
        assert!(sparse
            .row_indices
            .shares_allocation_with(&clone.row_indices));
        assert!(sparse
            .numeric_host_buffer()
            .unwrap()
            .shares_allocation_with(clone.numeric_host_buffer().unwrap()));

        let expanded = sparse.with_expanded_shape(2, 3).unwrap();
        assert!(!sparse.col_ptrs.shares_allocation_with(&expanded.col_ptrs));
        assert!(sparse
            .row_indices
            .shares_allocation_with(&expanded.row_indices));
        assert!(sparse
            .numeric_host_buffer()
            .unwrap()
            .shares_allocation_with(expanded.numeric_host_buffer().unwrap()));
    }

    #[test]
    fn complex_sparse_storage_is_canonical_and_survives_structural_paths() {
        let sparse = SparseTensor::new_complex(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![(1.0, -1.0), (3.0, 0.5), (2.0, 4.0), (4.0, -2.0), (5.0, 6.0)],
        )
        .expect("complex sparse");
        assert!(sparse.is_complex());
        assert_eq!(sparse.class_name(), "double");
        assert_eq!(sparse.numeric_dtype(), Some(NumericDType::F64));
        assert_eq!(sparse.value_byte_size(), 16);
        assert_eq!(sparse.complex_at(2, 0), Some((3.0, 0.5)));
        assert_eq!(sparse.complex_at(1, 2), None);
        assert_eq!(sparse.complex_value_at(2), Some(ComplexElement(2.0, 4.0)));
        assert!(sparse.as_f64_slice().is_none());

        let clone = sparse.clone();
        assert!(sparse
            .complex_host_buffer()
            .expect("complex host buffer")
            .shares_allocation_with(clone.complex_host_buffer().expect("clone host buffer")));

        let dense = sparse.to_dense_complex().expect("dense complex");
        assert_eq!(
            dense.materialize_f64(),
            vec![
                (1.0, -1.0),
                (0.0, 0.0),
                (3.0, 0.5),
                (0.0, 0.0),
                (2.0, 4.0),
                (0.0, 0.0),
                (4.0, -2.0),
                (0.0, 0.0),
                (5.0, 6.0),
            ]
        );

        let updated = sparse
            .with_updated_complex_value(2, 0, (0.0, 0.0))
            .expect("remove complex zero")
            .with_updated_complex_value(1, 2, (7.0, -8.0))
            .expect("insert complex value");
        assert_eq!(updated.row_indices, vec![0, 1, 0, 1, 2]);
        assert_eq!(updated.complex_at(1, 2), Some((7.0, -8.0)));

        let expanded = sparse.with_expanded_shape(4, 4).expect("expand complex");
        assert_eq!(expanded.col_ptrs, vec![0, 2, 3, 5, 5]);
        assert!(sparse
            .complex_host_buffer()
            .expect("complex host buffer")
            .shares_allocation_with(
                expanded
                    .complex_host_buffer()
                    .expect("expanded host buffer")
            ));

        let rows = sparse.with_deleted_rows(&[1]).expect("delete complex row");
        assert_eq!(rows.shape(), vec![2, 3]);
        assert_eq!(
            rows.materialize_complex_f64().expect("complex values"),
            vec![(1.0, -1.0), (3.0, 0.5), (4.0, -2.0), (5.0, 6.0)]
        );

        let columns = sparse
            .with_deleted_columns(&[1])
            .expect("delete complex column");
        assert_eq!(columns.shape(), vec![3, 2]);
        assert_eq!(
            columns.materialize_complex_f64().expect("complex values"),
            vec![(1.0, -1.0), (3.0, 0.5), (4.0, -2.0), (5.0, 6.0)]
        );
        assert!(sparse.to_dense().is_err());
        assert!(sparse.to_dense_logical().is_err());
    }

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
            col_ptrs: vec![0, 0, 0].into(),
            row_indices: Vec::new().into(),
            storage: SparseValueStorage::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F64(Vec::new()),
            )),
        };

        let err = sparse.to_dense().unwrap_err();
        assert!(err.contains("overflow"));
    }
}

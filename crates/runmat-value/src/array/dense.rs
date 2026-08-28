use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    storage: TensorStorage,
    pub shape: Vec<usize>, // Column-major layout
    pub rows: usize,       // Compatibility for 2D usage
    pub cols: usize,       // Compatibility for 2D usage
}

#[derive(Debug, Clone, PartialEq)]
enum TensorStorage {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer(IntegerStorage),
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F64(data), shape)
    }

    /// Constructs a dense tensor from one authoritative native numeric buffer.
    pub fn from_numeric_storage(
        storage: NumericStorage,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        storage.validate_shape(&shape)?;
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        let storage = match storage {
            NumericStorage::F64(values) => TensorStorage::F64(values),
            NumericStorage::F32(values) => TensorStorage::F32(values),
            storage @ (NumericStorage::I8(_)
            | NumericStorage::I16(_)
            | NumericStorage::I32(_)
            | NumericStorage::I64(_)
            | NumericStorage::U8(_)
            | NumericStorage::U16(_)
            | NumericStorage::U32(_)
            | NumericStorage::U64(_)) => {
                let integer_data = storage
                    .into_integer_storage()
                    .expect("integer NumericStorage variant");
                TensorStorage::Integer(integer_data)
            }
        };
        Ok(Tensor {
            storage,
            shape,
            rows,
            cols,
        })
    }

    pub fn new_2d(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F32(data), shape)
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F32(data.to_vec()), shape.to_vec())
    }

    pub fn new_with_dtype(
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: NumericDType,
    ) -> Result<Self, String> {
        match dtype {
            NumericDType::F64 => Self::from_numeric_storage(NumericStorage::F64(data), shape),
            NumericDType::F32 => Self::from_numeric_storage(
                NumericStorage::F32(data.into_iter().map(|value| value as f32).collect()),
                shape,
            ),
            integer_dtype => {
                let prototype =
                    integer_storage_prototype(integer_dtype).expect("integer dtype prototype");
                let values = data
                    .into_iter()
                    .map(|value| prototype.cast_f64_assignment(value))
                    .collect();
                Self::new_integer(prototype.from_same_class_values(values)?, shape)
            }
        }
    }

    /// Construct a tensor backed by an exact homogeneous integer buffer.
    pub fn new_integer(storage: IntegerStorage, shape: Vec<usize>) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::from_integer_storage(storage), shape)
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        match &self.storage {
            TensorStorage::Integer(storage) => Some(storage),
            TensorStorage::F64(_) | TensorStorage::F32(_) => None,
        }
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        match &self.storage {
            TensorStorage::F64(_) => NumericDType::F64,
            TensorStorage::F32(_) => NumericDType::F32,
            TensorStorage::Integer(storage) => storage.numeric_dtype(),
        }
    }

    /// Returns the authoritative number of stored numeric elements.
    pub fn len(&self) -> usize {
        match &self.storage {
            TensorStorage::F64(values) => values.len(),
            TensorStorage::F32(values) => values.len(),
            TensorStorage::Integer(storage) => storage.len(),
        }
    }

    /// Returns whether the authoritative numeric storage contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows native double storage when this tensor's authoritative class is double.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match &self.storage {
            TensorStorage::F64(values) => Some(values),
            TensorStorage::F32(_) | TensorStorage::Integer(_) => None,
        }
    }

    /// Borrows native single storage when this tensor's authoritative class is single.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            TensorStorage::F32(values) => Some(values),
            TensorStorage::F64(_) | TensorStorage::Integer(_) => None,
        }
    }

    /// Explicitly materializes this tensor in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_f64(&self) -> Vec<f64> {
        match &self.storage {
            TensorStorage::F64(values) => values.clone(),
            TensorStorage::F32(values) => values.iter().copied().map(f64::from).collect(),
            TensorStorage::Integer(storage) => storage.to_f64_vec(),
        }
    }

    /// Read one element without routing an integer through floating-point storage.
    pub fn numeric_value_at(&self, index: usize) -> Option<NumericScalar> {
        match &self.storage {
            TensorStorage::F64(values) => values.get(index).copied().map(NumericScalar::F64),
            TensorStorage::F32(values) => values.get(index).copied().map(NumericScalar::F32),
            TensorStorage::Integer(storage) => storage.value_at(index).map(NumericScalar::from),
        }
    }

    /// Assign one numeric scalar using the destination array's class semantics.
    ///
    /// Integer-to-integer assignments remain exact and floating assignments use
    /// the same round-and-saturate conversion as MATLAB integer arrays.
    pub fn set_numeric_assignment_at(
        &mut self,
        index: usize,
        value: NumericScalar,
    ) -> Result<(), String> {
        match &mut self.storage {
            TensorStorage::F64(values) => {
                let value = match value {
                    NumericScalar::F64(value) => value,
                    NumericScalar::F32(value) => f64::from(value),
                    value => value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer")
                        .to_f64(),
                };
                let destination = values
                    .get_mut(index)
                    .ok_or_else(|| format!("Tensor index {index} out of bounds"))?;
                *destination = value;
            }
            TensorStorage::F32(values) => {
                let value = match value {
                    NumericScalar::F64(value) => value as f32,
                    NumericScalar::F32(value) => value,
                    value => value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer")
                        .to_f64() as f32,
                };
                let destination = values
                    .get_mut(index)
                    .ok_or_else(|| format!("Tensor index {index} out of bounds"))?;
                *destination = value;
            }
            TensorStorage::Integer(storage) => {
                let exact = match value {
                    NumericScalar::F64(value) => storage.cast_f64_assignment(value),
                    NumericScalar::F32(value) => storage.cast_f64_assignment(f64::from(value)),
                    value => storage.cast_exact_assignment(
                        &value
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                    ),
                };
                storage.set_value(index, exact)?;
            }
        }
        Ok(())
    }

    /// Consumes this tensor into one public all-class native numeric buffer.
    pub fn into_numeric_storage(self) -> Result<NumericStorage, String> {
        let storage = match self.storage {
            TensorStorage::F64(values) => NumericStorage::F64(values),
            TensorStorage::F32(values) => NumericStorage::F32(values),
            TensorStorage::Integer(storage) => NumericStorage::from_integer_storage(storage),
        };
        storage.validate_shape(&self.shape)?;
        Ok(storage)
    }

    /// Change only shape metadata while retaining the underlying numeric storage.
    pub fn reshape(mut self, shape: Vec<usize>) -> Result<Self, String> {
        let expected = shape.iter().product::<usize>();
        if self.len() != expected {
            return Err(format!(
                "{} tensor data length {} doesn't match shape {:?} ({} elements)",
                self.numeric_dtype().class_name(),
                self.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        self.shape = shape;
        self.rows = rows;
        self.cols = cols;
        Ok(self)
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self::from_numeric_storage(NumericStorage::zeros(NumericDType::F64, size), shape)
            .expect("zero storage length matches shape")
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self::from_numeric_storage(NumericStorage::ones(NumericDType::F64, size), shape)
            .expect("one storage length matches shape")
    }

    // 2D helpers for transitional call sites
    pub fn zeros2(rows: usize, cols: usize) -> Self {
        Self::zeros(vec![rows, cols])
    }
    pub fn ones2(rows: usize, cols: usize) -> Self {
        Self::ones(vec![rows, cols])
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    pub fn get2(&self, row: usize, col: usize) -> Result<f64, String> {
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!(
                "Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"
            ));
        }
        // Column-major linearization: lin = row + col*rows
        let index = row + col * rows;
        Ok(self
            .numeric_value_at(index)
            .expect("validated tensor index")
            .materialize_f64())
    }

    pub fn set2(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!(
                "Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"
            ));
        }
        // Column-major linearization
        let index = row + col * rows;
        self.set_numeric_assignment_at(index, NumericScalar::F64(value))
    }

    pub fn scalar_to_tensor2(scalar: f64, rows: usize, cols: usize) -> Tensor {
        Self::from_numeric_storage(
            NumericStorage::F64(vec![scalar; rows * cols]),
            vec![rows, cols],
        )
        .expect("scalar expansion length matches shape")
    }
    // No-compat constructors: prefer new/new_2d/zeros/zeros2/ones/ones2
}

fn integer_storage_prototype(dtype: NumericDType) -> Option<IntegerStorage> {
    match dtype {
        NumericDType::I8 => Some(IntegerStorage::I8(Vec::new())),
        NumericDType::I16 => Some(IntegerStorage::I16(Vec::new())),
        NumericDType::I32 => Some(IntegerStorage::I32(Vec::new())),
        NumericDType::I64 => Some(IntegerStorage::I64(Vec::new())),
        NumericDType::U8 => Some(IntegerStorage::U8(Vec::new())),
        NumericDType::U16 => Some(IntegerStorage::U16(Vec::new())),
        NumericDType::U32 => Some(IntegerStorage::U32(Vec::new())),
        NumericDType::U64 => Some(IntegerStorage::U64(Vec::new())),
        NumericDType::F64 | NumericDType::F32 => None,
    }
}

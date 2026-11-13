use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: SharedShape, // Column-major layout
    // Payload and immutable shape each retain their own allocation claim.
    storage: HostNumericBuffer,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: impl Into<HostShapeInput>) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F64(data), shape)
    }

    /// Constructs a dense tensor from one authoritative native numeric buffer.
    pub fn from_numeric_storage(
        storage: NumericStorage,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        match shape.into() {
            HostShapeInput::Owned(shape) => {
                Ok(NumericTensorAdmission::from_numeric_storage(storage, shape)
                    .finish_synchronously()?
                    .into_tensor())
            }
            HostShapeInput::Shared(shape) => Self::from_shared_host_buffer(
                HostNumericBuffer::from_numeric_storage(storage),
                shape,
            ),
        }
    }

    pub fn new_2d(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn from_f32(data: Vec<f32>, shape: impl Into<HostShapeInput>) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F32(data), shape)
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::F32(data.to_vec()), shape.to_vec())
    }

    pub fn new_with_dtype(
        data: Vec<f64>,
        shape: impl Into<HostShapeInput>,
        dtype: NumericDType,
    ) -> Result<Self, String> {
        let shape = shape.into();
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
    pub fn new_integer(
        storage: IntegerStorage,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        Self::from_numeric_storage(NumericStorage::from_integer_storage(storage), shape)
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        self.storage.integer_storage()
    }

    pub fn host_buffer(&self) -> &HostNumericBuffer {
        &self.storage
    }

    pub fn from_host_buffer(
        storage: HostNumericBuffer,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        match shape.into() {
            HostShapeInput::Owned(shape) => {
                Ok(NumericTensorAdmission::from_host_buffer(storage, shape)
                    .finish_synchronously()?
                    .into_tensor())
            }
            HostShapeInput::Shared(shape) => Self::from_shared_host_buffer(storage, shape),
        }
    }

    /// Reuses an already admitted shape owner without materializing a second
    /// dimension vector or retaining the source payload's quota claim.
    pub fn from_shared_host_buffer(
        storage: HostNumericBuffer,
        shape: SharedShape,
    ) -> Result<Self, String> {
        let tensor = Self { shape, storage };
        tensor.validate_shape()?;
        Ok(tensor)
    }

    pub(super) fn from_validated_parts(parts: ValidatedNumericTensorParts) -> Self {
        let (storage, shape, _, _) = parts.into_components();
        Self {
            storage,
            shape: SharedShape::new_unmanaged(shape),
        }
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        self.storage.numeric_dtype()
    }

    /// Returns the authoritative number of stored numeric elements.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns whether the authoritative numeric storage contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows native double storage when this tensor's authoritative class is double.
    /// Borrow unique native storage without growing it or triggering a COW copy.
    pub fn try_as_f64_slice_mut_unique(&mut self) -> Option<&mut [f64]> {
        self.storage.try_as_f64_slice_mut_unique()
    }

    /// Borrow the existing native payload without allocation or implicit COW.
    pub fn try_unique_numeric_mutation(&mut self) -> Option<super::UniqueHostNumericMutation<'_>> {
        self.storage.try_unique_mutation()
    }

    /// Borrow unique native storage without growing it or triggering a COW copy.
    pub fn try_as_f32_slice_mut_unique(&mut self) -> Option<&mut [f32]> {
        self.storage.try_as_f32_slice_mut_unique()
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        self.storage.as_f64_slice()
    }

    /// Borrows native single storage when this tensor's authoritative class is single.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        self.storage.as_f32_slice()
    }

    /// Explicitly materializes this tensor in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_f64(&self) -> Vec<f64> {
        self.storage.materialize_f64()
    }

    /// Read one element without routing an integer through floating-point storage.
    pub fn numeric_value_at(&self, index: usize) -> Option<NumericScalar> {
        self.storage.value_at(index)
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
        self.storage.set_value(index, value)
    }

    /// Consumes this tensor into one public all-class native numeric buffer.
    pub fn into_numeric_storage(self) -> Result<NumericStorage, String> {
        let storage = self.storage.into_numeric_storage();
        storage.validate_shape(&self.shape)?;
        Ok(storage)
    }

    /// Validate retained dimensions against the current payload without
    /// cloning either owner. This preserves the pre-mutation guard used by
    /// indexing while dimension fields remain publicly writable.
    pub fn validate_shape(&self) -> Result<(), String> {
        let expected = BorrowedShapeAdmission::new(&self.shape).finish_synchronously(|shape| {
            format!("numeric tensor shape {shape:?} overflows usize")
        })?;
        if self.len() != expected {
            return Err(format!(
                "{} storage length {} doesn't match shape {:?} ({} elements)",
                self.numeric_dtype().class_name(),
                self.len(),
                self.shape,
                expected
            ));
        }
        Ok(())
    }

    /// Change only shape metadata while retaining the underlying numeric storage.
    pub fn reshape(mut self, shape: Vec<usize>) -> Result<Self, String> {
        self.reshape_in_place(shape)?;
        Ok(self)
    }

    /// Installs shape metadata admitted by the *current* operation. The
    /// source tensor's retained claim does not authorize this new shape.
    pub fn reshape_with_shared_shape(&mut self, shape: SharedShape) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let expected = BorrowedShapeAdmission::new(&shape).finish_synchronously(|dimensions| {
            format!("{class} tensor shape {dimensions:?} overflows usize")
        })?;
        if expected != self.len() {
            return Err(format!(
                "{class} tensor data length {} doesn't match shape {:?} ({expected} elements)",
                self.len(),
                shape
            ));
        }
        self.shape = shape;
        Ok(())
    }

    /// Atomically replace shape metadata after canonical checked admission.
    ///
    /// A failed admission leaves both the payload and existing dimensions
    /// unchanged.
    pub fn reshape_in_place(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let len = self.len();
        let shape = admit_shape(
            shape,
            len,
            |shape| format!("{class} tensor shape {shape:?} overflows usize"),
            |shape, expected| {
                format!(
                    "{class} tensor data length {len} doesn't match shape {shape:?} ({expected} elements)"
                )
            },
        )?;
        self.apply_validated_shape(shape);
        Ok(())
    }

    /// Resize in the current numeric class and atomically publish the admitted
    /// shape after storage growth succeeds.
    pub fn resize_zeroed_for_shape(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let shape = admit_shape_for_resize(shape, |shape| {
            format!("{class} tensor shape {shape:?} overflows usize")
        })?;
        let target_len = shape.elements();
        self.storage.resize_zeroed(target_len);
        self.apply_validated_shape(shape);
        Ok(())
    }

    fn apply_validated_shape(&mut self, shape: ValidatedShape) {
        let (shape, _, _) = shape.into_components();
        self.shape = SharedShape::new_unmanaged(shape);
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

    /// Borrows the authoritative column-major dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Share the immutable dimension allocation with another value. This
    /// carries only the shape claim, never this tensor's payload claim.
    pub fn shared_shape(&self) -> SharedShape {
        self.shape.clone()
    }

    #[cfg(test)]
    pub(crate) fn shape_capacity_for_testing(&self) -> usize {
        self.shape.capacity()
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

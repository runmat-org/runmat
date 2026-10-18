use super::*;

/// One complex numeric element in the canonical host interleaved layout.
///
/// The explicit C representation makes this layout suitable for in-process
/// foreign buffer leases. Callers must still use the owning buffer APIs for
/// lifetime and mutation; this type establishes element layout only.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct ComplexElement<T>(pub T, pub T);

impl<T> From<(T, T)> for ComplexElement<T> {
    fn from((real, imaginary): (T, T)) -> Self {
        Self(real, imaginary)
    }
}

impl<T> From<ComplexElement<T>> for (T, T) {
    fn from(value: ComplexElement<T>) -> Self {
        (value.0, value.1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexTensor {
    // Payload and immutable shape each retain their own allocation claim.
    shape: SharedShape,
    storage: ComplexStorage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexStorage {
    F64(HostComplexBuffer<f64>),
    F32(HostComplexBuffer<f32>),
    Integer(IntegerComplexStorage),
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegerComplexStorage {
    pub real: IntegerStorage,
    pub imag: IntegerStorage,
}

impl IntegerComplexStorage {
    pub fn new(real: IntegerStorage, imag: IntegerStorage) -> Result<Self, String> {
        if real.numeric_dtype() != imag.numeric_dtype() || real.len() != imag.len() {
            return Err("complex integer components must have matching class and length".into());
        }
        Ok(Self { real, imag })
    }

    pub fn len(&self) -> usize {
        self.real.len()
    }

    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    pub fn class_name(&self) -> &'static str {
        self.real.class_name()
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        self.real.numeric_dtype()
    }

    /// Tests a paired complex integer element without consulting its lossy
    /// floating compatibility representation.
    pub fn is_nonzero_at(&self, index: usize) -> Option<bool> {
        let real = self.real.value_at(index)?;
        let imag = self.imag.value_at(index)?;
        Some(!real.is_zero() || !imag.is_zero())
    }

    /// Applies the same structural reorder independently to both exact components.
    pub fn reorder(
        &self,
        reorder: impl Fn(&[IntValue]) -> Result<Vec<IntValue>, String>,
    ) -> Result<Self, String> {
        let real = self
            .real
            .from_exact_values_like(reorder(&self.real.exact_values())?)?;
        let imag = self
            .imag
            .from_exact_values_like(reorder(&self.imag.exact_values())?)?;
        Self::new(real, imag)
    }
}

impl ComplexTensor {
    pub fn new(data: Vec<(f64, f64)>, shape: impl Into<HostShapeInput>) -> Result<Self, String> {
        Self::from_complex_storage(ComplexStorage::F64(data.into()), shape)
    }

    pub fn from_f32(
        data: Vec<(f32, f32)>,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        Self::from_complex_storage(ComplexStorage::F32(data.into()), shape)
    }

    /// Reconstructs floating complex values in the requested floating class.
    pub fn from_f64_values_with_dtype(
        data: Vec<(f64, f64)>,
        shape: impl Into<HostShapeInput>,
        dtype: NumericDType,
    ) -> Result<Self, String> {
        let shape = shape.into();
        match dtype {
            NumericDType::F64 => Self::new(data, shape),
            NumericDType::F32 => Self::from_f32(
                data.into_iter()
                    .map(|(real, imag)| (real as f32, imag as f32))
                    .collect(),
                shape,
            ),
            _ => Err(format!(
                "complex floating reconstruction requires single or double, got {}",
                dtype.class_name()
            )),
        }
    }

    pub fn from_complex_storage(
        storage: ComplexStorage,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        match shape.into() {
            HostShapeInput::Owned(shape) => Ok(ComplexTensorAdmission::new(storage, shape)?
                .finish_synchronously()?
                .into_tensor()),
            HostShapeInput::Shared(shape) => Self::from_shared_host_buffer(storage, shape),
        }
    }

    /// Reuses an independently admitted shape without cloning its backing.
    pub fn from_shared_host_buffer(
        storage: ComplexStorage,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if let ComplexStorage::Integer(integer) = &storage {
            if integer.real.numeric_dtype() != integer.imag.numeric_dtype()
                || integer.real.len() != integer.imag.len()
            {
                return Err(
                    "complex integer components must have matching class and length".into(),
                );
            }
        }
        let tensor = Self { shape, storage };
        tensor.validate_shape()?;
        Ok(tensor)
    }

    pub(super) fn from_validated_parts(parts: ValidatedComplexTensorParts) -> Self {
        let (storage, shape, _, _) = parts.into_components();
        Self {
            storage,
            shape: SharedShape::new_unmanaged(shape),
        }
    }
    pub fn new_integer(
        storage: IntegerComplexStorage,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        Self::from_complex_storage(ComplexStorage::Integer(storage), shape)
    }
    pub fn new_2d(data: Vec<(f64, f64)>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .expect("complex zero shape must fit usize");
        Self::from_complex_storage(
            ComplexStorage::F64(vec![ComplexElement(0.0, 0.0); size].into()),
            shape,
        )
        .expect("complex zero storage length matches shape")
    }

    pub fn complex_storage(&self) -> &ComplexStorage {
        &self.storage
    }

    pub fn into_complex_storage(self) -> ComplexStorage {
        self.into_complex_storage_and_shape().0
    }

    /// Consumes this tensor into its native storage and authoritative shape.
    ///
    /// Both owners are transferred together without exposing mutable metadata
    /// or reconstructing either allocation.
    pub fn into_complex_storage_and_shape(self) -> (ComplexStorage, SharedShape) {
        let Self { storage, shape } = self;
        (storage, shape)
    }

    /// Validate retained dimensions against the current component storage
    /// without cloning either owner.
    pub fn validate_shape(&self) -> Result<(), String> {
        self.storage.validate_shape(&self.shape)
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        self.storage.numeric_dtype()
    }

    pub fn as_f64_slice(&self) -> Option<&[ComplexElement<f64>]> {
        match &self.storage {
            ComplexStorage::F64(values) => Some(values),
            ComplexStorage::F32(_) | ComplexStorage::Integer(_) => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[ComplexElement<f32>]> {
        match &self.storage {
            ComplexStorage::F32(values) => Some(values),
            ComplexStorage::F64(_) | ComplexStorage::Integer(_) => None,
        }
    }

    /// Explicitly materializes complex values in the `f64` computation domain.
    ///
    /// Integer components outside the exact binary64 range may lose precision.
    pub fn materialize_f64(&self) -> Vec<(f64, f64)> {
        self.storage.materialize_f64()
    }

    pub fn integer_storage(&self) -> Option<&IntegerComplexStorage> {
        match &self.storage {
            ComplexStorage::Integer(storage) => Some(storage),
            ComplexStorage::F64(_) | ComplexStorage::F32(_) => None,
        }
    }

    /// Read one complex element without routing integer components through floating point.
    pub fn numeric_value_at(&self, index: usize) -> Option<(NumericScalar, NumericScalar)> {
        if let Some(storage) = self.integer_storage() {
            return Some((
                NumericScalar::from(storage.real.value_at(index)?),
                NumericScalar::from(storage.imag.value_at(index)?),
            ));
        }
        match &self.storage {
            ComplexStorage::F64(values) => {
                values
                    .get(index)
                    .copied()
                    .map(|ComplexElement(real, imag)| {
                        (NumericScalar::F64(real), NumericScalar::F64(imag))
                    })
            }
            ComplexStorage::F32(values) => {
                values
                    .get(index)
                    .copied()
                    .map(|ComplexElement(real, imag)| {
                        (NumericScalar::F32(real), NumericScalar::F32(imag))
                    })
            }
            ComplexStorage::Integer(_) => unreachable!("integer storage returned above"),
        }
    }

    /// Assigns one complex floating scalar using the destination component class.
    pub fn set_f64_assignment_at(
        &mut self,
        index: usize,
        real: f64,
        imag: f64,
    ) -> Result<(), String> {
        match &mut self.storage {
            ComplexStorage::F64(values) => {
                let destination = values
                    .get_mut(index)
                    .ok_or_else(|| format!("ComplexTensor index {index} out of bounds"))?;
                *destination = ComplexElement(real, imag);
            }
            ComplexStorage::F32(values) => {
                let destination = values
                    .get_mut(index)
                    .ok_or_else(|| format!("ComplexTensor index {index} out of bounds"))?;
                *destination = ComplexElement(real as f32, imag as f32);
            }
            ComplexStorage::Integer(storage) => {
                storage.real.set_f64_assignment(index, real)?;
                storage.imag.set_f64_assignment(index, imag)?;
            }
        }
        Ok(())
    }

    /// Atomically replace shape metadata after canonical checked admission.
    pub fn reshape_in_place(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let len = self.len();
        let shape = admit_shape(
            shape,
            len,
            |shape| format!("complex {class} tensor shape {shape:?} overflows usize"),
            |shape, expected| {
                format!(
                    "complex {class} storage length {len} doesn't match shape {shape:?} ({expected} elements)"
                )
            },
        )?;
        self.apply_validated_shape(shape);
        Ok(())
    }

    /// Installs an independently admitted shape for this operation without
    /// borrowing the source payload's retained allocation authority.
    pub fn reshape_with_shared_shape(&mut self, shape: SharedShape) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let expected = BorrowedShapeAdmission::new(&shape).finish_synchronously(|dimensions| {
            format!("complex {class} tensor shape {dimensions:?} overflows usize")
        })?;
        if expected != self.len() {
            return Err(format!(
                "complex {class} storage length {} doesn't match shape {:?} ({expected} elements)",
                self.len(),
                shape
            ));
        }
        self.shape = shape;
        Ok(())
    }

    /// Resize in the current complex class and publish admitted dimensions
    /// only after both component stores have grown successfully.
    pub fn resize_zeroed_for_shape(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let class = self.numeric_dtype().class_name();
        let shape = admit_shape_for_resize(shape, |shape| {
            format!("complex {class} tensor shape {shape:?} overflows usize")
        })?;
        let target_len = shape.elements();
        self.storage.resize_zeroed(target_len);
        self.apply_validated_shape(shape);
        Ok(())
    }

    /// Resize exact integer components and assign one destination-class value
    /// without cloning or materializing the component buffers.
    pub fn resize_and_set_integer_assignment(
        &mut self,
        shape: Vec<usize>,
        index: usize,
        real: IntValue,
        imag: IntValue,
    ) -> Result<(), String> {
        let storage = self
            .integer_storage()
            .ok_or_else(|| "complex integer assignment requires integer storage".to_string())?;
        let dtype = storage.numeric_dtype();
        if real.numeric_dtype() != dtype || imag.numeric_dtype() != dtype {
            return Err(format!(
                "complex integer assignment components must match {} storage",
                dtype.class_name()
            ));
        }
        let shape = admit_shape_for_resize(shape, |shape| {
            format!(
                "complex {} tensor shape {shape:?} overflows usize",
                dtype.class_name()
            )
        })?;
        if index >= shape.elements() {
            return Err(format!(
                "ComplexTensor index {index} out of bounds for {} elements",
                shape.elements()
            ));
        }

        let ComplexStorage::Integer(storage) = &mut self.storage else {
            unreachable!("integer storage checked before mutation")
        };
        storage.real.resize_zeroed(shape.elements());
        storage.imag.resize_zeroed(shape.elements());
        storage
            .real
            .set_value(index, real)
            .expect("destination class and index validated before mutation");
        storage
            .imag
            .set_value(index, imag)
            .expect("destination class and index validated before mutation");
        self.apply_validated_shape(shape);
        Ok(())
    }

    fn apply_validated_shape(&mut self, shape: ValidatedShape) {
        let (shape, _, _) = shape.into_components();
        self.shape = SharedShape::new_unmanaged(shape);
    }

    /// Borrows the authoritative column-major dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Shares the independently retained shape without retaining storage.
    pub fn shared_shape(&self) -> SharedShape {
        self.shape.clone()
    }

    pub fn rows(&self) -> usize {
        shape_dimensions(&self.shape).0
    }

    pub fn cols(&self) -> usize {
        shape_dimensions(&self.shape).1
    }

    /// Formats one element using its exact integer components when present.
    ///
    pub fn format_element(&self, index: usize) -> String {
        match &self.storage {
            ComplexStorage::Integer(storage) => {
                let real = storage
                    .real
                    .value_at(index)
                    .expect("complex integer real storage must match tensor shape");
                let imag = storage
                    .imag
                    .value_at(index)
                    .expect("complex integer imaginary storage must match tensor shape");
                format_integer_complex_value(&real, &imag)
            }
            ComplexStorage::F64(values) => {
                let ComplexElement(real, imag) = values[index];
                Value::Complex(real, imag).to_string()
            }
            ComplexStorage::F32(values) => {
                let ComplexElement(real, imag) = values[index];
                Value::Complex(f64::from(real), f64::from(imag)).to_string()
            }
        }
    }
}

impl ComplexStorage {
    pub fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Integer(storage) => storage.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn resize_zeroed(&mut self, len: usize) {
        match self {
            Self::F64(values) => values.resize(len, ComplexElement(0.0, 0.0)),
            Self::F32(values) => values.resize(len, ComplexElement(0.0, 0.0)),
            Self::Integer(storage) => {
                storage.real.resize_zeroed(len);
                storage.imag.resize_zeroed(len);
            }
        }
    }

    pub fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::Integer(storage) => storage.real.numeric_dtype(),
        }
    }

    pub fn validate_shape(&self, shape: &[usize]) -> Result<(), String> {
        let expected = BorrowedShapeAdmission::new(shape).finish_synchronously(|shape| {
            format!("complex tensor shape {shape:?} overflows usize")
        })?;
        if self.len() != expected {
            return Err(format!(
                "complex {} storage length {} doesn't match shape {:?} ({} elements)",
                self.numeric_dtype().class_name(),
                self.len(),
                shape,
                expected
            ));
        }
        Ok(())
    }

    pub fn materialize_f64(&self) -> Vec<(f64, f64)> {
        match self {
            Self::F64(values) => values.iter().copied().map(Into::into).collect(),
            Self::F32(values) => values
                .iter()
                .map(|&ComplexElement(real, imag)| (f64::from(real), f64::from(imag)))
                .collect(),
            Self::Integer(storage) => storage
                .real
                .to_f64_vec()
                .into_iter()
                .zip(storage.imag.to_f64_vec())
                .collect(),
        }
    }

    pub fn gather(&self, indices: &[usize]) -> Result<Self, String> {
        match self {
            Self::F64(values) => indices
                .iter()
                .map(|&index| {
                    values
                        .get(index)
                        .copied()
                        .ok_or_else(|| format!("complex storage index {index} out of bounds"))
                })
                .collect::<Result<HostComplexBuffer<_>, _>>()
                .map(Self::F64),
            Self::F32(values) => indices
                .iter()
                .map(|&index| {
                    values
                        .get(index)
                        .copied()
                        .ok_or_else(|| format!("complex storage index {index} out of bounds"))
                })
                .collect::<Result<HostComplexBuffer<_>, _>>()
                .map(Self::F32),
            Self::Integer(storage) => {
                let real = indices
                    .iter()
                    .map(|&index| {
                        storage
                            .real
                            .value_at(index)
                            .ok_or_else(|| format!("complex storage index {index} out of bounds"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let imag = indices
                    .iter()
                    .map(|&index| {
                        storage
                            .imag
                            .value_at(index)
                            .ok_or_else(|| format!("complex storage index {index} out of bounds"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let real = storage.real.from_exact_values_like(real)?;
                let imag = storage.imag.from_exact_values_like(imag)?;
                IntegerComplexStorage::new(real, imag).map(Self::Integer)
            }
        }
    }
}

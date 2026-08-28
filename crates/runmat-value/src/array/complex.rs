use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexTensor {
    storage: ComplexStorage,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexStorage {
    F64(Vec<(f64, f64)>),
    F32(Vec<(f32, f32)>),
    Integer(IntegerComplexStorage),
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegerComplexStorage {
    pub real: IntegerStorage,
    pub imag: IntegerStorage,
}

impl IntegerComplexStorage {
    pub fn new(real: IntegerStorage, imag: IntegerStorage) -> Result<Self, String> {
        if real.class_name() != imag.class_name() || real.len() != imag.len() {
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
    pub fn new(data: Vec<(f64, f64)>, shape: Vec<usize>) -> Result<Self, String> {
        Self::from_complex_storage(ComplexStorage::F64(data), shape)
    }

    pub fn from_f32(data: Vec<(f32, f32)>, shape: Vec<usize>) -> Result<Self, String> {
        Self::from_complex_storage(ComplexStorage::F32(data), shape)
    }

    /// Reconstructs floating complex values in the requested floating class.
    pub fn from_f64_values_with_dtype(
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        dtype: NumericDType,
    ) -> Result<Self, String> {
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
        Ok(ComplexTensor {
            storage,
            shape,
            rows,
            cols,
        })
    }
    pub fn new_integer(storage: IntegerComplexStorage, shape: Vec<usize>) -> Result<Self, String> {
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
        Self::from_complex_storage(ComplexStorage::F64(vec![(0.0, 0.0); size]), shape)
            .expect("complex zero storage length matches shape")
    }

    pub fn complex_storage(&self) -> &ComplexStorage {
        &self.storage
    }

    pub fn into_complex_storage(self) -> ComplexStorage {
        self.storage
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

    pub fn as_f64_slice(&self) -> Option<&[(f64, f64)]> {
        match &self.storage {
            ComplexStorage::F64(values) => Some(values),
            ComplexStorage::F32(_) | ComplexStorage::Integer(_) => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[(f32, f32)]> {
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
            ComplexStorage::F64(values) => values
                .get(index)
                .copied()
                .map(|(real, imag)| (NumericScalar::F64(real), NumericScalar::F64(imag))),
            ComplexStorage::F32(values) => values
                .get(index)
                .copied()
                .map(|(real, imag)| (NumericScalar::F32(real), NumericScalar::F32(imag))),
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
                *destination = (real, imag);
            }
            ComplexStorage::F32(values) => {
                let destination = values
                    .get_mut(index)
                    .ok_or_else(|| format!("ComplexTensor index {index} out of bounds"))?;
                *destination = (real as f32, imag as f32);
            }
            ComplexStorage::Integer(storage) => {
                storage.real.set_f64_assignment(index, real)?;
                storage.imag.set_f64_assignment(index, imag)?;
            }
        }
        Ok(())
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
                let (real, imag) = values[index];
                Value::Complex(real, imag).to_string()
            }
            ComplexStorage::F32(values) => {
                let (real, imag) = values[index];
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

    pub fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::Integer(storage) => storage.real.numeric_dtype(),
        }
    }

    pub fn validate_shape(&self, shape: &[usize]) -> Result<(), String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        let Some(expected) = expected else {
            return Err(format!("complex tensor shape {shape:?} overflows usize"));
        };
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
            Self::F64(values) => values.clone(),
            Self::F32(values) => values
                .iter()
                .map(|&(real, imag)| (f64::from(real), f64::from(imag)))
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
                .collect::<Result<Vec<_>, _>>()
                .map(Self::F64),
            Self::F32(values) => indices
                .iter()
                .map(|&index| {
                    values
                        .get(index)
                        .copied()
                        .ok_or_else(|| format!("complex storage index {index} out of bounds"))
                })
                .collect::<Result<Vec<_>, _>>()
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

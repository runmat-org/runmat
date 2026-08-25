use runmat_value::{NumericDType, NumericScalar, NumericStorage};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct MxComplex64 {
    pub real: f64,
    pub imag: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct MxComplex32 {
    pub real: f32,
    pub imag: f32,
}

/// Interleaved-complex storage is kept separate from RunMat's canonical
/// storage because its C layout is part of the `-R2018a` compatibility API.
#[derive(Debug, Clone, PartialEq)]
pub enum MxInterleavedStorage {
    F64(Vec<MxComplex64>),
    F32(Vec<MxComplex32>),
    I8(Vec<(i8, i8)>),
    I16(Vec<(i16, i16)>),
    I32(Vec<(i32, i32)>),
    I64(Vec<(i64, i64)>),
    U8(Vec<(u8, u8)>),
    U16(Vec<(u16, u16)>),
    U32(Vec<(u32, u32)>),
    U64(Vec<(u64, u64)>),
}

impl MxInterleavedStorage {
    pub fn zeros(dtype: NumericDType, len: usize) -> Self {
        match dtype {
            NumericDType::F64 => Self::F64(vec![
                MxComplex64 {
                    real: 0.0,
                    imag: 0.0
                };
                len
            ]),
            NumericDType::F32 => Self::F32(vec![
                MxComplex32 {
                    real: 0.0,
                    imag: 0.0
                };
                len
            ]),
            NumericDType::I8 => Self::I8(vec![(0, 0); len]),
            NumericDType::I16 => Self::I16(vec![(0, 0); len]),
            NumericDType::I32 => Self::I32(vec![(0, 0); len]),
            NumericDType::I64 => Self::I64(vec![(0, 0); len]),
            NumericDType::U8 => Self::U8(vec![(0, 0); len]),
            NumericDType::U16 => Self::U16(vec![(0, 0); len]),
            NumericDType::U32 => Self::U32(vec![(0, 0); len]),
            NumericDType::U64 => Self::U64(vec![(0, 0); len]),
        }
    }

    pub const fn dtype(&self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::I8(_) => NumericDType::I8,
            Self::I16(_) => NumericDType::I16,
            Self::I32(_) => NumericDType::I32,
            Self::I64(_) => NumericDType::I64,
            Self::U8(_) => NumericDType::U8,
            Self::U16(_) => NumericDType::U16,
            Self::U32(_) => NumericDType::U32,
            Self::U64(_) => NumericDType::U64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn components(&self) -> (NumericStorage, NumericStorage) {
        macro_rules! split {
            ($values:expr, $variant:ident) => {{
                let (real, imag) = $values.iter().copied().unzip();
                (
                    NumericStorage::$variant(real),
                    NumericStorage::$variant(imag),
                )
            }};
        }
        match self {
            Self::F64(values) => (
                NumericStorage::F64(values.iter().map(|value| value.real).collect()),
                NumericStorage::F64(values.iter().map(|value| value.imag).collect()),
            ),
            Self::F32(values) => (
                NumericStorage::F32(values.iter().map(|value| value.real).collect()),
                NumericStorage::F32(values.iter().map(|value| value.imag).collect()),
            ),
            Self::I8(values) => split!(values, I8),
            Self::I16(values) => split!(values, I16),
            Self::I32(values) => split!(values, I32),
            Self::I64(values) => split!(values, I64),
            Self::U8(values) => split!(values, U8),
            Self::U16(values) => split!(values, U16),
            Self::U32(values) => split!(values, U32),
            Self::U64(values) => split!(values, U64),
        }
    }

    pub fn from_components(real: &NumericStorage, imag: &NumericStorage) -> Result<Self, String> {
        if real.numeric_dtype() != imag.numeric_dtype() || real.len() != imag.len() {
            return Err(
                "interleaved complex components must have matching class and length".into(),
            );
        }
        let mut output = Self::zeros(real.numeric_dtype(), real.len());
        for index in 0..real.len() {
            output.set(
                index,
                real.value_at(index).unwrap(),
                imag.value_at(index).unwrap(),
            )?;
        }
        Ok(output)
    }

    fn set(
        &mut self,
        index: usize,
        real: NumericScalar,
        imag: NumericScalar,
    ) -> Result<(), String> {
        macro_rules! assign {
            ($values:expr, $real:pat => $real_value:expr, $imag:pat => $imag_value:expr) => {{
                let destination = $values
                    .get_mut(index)
                    .ok_or_else(|| "complex index out of bounds".to_string())?;
                let ($real, $imag) = (real, imag) else {
                    return Err("complex component class mismatch".into());
                };
                *destination = ($real_value, $imag_value);
                Ok(())
            }};
        }
        match self {
            Self::F64(values) => {
                let (NumericScalar::F64(real), NumericScalar::F64(imag)) = (real, imag) else {
                    return Err("complex component class mismatch".into());
                };
                values[index] = MxComplex64 { real, imag };
                Ok(())
            }
            Self::F32(values) => {
                let (NumericScalar::F32(real), NumericScalar::F32(imag)) = (real, imag) else {
                    return Err("complex component class mismatch".into());
                };
                values[index] = MxComplex32 { real, imag };
                Ok(())
            }
            Self::I8(values) => {
                assign!(values, NumericScalar::I8(value_real) => value_real, NumericScalar::I8(value_imag) => value_imag)
            }
            Self::I16(values) => {
                assign!(values, NumericScalar::I16(value_real) => value_real, NumericScalar::I16(value_imag) => value_imag)
            }
            Self::I32(values) => {
                assign!(values, NumericScalar::I32(value_real) => value_real, NumericScalar::I32(value_imag) => value_imag)
            }
            Self::I64(values) => {
                assign!(values, NumericScalar::I64(value_real) => value_real, NumericScalar::I64(value_imag) => value_imag)
            }
            Self::U8(values) => {
                assign!(values, NumericScalar::U8(value_real) => value_real, NumericScalar::U8(value_imag) => value_imag)
            }
            Self::U16(values) => {
                assign!(values, NumericScalar::U16(value_real) => value_real, NumericScalar::U16(value_imag) => value_imag)
            }
            Self::U32(values) => {
                assign!(values, NumericScalar::U32(value_real) => value_real, NumericScalar::U32(value_imag) => value_imag)
            }
            Self::U64(values) => {
                assign!(values, NumericScalar::U64(value_real) => value_real, NumericScalar::U64(value_imag) => value_imag)
            }
        }
    }
}

use super::super::*;
use super::text::ShapeCopy;
use runmat_execution::value::{DenseValue, ElementType};

pub(in crate::execution::value_codec::decode::machine) struct DenseTask<'a> {
    source: &'a DenseValue,
    path: String,
    offset: usize,
    output: DenseOutput,
    shape: ShapeCopy<'a>,
}

pub(super) enum DenseOutput {
    Logical(Vec<u8>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    ComplexF32(Vec<(f32, f32)>),
    ComplexF64(Vec<(f64, f64)>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    ComplexI8(Vec<i8>, Vec<i8>),
    ComplexI16(Vec<i16>, Vec<i16>),
    ComplexI32(Vec<i32>, Vec<i32>),
    ComplexI64(Vec<i64>, Vec<i64>),
    ComplexU8(Vec<u8>, Vec<u8>),
    ComplexU16(Vec<u16>, Vec<u16>),
    ComplexU32(Vec<u32>, Vec<u32>),
    ComplexU64(Vec<u64>, Vec<u64>),
}

impl<'a> DenseTask<'a> {
    pub(super) fn new(source: &'a DenseValue, path: String) -> Result<Self, ValueCodecError> {
        let width = element_width(source.element_type);
        if !source.little_endian_data.len().is_multiple_of(width) {
            return Err(ValueCodecError::invalid(
                &path,
                if is_complex_integer(source.element_type) {
                    "complex integer byte length is not element aligned"
                } else {
                    "encoded data is not aligned to its element width"
                },
            ));
        }
        let elements = source.little_endian_data.len() / width;
        let output = DenseOutput::new(source.element_type, elements);
        let shape = ShapeCopy::new(&source.shape, path.clone());
        Ok(Self {
            source,
            path,
            offset: 0,
            output,
            shape,
        })
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }

    pub(super) fn is_complete(&self) -> bool {
        self.offset == self.source.little_endian_data.len() && self.shape.is_complete()
    }

    pub(super) fn advance(&mut self, maximum_work: usize) -> Result<usize, ValueCodecError> {
        let width = element_width(self.source.element_type);
        let available = self.source.little_endian_data.len() - self.offset;
        if available == 0 {
            return self.shape.advance(maximum_work);
        }
        let available_elements = available / width;
        let elements = available_elements.min(maximum_work);
        let end = self.offset + elements * width;
        for element in self.source.little_endian_data[self.offset..end].chunks_exact(width) {
            self.output.push(element);
        }
        self.offset = end;
        Ok(elements)
    }

    pub(super) fn finish(self) -> Result<Value, ValueCodecError> {
        let shape = self.shape.finish();
        self.output.finish(shape, &self.path)
    }
}

pub(super) fn is_complex_integer(element_type: ElementType) -> bool {
    matches!(
        element_type,
        ElementType::ComplexI8
            | ElementType::ComplexI16
            | ElementType::ComplexI32
            | ElementType::ComplexI64
            | ElementType::ComplexU8
            | ElementType::ComplexU16
            | ElementType::ComplexU32
            | ElementType::ComplexU64
    )
}

pub(super) fn element_width(element_type: ElementType) -> usize {
    match element_type {
        ElementType::Logical | ElementType::I8 | ElementType::U8 => 1,
        ElementType::I16 | ElementType::U16 | ElementType::ComplexI8 | ElementType::ComplexU8 => 2,
        ElementType::F32
        | ElementType::I32
        | ElementType::U32
        | ElementType::ComplexI16
        | ElementType::ComplexU16 => 4,
        ElementType::F64
        | ElementType::I64
        | ElementType::U64
        | ElementType::ComplexF32
        | ElementType::ComplexI32
        | ElementType::ComplexU32 => 8,
        ElementType::ComplexF64 | ElementType::ComplexI64 | ElementType::ComplexU64 => 16,
    }
}

impl DenseOutput {
    pub(super) fn new(element_type: ElementType, elements: usize) -> Self {
        match element_type {
            ElementType::Logical => Self::Logical(Vec::with_capacity(elements)),
            ElementType::F32 => Self::F32(Vec::with_capacity(elements)),
            ElementType::F64 => Self::F64(Vec::with_capacity(elements)),
            ElementType::ComplexF32 => Self::ComplexF32(Vec::with_capacity(elements)),
            ElementType::ComplexF64 => Self::ComplexF64(Vec::with_capacity(elements)),
            ElementType::I8 => Self::I8(Vec::with_capacity(elements)),
            ElementType::I16 => Self::I16(Vec::with_capacity(elements)),
            ElementType::I32 => Self::I32(Vec::with_capacity(elements)),
            ElementType::I64 => Self::I64(Vec::with_capacity(elements)),
            ElementType::U8 => Self::U8(Vec::with_capacity(elements)),
            ElementType::U16 => Self::U16(Vec::with_capacity(elements)),
            ElementType::U32 => Self::U32(Vec::with_capacity(elements)),
            ElementType::U64 => Self::U64(Vec::with_capacity(elements)),
            ElementType::ComplexI8 => {
                Self::ComplexI8(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexI16 => {
                Self::ComplexI16(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexI32 => {
                Self::ComplexI32(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexI64 => {
                Self::ComplexI64(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexU8 => {
                Self::ComplexU8(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexU16 => {
                Self::ComplexU16(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexU32 => {
                Self::ComplexU32(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
            ElementType::ComplexU64 => {
                Self::ComplexU64(Vec::with_capacity(elements), Vec::with_capacity(elements))
            }
        }
    }

    pub(super) fn push(&mut self, bytes: &[u8]) {
        macro_rules! word {
            ($n:literal, $ty:ty) => {{
                <$ty>::from_le_bytes(bytes.try_into().expect("validated element width"))
            }};
        }
        match self {
            Self::Logical(values) | Self::U8(values) => values.push(bytes[0]),
            Self::I8(values) => values.push(bytes[0] as i8),
            Self::F32(values) => values.push(f32::from_bits(word!(4, u32))),
            Self::F64(values) => values.push(f64::from_bits(word!(8, u64))),
            Self::I16(values) => values.push(word!(2, i16)),
            Self::I32(values) => values.push(word!(4, i32)),
            Self::I64(values) => values.push(word!(8, i64)),
            Self::U16(values) => values.push(word!(2, u16)),
            Self::U32(values) => values.push(word!(4, u32)),
            Self::U64(values) => values.push(word!(8, u64)),
            Self::ComplexF32(values) => values.push((
                f32::from_bits(u32::from_le_bytes(bytes[..4].try_into().unwrap())),
                f32::from_bits(u32::from_le_bytes(bytes[4..].try_into().unwrap())),
            )),
            Self::ComplexF64(values) => values.push((
                f64::from_bits(u64::from_le_bytes(bytes[..8].try_into().unwrap())),
                f64::from_bits(u64::from_le_bytes(bytes[8..].try_into().unwrap())),
            )),
            Self::ComplexI8(real, imag) => {
                real.push(bytes[0] as i8);
                imag.push(bytes[1] as i8);
            }
            Self::ComplexU8(real, imag) => {
                real.push(bytes[0]);
                imag.push(bytes[1]);
            }
            Self::ComplexI16(real, imag) => {
                push_pair::<2, i16>(bytes, real, imag, i16::from_le_bytes)
            }
            Self::ComplexI32(real, imag) => {
                push_pair::<4, i32>(bytes, real, imag, i32::from_le_bytes)
            }
            Self::ComplexI64(real, imag) => {
                push_pair::<8, i64>(bytes, real, imag, i64::from_le_bytes)
            }
            Self::ComplexU16(real, imag) => {
                push_pair::<2, u16>(bytes, real, imag, u16::from_le_bytes)
            }
            Self::ComplexU32(real, imag) => {
                push_pair::<4, u32>(bytes, real, imag, u32::from_le_bytes)
            }
            Self::ComplexU64(real, imag) => {
                push_pair::<8, u64>(bytes, real, imag, u64::from_le_bytes)
            }
        }
    }

    pub(super) fn into_sparse(
        self,
        parts: runmat_value::ValidatedSparseCscParts,
        path: &str,
    ) -> Result<Value, ValueCodecError> {
        use runmat_value::{HostComplexBuffer, HostNumericBuffer, NumericStorage};
        let sparse = match self {
            Self::Logical(values) => {
                if values.iter().any(|entry| *entry != 1) {
                    return Err(ValueCodecError::invalid(
                        path,
                        "logical sparse payload contains a stored false value",
                    ));
                }
                SparseTensor::from_validated_logical(parts)
            }
            Self::F32(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::F32(values)),
            ),
            Self::F64(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::F64(values)),
            ),
            Self::I8(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::I8(values)),
            ),
            Self::I16(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::I16(values)),
            ),
            Self::I32(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::I32(values)),
            ),
            Self::I64(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::I64(values)),
            ),
            Self::U8(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::U8(values)),
            ),
            Self::U16(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::U16(values)),
            ),
            Self::U32(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::U32(values)),
            ),
            Self::U64(values) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::U64(values)),
            ),
            Self::ComplexF32(values) => {
                SparseTensor::from_validated_complex_f32(parts, HostComplexBuffer::from(values))
            }
            Self::ComplexF64(values) => {
                SparseTensor::from_validated_complex_f64(parts, HostComplexBuffer::from(values))
            }
            Self::ComplexI8(..)
            | Self::ComplexI16(..)
            | Self::ComplexI32(..)
            | Self::ComplexI64(..)
            | Self::ComplexU8(..)
            | Self::ComplexU16(..)
            | Self::ComplexU32(..)
            | Self::ComplexU64(..) => {
                return Err(ValueCodecError::unsupported(
                    path,
                    "the runtime has no matching sparse value class",
                ));
            }
        }
        .map_err(|error| ValueCodecError::invalid(path, error))?;
        Ok(Value::SparseTensor(sparse))
    }

    fn finish(self, shape: Vec<usize>, path: &str) -> Result<Value, ValueCodecError> {
        macro_rules! tensor_int {
            ($v:expr, $k:ident) => {
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::$k($v), shape)
                        .map_err(|e| ValueCodecError::invalid(path, e))?,
                )
            };
        }
        macro_rules! complex_int {
            ($r:expr,$i:expr,$k:ident) => {
                Value::ComplexTensor(
                    ComplexTensor::new_integer(
                        IntegerComplexStorage::new(IntegerStorage::$k($r), IntegerStorage::$k($i))
                            .map_err(|e| ValueCodecError::invalid(path, e))?,
                        shape,
                    )
                    .map_err(|e| ValueCodecError::invalid(path, e))?,
                )
            };
        }
        Ok(match self {
            Self::Logical(v) => Value::LogicalArray(LogicalArray {
                data: runmat_value::HostLogicalBuffer::new(v),
                shape,
            }),
            Self::F32(v) => Value::Tensor(
                Tensor::from_f32(v, shape).map_err(|e| ValueCodecError::invalid(path, e))?,
            ),
            Self::F64(v) => {
                Value::Tensor(Tensor::new(v, shape).map_err(|e| ValueCodecError::invalid(path, e))?)
            }
            Self::ComplexF32(v) => Value::ComplexTensor(
                ComplexTensor::from_f32(v, shape).map_err(|e| ValueCodecError::invalid(path, e))?,
            ),
            Self::ComplexF64(v) => Value::ComplexTensor(
                ComplexTensor::new(v, shape).map_err(|e| ValueCodecError::invalid(path, e))?,
            ),
            Self::I8(v) => tensor_int!(v, I8),
            Self::I16(v) => tensor_int!(v, I16),
            Self::I32(v) => tensor_int!(v, I32),
            Self::I64(v) => tensor_int!(v, I64),
            Self::U8(v) => tensor_int!(v, U8),
            Self::U16(v) => tensor_int!(v, U16),
            Self::U32(v) => tensor_int!(v, U32),
            Self::U64(v) => tensor_int!(v, U64),
            Self::ComplexI8(r, i) => complex_int!(r, i, I8),
            Self::ComplexI16(r, i) => complex_int!(r, i, I16),
            Self::ComplexI32(r, i) => complex_int!(r, i, I32),
            Self::ComplexI64(r, i) => complex_int!(r, i, I64),
            Self::ComplexU8(r, i) => complex_int!(r, i, U8),
            Self::ComplexU16(r, i) => complex_int!(r, i, U16),
            Self::ComplexU32(r, i) => complex_int!(r, i, U32),
            Self::ComplexU64(r, i) => complex_int!(r, i, U64),
        })
    }
}

fn push_pair<const N: usize, T>(
    bytes: &[u8],
    real: &mut Vec<T>,
    imag: &mut Vec<T>,
    decode: fn([u8; N]) -> T,
) {
    real.push(decode(bytes[..N].try_into().unwrap()));
    imag.push(decode(bytes[N..].try_into().unwrap()));
}

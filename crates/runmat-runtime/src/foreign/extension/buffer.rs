use runmat_extension_abi::{
    RunMatBufferView, RunMatElementType, RUNMAT_BUFFER_COLUMN_MAJOR,
    RUNMAT_BUFFER_INTERLEAVED_COMPLEX, RUNMAT_BUFFER_READ_ONLY,
};
use runmat_value::{
    ComplexElement, ComplexStorage, HostComplexBuffer, HostLogicalBuffer, HostNumericBuffer,
    IntValue, Value,
};

#[derive(Debug)]
pub(super) enum LeasePayload {
    NumericScalar(f64),
    IntegerScalar(IntValue),
    LogicalScalar(u8),
    ComplexScalar(ComplexElement<f64>),
    Numeric(HostNumericBuffer),
    Logical(HostLogicalBuffer),
    ComplexF64(HostComplexBuffer<f64>),
    ComplexF32(HostComplexBuffer<f32>),
}

#[derive(Debug)]
pub(super) struct BufferLease {
    payload: LeasePayload,
    shape: Box<[usize]>,
}

impl BufferLease {
    pub(super) fn from_value(value: &Value) -> Result<Self, ()> {
        let (payload, shape) = match value {
            Value::Num(value) => (LeasePayload::NumericScalar(*value), vec![1, 1]),
            Value::Int(value) => (LeasePayload::IntegerScalar(value.clone()), vec![1, 1]),
            Value::Bool(value) => (LeasePayload::LogicalScalar(u8::from(*value)), vec![1, 1]),
            Value::Complex(real, imaginary) => (
                LeasePayload::ComplexScalar(ComplexElement(*real, *imaginary)),
                vec![1, 1],
            ),
            Value::Tensor(value) => (
                LeasePayload::Numeric(value.host_buffer().clone()),
                value.shape.clone(),
            ),
            Value::LogicalArray(value) => (
                LeasePayload::Logical(value.data.clone()),
                value.shape.clone(),
            ),
            Value::ComplexTensor(value) => match value.complex_storage() {
                ComplexStorage::F64(values) => (
                    LeasePayload::ComplexF64(values.clone()),
                    value.shape.clone(),
                ),
                ComplexStorage::F32(values) => (
                    LeasePayload::ComplexF32(values.clone()),
                    value.shape.clone(),
                ),
                ComplexStorage::Integer(_) => {
                    return Err(());
                }
            },
            _ => return Err(()),
        };
        Ok(Self {
            payload,
            shape: shape.into_boxed_slice(),
        })
    }

    pub(super) fn view(&self) -> RunMatBufferView {
        let (data, byte_length, element_type, complex) = match &self.payload {
            LeasePayload::LogicalScalar(value) => (
                std::ptr::from_ref(value).cast::<u8>(),
                1,
                RunMatElementType::Logical,
                false,
            ),
            LeasePayload::ComplexScalar(value) => (
                std::ptr::from_ref(value).cast::<u8>(),
                std::mem::size_of::<ComplexElement<f64>>(),
                RunMatElementType::ComplexF64,
                true,
            ),
            LeasePayload::NumericScalar(value) => (
                std::ptr::from_ref(value).cast::<u8>(),
                std::mem::size_of::<f64>(),
                RunMatElementType::F64,
                false,
            ),
            LeasePayload::IntegerScalar(value) => int_view(value),
            LeasePayload::Numeric(value) => {
                let element_type = element_type(value.numeric_dtype());
                // SAFETY: this lease owns a shared copy of the canonical host buffer.
                let data = unsafe { value.foreign_data_pointer() }
                    .cast::<u8>()
                    .cast_const();
                (
                    data,
                    value.checked_byte_len().unwrap_or(0),
                    element_type,
                    false,
                )
            }
            LeasePayload::Logical(value) => {
                // SAFETY: this lease owns a shared copy of the canonical host buffer.
                let data = unsafe { value.foreign_data_pointer() }
                    .cast::<u8>()
                    .cast_const();
                (data, value.len(), RunMatElementType::Logical, false)
            }
            LeasePayload::ComplexF64(values) => {
                // SAFETY: this lease retains the canonical interleaved buffer.
                let data = unsafe { values.foreign_data_pointer() }
                    .cast::<u8>()
                    .cast_const();
                (
                    data,
                    values.len() * std::mem::size_of::<ComplexElement<f64>>(),
                    RunMatElementType::ComplexF64,
                    true,
                )
            }
            LeasePayload::ComplexF32(values) => {
                // SAFETY: this lease retains the canonical interleaved buffer.
                let data = unsafe { values.foreign_data_pointer() }
                    .cast::<u8>()
                    .cast_const();
                (
                    data,
                    values.len() * std::mem::size_of::<ComplexElement<f32>>(),
                    RunMatElementType::ComplexF32,
                    true,
                )
            }
        };
        let mut flags = RUNMAT_BUFFER_READ_ONLY | RUNMAT_BUFFER_COLUMN_MAJOR;
        if complex {
            flags |= RUNMAT_BUFFER_INTERLEAVED_COMPLEX;
        }
        RunMatBufferView {
            data,
            byte_length,
            shape: self.shape.as_ptr(),
            rank: self.shape.len(),
            element_type,
            flags,
        }
    }
}

fn element_type(dtype: runmat_value::NumericDType) -> RunMatElementType {
    use runmat_value::NumericDType;
    match dtype {
        NumericDType::F64 => RunMatElementType::F64,
        NumericDType::F32 => RunMatElementType::F32,
        NumericDType::I8 => RunMatElementType::I8,
        NumericDType::I16 => RunMatElementType::I16,
        NumericDType::I32 => RunMatElementType::I32,
        NumericDType::I64 => RunMatElementType::I64,
        NumericDType::U8 => RunMatElementType::U8,
        NumericDType::U16 => RunMatElementType::U16,
        NumericDType::U32 => RunMatElementType::U32,
        NumericDType::U64 => RunMatElementType::U64,
    }
}

fn int_view(value: &IntValue) -> (*const u8, usize, RunMatElementType, bool) {
    macro_rules! scalar {
        ($value:expr, $type:expr) => {
            (
                std::ptr::from_ref($value).cast::<u8>(),
                std::mem::size_of_val($value),
                $type,
                false,
            )
        };
    }
    match value {
        IntValue::I8(value) => scalar!(value, RunMatElementType::I8),
        IntValue::I16(value) => scalar!(value, RunMatElementType::I16),
        IntValue::I32(value) => scalar!(value, RunMatElementType::I32),
        IntValue::I64(value) => scalar!(value, RunMatElementType::I64),
        IntValue::U8(value) => scalar!(value, RunMatElementType::U8),
        IntValue::U16(value) => scalar!(value, RunMatElementType::U16),
        IntValue::U32(value) => scalar!(value, RunMatElementType::U32),
        IntValue::U64(value) => scalar!(value, RunMatElementType::U64),
    }
}

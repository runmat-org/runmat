use super::*;

pub(super) enum DenseAdmission {
    Numeric(NumericTensorAdmission),
    Complex(ComplexTensorAdmission),
    Ready(Value),
}

impl DenseAdmission {
    pub(super) fn advance(
        &mut self,
        maximum_work: NonZeroUsize,
        path: &str,
    ) -> Result<usize, ValueCodecError> {
        match self {
            Self::Numeric(admission) => admission
                .advance(maximum_work)
                .map_err(|error| ValueCodecError::invalid(path, error)),
            Self::Complex(admission) => admission
                .advance(maximum_work)
                .map_err(|error| ValueCodecError::invalid(path, error)),
            Self::Ready(_) => Ok(1),
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        match self {
            Self::Numeric(admission) => admission.is_complete(),
            Self::Complex(admission) => admission.is_complete(),
            Self::Ready(_) => true,
        }
    }

    pub(super) fn into_value(self) -> Value {
        match self {
            Self::Numeric(admission) => {
                let tensor: Tensor = admission
                    .into_validated()
                    .ok()
                    .expect("completed numeric admission")
                    .into_tensor();
                Value::Tensor(tensor)
            }
            Self::Complex(admission) => {
                let tensor: ComplexTensor = admission
                    .into_validated()
                    .ok()
                    .expect("completed complex admission")
                    .into_tensor();
                Value::ComplexTensor(tensor)
            }
            Self::Ready(value) => value,
        }
    }
}

impl DenseOutput {
    pub(super) fn into_admission(
        self,
        shape: Vec<usize>,
        path: &str,
    ) -> Result<DenseAdmission, ValueCodecError> {
        macro_rules! tensor_int {
            ($v:expr, $k:ident) => {
                DenseAdmission::Numeric(NumericTensorAdmission::from_numeric_storage(
                    NumericStorage::$k($v),
                    shape,
                ))
            };
        }
        macro_rules! complex_int {
            ($r:expr,$i:expr,$k:ident) => {
                DenseAdmission::Complex(
                    ComplexTensorAdmission::new(
                        ComplexStorage::Integer(
                            IntegerComplexStorage::new(
                                IntegerStorage::$k($r),
                                IntegerStorage::$k($i),
                            )
                            .map_err(|error| ValueCodecError::invalid(path, error))?,
                        ),
                        shape,
                    )
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
                )
            };
        }
        Ok(match self {
            Self::Logical(v) => DenseAdmission::Ready(Value::LogicalArray(LogicalArray {
                data: runmat_value::HostLogicalBuffer::new(v),
                shape: shape.into(),
            })),
            Self::F32(v) => DenseAdmission::Numeric(NumericTensorAdmission::from_numeric_storage(
                NumericStorage::F32(v),
                shape,
            )),
            Self::F64(v) => DenseAdmission::Numeric(NumericTensorAdmission::from_numeric_storage(
                NumericStorage::F64(v),
                shape,
            )),
            Self::ComplexF32(v) => DenseAdmission::Complex(
                ComplexTensorAdmission::new(ComplexStorage::F32(v.into()), shape)
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
            ),
            Self::ComplexF64(v) => DenseAdmission::Complex(
                ComplexTensorAdmission::new(ComplexStorage::F64(v.into()), shape)
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_decode_allocations_become_tensor_storage_without_terminal_repack() {
        let values = vec![ComplexElement(1.0_f64, -2.0), ComplexElement(3.0_f64, -4.0)];
        let original_pointer = values.as_ptr();
        let mut admission = DenseOutput::ComplexF64(values)
            .into_admission(vec![1, 2], "$")
            .expect("complex admission");
        while !admission.is_complete() {
            admission
                .advance(NonZeroUsize::new(1).unwrap(), "$")
                .expect("admission step");
        }
        let Value::ComplexTensor(tensor) = admission.into_value() else {
            panic!("complex tensor result");
        };
        assert_eq!(
            tensor.as_f64_slice().expect("double complex").as_ptr(),
            original_pointer
        );

        let values = vec![ComplexElement(1.0_f32, -2.0), ComplexElement(3.0_f32, -4.0)];
        let original_pointer = values.as_ptr();
        let mut admission = DenseOutput::ComplexF32(values)
            .into_admission(vec![1, 2], "$")
            .expect("single complex admission");
        while !admission.is_complete() {
            admission
                .advance(NonZeroUsize::new(1).unwrap(), "$")
                .expect("admission step");
        }
        let Value::ComplexTensor(tensor) = admission.into_value() else {
            panic!("complex tensor result");
        };
        assert_eq!(
            tensor.as_f32_slice().expect("single complex").as_ptr(),
            original_pointer
        );
    }
}

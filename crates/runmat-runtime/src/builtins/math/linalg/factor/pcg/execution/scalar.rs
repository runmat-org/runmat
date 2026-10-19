use super::{
    invalid, BuiltinResult, ComplexElement, ComplexStorage, ComplexTensor, Diagonal,
    ExactHostAllocationPermit, ExactHostComplexTensorAllocation, ExactHostNumericAllocationError,
    ExactHostTensorAllocation, HostAllocationError, HostAllocationPlan, HostNumericWorkspace,
    NativePcgVector, NumericDType, RuntimeContext, SparseOperatorError, Tensor, Triangle,
    TriangularLinearOperator, TriangularOperator, TriangularOperatorError, TypedLinearOperator,
    Value, VectorClass,
};

pub(super) trait PcgValue: NativePcgVector {
    const DTYPE: NumericDType;
    const LANES: usize;
    fn vector(value: &Value) -> Option<&[Self]>;
    fn scratch(workspace: &mut HostNumericWorkspace) -> Option<&mut [Self]>;
    fn output_bytes(count: usize) -> Result<u64, HostAllocationError>;
    fn admitted_output(
        values: &[Self],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Value, ExactHostNumericAllocationError>;
    fn unbounded_output(values: Vec<Self>) -> BuiltinResult<Value>;
    fn matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: Option<&RuntimeContext>,
    ) -> Result<(), SparseOperatorError>;
    fn triangle<'a>(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError>;
    const VECTOR_CLASS: VectorClass;
}

fn select_triangle<'a>(
    value: &'a Value,
    context: &RuntimeContext,
) -> Result<TriangularLinearOperator<'a>, TriangularOperatorError> {
    match TriangularLinearOperator::borrow_with_context(
        value,
        Triangle::Lower,
        Diagonal::Explicit,
        context,
    ) {
        Ok(operator) => Ok(operator),
        Err(TriangularOperatorError::NonTriangular { .. }) => {
            TriangularLinearOperator::borrow_with_context(
                value,
                Triangle::Upper,
                Diagonal::Explicit,
                context,
            )
        }
        Err(error) => Err(error),
    }
}

pub(super) fn vector_shape(shape: &[usize], count: usize) -> bool {
    shape.first() == Some(&count) && shape.iter().skip(1).all(|&extent| extent == 1)
}

impl PcgValue for f32 {
    const DTYPE: NumericDType = NumericDType::F32;
    const LANES: usize = 1;
    const VECTOR_CLASS: VectorClass = VectorClass::RealSingle;
    fn vector(value: &Value) -> Option<&[Self]> {
        let Value::Tensor(tensor) = value else {
            return None;
        };
        tensor.as_f32_slice()
    }
    fn scratch(workspace: &mut HostNumericWorkspace) -> Option<&mut [Self]> {
        workspace.as_f32_slice_mut()
    }
    fn output_bytes(count: usize) -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_tensor_bytes_for_dimensions(Self::DTYPE, count, 2)
    }
    fn admitted_output(
        values: &[Self],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Value, ExactHostNumericAllocationError> {
        let mut output = ExactHostTensorAllocation::new(Self::DTYPE, &[values.len(), 1], permit)?;
        output
            .as_f32_slice_mut()
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?
            .copy_from_slice(values);
        output.into_tensor().map(Value::Tensor)
    }
    fn unbounded_output(values: Vec<Self>) -> BuiltinResult<Value> {
        let count = values.len();
        Tensor::from_f32(values, vec![count, 1])
            .map(Value::Tensor)
            .map_err(|_| invalid("unable to retain result vector"))
    }
    fn matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: Option<&RuntimeContext>,
    ) -> Result<(), SparseOperatorError> {
        match context {
            Some(context) => operator.apply_into(input, output, context),
            None => operator.apply_into_unbounded(input, output),
        }
    }
    fn triangle<'a>(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError> {
        match select_triangle(value, context)? {
            TriangularLinearOperator::RealF32(operator) => Ok(operator),
            _ => Err(TriangularOperatorError::UnsupportedClass),
        }
    }
}

impl PcgValue for f64 {
    const DTYPE: NumericDType = NumericDType::F64;
    const LANES: usize = 1;
    const VECTOR_CLASS: VectorClass = VectorClass::RealDouble;
    fn vector(value: &Value) -> Option<&[Self]> {
        let Value::Tensor(tensor) = value else {
            return None;
        };
        tensor.as_f64_slice()
    }
    fn scratch(workspace: &mut HostNumericWorkspace) -> Option<&mut [Self]> {
        workspace.as_f64_slice_mut()
    }
    fn output_bytes(count: usize) -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_tensor_bytes_for_dimensions(Self::DTYPE, count, 2)
    }
    fn admitted_output(
        values: &[Self],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Value, ExactHostNumericAllocationError> {
        let mut output = ExactHostTensorAllocation::new(Self::DTYPE, &[values.len(), 1], permit)?;
        output
            .as_f64_slice_mut()
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?
            .copy_from_slice(values);
        output.into_tensor().map(Value::Tensor)
    }
    fn unbounded_output(values: Vec<Self>) -> BuiltinResult<Value> {
        let count = values.len();
        Tensor::new(values, vec![count, 1])
            .map(Value::Tensor)
            .map_err(|_| invalid("unable to retain result vector"))
    }
    fn matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: Option<&RuntimeContext>,
    ) -> Result<(), SparseOperatorError> {
        match context {
            Some(context) => operator.apply_into(input, output, context),
            None => operator.apply_into_unbounded(input, output),
        }
    }
    fn triangle<'a>(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError> {
        match select_triangle(value, context)? {
            TriangularLinearOperator::RealF64(operator) => Ok(operator),
            _ => Err(TriangularOperatorError::UnsupportedClass),
        }
    }
}

macro_rules! complex_value {
    ($scalar:ty, $dtype:expr, $view:ident, $scratch:ident, $storage:ident, $class:expr, $variant:ident) => {
        impl PcgValue for ComplexElement<$scalar> {
            const DTYPE: NumericDType = $dtype;
            const LANES: usize = 2;
            const VECTOR_CLASS: VectorClass = $class;
            fn vector(value: &Value) -> Option<&[Self]> {
                let Value::ComplexTensor(tensor) = value else {
                    return None;
                };
                tensor.$view()
            }
            fn scratch(workspace: &mut HostNumericWorkspace) -> Option<&mut [Self]> {
                workspace.$scratch()
            }
            fn output_bytes(count: usize) -> Result<u64, HostAllocationError> {
                HostAllocationPlan::retained_complex_tensor_bytes_for_dimensions($dtype, count, 2)
            }
            fn admitted_output(
                values: &[Self],
                permit: &dyn ExactHostAllocationPermit,
            ) -> Result<Value, ExactHostNumericAllocationError> {
                let mut output =
                    ExactHostComplexTensorAllocation::new($dtype, &[values.len(), 1], permit)?;
                for (index, value) in values.iter().enumerate() {
                    output.set(index, f64::from(value.0), f64::from(value.1))?;
                }
                output.into_tensor().map(Value::ComplexTensor)
            }
            fn unbounded_output(values: Vec<Self>) -> BuiltinResult<Value> {
                let count = values.len();
                ComplexTensor::from_complex_storage(
                    ComplexStorage::$storage(values.into()),
                    vec![count, 1],
                )
                .map(Value::ComplexTensor)
                .map_err(|_| invalid("unable to retain complex result vector"))
            }
            fn matvec(
                operator: &TypedLinearOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                context: Option<&RuntimeContext>,
            ) -> Result<(), SparseOperatorError> {
                match context {
                    Some(context) => operator.apply_into(input, output, context),
                    None => operator.apply_into_unbounded(input, output),
                }
            }
            fn triangle<'a>(
                value: &'a Value,
                context: &RuntimeContext,
            ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError> {
                match select_triangle(value, context)? {
                    TriangularLinearOperator::$variant(operator) => Ok(operator),
                    _ => Err(TriangularOperatorError::UnsupportedClass),
                }
            }
        }
    };
}

complex_value!(
    f32,
    NumericDType::F32,
    as_f32_slice,
    as_complex_f32_slice_mut,
    F32,
    VectorClass::ComplexSingle,
    ComplexF32
);
complex_value!(
    f64,
    NumericDType::F64,
    as_f64_slice,
    as_complex_f64_slice_mut,
    F64,
    VectorClass::ComplexDouble,
    ComplexF64
);

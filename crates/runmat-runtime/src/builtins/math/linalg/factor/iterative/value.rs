use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostAllocationPermit,
    ExactHostComplexTensorAllocation, ExactHostNumericAllocationError, ExactHostTensorAllocation,
    NumericDType, Tensor, Value,
};

use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator;
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostNumericWorkspace,
};
use crate::context::RuntimeContext;

pub(in crate::builtins::math::linalg::factor) trait IterativeValue:
    crate::builtins::math::linalg::ops::linear_operator::NativeScalarView
{
    fn input(value: &Value) -> Option<NativeVector<'_, Self>> {
        match Self::scalar(value) {
            Some(value) => Some(NativeVector::Scalar(value)),
            None => Self::vector(value).map(NativeVector::Borrowed),
        }
    }
    const DTYPE: NumericDType;
    const LANES: usize;
    fn vector(value: &Value) -> Option<&[Self]>;
    fn scratch(workspace: &mut HostNumericWorkspace) -> Option<&mut [Self]>;
    fn output_bytes(count: usize) -> Result<u64, HostAllocationError>;
    type Output;
    fn allocate_output(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self::Output, ExactHostNumericAllocationError>;
    fn copy_output(
        output: &mut Self::Output,
        offset: usize,
        values: &[Self],
    ) -> Result<(), ExactHostNumericAllocationError>;
    fn finish_output(output: Self::Output) -> Result<Value, ExactHostNumericAllocationError>;
    fn unbounded_output(values: Vec<Self>) -> Result<Value, &'static str>;
    async fn cooperative_matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: &RuntimeContext,
        budget: &mut crate::execution::CpuCooperationBudget,
    ) -> Result<(), crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError>;
    fn matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: Option<&RuntimeContext>,
    ) -> Result<(), SparseOperatorError>;
}

pub(in crate::builtins::math::linalg::factor) enum NativeVector<'a, T> {
    Scalar(T),
    Borrowed(&'a [T]),
}
impl<T> NativeVector<'_, T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Scalar(value) => std::slice::from_ref(value),
            Self::Borrowed(values) => values,
        }
    }
}

pub(in crate::builtins::math::linalg::factor) fn vector_shape(
    shape: &[usize],
    count: usize,
) -> bool {
    shape.first() == Some(&count) && shape.iter().skip(1).all(|&extent| extent == 1)
}

impl IterativeValue for f32 {
    const DTYPE: NumericDType = NumericDType::F32;
    const LANES: usize = 1;
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
    type Output = ExactHostTensorAllocation;
    fn allocate_output(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self::Output, ExactHostNumericAllocationError> {
        ExactHostTensorAllocation::new(Self::DTYPE, &[count, 1], permit)
    }
    fn copy_output(
        output: &mut Self::Output,
        offset: usize,
        values: &[Self],
    ) -> Result<(), ExactHostNumericAllocationError> {
        let target = output
            .as_f32_slice_mut()
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?;
        target[offset..offset + values.len()].copy_from_slice(values);
        Ok(())
    }
    fn finish_output(output: Self::Output) -> Result<Value, ExactHostNumericAllocationError> {
        output.into_tensor().map(Value::Tensor)
    }
    fn unbounded_output(values: Vec<Self>) -> Result<Value, &'static str> {
        let count = values.len();
        Tensor::from_f32(values, vec![count, 1])
            .map(Value::Tensor)
            .map_err(|_| "unable to retain result vector")
    }
    async fn cooperative_matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: &RuntimeContext,
        budget: &mut crate::execution::CpuCooperationBudget,
    ) -> Result<(), crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError>
    {
        operator
            .apply_cooperatively(input, output, context, budget)
            .await
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
}

impl IterativeValue for f64 {
    const DTYPE: NumericDType = NumericDType::F64;
    const LANES: usize = 1;
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
    type Output = ExactHostTensorAllocation;
    fn allocate_output(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self::Output, ExactHostNumericAllocationError> {
        ExactHostTensorAllocation::new(Self::DTYPE, &[count, 1], permit)
    }
    fn copy_output(
        output: &mut Self::Output,
        offset: usize,
        values: &[Self],
    ) -> Result<(), ExactHostNumericAllocationError> {
        let target = output
            .as_f64_slice_mut()
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?;
        target[offset..offset + values.len()].copy_from_slice(values);
        Ok(())
    }
    fn finish_output(output: Self::Output) -> Result<Value, ExactHostNumericAllocationError> {
        output.into_tensor().map(Value::Tensor)
    }
    fn unbounded_output(values: Vec<Self>) -> Result<Value, &'static str> {
        let count = values.len();
        Tensor::new(values, vec![count, 1])
            .map(Value::Tensor)
            .map_err(|_| "unable to retain result vector")
    }
    async fn cooperative_matvec(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: &RuntimeContext,
        budget: &mut crate::execution::CpuCooperationBudget,
    ) -> Result<(), crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError>
    {
        operator
            .apply_cooperatively(input, output, context, budget)
            .await
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
}

macro_rules! complex_value {
    ($scalar:ty, $dtype:expr, $view:ident, $scratch:ident, $storage:ident) => {
        impl IterativeValue for ComplexElement<$scalar> {
            const DTYPE: NumericDType = $dtype;
            const LANES: usize = 2;
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
            type Output = ExactHostComplexTensorAllocation;
            fn allocate_output(
                count: usize,
                permit: &dyn ExactHostAllocationPermit,
            ) -> Result<Self::Output, ExactHostNumericAllocationError> {
                ExactHostComplexTensorAllocation::new($dtype, &[count, 1], permit)
            }
            fn copy_output(
                output: &mut Self::Output,
                offset: usize,
                values: &[Self],
            ) -> Result<(), ExactHostNumericAllocationError> {
                for (index, value) in values.iter().enumerate() {
                    output.set(offset + index, f64::from(value.0), f64::from(value.1))?;
                }
                Ok(())
            }
            fn finish_output(
                output: Self::Output,
            ) -> Result<Value, ExactHostNumericAllocationError> {
                output.into_tensor().map(Value::ComplexTensor)
            }
            fn unbounded_output(values: Vec<Self>) -> Result<Value, &'static str> {
                let count = values.len();
                ComplexTensor::from_complex_storage(
                    ComplexStorage::$storage(values.into()),
                    vec![count, 1],
                )
                .map(Value::ComplexTensor)
                .map_err(|_| "unable to retain complex result vector")
            }
            async fn cooperative_matvec(
                operator: &TypedLinearOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                context: &RuntimeContext,
                budget: &mut crate::execution::CpuCooperationBudget,
            ) -> Result<
                (),
                crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError,
            > {
                operator
                    .apply_cooperatively(input, output, context, budget)
                    .await
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
        }
    };
}

complex_value!(
    f32,
    NumericDType::F32,
    as_f32_slice,
    as_complex_f32_slice_mut,
    F32
);
complex_value!(
    f64,
    NumericDType::F64,
    as_f64_slice,
    as_complex_f64_slice_mut,
    F64
);

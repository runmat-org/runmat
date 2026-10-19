use super::*;
use runmat_accelerate_api::{
    NumericElementType, ProviderElementType, ProviderExecutedRealBinary, ProviderExecutionError,
    ProviderExecutionResult, ProviderRealBinaryObserved, ProviderRealBinaryOperation,
    ProviderRealBinaryPhysicalProposal, ProviderRealBinaryRecipe, ProviderRealBinaryRequest,
    ProviderRealBinaryRoute,
};
use runmat_runtime::builtins::common::broadcast::{broadcast_index, compute_strides};

fn concrete_shape(shape: &[u64]) -> Option<Vec<usize>> {
    shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()
}

fn element_type(class: ProviderElementType) -> Option<NumericElementType> {
    match class {
        ProviderElementType::F32 => Some(NumericElementType::F32),
        ProviderElementType::F64 => Some(NumericElementType::F64),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(in super::super) enum RealBinaryArithmetic {
    Sub,
    Div,
    Mul,
}

impl From<ProviderRealBinaryOperation> for RealBinaryArithmetic {
    fn from(operation: ProviderRealBinaryOperation) -> Self {
        match operation {
            ProviderRealBinaryOperation::Sub => Self::Sub,
            ProviderRealBinaryOperation::Div => Self::Div,
        }
    }
}

fn binary_values(
    left: &[f64],
    left_shape: &[usize],
    right: &[f64],
    right_shape: &[usize],
    output_shape: &[usize],
    class: NumericElementType,
    operation: RealBinaryArithmetic,
) -> ProviderExecutionResult<Vec<f64>> {
    let count = output_shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| ProviderExecutionError::from(anyhow!("binary output size overflows")))?;
    let left_stride = compute_strides(left_shape);
    let right_stride = compute_strides(right_shape);
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(ProviderExecutionError::from_source)?;
    for index in 0..count {
        let l = left[broadcast_index(index, output_shape, left_shape, &left_stride)];
        let r = right[broadcast_index(index, output_shape, right_shape, &right_stride)];
        let value = match (class, operation) {
            (NumericElementType::F32, RealBinaryArithmetic::Sub) => {
                f64::from((l as f32) - (r as f32))
            }
            (NumericElementType::F32, RealBinaryArithmetic::Div) => {
                f64::from((l as f32) / (r as f32))
            }
            (NumericElementType::F32, RealBinaryArithmetic::Mul) => {
                f64::from((l as f32) * (r as f32))
            }
            (NumericElementType::F64, RealBinaryArithmetic::Sub) => l - r,
            (NumericElementType::F64, RealBinaryArithmetic::Div) => l / r,
            (NumericElementType::F64, RealBinaryArithmetic::Mul) => l * r,
            _ => {
                return Err(ProviderExecutionError::from(anyhow!(
                    "unsupported real binary class"
                )))
            }
        };
        values.push(value);
    }
    Ok(values)
}

impl TestProvider {
    pub(in super::super) fn ordinary_real_repmat(
        &self,
        input: &GpuTensorHandle,
        repetitions: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        let class = input.physical().element_type();
        if !matches!(class, NumericElementType::F32 | NumericElementType::F64)
            || input.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
        {
            return Ok(None);
        }
        let (data, shape) = self.pull(input).map_err(ProviderExecutionError::from)?;
        let rank = shape.len().max(repetitions.len());
        let mut output_shape = Vec::new();
        output_shape
            .try_reserve_exact(rank)
            .map_err(ProviderExecutionError::from_source)?;
        for axis in 0..rank {
            let extent = shape.get(axis).copied().unwrap_or(1);
            let repeat = repetitions.get(axis).copied().unwrap_or(1);
            let Some(total) = extent.checked_mul(repeat) else {
                return Ok(None);
            };
            output_shape.push(total);
        }
        let Some(count) = output_shape
            .iter()
            .try_fold(1usize, |total, &extent| total.checked_mul(extent))
        else {
            return Ok(None);
        };
        let source_strides = compute_strides(&shape);
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(ProviderExecutionError::from_source)?;
        for linear in 0..count {
            let mut remaining = linear;
            let mut source_index = 0usize;
            for (axis, &extent) in output_shape.iter().enumerate() {
                let coordinate = remaining % extent;
                remaining /= extent;
                if let Some(&source_extent) = shape.get(axis) {
                    source_index += (coordinate % source_extent) * source_strides[axis];
                }
            }
            values.push(data[source_index]);
        }
        Ok(Some(self.push_typed(values, output_shape, class)))
    }

    pub(in super::super) fn ordinary_real_binary(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        operation: ProviderRealBinaryOperation,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.ordinary_real_arithmetic(lhs, rhs, operation.into())
    }

    pub(in super::super) fn ordinary_real_arithmetic(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        operation: RealBinaryArithmetic,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let class = lhs.physical().element_type();
        if class != rhs.physical().element_type()
            || !matches!(class, NumericElementType::F32 | NumericElementType::F64)
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "ordinary real binary operands have incompatible classes"
            )));
        }
        let (left, left_shape) = self.pull(lhs).map_err(ProviderExecutionError::from)?;
        let (right, right_shape) = self.pull(rhs).map_err(ProviderExecutionError::from)?;
        let output_shape =
            runmat_types::geometry::broadcast_concrete_shape(&left_shape, &right_shape)
                .map_err(|error| ProviderExecutionError::from(anyhow!("{error}")))?;
        let values = binary_values(
            &left,
            &left_shape,
            &right,
            &right_shape,
            &output_shape,
            class,
            operation,
        )?;
        Ok(self.push_typed(values, output_shape, class))
    }

    pub(in super::super) fn staged_binary_proposal(
        &self,
        request: &ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealBinaryPhysicalProposal>> {
        let lhs = &request.lhs.representation;
        let rhs = &request.rhs.representation;
        if lhs.element_type != rhs.element_type
            || element_type(lhs.element_type).is_none()
            || lhs
                .shape
                .iter()
                .chain(&rhs.shape)
                .any(|&extent| extent == 0)
            || lhs.checked_byte_len() != Some(request.lhs.buffer_bytes)
            || rhs.checked_byte_len() != Some(request.rhs.buffer_bytes)
        {
            return Ok(None);
        }
        let (Some(lhs_shape), Some(rhs_shape)) =
            (concrete_shape(&lhs.shape), concrete_shape(&rhs.shape))
        else {
            return Ok(None);
        };
        let Ok(shape) = runmat_types::geometry::broadcast_concrete_shape(&lhs_shape, &rhs_shape)
        else {
            return Ok(None);
        };
        let Some(shape) = shape
            .into_iter()
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()
        else {
            return Ok(None);
        };
        let mut output = lhs.clone();
        output.shape = shape;
        let Some(output_buffer_bytes) = output.checked_byte_len() else {
            return Ok(None);
        };
        Ok(Some(ProviderRealBinaryPhysicalProposal {
            route: if lhs.shape == rhs.shape {
                ProviderRealBinaryRoute::SameShape
            } else {
                ProviderRealBinaryRoute::Broadcast
            },
            output,
            output_buffer_bytes,
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: 0,
            uniform_buffer_bytes: 0,
            dispatches: 1,
            pre_output_submissions: 0,
            queue_submissions: 1,
        }))
    }

    pub(in super::super) fn staged_binary_execute(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        recipe: &ProviderRealBinaryRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedRealBinary> {
        if !recipe.is_owned_by(self) {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected binary recipe has a foreign owner"
            )));
        }
        let (left, left_shape) = self.pull(lhs).map_err(ProviderExecutionError::from)?;
        let (right, right_shape) = self.pull(rhs).map_err(ProviderExecutionError::from)?;
        let physical = recipe.physical();
        let output_shape = concrete_shape(&physical.output.shape).ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected binary output shape overflows"))
        })?;
        let Some(class) = element_type(physical.output.element_type) else {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected binary output class is unsupported"
            )));
        };
        let values = binary_values(
            &left,
            &left_shape,
            &right,
            &right_shape,
            &output_shape,
            class,
            recipe.request().operation.into(),
        )?;
        let output = self.push_typed(values, output_shape, class);
        Ok(ProviderExecutedRealBinary {
            output,
            observed: ProviderRealBinaryObserved {
                output_buffer_bytes: Some(physical.output_buffer_bytes),
                uniform_allocations: physical.uniform_allocations,
                uniform_buffer_bytes: physical.uniform_buffer_bytes,
                dispatches: physical.dispatches,
                pre_output_submissions: physical.pre_output_submissions,
                queue_submissions: physical.queue_submissions,
            },
        })
    }
}

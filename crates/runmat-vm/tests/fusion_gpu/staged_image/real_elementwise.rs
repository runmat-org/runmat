use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, NumericElementType, ProviderElementType,
    ProviderExecutedRealElementwise, ProviderExecutionError, ProviderExecutionResult,
    ProviderPrecision, ProviderRealElementwiseObserved, ProviderRealElementwiseOperation,
    ProviderRealElementwiseOutputRoute, ProviderRealElementwisePhysicalProposal,
    ProviderRealElementwiseRecipe, ProviderRealElementwiseRequest,
};

fn scalar(control: ScalarLiteral) -> f64 {
    match control {
        ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
        ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
    }
}

fn output_class(request: &ProviderRealElementwiseRequest) -> Option<NumericElementType> {
    match request.operation {
        ProviderRealElementwiseOperation::FloatingCast {
            target: ProviderPrecision::F32,
        } => Some(NumericElementType::F32),
        ProviderRealElementwiseOperation::FloatingCast {
            target: ProviderPrecision::F64,
        } => Some(NumericElementType::F64),
        _ => match request.source.element_type {
            ProviderElementType::F32 => Some(NumericElementType::F32),
            ProviderElementType::F64 => Some(NumericElementType::F64),
            _ => None,
        },
    }
}

fn supported(operation: ProviderRealElementwiseOperation) -> bool {
    if let ProviderRealElementwiseOperation::ExactSingleBasePower { control } = operation {
        return runmat_accelerate_api::ExactSingleBasePower::for_control(control).is_some();
    }
    matches!(
        operation,
        ProviderRealElementwiseOperation::ScalarAdd { .. }
            | ProviderRealElementwiseOperation::ScalarMul { .. }
            | ProviderRealElementwiseOperation::ScalarMax { .. }
            | ProviderRealElementwiseOperation::Sqrt
            | ProviderRealElementwiseOperation::FloatingCast { .. }
    )
}

fn evaluate(
    value: f64,
    class: NumericElementType,
    operation: ProviderRealElementwiseOperation,
) -> f64 {
    let value32 = value as f32;
    let result = match operation {
        ProviderRealElementwiseOperation::ExactSingleBasePower { control } => {
            let exact = runmat_accelerate_api::ExactSingleBasePower::for_control(control)
                .expect("proposal admitted exact Double scalar control");
            f64::from(exact.evaluate(value32))
        }
        ProviderRealElementwiseOperation::ScalarAdd { control, mode } => {
            let rhs = scalar(control);
            match mode {
                runmat_types::RealFloatingAddMode::Single => f64::from(value32 + rhs as f32),
                runmat_types::RealFloatingAddMode::Double => value + rhs,
                runmat_types::RealFloatingAddMode::MixedDoubleThenSingle => {
                    f64::from((value + rhs) as f32)
                }
            }
        }
        ProviderRealElementwiseOperation::ScalarMul { control } => {
            let rhs = scalar(control);
            if class == NumericElementType::F32 {
                f64::from(value32 * rhs as f32)
            } else {
                value * rhs
            }
        }
        ProviderRealElementwiseOperation::ScalarMax { control } => {
            let rhs = scalar(control);
            if class == NumericElementType::F32 {
                let rhs32 = rhs as f32;
                if value32.is_nan() || rhs32.is_nan() {
                    f64::from(f32::NAN)
                } else {
                    f64::from(value32.max(rhs32))
                }
            } else if value.is_nan() || rhs.is_nan() {
                f64::NAN
            } else {
                value.max(rhs)
            }
        }
        ProviderRealElementwiseOperation::Sqrt => {
            if class == NumericElementType::F32 {
                f64::from(value32.sqrt())
            } else {
                value.sqrt()
            }
        }
        ProviderRealElementwiseOperation::FloatingCast { .. } => value,
        _ => unreachable!("only proposed operations reach software execution"),
    };
    if class == NumericElementType::F32 {
        f64::from(result as f32)
    } else {
        result
    }
}

impl TestProvider {
    pub(in super::super) fn staged_real_proposal(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        let Some(class) = output_class(request) else {
            return Ok(None);
        };
        if !supported(request.operation)
            || request.source.checked_byte_len() != Some(request.source_buffer_bytes)
            || matches!(
                request.operation,
                ProviderRealElementwiseOperation::ExactSingleBasePower { .. }
            ) && request.source.element_type != ProviderElementType::F32
        {
            return Ok(None);
        }
        let output_route = match request.operation {
            ProviderRealElementwiseOperation::FloatingCast { .. }
                if class.element_size()
                    == match request.source.element_type {
                        ProviderElementType::F32 => 4,
                        ProviderElementType::F64 => 8,
                        _ => return Ok(None),
                    } =>
            {
                ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
            }
            _ => ProviderRealElementwiseOutputRoute::FreshAllocation,
        };
        let Some(count) = request.source.checked_element_count() else {
            return Ok(None);
        };
        let Some(output_buffer_bytes) = count.checked_mul(class.element_size() as u64) else {
            return Ok(None);
        };
        let fresh = output_route == ProviderRealElementwiseOutputRoute::FreshAllocation;
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route,
            output_buffer_bytes,
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: 0,
            uniform_buffer_bytes: 0,
            dispatches: u32::from(fresh),
            queue_submissions: u32::from(fresh),
            pipeline_key: fresh.then_some(1),
        }))
    }

    pub(in super::super) fn staged_real_execute(
        &self,
        input: &GpuTensorHandle,
        recipe: &ProviderRealElementwiseRecipe,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        if !recipe.is_owned_by(self) {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected real recipe has a foreign owner"
            )));
        }
        let physical = recipe.physical();
        let output = if physical.output_route
            == ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
        {
            input
                .with_view(input.view().clone())
                .map_err(ProviderExecutionError::from_source)?
        } else {
            let (data, shape) = self.pull(input).map_err(ProviderExecutionError::from)?;
            let class = output_class(recipe.request()).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("selected real output class is unavailable"))
            })?;
            let values = data
                .into_iter()
                .map(|value| evaluate(value, class, recipe.request().operation))
                .collect();
            self.push_typed(values, shape, class)
        };
        Ok(ProviderExecutedRealElementwise {
            output,
            observed: ProviderRealElementwiseObserved {
                output_buffer_bytes: Some(physical.output_buffer_bytes),
                uniform_allocations: physical.uniform_allocations,
                uniform_buffer_bytes: physical.uniform_buffer_bytes,
                dispatches: physical.dispatches,
                queue_submissions: physical.queue_submissions,
            },
        })
    }
}

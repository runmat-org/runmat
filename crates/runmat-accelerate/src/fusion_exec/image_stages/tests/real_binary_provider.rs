//! Deterministic held-stage boundary for the ordinary selected binary owner.
//! This fixture models receipts and faults, not a numerical GPU oracle.
use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderExecutedRealBinary, ProviderRealBinaryObserved,
    ProviderRealBinaryOperation, ProviderRealBinaryPhysicalProposal, ProviderRealBinaryRecipe,
    ProviderRealBinaryRequest, ProviderRealBinaryRoute,
};

#[derive(Clone)]
pub(super) enum BinaryFault {
    RejectPlan,
    WrongShape,
    WrongClass,
    WrongOwner,
    Alias(GpuTensorHandle),
    ObservedOutput,
    ObservedSubmission,
    Pending,
    HookError,
}

impl StagedProvider {
    pub(super) fn binary_proposal(
        &self,
        request: &ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealBinaryPhysicalProposal>> {
        if matches!(
            self.binary_fault.lock().unwrap().as_ref(),
            Some(BinaryFault::RejectPlan)
        ) {
            return Ok(None);
        }
        let lhs = &request.lhs.representation;
        let rhs = &request.rhs.representation;
        if lhs.element_type != rhs.element_type
            || !matches!(
                lhs.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || (lhs.element_type == ProviderElementType::F64
                && self.precision != ProviderPrecision::F64)
            || lhs.shape.len() != rhs.shape.len()
            || lhs.shape.iter().chain(&rhs.shape).any(|n| *n == 0)
        {
            return Ok(None);
        }
        let shape = lhs
            .shape
            .iter()
            .zip(&rhs.shape)
            .map(|(&a, &b)| (a == b || a == 1 || b == 1).then_some(a.max(b)))
            .collect::<Option<Vec<_>>>();
        let Some(shape) = shape else {
            return Ok(None);
        };
        let route = if lhs.shape == rhs.shape {
            ProviderRealBinaryRoute::SameShape
        } else {
            ProviderRealBinaryRoute::Broadcast
        };
        let mut output = lhs.clone();
        output.shape = shape;
        let Some(elements) = output.checked_element_count() else {
            return Ok(None);
        };
        let Some(output_buffer_bytes) = output.checked_byte_len() else {
            return Ok(None);
        };
        let dispatches = u32::try_from(elements.div_ceil(64)).ok();
        let Some(dispatches) = dispatches else {
            return Ok(None);
        };
        let (uniform_allocations, pre_output_submissions) = match route {
            ProviderRealBinaryRoute::SameShape => (dispatches, 2_u32),
            ProviderRealBinaryRoute::Broadcast => (1, 0),
        };
        Ok(Some(ProviderRealBinaryPhysicalProposal {
            route,
            output,
            output_buffer_bytes,
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations,
            uniform_buffer_bytes: u64::from(uniform_allocations) * 256,
            dispatches,
            pre_output_submissions,
            queue_submissions: pre_output_submissions + dispatches,
        }))
    }

    pub(super) fn binary_execution<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        recipe: &'a ProviderRealBinaryRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderExecutedRealBinary> {
        Box::pin(async move {
            let fault = self.binary_fault.lock().unwrap().clone();
            if matches!(&fault, Some(BinaryFault::Pending)) {
                std::future::pending::<()>().await;
            }
            if matches!(&fault, Some(BinaryFault::HookError)) {
                return Err(ProviderExecutionError::from(std::io::Error::other(
                    "original selected real binary hook cause",
                ))
                .after_effect());
            }
            let physical = recipe.physical();
            let shape = physical
                .output
                .shape
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    ProviderExecutionError::from(anyhow::Error::from(error)).after_effect()
                })?;
            let hook = match recipe.request().operation {
                ProviderRealBinaryOperation::Sub => StageHook::Sub,
                ProviderRealBinaryOperation::Div => StageHook::Div,
            };
            let class = match physical.output.element_type {
                ProviderElementType::F32 => NumericElementType::F32,
                ProviderElementType::F64 => NumericElementType::F64,
                _ => {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "binary fixture class"
                    )))
                }
            };
            let output = match &fault {
                Some(BinaryFault::Alias(handle)) => handle
                    .with_view(handle.view().clone())
                    .map_err(ProviderExecutionError::from_source)?,
                Some(BinaryFault::WrongShape) => {
                    let mut wrong = shape.clone();
                    wrong[1] += 1;
                    self.produce_class(wrong, hook, class)
                        .map_err(|error| ProviderExecutionError::from(error).after_effect())?
                }
                Some(BinaryFault::WrongClass) => self
                    .produce_class(
                        shape,
                        hook,
                        if class == NumericElementType::F32 {
                            NumericElementType::F64
                        } else {
                            NumericElementType::F32
                        },
                    )
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())?,
                Some(BinaryFault::WrongOwner) => {
                    let owned = self
                        .foreign_allocations
                        .allocate_zeroed(shape, class, GpuTensorStorage::Real)
                        .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
                    self.probes.lock().unwrap().push(owned.probe);
                    owned.handle
                }
                _ => self
                    .produce_class(shape, hook, class)
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())?,
            };
            let mut observed = ProviderRealBinaryObserved {
                output_buffer_bytes: Some(output.physical().requested_bytes()),
                uniform_allocations: physical.uniform_allocations,
                uniform_buffer_bytes: physical.uniform_buffer_bytes,
                dispatches: physical.dispatches,
                pre_output_submissions: physical.pre_output_submissions,
                queue_submissions: physical.queue_submissions,
            };
            if matches!(&fault, Some(BinaryFault::ObservedOutput)) {
                observed.output_buffer_bytes =
                    observed.output_buffer_bytes.and_then(|n| n.checked_add(1));
            }
            if matches!(&fault, Some(BinaryFault::ObservedSubmission)) {
                observed.queue_submissions += 1;
            }
            Ok(ProviderExecutedRealBinary { output, observed })
        })
    }
}

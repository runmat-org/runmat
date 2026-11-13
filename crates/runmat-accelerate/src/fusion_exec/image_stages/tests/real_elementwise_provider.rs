//! Deterministic image lifetime/resource boundary fixture, not GPU evidence.
use super::*;
use runmat_accelerate_api::{
    ProviderExecutedRealElementwise, ProviderRealElementwiseObserved,
    ProviderRealElementwiseOperation as Operation, ProviderRealElementwisePhysicalProposal,
    ProviderRealElementwiseRecipe, ProviderRealElementwiseRequest,
};

#[derive(Clone)]
pub(super) enum RealFault {
    RejectPlan,
    ObservedOutput,
    ObservedUniformBytes,
    ObservedUniformCount,
    ObservedDispatches,
    ObservedSubmissions,
    Alias(GpuTensorHandle),
    Pending,
    HookError,
    WrongShape,
    WrongClass,
    WrongOwner,
}

impl StagedProvider {
    pub(super) fn real_proposal(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        let sqrt = match request.operation {
            Operation::UnaryMath(_) | Operation::Spacing | Operation::Expm1 => return Ok(None),
            Operation::ExactSingleBasePower { .. } => return Ok(None),
            Operation::ScalarAdd { .. }
            | Operation::ScalarMul { .. }
            | Operation::ScalarMax { .. } => false,
            Operation::Sqrt => true,
            Operation::FloatingCast { target } => {
                let same_class = matches!(
                    (request.source.element_type, target),
                    (
                        runmat_accelerate_api::ProviderElementType::F32,
                        ProviderPrecision::F32
                    ) | (
                        runmat_accelerate_api::ProviderElementType::F64,
                        ProviderPrecision::F64
                    )
                );
                if !same_class {
                    return Ok(None);
                }
                return Ok(Some(ProviderRealElementwisePhysicalProposal {
                    output_route:
                        runmat_accelerate_api::ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias,
                    output_buffer_bytes: request.source_buffer_bytes,
                    workgroup_size: 64,
                    chunk_capacity: 64,
                    uniform_allocations: 0,
                    uniform_buffer_bytes: 0,
                    dispatches: 0,
                    queue_submissions: 0,
                    pipeline_key: None,
                }));
            }
        };
        if self
            .real_fault
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|(selected, fault)| {
                *selected == sqrt && matches!(fault, RealFault::RejectPlan)
            })
        {
            return Ok(None);
        }
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route:
                runmat_accelerate_api::ProviderRealElementwiseOutputRoute::FreshAllocation,
            output_buffer_bytes: request.source.checked_byte_len().unwrap(),
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: 1,
            uniform_buffer_bytes: if sqrt { 16 } else { 256 },
            dispatches: 1,
            queue_submissions: if sqrt { 2 } else { 1 },
            pipeline_key: Some(if sqrt { 71 } else { 70 }),
        }))
    }

    pub(super) fn real_execution<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a ProviderRealElementwiseRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        Box::pin(async move {
            let operation = recipe.request().operation;
            let sqrt = match operation {
                Operation::UnaryMath(_) | Operation::Spacing | Operation::Expm1 => {
                    return Err(ProviderExecutionError::from(
                        runmat_accelerate_api::unsupported_provider_operation(
                            "fixture has no ordinary unary kernel",
                        ),
                    ));
                }
                Operation::ExactSingleBasePower { .. } => {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "fixture has no exact POWER kernel"
                    )))
                }
                Operation::ScalarAdd { .. }
                | Operation::ScalarMul { .. }
                | Operation::ScalarMax { .. } => false,
                Operation::Sqrt => true,
                Operation::FloatingCast { target } => {
                    let output = match target {
                        ProviderPrecision::F32 => self.unary_single(input).await?,
                        ProviderPrecision::F64 => self.unary_double(input).await?,
                    };
                    return Ok(ProviderExecutedRealElementwise {
                        observed: ProviderRealElementwiseObserved {
                            output_buffer_bytes: Some(output.physical().requested_bytes()),
                            ..ProviderRealElementwiseObserved::default()
                        },
                        output,
                    });
                }
            };
            self.real_controls.lock().unwrap().push(operation);
            self.mutate_cast_metadata(input);
            if let Some(handle) = self.mutate_protected_residency.lock().unwrap().as_ref() {
                runmat_accelerate_api::clear_residency(handle).unwrap();
                assert!(!crate::fusion_residency::is_resident(handle));
            }
            let fault = self
                .real_fault
                .lock()
                .unwrap()
                .clone()
                .filter(|(selected, _)| *selected == sqrt)
                .map(|(_, fault)| fault);
            let output_fault = match &fault {
                Some(RealFault::WrongShape) => Some(StageFault::WrongShape),
                Some(RealFault::WrongClass) => Some(StageFault::WrongClass),
                Some(RealFault::WrongOwner) => Some(StageFault::WrongOwner),
                _ => None,
            };
            if let Some(output_fault) = output_fault {
                *self.fault.lock().unwrap() =
                    Some((self.calls.load(Ordering::SeqCst), output_fault));
            }
            let output = if let Some(RealFault::Alias(handle)) = &fault {
                handle.with_view(handle.view().clone()).unwrap()
            } else {
                self.produce_class(
                    input.shape().to_vec(),
                    match operation {
                        Operation::ExactSingleBasePower { .. } => {
                            unreachable!("fixture declined exact POWER")
                        }
                        Operation::Sqrt => StageHook::Sqrt,
                        Operation::ScalarAdd { .. } => StageHook::Add,
                        Operation::ScalarMul { .. } => StageHook::Mul,
                        Operation::ScalarMax { .. } => StageHook::Max,
                        Operation::FloatingCast { .. } => unreachable!("handled cast"),
                        Operation::UnaryMath(_) | Operation::Spacing | Operation::Expm1 => {
                            unreachable!("unary recipe rejected before execution")
                        }
                    },
                    input.physical().element_type(),
                )
                .map_err(|error| ProviderExecutionError::from(error).after_effect())?
            };
            if matches!(&fault, Some(RealFault::HookError)) {
                return Err(ProviderExecutionError::from(std::io::Error::other(
                    "original selected Sigma hook cause",
                ))
                .after_effect());
            }
            if matches!(&fault, Some(RealFault::Pending)) {
                std::future::pending::<()>().await;
            }
            let physical = recipe.physical();
            Ok(ProviderExecutedRealElementwise {
                observed: ProviderRealElementwiseObserved {
                    output_buffer_bytes: Some(
                        output.physical().requested_bytes()
                            + u64::from(matches!(&fault, Some(RealFault::ObservedOutput))),
                    ),
                    uniform_allocations: physical.uniform_allocations
                        + u32::from(matches!(&fault, Some(RealFault::ObservedUniformCount))),
                    uniform_buffer_bytes: physical.uniform_buffer_bytes
                        + u64::from(matches!(
                            &fault,
                            Some(RealFault::ObservedUniformBytes | RealFault::Alias(_))
                        )),
                    dispatches: physical.dispatches
                        + u32::from(matches!(&fault, Some(RealFault::ObservedDispatches))),
                    queue_submissions: physical.queue_submissions
                        + u32::from(matches!(&fault, Some(RealFault::ObservedSubmissions))),
                },
                output,
            })
        })
    }
}

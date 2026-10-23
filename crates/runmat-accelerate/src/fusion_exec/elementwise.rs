//! Typed elementwise request execution and transactional publication.
use super::*;
use crate::precision::ensure_provider_supports_dtype;
use runmat_accelerate_api::{ProviderElementType, ProviderExecutionResult as ExecutionResult};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;

#[path = "elementwise/effects.rs"]
mod effects;
use effects::ElementwiseEffects;
mod abi;
#[path = "elementwise/host_candidate.rs"]
mod host_candidate;
#[path = "elementwise/host_liveness.rs"]
mod host_liveness;
#[path = "elementwise/host_readback.rs"]
pub(super) mod host_readback;

pub(super) fn host_liveness_preparation_bytes(
    value_ids_upper: usize,
    operations: usize,
    max_rank: usize,
) -> Option<u64> {
    host_liveness::preparation_bytes(value_ids_upper, operations, max_rank)
}

#[cfg(test)]
#[path = "elementwise/effect_tests.rs"]
mod effect_tests;

struct ElementwiseHostContinuation {
    operations: Option<Vec<runmat_runtime::context::VmFusionHostOperation>>,
    readback: Option<super::host_readback::CpuReadbackPlan>,
    outputs: Option<host_readback::HostOutputPlan>,
}

struct TypedProducerPreparation {
    receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    primary: abi::ShaderBodyPreparation,
    fallback: abi::ShaderBodyPreparation,
}

fn classify_producer_preparation_error(
    program: &crate::fusion::semantics::ResolvedProgram,
    error: anyhow::Error,
) -> crate::fusion::semantics::FusionResolveFailure {
    use crate::fusion::semantics::FusionResolveFailure;
    if error
        .downcast_ref::<crate::fusion::typed_producer::CountedRecipeUnavailable>()
        .is_some()
    {
        let failure = FusionResolveFailure::Structural(error);
        if program.observed_provider_probe() {
            failure.after_observed_probe()
        } else {
            failure
        }
    } else {
        FusionResolveFailure::Admission(error)
    }
}

/// Typed admission for the floating elementwise implementation.
///
/// Native integer values are valid runtime inputs, but this implementation
/// cannot execute them without changing their representation. VM callers with
/// a certified decline right may retain the original typed operations instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementwiseFloatingAdmission {
    Eligible,
    NativeIntegerIneligible,
}

pub fn elementwise_floating_admission(
    request: &FusionExecutionRequest<'_>,
) -> ExecutionResult<ElementwiseFloatingAdmission> {
    use crate::fusion::semantics::authenticate_resident_input;
    crate::ensure_residency_hooks();
    let invalid = || anyhow!("fusion input has invalid floating metadata or provider ownership");
    let mut native_integer = false;
    for value in request
        .inputs
        .iter()
        .chain(unbound_constant_values(request.plan))
    {
        if matches!(value, Value::GpuTensor(_)) {
            authenticate_resident_input(value)
                .map_err(|_| invalid())?
                .ok_or_else(invalid)?;
        }
        native_integer |= native_integer_fusion_value_kind(value).is_some();
    }
    Ok(if native_integer {
        ElementwiseFloatingAdmission::NativeIntegerIneligible
    } else {
        ElementwiseFloatingAdmission::Eligible
    })
}

/// The semantic receipt stays live while this additive producer reservation
/// is admitted. A retained multi-output shader can overlap one single-output
/// producer if the provider rejects multi-output dispatch after preparation.
fn admit_typed_producer_preparation(
    request: &FusionExecutionRequest<'_>,
    outputs: &[ValueId],
    program: &crate::fusion::semantics::ResolvedProgram,
    provider: &dyn AccelProvider,
    finite: bool,
) -> std::result::Result<TypedProducerPreparation, crate::fusion::semantics::FusionResolveFailure> {
    use crate::fusion::semantics::FusionResolveFailure;
    use runmat_runtime::builtins::common::spec::ScalarType;
    if !finite {
        return Ok(TypedProducerPreparation {
            receipt: None,
            primary: abi::ShaderBodyPreparation::Unbounded,
            fallback: abi::ShaderBodyPreparation::Unbounded,
        });
    }
    let default_precision = provider.precision();
    let default_scalar = match default_precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    let default_indices = abi::physical_indices(request.plan, outputs, program, default_precision)
        .map_err(FusionResolveFailure::Admission)?;
    let mut peak = 0_u64;
    let mut f32_body_capacity = None;
    let mut f32_route = None;
    if default_precision == ProviderPrecision::F64 {
        let single_indices =
            abi::physical_indices(request.plan, outputs, program, ProviderPrecision::F32)
                .map_err(FusionResolveFailure::Admission)?;
        if abi::may_try_single(
            request.plan,
            outputs,
            program,
            provider,
            &request.inputs,
            &single_indices,
        )
        .map_err(FusionResolveFailure::Admission)?
        {
            let single = request.plan.measured_typed_shader_preparation_bytes(
                outputs,
                program,
                ScalarType::F32,
                single_indices.len(),
            );
            match single {
                Ok(single) => {
                    f32_body_capacity = Some(single.body_bytes);
                    peak = peak.max(single.requested_bytes);
                    f32_route = Some((single_indices, single));
                }
                Err(error)
                    if error
                        .downcast_ref::<crate::fusion::typed_producer::CountedRecipeUnavailable>()
                        .is_some() =>
                {
                    log::debug!("finite fusion Single ABI has no counted recipe: {error}");
                }
                Err(error) => return Err(FusionResolveFailure::Admission(error)),
            }
        }
    }
    let default_bound = request.plan.measured_typed_shader_preparation_bytes(
        outputs,
        program,
        default_scalar,
        default_indices.len(),
    );
    let default_bound = match default_bound {
        Ok(bound) => {
            peak = peak.max(bound.requested_bytes);
            Some(bound)
        }
        Err(error)
            if f32_route.is_some()
                && error
                    .downcast_ref::<crate::fusion::typed_producer::CountedRecipeUnavailable>()
                    .is_some() =>
        {
            log::debug!("finite fusion default ABI has no counted recipe: {error}");
            None
        }
        Err(error) => return Err(classify_producer_preparation_error(program, error)),
    };
    let mut default_fallback_body_capacity = None;
    let mut f32_fallback_body_capacity = None;
    if outputs.len() > 1 {
        let mut fallback = |scalar, indices: &[usize], retained: u64| {
            let mut body = 0usize;
            for output in outputs {
                let single = request
                    .plan
                    .measured_typed_shader_preparation_bytes(
                        std::slice::from_ref(output),
                        program,
                        scalar,
                        indices.len(),
                    )
                    .map_err(|error| classify_producer_preparation_error(program, error))?;
                body = body.max(single.body_bytes);
                let overlap = retained
                    .checked_add(single.requested_bytes)
                    .ok_or_else(|| {
                        FusionResolveFailure::Admission(anyhow!(
                            "finite fusion multi-output fallback peak overflows"
                        ))
                    })?;
                peak = peak.max(overlap);
            }
            Ok::<usize, FusionResolveFailure>(body)
        };
        if let Some(bound) = default_bound {
            default_fallback_body_capacity = Some(fallback(
                default_scalar,
                &default_indices,
                bound.retained_bytes,
            )?);
        }
        if let Some((indices, bound)) = &f32_route {
            f32_fallback_body_capacity =
                Some(fallback(ScalarType::F32, indices, bound.retained_bytes)?);
        }
    }
    let plan = runmat_runtime::context::host_allocation::HostAllocationPlan::new(peak, 0)
        .map_err(|error| FusionResolveFailure::Admission(error.into()))?;
    let receipt = runmat_runtime::context::host_allocation::HostAllocationReceipt::admit(
        request.runtime.as_ref(),
        plan,
    )
    .map_err(|error| FusionResolveFailure::Admission(error.into()))?;
    Ok(TypedProducerPreparation {
        receipt,
        primary: abi::ShaderBodyPreparation::Counted {
            single: f32_body_capacity,
            provider_default: default_bound.map(|bound| bound.body_bytes),
        },
        fallback: abi::ShaderBodyPreparation::Counted {
            single: f32_fallback_body_capacity,
            provider_default: default_fallback_body_capacity,
        },
    })
}

async fn execute_elementwise_outputs(
    request: &FusionExecutionRequest<'_>,
    output_ids: &[ValueId],
    vm_selection: Option<super::VmFusionSelectionRight<'_>>,
    input_uses: Option<&super::VmHostInputUses>,
    host_continuation: &mut Option<ElementwiseHostContinuation>,
    native: Option<(
        Option<&runmat_runtime::numeric_region::service::NativeNumericRegionCpuCandidates>,
        &runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        &runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
    )>,
) -> ExecutionResult<FusionExecutionOutcome<(HashMap<ValueId, Value>, FusionPlacementSelection)>> {
    let vm_right = vm_selection.and_then(super::VmFusionSelectionRight::completion);
    crate::ensure_residency_hooks();
    if !request.plan.group.kind.is_elementwise() {
        return Err(anyhow!("unsupported fusion kind").into());
    }
    if !request.plan.kernel.supported {
        return Err(anyhow!("fusion kernel not supported for this plan").into());
    }
    if request.inputs.len() != request.plan.inputs.len() {
        return Err(anyhow!(
            "fusion input mismatch: expected {}, got {}",
            request.plan.inputs.len(),
            request.inputs.len()
        )
        .into());
    }
    let floating_admission = elementwise_floating_admission(request)?;
    let may_decline = vm_selection
        .and_then(|selection| selection.certified_decline(request))
        .is_some();
    let finite_host_preparation = request
        .runtime
        .as_ref()
        .and_then(|runtime| runtime.service_ports().parallel())
        .is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
    use crate::fusion::semantics::{authenticate_resident_input, FloatingFusionEligibility};
    let invalid = || anyhow!("fusion input has invalid floating metadata or provider ownership");
    let selected_provider = provider();
    if floating_admission == ElementwiseFloatingAdmission::NativeIntegerIneligible
        && !may_decline
        && selected_provider.is_none()
    {
        reject_native_integer_fusion_inputs(request)?;
        unreachable!("native integer rejection always returns an error");
    }
    let provider = match admit_provider_availability(request, selected_provider)? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            if floating_admission == ElementwiseFloatingAdmission::NativeIntegerIneligible
                && !may_decline
            {
                reject_native_integer_fusion_inputs(request)?;
                unreachable!("native integer rejection always returns an error");
            }
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let mut floating_route_ineligible = false;
    for value in request
        .inputs
        .iter()
        .chain(unbound_constant_values(request.plan))
    {
        match value {
            Value::GpuTensor(_) => {
                use runmat_runtime::builtins::common::gpu_helpers::gpu_class_metadata_matches;
                let authenticated = authenticate_resident_input(value)
                    .map_err(|_| invalid())?
                    .ok_or_else(invalid)?;
                if !authenticated.belongs_to(provider) {
                    return Err(invalid().into());
                }
                let precision = match authenticated.floating_eligibility() {
                    FloatingFusionEligibility::Eligible(precision) => precision,
                    FloatingFusionEligibility::Ineligible => {
                        floating_route_ineligible = true;
                        continue;
                    }
                };
                let handle = authenticated.handle();
                let element = Some(match precision {
                    ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                    ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                });
                if handle.descriptor().storage
                    != Some(runmat_accelerate_api::GpuTensorStorage::Real)
                    || handle.descriptor().element_type != element
                    || !gpu_class_metadata_matches(handle, Some(precision), None, false)
                {
                    return Err(invalid().into());
                }
            }
            Value::Num(_) | Value::Tensor(_) => {}
            _ => floating_route_ineligible = true,
        }
    }
    if floating_admission == ElementwiseFloatingAdmission::NativeIntegerIneligible {
        if may_decline {
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ));
        }
        reject_native_integer_fusion_inputs(request)?;
        unreachable!("native integer rejection always returns an error");
    }
    if floating_route_ineligible {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let preparation_receipt = if finite_host_preparation {
        let geometry = crate::fusion::semantics::preparation_resources::PureSourceGeometry::admit(
            request.plan,
            &request.inputs,
        )
        .map_err(ProviderExecutionError::from_source)?;
        let host_uses = vm_right
            .zip(input_uses)
            .map(|(_, uses)| (uses.entry.len(), uses.loads.len()))
            .or_else(|| native.map(|_| (request.inputs.len(), 0)));
        let bytes = geometry
            .overlapping_preparation_bytes(host_uses)
            .map_err(ProviderExecutionError::from_source)?;
        let plan = runmat_runtime::context::host_allocation::HostAllocationPlan::new(bytes, 0)
            .map_err(ProviderExecutionError::from_source)?;
        runmat_runtime::context::host_allocation::HostAllocationReceipt::admit(
            request.runtime.as_ref(),
            plan,
        )
        .map_err(ProviderExecutionError::from_source)?
    } else {
        None
    };
    let Some(bindings) = AdmittedBindings::new(request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let typed = async {
        let program = crate::fusion::semantics::ResolvedProgram::resolve(
            request.plan,
            &request.inputs,
            provider.precision(),
            request.placement,
        )
        .await?;
        let producer_preparation = admit_typed_producer_preparation(
            request,
            output_ids,
            &program,
            provider,
            finite_host_preparation,
        )?;
        let mut classes = Vec::new();
        classes
            .try_reserve_exact(output_ids.len())
            .map_err(|error| {
                crate::fusion::semantics::FusionResolveFailure::Admission(error.into())
            })?;
        for id in output_ids {
            classes.push(program.floating_class(*id)?);
        }
        let started = Instant::now();
        let (generated_shader, typed_abi) = abi::select_shader(
            request.plan,
            output_ids,
            &program,
            provider,
            &request.inputs,
            producer_preparation.primary,
        )
        .map_err(|error| {
            if finite_host_preparation {
                classify_producer_preparation_error(&program, error)
            } else {
                let failure = crate::fusion::semantics::FusionResolveFailure::Structural(error);
                if program.observed_provider_probe() {
                    failure.after_observed_probe()
                } else {
                    failure
                }
            }
        })?;
        FusionPlacementObserver::new(request.placement).shader_source_generation(started);
        Ok::<_, crate::fusion::semantics::FusionResolveFailure>((
            program,
            classes,
            generated_shader,
            typed_abi,
            producer_preparation,
        ))
    }
    .await;
    let (program, output_classes, generated_shader, typed_abi, producer_preparation) = match typed {
        Ok(typed) => typed,
        Err(error) => {
            let error = match error.into_execution_or_structural() {
                Ok(error) => error,
                Err(failure) => return Err(failure),
            };
            log::debug!("fusion retained shared runtime: {error}");
            FusionPlacementObserver::new(request.placement).selected(
                PlacementVariant::SharedRuntime,
                "typed_representation_unsupported",
                None,
            );
            if let (Some(receipt), Some(runtime)) = (&preparation_receipt, request.runtime.as_ref())
            {
                receipt
                    .checkpoint(runtime)
                    .map_err(ProviderExecutionError::from_source)?;
            }
            return host_candidate::without_shader(
                request,
                output_ids,
                &bindings,
                provider,
                native
                    .map(|(cpu, right, program)| host_candidate::Source::Native {
                        cpu,
                        right,
                        program,
                    })
                    .or_else(|| {
                        vm_right
                            .zip(input_uses)
                            .map(|(right, uses)| host_candidate::Source::Vm { right, uses })
                    }),
                preparation_receipt,
                host_continuation,
            );
        }
    };
    if let (Some(receipt), Some(runtime)) = (&preparation_receipt, request.runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    if let (Some(receipt), Some(runtime)) =
        (&producer_preparation.receipt, request.runtime.as_ref())
    {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    let physical_precision = generated_shader.abi().precision();
    let exact_product_words = generated_shader.uses_exact_product_words();
    let emitted_implementations = generated_shader.implementations().to_vec();
    let scalar_type = match physical_precision {
        ProviderPrecision::F32 => runmat_runtime::builtins::common::spec::ScalarType::F32,
        ProviderPrecision::F64 => runmat_runtime::builtins::common::spec::ScalarType::F64,
    };
    let (shader, shader_requirements, input_indices) = generated_shader.into_parts();
    // Determine output shape from the fusion plan; if unknown, derive from runtime inputs via broadcasting.
    // Determine output shape from the fusion plan and derive the element count from it.
    let runtime_shape = runtime_broadcast_shape(&request.inputs);
    let mut output_shape = match &request.plan.group.shape {
        ShapeInfo::Tensor(dims) if !dims.is_empty() && dims.iter().all(|d| d.is_some()) => {
            dims.iter().map(|d| d.unwrap_or(1)).collect()
        }
        ShapeInfo::Tensor(dims) if !dims.is_empty() => {
            let rt_shape = runtime_shape
                .clone()
                .ok_or_else(|| anyhow!("fusion: unknown output shape"))?;
            if rt_shape.len() != dims.len() {
                rt_shape
            } else {
                dims.iter()
                    .zip(rt_shape.iter())
                    .map(|(planned, runtime)| planned.unwrap_or(*runtime))
                    .collect()
            }
        }
        _ => runtime_shape.ok_or_else(|| anyhow!("fusion: unknown output shape"))?,
    };
    let typed_shape = output_ids
        .first()
        .and_then(|id| program.fact(*id).ok())
        .and_then(|fact| fact.shape.known_dims())
        .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>());
    if let Some(shape) = typed_shape {
        output_shape = shape;
    }
    if output_ids.iter().any(|id| {
        program
            .fact(*id)
            .ok()
            .and_then(|fact| fact.shape.known_dims())
            .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
            .is_none_or(|shape| shape != output_shape)
    }) {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let len: usize = output_shape
        .iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
        .ok_or_else(|| anyhow!("fusion output shape exceeds addressable size"))?;
    if len == 0 {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let representations = output_classes
        .iter()
        .map(|class| {
            let precision = match class {
                runmat_types::NumericClass::Single => ProviderPrecision::F32,
                _ => ProviderPrecision::F64,
            };
            dense_output_representation(precision, &output_shape)
        })
        .collect();
    let scalar_work_units = match u64::try_from(len).ok().and_then(|elements| {
        program
            .physical_scalar_work_units(request.plan, elements, &emitted_implementations)
            .ok()
    }) {
        Some(units) => units,
        None => {
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ))
        }
    };
    let has_host_operations = bindings.has_host_operations(request.runtime.as_ref());
    let has_resident = request
        .inputs
        .iter()
        .chain(unbound_constant_values(request.plan))
        .any(runmat_runtime::dispatcher::value_contains_gpu);
    let managed_host = runmat_runtime::context::host_allocation::HostAllocationReceipt::policy(
        request.runtime.as_ref(),
    )
    .ok()
        == Some(runmat_runtime::context::host_allocation::HostAllocationPolicy::Managed);
    let admission = if let Some((_, right, program)) = native {
        has_host_operations
            .then(|| host_readback::admit_native(request, &bindings, provider, program, right))
            .flatten()
    } else {
        vm_right
            .and(input_uses)
            .filter(|_| has_host_operations)
            .and_then(|uses| {
                if managed_host {
                    host_readback::admit_host(request, &bindings, provider, uses)
                } else {
                    host_readback::admit(request, &bindings, provider, uses)
                }
            })
    };
    let original_host_program = admission
        .is_none()
        .then(|| bindings.host_facts(request, provider.precision()))
        .flatten();
    let original_host_program = admission
        .as_ref()
        .map(|admission| &admission.program)
        .or(original_host_program.as_ref());
    let host_scalar_work_units =
        original_host_program.and_then(|program| program.host_scalar_work_units(request.plan).ok());
    let semantic_identity = runmat_execution::Digest::sha256_serialized(&(
        "elementwise-independent-host-provider-work-v1",
        program.cache_identity()?,
        original_host_program
            .map(|program| program.cache_identity())
            .transpose()?,
        host_scalar_work_units,
        &shader,
        exact_product_words,
        &emitted_implementations,
        scalar_work_units,
    ))
    .map_err(anyhow::Error::from)?;
    let host = if let Some((cpu, right, _)) = native {
        super::host_readback::CpuHostPlacement::Native {
            cpu,
            right,
            source_host: admission.as_ref().map(|admission| &admission.resources),
        }
    } else if (has_resident || managed_host) && vm_right.is_some() {
        admission.as_ref().map_or(
            super::host_readback::CpuHostPlacement::Unavailable,
            |admission| super::host_readback::CpuHostPlacement::Readbacks(&admission.resources),
        )
    } else {
        super::host_readback::CpuHostPlacement::Legacy
    };
    if let (Some(receipt), Some(runtime)) = (&preparation_receipt, request.runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    if let (Some(receipt), Some(runtime)) =
        (&producer_preparation.receipt, request.runtime.as_ref())
    {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    let selection = select_fusion_provider_with_contract_vm_cpu(
        request,
        &bindings,
        provider,
        FusionProviderContract {
            elementwise_abi: typed_abi,
            input_indices: Some(&input_indices),
            outputs: representations,
            semantic_identity: Some(semantic_identity),
            scalar_work_units: Some(scalar_work_units),
            host_scalar_work_units,
            matmul: None,
            preparation_program: Some(
                runmat_accelerate_api::ProviderPreparationProgram::Elementwise { shader: &shader },
            ),
            shader_requirements: Some(shader_requirements),
        },
        vm_selection,
        host,
    )?;
    if let (Some(receipt), Some(runtime)) = (&preparation_receipt, request.runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    if let (Some(receipt), Some(runtime)) =
        (&producer_preparation.receipt, request.runtime.as_ref())
    {
        receipt
            .checkpoint(runtime)
            .map_err(ProviderExecutionError::from_source)?;
    }
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(mut observation) => {
            if native.is_some() {
                observation.attach_native_host_admission(
                    request
                        .runtime
                        .as_ref()
                        .expect("native request has owning Runtime"),
                    admission,
                    preparation_receipt,
                )?;
                return Ok(FusionExecutionOutcome::SharedRuntime(observation));
            }
            if vm_right.is_some() {
                let (readback, outputs) = admission
                    .map(|admission| {
                        let outputs = (!admission.outputs.is_empty()).then_some(admission.outputs);
                        let readback =
                            (!admission.plan.sources.is_empty()).then_some(admission.plan);
                        (readback, outputs)
                    })
                    .unwrap_or((None, None));
                *host_continuation = Some(ElementwiseHostContinuation {
                    operations: bindings.host_operations(request.runtime.as_ref()),
                    readback,
                    outputs,
                });
            }
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let recorder = selection.recorder();
    runmat_accelerate_api::observation::observe_provider_future(recorder, async move {
        let mut effects = ElementwiseEffects::default();
        let result: ExecutionResult<_> = async {
        program.require_exact_product_controls(request.plan, &request.inputs)?;
        if let Some(work) = typed_abi { abi::validate_inputs(request.plan, &program, &input_indices, &request.inputs, work)?; }
        let observer = selection.observer;
        let mut timer = FusionStageTimer::new("elementwise", request.plan.index, len);
        let scalar_shape = normalize_scalar_shape(&vec![1; output_shape.len()]);
        let mut prepared = Vec::with_capacity(request.inputs.len());
        let mut allocations = outputs::FusionOutputAdmission::new(&request.inputs);
        let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
        let scalar_dtype = match physical_precision { ProviderPrecision::F32 => NumericDType::F32, ProviderPrecision::F64 => NumericDType::F64 };
        if program.uses_implicit_downcast_for(&input_indices) {
            // Reuse the existing policy warning and admission authority. Explicit
            // Single conversions need no implicit-policy warning: typed codegen
            // has already proved every operation on an F32 backend is F32.
            ensure_provider_supports_dtype(provider, NumericDType::F64)
                .map_err(anyhow::Error::msg)?;
        }
        for &input_index in &input_indices {
            let value = &request.inputs[input_index];
            match value {
                Value::GpuTensor(handle) => {
                    use runmat_runtime::builtins::common::gpu_helpers::{
                        exact_provider_for_handle, ResidencyRestoreGuard,
                    };
                    if !exact_provider_for_handle(handle)
                        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
                    {
                        return Err(anyhow!("fusion input has no matching exact provider owner").into());
                    }
                    let expected = match physical_precision {
                        ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                        ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                    };
                    let contiguous = if runmat_accelerate_api::handle_is_transposed(handle)
                        && handle.descriptor().element_type != Some(expected)
                    {
                        let metadata = runmat_runtime::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(handle);
                        let started = Instant::now();
                        let materialized = provider.reshape(handle, handle.shape());
                        drop(metadata);
                        observer.queue(started);
                        let materialized = match materialized {
                            Ok(handle) => {
                                effects.succeeded();
                                handle
                            }
                            Err(error)
                                if runmat_accelerate_api::is_unsupported_provider_operation(
                                    &error,
                                ) && effects.can_replay_unsupported() =>
                            {
                                return Ok(FusionExecutionOutcome::SharedRuntime(
                                    FusionCpuObservation::new(None),
                                ));
                            }
                            Err(error) => return Err(error.into()),
                        };
                        observer.kernel_unmeasured();
                        allocations.admit(
                            materialized,
                            provider,
                            handle.shape(),
                            runmat_accelerate_api::handle_precision(handle)
                                .ok_or_else(|| anyhow!("missing source precision"))?,
                        )?
                    } else {
                        handle.clone()
                    };
                    let prepared_handle = if contiguous.descriptor().element_type == Some(expected) {
                        contiguous
                    } else {
                        let metadata = ResidencyRestoreGuard::new(&contiguous);
                        let converted = match physical_precision {
                            ProviderPrecision::F64 => provider
                                .unary_double(&contiguous)
                                .await,
                            ProviderPrecision::F32 => provider.unary_single(&contiguous).await,
                        };
                        drop(metadata);
                        let converted = converted?;
                        effects.succeeded();
                        allocations.admit(
                            converted,
                            provider,
                            handle.shape(),
                            physical_precision,
                        )?
                    };
                    prepared.push(PreparedInput {
                        handle: prepared_handle,
                    });
                }
                Value::Tensor(t) => {
                    if typed_abi.is_some() {
                        let owned;
                        let data = if let Some(data) = t.as_f32_slice() { data } else {
                            owned = (0..t.len()).map(|index| match t.numeric_value_at(index) {
                                Some(runmat_value::NumericScalar::F32(value)) => Ok(value),
                                _ => Err(anyhow!("typed Single host materialization changed class")),
                            }).collect::<Result<Vec<_>>>()?;
                            &owned
                        };
                        let started = Instant::now();
                        let handle = allocations.upload_typed(provider, &runmat_accelerate_api::HostNumericTensorView { data: runmat_accelerate_api::HostNumericDataView::F32(data), shape: t.shape(), storage: runmat_accelerate_api::GpuTensorStorage::Real })?;
                        effects.succeeded();
                        observer.upload(started, physical_upload_bytes(physical_precision, data.len()));
                        prepared.push(PreparedInput { handle });
                        continue;
                    }
                    if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                        return Err(anyhow!(
                            "fusion: tensor input requires unsupported precision ({msg})"
                        ).into());
                    }
                    let data = t.materialize_f64();
                    let started = Instant::now();
                    let handle = allocations.upload(provider, &data, t.shape())?;
                    effects.succeeded();
                    observer.upload(
                        started,
                        physical_upload_bytes(provider.precision(), data.len()),
                    );
                    prepared.push(PreparedInput { handle });
                }
                Value::Num(n) => {
                    if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                        return Err(anyhow!(
                            "fusion: scalar input requires unsupported precision ({msg})"
                        ).into());
                    }
                    let scalar = match provider.precision() {
                        ProviderPrecision::F32 => (*n as f32) as f64,
                        ProviderPrecision::F64 => *n,
                    };
                    temp_scalars.push(vec![scalar]);
                    let data = temp_scalars.last().unwrap();
                    let started = Instant::now();
                    let handle = allocations.upload(provider, data, &scalar_shape)?;
                    effects.succeeded();
                    observer.upload(
                        started,
                        physical_upload_bytes(provider.precision(), data.len()),
                    );
                    prepared.push(PreparedInput { handle });
                }
                Value::Int(_) => unreachable!("integer inputs rejected before fusion upload"),
                _ => {
                    return Err(anyhow!("fusion: unsupported value type").into());
                }
            }
        }
        timer.mark("prepare_inputs");

        let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
        let mut outputs = HashMap::new();
        // Shader stores use backend precision; semantic Single adaptation follows dispatch.
        let physical_output_class = match physical_precision {
            ProviderPrecision::F32 => ProviderElementType::F32,
            ProviderPrecision::F64 => ProviderElementType::F64,
        };

        if output_ids.len() == 1 {
            // Fast path: single output, use the existing single-dispatch API.
            let output_id = output_ids[0];
            timer.mark("generate_wgsl");
            let started = Instant::now();
            let metadata = protect_kernel_inputs(&handles);
            let output = abi::dispatch_single(provider, typed_abi, runmat_accelerate_api::ProviderFusedElementwiseRequest {
                shader: &shader, requirements: shader_requirements, inputs: &handles, output_shape: &output_shape, len, output_classes: &[physical_output_class], abi: runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal { precision: physical_precision },
            }).await;
            drop(metadata);
            let output = output?;
            effects.succeeded();
            let output =
                allocations.admit(output, provider, &output_shape, physical_precision)?;
            observer.queue(started);
            observer.kernel_unmeasured();
            outputs.insert(output_id, Value::GpuTensor(output));
        } else {
            // Multi-output path: generate one shader that writes all outputs in a single dispatch,
            // reducing O(N²) GPU work (N full-chain recomputations) to O(N).
            timer.mark("generate_wgsl");
            let started = Instant::now();
            let metadata = protect_kernel_inputs(&handles);
            let physical_output_classes = vec![physical_output_class; output_ids.len()];
            let multi_result = abi::dispatch_multi(provider, typed_abi, runmat_accelerate_api::ProviderFusedElementwiseRequest {
                shader: &shader, requirements: shader_requirements, inputs: &handles, output_shape: &output_shape, len, output_classes: &physical_output_classes, abi: runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal { precision: physical_precision },
            }).await;
            drop(metadata);
            observer.queue(started);
            match multi_result {
                Ok(out_handles) => {
                    effects.succeeded();
                    let out_handles = allocations.admit_batch(
                        out_handles,
                        provider,
                        output_ids.len(),
                        &output_shape,
                        physical_precision,
                    )?;
                    observer.kernel_unmeasured();
                    for (output_id, handle) in output_ids.iter().copied().zip(out_handles) {
                        outputs.insert(output_id, Value::GpuTensor(handle));
                    }
                }
                Err(error) if error.is_unsupported() => {
                    // Provider does not support multi-output dispatch; fall back to N single dispatches.
                    for output_id in output_ids.iter().copied() {
                        let started = Instant::now();
                        let single_shader = request.plan.prepare_typed_shader_with_inputs_bounded(
                            &[output_id],
                            &program,
                            scalar_type,
                            &input_indices,
                            if typed_abi.is_some() {
                                producer_preparation.fallback.single_capacity()
                            } else {
                                producer_preparation.fallback.default_capacity()
                            },
                        )?;
                        observer.shader_source_generation(started);
                        if !provider.supports_shader_requirements(single_shader.requirements()) {
                            return Err(runmat_accelerate_api::unsupported_provider_operation(
                                "selected single-output shader requires unavailable features",
                            )
                            .into());
                        }
                        let started = Instant::now();
                        let metadata = protect_kernel_inputs(&handles);
                        let output = abi::dispatch_single(provider, typed_abi, runmat_accelerate_api::ProviderFusedElementwiseRequest {
                            shader: single_shader.source(), requirements: single_shader.requirements(), inputs: &handles, output_shape: &output_shape, len, output_classes: &[physical_output_class], abi: single_shader.abi(),
                        }).await;
                        drop(metadata);
                        let output = output?;
                        effects.succeeded();
                        let output = allocations.admit(
                            output,
                            provider,
                            &output_shape,
                            physical_precision,
                        )?;
                        observer.queue(started);
                        observer.kernel_unmeasured();
                        outputs.insert(output_id, Value::GpuTensor(output));
                    }
                }
                Err(error) => return Err(error),
            }
        }
        timer.mark("dispatch");

        for (id, class) in output_ids.iter().zip(output_classes) {
            let Value::GpuTensor(raw) = outputs
                .remove(id)
                .ok_or_else(|| anyhow!("missing fused output"))?
            else {
                return Err(anyhow!("fusion returned a nonresident output").into());
            };
            let output = if class == runmat_types::NumericClass::Single
                && physical_precision == ProviderPrecision::F64
            {
                let metadata =
                    runmat_runtime::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(
                        &raw,
                    );
                let converted = provider.unary_single(&raw).await;
                drop(metadata);
                let converted = converted?;
                effects.succeeded();
                allocations.admit(converted, provider, &output_shape, ProviderPrecision::F32)?
            } else {
                raw
            };
            outputs.insert(*id, Value::GpuTensor(output));
        }
        for output in outputs.values_mut() {
            if let Value::GpuTensor(handle) = output {
                mark_fusion_output(handle, &request.inputs)?;
            }
        }
        timer.mark("cleanup");
        timer.finish();

        Ok(FusionExecutionOutcome::Executed((outputs, selection)))
        }
        .await;
        result.map_err(|error| effects.classify(error))
    })
    .await
}

fn physical_upload_bytes(precision: ProviderPrecision, elements: usize) -> Option<u64> {
    let width = match precision {
        ProviderPrecision::F32 => 4,
        ProviderPrecision::F64 => 8,
    };
    u64::try_from(elements).ok()?.checked_mul(width)
}

fn protect_kernel_inputs(
    handles: &[GpuTensorHandle],
) -> Vec<runmat_runtime::builtins::common::gpu_helpers::ResidencyRestoreGuard<'_>> {
    handles
        .iter()
        .map(runmat_runtime::builtins::common::gpu_helpers::ResidencyRestoreGuard::new)
        .collect()
}

/// Execute a direct provider request. VM placement uses the typed entrypoint.
pub async fn execute_elementwise(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<ElementwiseExecutionResult> {
    match execute_elementwise_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

/// Native regions retain their original executor plan and input generations;
/// they do not borrow VM installation, decline, readback, or host-route rights.
pub(crate) async fn execute_elementwise_native(
    request: FusionExecutionRequest<'_>,
    outputs: &[ValueId],
    right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
    native_program: &runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
) -> ExecutionResult<runmat_runtime::numeric_region::service::NativeNumericRegionOutcome> {
    use runmat_runtime::numeric_region::service::{root_values, NativeNumericRegionOutcome};
    let runtime = request
        .runtime
        .as_ref()
        .ok_or_else(|| anyhow!("native fusion requires its owning runtime"))?;
    if !right.is_current(runtime) {
        return Err(anyhow!("native numeric region authority is no longer current").into());
    }
    let mut host_continuation = None;
    let (mut values, selection) = match execute_elementwise_outputs(
        &request,
        outputs,
        None,
        None,
        &mut host_continuation,
        Some((native_program.cpu_candidates(), &right, native_program)),
    )
    .await?
    {
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return observation
                .into_native_selection(runtime, right, native_program, &request.inputs)
                .await
        }
        FusionExecutionOutcome::Executed(result) => result,
    };
    let rooted = root_values(runtime, outputs.len(), |index| {
        values.remove(&outputs[index]).ok_or_else(|| {
            runmat_runtime::runtime_error::semantic_error(
                "RunMat:Native:NumericRegion",
                "native fusion output is missing or duplicated",
            )
        })
    })
    .map_err(|source| ProviderExecutionError::from_source(source).after_effect())?;
    let publication = selection.into_native_publication(right.identity())?;
    Ok(NativeNumericRegionOutcome::Completed {
        outputs: rooted,
        right,
        publication,
    })
}

pub async fn execute_elementwise_placed(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseExecutionResult>> {
    Ok(
        match execute_elementwise_placed_inner(request, None, None).await? {
            FusionExecutionOutcome::Executed(output) => {
                FusionExecutionOutcome::Executed(output.result)
            }
            FusionExecutionOutcome::SharedRuntime(observation) => {
                FusionExecutionOutcome::SharedRuntime(observation)
            }
        },
    )
}

pub async fn execute_elementwise_placed_vm(
    request: FusionExecutionRequest<'_>,
    right: runmat_runtime::context::VmFusionCompletionRight,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseVmExecutionResult>> {
    let mut right = Some(right);
    execute_elementwise_placed_inner(
        request,
        Some(super::VmFusionExecutionRight::Completion(&mut right)),
        None,
    )
    .await
}

/// VM-only sibling retaining exact ordinary-load occurrences from the same
/// installed bytecode plan. Direct APIs do not manufacture that transcript.
pub async fn execute_elementwise_placed_vm_inputs(
    request: FusionExecutionRequest<'_>,
    right: runmat_runtime::context::VmFusionCompletionRight,
    input_uses: &super::VmHostInputUses,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseVmExecutionResult>> {
    let mut right = Some(right);
    execute_elementwise_placed_inner(
        request,
        Some(super::VmFusionExecutionRight::Completion(&mut right)),
        Some(input_uses),
    )
    .await
}

/// An installed VM region may decline before effects and resume its original
/// instructions. This right does not grant a host readback or FullCall route.
pub async fn execute_elementwise_placed_vm_decline(
    request: FusionExecutionRequest<'_>,
    decline: runmat_runtime::context::VmFusionDeclineRight,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseVmExecutionResult>> {
    execute_elementwise_placed_inner(
        request,
        Some(super::VmFusionExecutionRight::Decline(&decline)),
        None,
    )
    .await
}

async fn execute_elementwise_placed_inner(
    request: FusionExecutionRequest<'_>,
    authority: Option<super::VmFusionExecutionRight<'_>>,
    input_uses: Option<&super::VmHostInputUses>,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseVmExecutionResult>> {
    let final_output = request
        .plan
        .output
        .ok_or_else(|| anyhow!("fusion: missing final output value id"))?;
    let mut output_ids = vec![final_output];
    for store in &request.plan.materialized_stores {
        if !output_ids.contains(&store.value_id) {
            output_ids.push(store.value_id);
        }
    }
    let mut host_continuation = None;
    let (mut outputs, selection) = match execute_elementwise_outputs(
        &request,
        &output_ids,
        authority
            .as_ref()
            .and_then(super::VmFusionExecutionRight::as_selection),
        input_uses,
        &mut host_continuation,
        None,
    )
    .await?
    {
        FusionExecutionOutcome::Executed(outputs) => outputs,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            let observation = if let Some(right) =
                authority.and_then(super::VmFusionExecutionRight::into_completion)
            {
                let continuation = host_continuation.unwrap_or(ElementwiseHostContinuation {
                    operations: None,
                    readback: None,
                    outputs: None,
                });
                observation.with_vm_host_operations_readback(
                    right,
                    request.runtime.as_ref(),
                    continuation.operations,
                    continuation.readback,
                    continuation.outputs,
                )
            } else {
                observation
            };
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let final_value = outputs
        .remove(&final_output)
        .ok_or_else(|| anyhow!("fusion: final output materialization missing"))?;
    let materialized_stores = request
        .plan
        .materialized_stores
        .iter()
        .map(|store| {
            // A store whose value_id equals final_output was already consumed above;
            // reuse a clone of final_value rather than silently dropping the store.
            let value = if store.value_id == final_output {
                Some(final_value.clone())
            } else {
                // Several bindings may materialize the same SSA value. They
                // share its allocation rather than consuming it on first use.
                outputs.get(&store.value_id).cloned()
            };
            value
                .map(|v| (store.clone(), v))
                .ok_or_else(|| anyhow!("fusion store output materialization missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let completion =
        if let Some(right) = authority.and_then(super::VmFusionExecutionRight::into_completion) {
            selection.into_vm_completion(right)
        } else {
            selection.submitted();
            None
        };
    Ok(FusionExecutionOutcome::Executed(
        ElementwiseVmExecutionResult {
            result: ElementwiseExecutionResult {
                final_value,
                materialized_stores,
            },
            completion,
        },
    ))
}

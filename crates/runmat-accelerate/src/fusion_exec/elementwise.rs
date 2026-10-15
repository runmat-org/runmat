//! Typed elementwise request execution and transactional publication.
use super::*;
use crate::precision::ensure_provider_supports_dtype;
use runmat_accelerate_api::{ProviderElementType, ProviderExecutionResult as ExecutionResult};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;

#[path = "elementwise/effects.rs"]
mod effects;
use effects::ElementwiseEffects;

#[cfg(test)]
#[path = "elementwise/effect_tests.rs"]
mod effect_tests;

async fn execute_elementwise_outputs(
    request: &FusionExecutionRequest<'_>,
    output_ids: &[ValueId],
    vm_right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
) -> ExecutionResult<FusionExecutionOutcome<(HashMap<ValueId, Value>, FusionPlacementSelection)>> {
    let Some(bindings) = AdmittedBindings::new(request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
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
    reject_native_integer_fusion_inputs(request)?;
    let provider = match admit_provider_availability(request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => {
                use runmat_runtime::builtins::common::gpu_helpers::{
                    exact_provider_for_handle, gpu_class_metadata_matches,
                };
                let precision = runmat_accelerate_api::handle_precision(handle);
                let element = precision.map(|precision| match precision {
                    ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                    ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                });
                if precision.is_none()
                    || handle.descriptor().storage
                        != Some(runmat_accelerate_api::GpuTensorStorage::Real)
                    || handle.descriptor().element_type != element
                    || !gpu_class_metadata_matches(handle, precision, None, false)
                    || !exact_provider_for_handle(handle).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, provider)
                    })
                {
                    return Err(anyhow!(
                        "fusion input has invalid floating metadata or provider ownership"
                    )
                    .into());
                }
            }
            Value::Num(_) | Value::Tensor(_) => {}
            _ => {
                return Ok(FusionExecutionOutcome::SharedRuntime(
                    FusionCpuObservation::new(None),
                ))
            }
        }
    }
    let scalar_type = match provider.precision() {
        ProviderPrecision::F32 => runmat_runtime::builtins::common::spec::ScalarType::F32,
        ProviderPrecision::F64 => runmat_runtime::builtins::common::spec::ScalarType::F64,
    };
    let typed = async {
        let program = crate::fusion::semantics::ResolvedProgram::resolve(
            request.plan,
            &request.inputs,
            provider.precision(),
            request.placement,
        )
        .await?;
        let classes = output_ids
            .iter()
            .map(|id| program.floating_class(*id))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let generated_shader =
            request
                .plan
                .prepare_typed_shader(output_ids, &program, scalar_type)?;
        FusionPlacementObserver::new(request.placement).shader_source_generation(started);
        Ok::<_, crate::fusion::semantics::FusionResolveFailure>((
            program,
            classes,
            generated_shader,
        ))
    }
    .await;
    let (program, output_classes, generated_shader) = match typed {
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
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ));
        }
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
    let scalar_work_units = match u64::try_from(len)
        .ok()
        .and_then(|elements| program.scalar_work_units(request.plan, elements).ok())
    {
        Some(units) => units,
        None => {
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ))
        }
    };
    let semantic_identity = runmat_execution::Digest::sha256(
        serde_json::to_vec(&(program.cache_identity()?, &shader, scalar_work_units))
            .map_err(anyhow::Error::from)?,
    );
    let selection = select_fusion_provider_with_contract_vm(
        request,
        &bindings,
        provider,
        FusionProviderContract {
            input_indices: Some(&input_indices),
            outputs: representations,
            semantic_identity: Some(semantic_identity),
            scalar_work_units: Some(scalar_work_units),
            matmul: None,
            preparation_program: Some(
                runmat_accelerate_api::ProviderPreparationProgram::Elementwise { shader: &shader },
            ),
            shader_requirements: Some(shader_requirements),
        },
        vm_right,
    )?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let recorder = selection.recorder();
    runmat_accelerate_api::observation::observe_provider_future(recorder, async move {
        let mut effects = ElementwiseEffects::default();
        let result: ExecutionResult<_> = async {
        let observer = selection.observer;
        let mut timer = FusionStageTimer::new("elementwise", request.plan.index, len);
        let scalar_shape = normalize_scalar_shape(&vec![1; output_shape.len()]);
        let mut prepared = Vec::with_capacity(request.inputs.len());
        let mut allocations = outputs::FusionOutputAdmission::new(&request.inputs);
        let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
        let scalar_dtype = scalar_upload_dtype(provider);
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
                    let expected = match provider.precision() {
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
                        let converted = match provider.precision() {
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
                            provider.precision(),
                        )?
                    };
                    prepared.push(PreparedInput {
                        handle: prepared_handle,
                    });
                }
                Value::Tensor(t) => {
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
        let physical_output_class = match provider.precision() {
            ProviderPrecision::F32 => ProviderElementType::F32,
            ProviderPrecision::F64 => ProviderElementType::F64,
        };

        if output_ids.len() == 1 {
            // Fast path: single output, use the existing single-dispatch API.
            let output_id = output_ids[0];
            timer.mark("generate_wgsl");
            let started = Instant::now();
            let metadata = protect_kernel_inputs(&handles);
            let output = provider.fused_elementwise(
                &shader,
                shader_requirements,
                &handles,
                &output_shape,
                len,
                physical_output_class,
            ).await;
            drop(metadata);
            let output = output?;
            effects.succeeded();
            let output =
                allocations.admit(output, provider, &output_shape, provider.precision())?;
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
            let multi_result = provider.fused_elementwise_multi(
                &shader,
                shader_requirements,
                &handles,
                &output_shape,
                len,
                &physical_output_classes,
            ).await;
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
                        provider.precision(),
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
                        let single_shader = request.plan.prepare_typed_shader_with_inputs(
                            &[output_id],
                            &program,
                            scalar_type,
                            &input_indices,
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
                        let output = provider.fused_elementwise(
                            single_shader.source(),
                            single_shader.requirements(),
                            &handles,
                            &output_shape,
                            len,
                            physical_output_class,
                        ).await;
                        drop(metadata);
                        let output = output?;
                        effects.succeeded();
                        let output = allocations.admit(
                            output,
                            provider,
                            &output_shape,
                            provider.precision(),
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
                && provider.precision() == ProviderPrecision::F64
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

pub async fn execute_elementwise_placed(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<FusionExecutionOutcome<ElementwiseExecutionResult>> {
    Ok(
        match execute_elementwise_placed_inner(request, None).await? {
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
    execute_elementwise_placed_inner(request, Some(right)).await
}

async fn execute_elementwise_placed_inner(
    request: FusionExecutionRequest<'_>,
    right: Option<runmat_runtime::context::VmFusionCompletionRight>,
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
    let (mut outputs, selection) =
        match execute_elementwise_outputs(&request, &output_ids, right.as_ref()).await? {
            FusionExecutionOutcome::Executed(outputs) => outputs,
            FusionExecutionOutcome::SharedRuntime(observation) => {
                let observation = if let Some(right) = right {
                    let operations = AdmittedBindings::new(&request)
                        .and_then(|bindings| bindings.host_operations(request.runtime.as_ref()));
                    observation.with_vm_host_operations(right, request.runtime.as_ref(), operations)
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
    let completion = if let Some(right) = right {
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

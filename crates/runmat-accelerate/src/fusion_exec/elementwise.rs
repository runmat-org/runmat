//! Typed elementwise request execution and transactional publication.
use super::*;

async fn execute_elementwise_outputs(
    request: &FusionExecutionRequest<'_>,
    output_ids: &[ValueId],
) -> Result<FusionExecutionOutcome<HashMap<ValueId, Value>>> {
    crate::ensure_residency_hooks();
    if !request.plan.group.kind.is_elementwise() {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    if !request.plan.kernel.supported {
        return Err(anyhow!("fusion kernel not supported for this plan"));
    }
    if request.inputs.len() != request.plan.inputs.len() {
        return Err(anyhow!(
            "fusion input mismatch: expected {}, got {}",
            request.plan.inputs.len(),
            request.inputs.len()
        ));
    }
    reject_native_integer_fusion_inputs(request)?;
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
                    || handle.descriptor.storage
                        != Some(runmat_accelerate_api::GpuTensorStorage::Real)
                    || handle.descriptor.element_type != element
                    || !gpu_class_metadata_matches(handle, precision, None, false)
                    || !exact_provider_for_handle(handle)
                        .is_some_and(|owner| std::ptr::eq(owner, provider))
                {
                    return Err(anyhow!(
                        "fusion input has invalid floating metadata or provider ownership"
                    ));
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
    let typed = (|| {
        let program =
            crate::fusion::semantics::ResolvedProgram::resolve(request.plan, &request.inputs)?;
        let classes = output_ids
            .iter()
            .map(|id| program.floating_class(*id))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let shader = request
            .plan
            .generate_typed_wgsl(output_ids, &program, scalar_type)?;
        FusionPlacementObserver::new(request.placement).compile(started);
        Ok::<_, anyhow::Error>((program, classes, shader))
    })();
    let (program, output_classes, shader) = match typed {
        Ok(typed) => typed,
        Err(error) => {
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
    let semantic_identity = runmat_execution::Digest::sha256(serde_json::to_vec(&(
        program.cache_identity()?,
        &shader,
    ))?);
    let selection = select_fusion_provider_with_contract(
        request,
        provider,
        representations,
        Some(semantic_identity),
    )?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
    let mut timer = FusionStageTimer::new("elementwise", request.plan.index, len);
    let scalar_shape = normalize_scalar_shape(&vec![1; output_shape.len()]);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut allocations = outputs::FusionAllocations::new(&request.inputs);
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    let scalar_dtype = scalar_upload_dtype(provider);
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => {
                use runmat_runtime::builtins::common::gpu_helpers::{
                    exact_provider_for_handle, HandleMetadataRestoreGuard,
                };
                if !exact_provider_for_handle(handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
                {
                    return Err(anyhow!("fusion input has no matching exact provider owner"));
                }
                let expected = match provider.precision() {
                    ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                    ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                };
                let prepared_handle = if handle.descriptor.element_type == Some(expected) {
                    handle.clone()
                } else {
                    let metadata = HandleMetadataRestoreGuard::new(handle);
                    let converted = match provider.precision() {
                        ProviderPrecision::F64 => provider.unary_double(handle).await,
                        ProviderPrecision::F32 => provider.unary_single(handle).await,
                    };
                    drop(metadata);
                    allocations.admit(converted?, provider, &handle.shape, provider.precision())?
                };
                prepared.push(PreparedInput {
                    handle: prepared_handle,
                    owned: None,
                });
            }
            Value::Tensor(t) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, t.numeric_dtype()) {
                    return Err(anyhow!(
                        "fusion: tensor input requires unsupported precision ({msg})"
                    ));
                }
                let data = t.materialize_f64();
                let view = HostTensorView {
                    data: &data,
                    shape: &t.shape,
                };
                let started = Instant::now();
                let handle = allocations.admit(
                    provider.upload(&view)?,
                    provider,
                    &t.shape,
                    provider.precision(),
                )?;
                observer.upload(started, host_f64_bytes(data.len()));
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: None,
                });
            }
            Value::Num(n) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    ));
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (*n as f32) as f64,
                    ProviderPrecision::F64 => *n,
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
                };
                let started = Instant::now();
                let handle = allocations.admit(
                    provider.upload(&view)?,
                    provider,
                    &scalar_shape,
                    provider.precision(),
                )?;
                observer.upload(started, host_f64_bytes(data.len()));
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: None,
                });
            }
            Value::Int(_) => unreachable!("integer inputs rejected before fusion upload"),
            _ => {
                return Err(anyhow!("fusion: unsupported value type"));
            }
        }
    }
    timer.mark("prepare_inputs");

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    let mut outputs = HashMap::new();

    if output_ids.len() == 1 {
        // Fast path: single output, use the existing single-dispatch API.
        let output_id = output_ids[0];
        timer.mark("generate_wgsl");
        let started = Instant::now();
        let output = provider.fused_elementwise(&shader, &handles, &output_shape, len)?;
        let output = allocations.admit(output, provider, &output_shape, provider.precision())?;
        observer.queue(started);
        observer.kernel_unmeasured();
        outputs.insert(output_id, Value::GpuTensor(output));
    } else {
        // Multi-output path: generate one shader that writes all outputs in a single dispatch,
        // reducing O(N²) GPU work (N full-chain recomputations) to O(N).
        timer.mark("generate_wgsl");
        let started = Instant::now();
        let multi_result = provider.fused_elementwise_multi(
            &shader,
            &handles,
            &output_shape,
            len,
            output_ids.len(),
        );
        observer.queue(started);
        match multi_result {
            Ok(out_handles) => {
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
            Err(error) if runmat_accelerate_api::is_unsupported_provider_operation(&error) => {
                // Provider does not support multi-output dispatch; fall back to N single dispatches.
                for output_id in output_ids.iter().copied() {
                    let started = Instant::now();
                    let single_shader =
                        request
                            .plan
                            .generate_typed_wgsl(&[output_id], &program, scalar_type)?;
                    observer.compile(started);
                    let started = Instant::now();
                    let output =
                        provider.fused_elementwise(&single_shader, &handles, &output_shape, len)?;
                    let output =
                        allocations.admit(output, provider, &output_shape, provider.precision())?;
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
            return Err(anyhow!("fusion returned a nonresident output"));
        };
        let output = if class == runmat_types::NumericClass::Single
            && provider.precision() == ProviderPrecision::F64
        {
            let metadata =
                runmat_runtime::builtins::common::gpu_helpers::HandleMetadataRestoreGuard::new(
                    &raw,
                );
            let converted = provider.unary_single(&raw).await;
            drop(metadata);
            allocations.admit(converted?, provider, &output_shape, ProviderPrecision::F32)?
        } else {
            raw
        };
        outputs.insert(*id, Value::GpuTensor(output));
    }
    for output in outputs.values_mut() {
        if let Value::GpuTensor(handle) = output {
            mark_fusion_output(handle, &request.inputs);
        }
    }
    let published: Vec<_> = outputs
        .values()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();
    allocations.publish(&published);
    timer.mark("cleanup");
    timer.finish();

    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(outputs))
}

/// Execute a direct provider request. VM placement uses the typed entrypoint.
pub async fn execute_elementwise(
    request: FusionExecutionRequest<'_>,
) -> Result<ElementwiseExecutionResult> {
    match execute_elementwise_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_elementwise_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<ElementwiseExecutionResult>> {
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
    let mut outputs = match execute_elementwise_outputs(&request, &output_ids).await? {
        FusionExecutionOutcome::Executed(outputs) => outputs,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
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
                outputs.remove(&store.value_id)
            };
            value
                .map(|v| (store.clone(), v))
                .ok_or_else(|| anyhow!("fusion store output materialization missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FusionExecutionOutcome::Executed(
        ElementwiseExecutionResult {
            final_value,
            materialized_stores,
        },
    ))
}

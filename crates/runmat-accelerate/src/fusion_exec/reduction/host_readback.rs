//! Checked host preparation for a selected typed VM reduction CPU route.
//! This is input ownership/resource admission, not a second placement policy.
pub use super::super::host_readback::CpuReadbackFailure;
pub(crate) use super::super::host_readback::CpuReadbackPlan;
use super::super::host_readback::Source;
use super::super::FusionExecutionRequest;
use super::super::ResolvedReductionProgram;
use runmat_accelerate_api::{
    checked_numeric_download_host_payload_bytes, handle_provenance, same_provider_instance,
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, NumericElementType,
};
use runmat_execution::{
    CandidateOutputPresentation, CandidateResourceDemand, ProviderReadbackDemand,
    ProviderReadbackDemands, ResourceByteBound,
};
use runmat_types::{FusionReductionOperation, NumericClass};
use runmat_value::Value;
use std::collections::HashSet;

pub(crate) enum CpuHostAdmission {
    Direct,
    Readback(Box<CpuReadbackAdmission>),
}

pub(crate) struct CpuReadbackAdmission {
    pub(super) plan: CpuReadbackPlan,
    pub(super) presentation: CandidateOutputPresentation,
    pub(super) resources: CandidateResourceDemand,
    pub(super) scratch_bytes: u64,
}

fn shape_bytes(shape: &[usize], element_bytes: u64) -> Option<u64> {
    shape.iter().try_fold(element_bytes, |bytes, &dim| {
        bytes.checked_mul(u64::try_from(dim).ok()?)
    })
}

fn checked_add_min_bindings<'a>(
    request: &FusionExecutionRequest<'_>,
    program: &'a ResolvedReductionProgram<'_>,
) -> Option<&'a [Vec<usize>]> {
    let contract = program.contract();
    if contract.operation != FusionReductionOperation::Minimum
        || contract.input_class != NumericClass::Single
        || contract.accumulator_class != NumericClass::Single
        || contract.output_class != NumericClass::Single
        || contract.axes.len() != 1
        || contract.axes[0] >= contract.input_shape.len()
    {
        return None;
    }
    let plus_entry = &runmat_builtins::PLUS_CATALOG_ENTRY;
    let min_entry = &runmat_builtins::MIN_CATALOG_ENTRY;
    let [plus_binding] = plus_entry.bindings else {
        return None;
    };
    let [min_binding] = min_entry.bindings else {
        return None;
    };
    let plus = plus_entry.binding_identity(plus_binding);
    let min = min_entry.binding_identity(min_binding);
    let bindings = program.host_bindings()?;
    let shapes = program.host_producer_shapes()?;
    if bindings.len() != request.plan.operations.len()
        || bindings.len() < 2
        || shapes.len() + 1 != bindings.len()
        || shapes.last()? != &contract.input_shape
    {
        return None;
    }
    for (index, (operation, binding)) in request.plan.operations.iter().zip(bindings).enumerate() {
        if index + 1 == bindings.len() {
            if !matches!(operation, crate::fusion::FusionOp::Builtin { inputs, .. } if inputs.len() == 1)
                || *binding != min
            {
                return None;
            }
        } else if !matches!(
            operation,
            crate::fusion::FusionOp::Primitive {
                op: crate::graph::PrimitiveOp::Add,
                ..
            }
        ) || *binding != plus
        {
            return None;
        }
    }
    Some(shapes)
}

/// No readback is admitted from an ordinal merged stack-pattern binding. Every
/// resident source must have exactly its ordinary operation-use occurrences
/// certified by the original compiler variable-stack analysis.
pub(crate) fn admit(
    request: &FusionExecutionRequest<'_>,
    program: &ResolvedReductionProgram<'_>,
    provider: &'static dyn AccelProvider,
) -> Option<CpuHostAdmission> {
    let data = program.data_inputs();
    if data.len() != program.data_classes().len() {
        return None;
    }
    let declared_resident = request
        .inputs
        .iter()
        .chain(request.plan.const_values.values())
        .filter(|value| runmat_runtime::dispatcher::value_contains_gpu(value))
        .count();
    if declared_resident
        != data
            .iter()
            .filter(|input| matches!(input.value, Value::GpuTensor(_)))
            .count()
    {
        return None;
    }
    if !data
        .iter()
        .any(|input| matches!(input.value, Value::GpuTensor(_)))
    {
        return data
            .iter()
            .all(|input| !runmat_runtime::dispatcher::value_contains_gpu(input.value))
            .then_some(CpuHostAdmission::Direct);
    }
    let add_shapes = checked_add_min_bindings(request, program)?;
    if data
        .iter()
        .zip(program.data_classes())
        .any(|(input, class)| {
            if *class != NumericClass::Single {
                return true;
            }
            match input.value {
                Value::Tensor(tensor) => tensor.numeric_dtype() != runmat_value::NumericDType::F32,
                Value::GpuTensor(handle) => {
                    handle.physical().element_type() != NumericElementType::F32
                }
                _ => true,
            }
        })
    {
        return None;
    }
    let layout = request.plan.group.stack_layout.as_ref()?;
    if layout.compiler_variable_bindings.is_empty() || layout.required_stack_operands == 0 {
        return None;
    }
    let mut certified_offsets = HashSet::new();
    if layout.compiler_variable_bindings.iter().any(|binding| {
        binding.stack_offset >= layout.required_stack_operands
            || !certified_offsets.insert(binding.stack_offset)
            || !layout.bindings.contains(binding)
    }) {
        return None;
    }
    if certified_offsets.len() != layout.required_stack_operands
        || layout
            .compiler_variable_bindings
            .iter()
            .any(|binding| !data.iter().any(|input| input.value_id == binding.value_id))
    {
        return None;
    }

    let mut sources = Vec::new();
    let mut readbacks = Vec::new();
    let mut retained_download_bytes = 0_u64;
    let mut staged_download_bytes = 0_u64;
    let mut host_peak = 0_u64;
    for (input, &bound_class) in data.iter().zip(program.data_classes()) {
        let Value::GpuTensor(handle) = input.value else {
            if runmat_runtime::dispatcher::value_contains_gpu(input.value) {
                return None;
            }
            continue;
        };
        let request_slot = input.request_slot?;
        if request.plan.inputs.get(request_slot) != Some(&input.value_id)
            || !matches!(request.inputs.get(request_slot), Some(Value::GpuTensor(actual))
                if actual.view_identity() == handle.view_identity()
                    && actual.wire_descriptor() == handle.wire_descriptor())
            || handle_provenance(handle) != Some(GpuHandleProvenance::Automatic)
            || handle.physical().storage() != GpuTensorStorage::Real
            || handle.view().validate(handle.physical()).is_err()
            || handle.allocation_lease().validate_live().is_err()
        {
            return None;
        }
        let actual_class = match handle.physical().element_type() {
            NumericElementType::F32 => NumericClass::Single,
            NumericElementType::F64 => NumericClass::Double,
            _ => return None,
        };
        if bound_class != actual_class {
            return None;
        }
        let owner =
            runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(handle)?;
        if !same_provider_instance(owner, provider) {
            // Session facts currently carry only the selected provider's
            // resource namespace; a foreign owner's staging cannot be priced.
            return None;
        }
        let offsets = layout
            .compiler_variable_bindings
            .iter()
            .filter(|binding| binding.value_id == input.value_id)
            .map(|binding| binding.stack_offset)
            .collect::<Vec<_>>();
        let uses = request
            .plan
            .operations
            .iter()
            .map(|operation| match operation {
                crate::fusion::FusionOp::Primitive { inputs, .. }
                | crate::fusion::FusionOp::Builtin { inputs, .. } => {
                    inputs.iter().filter(|&&id| id == input.value_id).count()
                }
            })
            .sum::<usize>();
        if uses == 0 || offsets.len() != uses {
            // In-span LoadVar or graph-produced leaves require a different
            // compiler certificate; never infer those from a matching handle.
            return None;
        }
        let payload_bytes = checked_numeric_download_host_payload_bytes(handle)?;
        let shape_bytes = u64::try_from(handle.shape().len())
            .ok()?
            .checked_mul(u64::try_from(std::mem::size_of::<usize>()).ok()?)?;
        let host_value_bytes = payload_bytes.checked_add(shape_bytes)?;
        let workspace = match owner.download_numeric_host_workspace_bound(handle) {
            ResourceByteBound::Known(bytes) => bytes,
            ResourceByteBound::Unavailable(_) => return None,
        };
        let provider_scratch = owner.download_numeric_provider_scratch_bound(handle);
        let queue_slots = owner.download_numeric_queue_slots(handle);
        // The returned payload may coexist with its provider's host
        // workspace until the awaited download has fully completed.
        let stage = retained_download_bytes
            .checked_add(host_value_bytes)?
            .checked_add(workspace)?;
        host_peak = host_peak.max(stage);
        retained_download_bytes = retained_download_bytes.checked_add(host_value_bytes)?;
        let occurrence_bytes = host_value_bytes.checked_mul(u64::try_from(offsets.len()).ok()?)?;
        staged_download_bytes = staged_download_bytes.checked_add(occurrence_bytes)?;
        readbacks.push(ProviderReadbackDemand {
            device_id: owner.device_id(),
            provider_scratch_bytes: provider_scratch,
            queue_slots,
        });
        sources.push(Source {
            handle: handle.clone(),
            owner,
            offsets,
            loads: Vec::new(),
            capabilities: owner
                .capability_snapshot_ref()
                .map(std::borrow::Cow::Borrowed)
                .unwrap_or_else(|| std::borrow::Cow::Owned(owner.capability_snapshot())),
            payload_bytes,
            host_workspace: workspace,
            provider_scratch,
            queue_slots,
        });
    }
    if sources.is_empty() {
        return None;
    }
    let readbacks = ProviderReadbackDemands::sequential(readbacks)?;
    // Runtime's canonical Single Add/min owners bound their actual host
    // allocations. Every downloaded occurrence remains live across the body.
    let operation_scratch = runmat_runtime::context::checked_single_add_min_host_scratch(
        &program.contract().input_shape,
        &program.contract().output_shape,
        add_shapes,
        program.host_producer_operand_shapes()?,
    )?;
    let host_compute = staged_download_bytes.checked_add(operation_scratch)?;
    host_peak = host_peak.max(host_compute);
    let output_bytes = shape_bytes(&program.contract().output_shape, 4)?.checked_add(
        u64::try_from(program.contract().output_shape.len())
            .ok()?
            .checked_mul(u64::try_from(std::mem::size_of::<usize>()).ok()?)?,
    )?;
    let resources = CandidateResourceDemand {
        cpu_millicores: 1_000,
        retained_bytes: ResourceByteBound::Known(output_bytes),
        scratch_bytes: ResourceByteBound::Known(host_peak),
        queue_slots: 0,
        provider_host_workspace: None,
        provider_input_stages: None,
        peak_live: None,
    };
    let mut resident_offsets = sources
        .iter()
        .flat_map(|source| source.offsets.iter().copied())
        .collect::<Vec<_>>();
    resident_offsets.sort_unstable();
    resident_offsets.dedup();
    let plan = CpuReadbackPlan {
        sources,
        stack_len: layout.required_stack_operands,
        resident_offsets,
    };
    Some(CpuHostAdmission::Readback(Box::new(CpuReadbackAdmission {
        plan,
        presentation: CandidateOutputPresentation::ProviderReadbacksHost {
            readbacks: Box::new(readbacks),
        },
        resources,
        scratch_bytes: host_peak,
    })))
}

#[cfg(test)]
#[path = "host_readback/tests.rs"]
mod tests;

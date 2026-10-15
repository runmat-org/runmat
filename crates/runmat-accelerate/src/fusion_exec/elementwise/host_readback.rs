//! Elementwise admission over shared physical readback and canonical host
//! allocation contracts. No policy or numerical semantics live here.
use super::*;
use crate::fusion_exec::host_readback::{CpuHostResources, CpuReadbackPlan, Source};
use crate::fusion_exec::VmHostInputUses;
use runmat_execution::{
    CandidateOutputPresentation, CandidateResourceDemand, ProviderReadbackDemand,
    ProviderReadbackDemands, ResourceByteBound,
};
use runmat_runtime::builtins::common::spec::{FusionHostResourceInput, FusionHostResourceQuery};
use std::collections::HashSet;

pub(super) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
}

fn shape(program: &crate::fusion::semantics::ResolvedProgram, id: ValueId) -> Option<Vec<usize>> {
    program
        .fact(id)
        .ok()?
        .shape
        .known_dims()?
        .into_iter()
        .collect()
}

pub(super) fn admit(
    request: &FusionExecutionRequest<'_>,
    program: &crate::fusion::semantics::ResolvedProgram,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    uses: &VmHostInputUses,
) -> Option<Admission> {
    // These exact compiler offsets can cover a proper subset of operation
    // inputs: later occurrences have separately certified ordinary load PCs.
    let layout = request.plan.group.stack_layout.as_ref()?;
    if uses.entry.iter().any(|&(id, offset)| {
        !layout
            .compiler_variable_bindings
            .iter()
            .any(|binding| binding.value_id == id && binding.stack_offset == offset)
    }) {
        return None;
    }
    let mut used_offsets = HashSet::new();
    let mut used_loads = HashSet::new();
    if uses.entry.iter().any(|&(_, offset)| {
        offset >= layout.required_stack_operands || !used_offsets.insert(offset)
    }) || uses.loads.iter().any(|load| {
        load.pc < request.plan.group.span.start
            || load.pc > request.plan.group.span.end
            || !used_loads.insert(load.pc)
    }) {
        return None;
    }
    let mut sources: Vec<Source> = Vec::new();
    for (&id, value) in request.plan.inputs.iter().zip(&request.inputs) {
        let Value::GpuTensor(handle) = value else {
            if runmat_runtime::dispatcher::value_contains_gpu(value) {
                return None;
            }
            continue;
        };
        let class = match handle.physical().element_type() {
            runmat_accelerate_api::NumericElementType::F32 => runmat_types::NumericClass::Single,
            runmat_accelerate_api::NumericElementType::F64 => runmat_types::NumericClass::Double,
            _ => return None,
        };
        if program.floating_class(id).ok()? != class || shape(program, id)? != handle.shape() {
            return None;
        }
        let offsets = uses
            .entry
            .iter()
            .filter_map(|&(value, offset)| (value == id).then_some(offset))
            .collect::<Vec<_>>();
        let loads = uses
            .loads
            .iter()
            .filter(|load| load.value_id == id)
            .cloned()
            .collect::<Vec<_>>();
        let occurrences = request
            .plan
            .operations
            .iter()
            .map(|operation| match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => {
                    inputs.iter().filter(|&&input| input == id).count()
                }
            })
            .sum::<usize>();
        if occurrences == 0 || offsets.len().checked_add(loads.len())? != occurrences {
            return None;
        }
        if let Some(source) = sources
            .iter_mut()
            .find(|source| source.matches_handle(handle))
        {
            source.offsets.extend(offsets);
            source.loads.extend(loads);
        } else {
            sources.push(Source::admit(handle, provider, offsets, loads)?);
        }
    }
    // Resident captured constants have no ordinary-load identity; never invent
    // a mirror from their buffer or bypass the exact input transcript.
    if unbound_constant_values(request.plan).any(runmat_runtime::dispatcher::value_contains_gpu)
        || sources.is_empty()
    {
        return None;
    }

    let mut retained_outputs = 0_u64;
    let mut active_scratch = 0_u64;
    let mut operand_metadata = 0_u64;
    for (index, operation) in request.plan.operations.iter().enumerate() {
        let (inputs, output) = match operation {
            FusionOp::Primitive { inputs, output, .. }
            | FusionOp::Builtin { inputs, output, .. } => (inputs, (*output)?),
        };
        let shapes = inputs
            .iter()
            .map(|&id| shape(program, id))
            .collect::<Option<Vec<_>>>()?;
        // Private producer fanout and retained stack operands can keep shape
        // clones alive while another operation executes. Charge every input
        // occurrence, not merely one clone per graph value or active operator.
        for shape in &shapes {
            operand_metadata = operand_metadata.checked_add(
                u64::try_from(shape.len())
                    .ok()?
                    .checked_mul(std::mem::size_of::<usize>() as u64)?,
            )?;
        }
        let args = inputs
            .iter()
            .zip(&shapes)
            .map(|(&id, shape)| {
                Some(FusionHostResourceInput {
                    shape,
                    class: program.floating_class(id).ok()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let output_shape = shape(program, output)?;
        let bound = bindings.host_resource_bound(
            index,
            request.runtime.as_ref(),
            &FusionHostResourceQuery {
                inputs: &args,
                output: FusionHostResourceInput {
                    shape: &output_shape,
                    class: program.floating_class(output).ok()?,
                },
            },
        )?;
        retained_outputs = retained_outputs.checked_add(bound.retained_bytes)?;
        active_scratch = active_scratch.max(bound.scratch_bytes);
    }
    let mut retained_mirrors = 0_u64;
    let mut host_peak = 0_u64;
    let mut download_bytes = 0_u64;
    let mut demands = Vec::with_capacity(sources.len());
    for source in &sources {
        let metadata = u64::try_from(source.handle.shape().len())
            .ok()?
            .checked_mul(std::mem::size_of::<usize>() as u64)?;
        let value_bytes = source.payload_bytes.checked_add(metadata)?;
        host_peak = host_peak.max(
            retained_mirrors
                .checked_add(value_bytes)?
                .checked_add(source.host_workspace)?,
        );
        // Numeric storage is COW. Each live ordinary load may retain its own
        // shape clone; numeric COW copies are bounded by its canonical body.
        let occurrences =
            u64::try_from(source.offsets.len().checked_add(source.loads.len())?).ok()?;
        retained_mirrors = retained_mirrors
            .checked_add(value_bytes)?
            .checked_add(metadata.checked_mul(occurrences)?)?;
        download_bytes = download_bytes.checked_add(source.payload_bytes)?;
        demands.push(ProviderReadbackDemand {
            device_id: source.owner.device_id(),
            provider_scratch_bytes: source.provider_scratch,
            queue_slots: source.queue_slots,
        });
    }
    let body_extra = retained_mirrors
        .checked_add(active_scratch)?
        .checked_add(operand_metadata)?;
    let scratch_bytes =
        crate::fusion_exec::checked_extra_host_scratch(retained_outputs, host_peak, body_extra)?;
    let readbacks = ProviderReadbackDemands::sequential(demands)?;
    let resident_offsets = sources
        .iter()
        .flat_map(|source| source.offsets.iter().copied())
        .collect();
    let readback_count = u64::try_from(sources.len()).ok()?;
    Some(Admission {
        plan: CpuReadbackPlan {
            sources,
            stack_len: layout.required_stack_operands,
            resident_offsets,
        },
        resources: CpuHostResources {
            // All operation outputs are conservatively retained across the
            // ordinary VM span, including observable stores and private DAGs.
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: ResourceByteBound::Known(retained_outputs),
                scratch_bytes: ResourceByteBound::Known(scratch_bytes),
                queue_slots: 0,
                provider_host_workspace: None,
                provider_input_stages: None,
                peak_live: None,
            },
            presentation: CandidateOutputPresentation::ProviderReadbacksHost {
                readbacks: Box::new(readbacks),
            },
            scratch_bytes,
            download_bytes,
            readback_count,
        },
    })
}

//! Elementwise admission over shared physical readback and canonical host
//! allocation contracts. No policy or numerical semantics live here.
use super::*;
use crate::fusion::FusionOp;
use crate::fusion_exec::host_readback::{CpuHostResources, CpuReadbackPlan, Source};
use crate::fusion_exec::VmHostInputUses;
use runmat_runtime::builtins::common::spec::{FusionHostResourceInput, FusionHostResourceQuery};

pub(super) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
    pub program: crate::fusion::semantics::ResolvedProgram,
}

fn shape(program: &crate::fusion::semantics::ResolvedProgram, id: ValueId) -> Option<Vec<usize>> {
    let known = program.fact(id).ok()?.shape.known_dims()?;
    let mut shape = Vec::with_capacity(known.len());
    for dimension in known {
        shape.push(dimension?);
    }
    Some(shape)
}

pub(super) fn admit(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    uses: &VmHostInputUses,
) -> Option<Admission> {
    admit_inputs(request, bindings, provider, uses, false)
}

/// A shader-unavailable route also needs an exact bound for entirely host
/// inputs; the normal shader path keeps its existing host-only fast path.
pub(super) fn admit_host(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    uses: &VmHostInputUses,
) -> Option<Admission> {
    admit_inputs(request, bindings, provider, uses, true)
}

fn admit_inputs(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    uses: &VmHostInputUses,
    allow_host_only: bool,
) -> Option<Admission> {
    // This route executes ordinary host operators. Provider-effective facts
    // may narrow Double under an opted F32 policy and cannot bound its host
    // allocations or determine the classes of intermediate CPU results.
    let host_program = bindings.host_facts(request, provider.precision())?;
    let program = &host_program;
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
    let mut used_offsets = uses
        .entry
        .iter()
        .map(|&(_, offset)| offset)
        .collect::<Vec<_>>();
    if used_offsets
        .iter()
        .any(|&offset| offset >= layout.required_stack_operands)
    {
        return None;
    }
    used_offsets.sort_unstable();
    if used_offsets.windows(2).any(|pair| pair[0] == pair[1]) {
        return None;
    }
    let mut used_loads = uses.loads.iter().map(|load| load.pc).collect::<Vec<_>>();
    if used_loads
        .iter()
        .any(|&pc| pc < request.plan.group.span.start || pc > request.plan.group.span.end)
    {
        return None;
    }
    used_loads.sort_unstable();
    if used_loads.windows(2).any(|pair| pair[0] == pair[1]) {
        return None;
    }
    let mut sources: Vec<Source> = Vec::with_capacity(
        request
            .inputs
            .iter()
            .filter(|value| matches!(value, Value::GpuTensor(_)))
            .count(),
    );
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
        let mut offsets = Vec::with_capacity(uses.entry.len());
        offsets.extend(
            uses.entry
                .iter()
                .filter_map(|&(value, offset)| (value == id).then_some(offset)),
        );
        let mut loads = Vec::with_capacity(uses.loads.len());
        loads.extend(
            uses.loads
                .iter()
                .filter(|load| load.value_id == id)
                .cloned(),
        );
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
        || (!allow_host_only && sources.is_empty())
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
        let mut shapes = Vec::with_capacity(inputs.len());
        for &id in inputs {
            shapes.push(shape(program, id)?);
        }
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
        let mut args = Vec::with_capacity(inputs.len());
        for (&id, shape) in inputs.iter().zip(&shapes) {
            args.push(FusionHostResourceInput {
                shape,
                class: program.floating_class(id).ok()?,
            });
        }
        let output_shape = shape(program, output)?;
        let bound = bindings.host_resource_bound(
            index,
            request.runtime.as_ref(),
            &FusionHostResourceQuery {
                inputs: &args,
                output_domain:
                    runmat_runtime::builtins::common::spec::FusionHostOutputDomain::Exact(
                        runmat_types::NumericDomain::Real,
                    ),
                output: FusionHostResourceInput {
                    shape: &output_shape,
                    class: program.floating_class(output).ok()?,
                },
            },
        )?;
        retained_outputs = retained_outputs.checked_add(bound.retained_bytes)?;
        active_scratch = active_scratch.max(bound.scratch_bytes);
    }
    let resources = crate::fusion_exec::host_readback::resources(
        &sources,
        retained_outputs,
        active_scratch,
        operand_metadata,
    )?;
    let mut resident_offsets = Vec::with_capacity(uses.entry.len());
    resident_offsets.extend(
        sources
            .iter()
            .flat_map(|source| source.offsets.iter().copied()),
    );
    resident_offsets.sort_unstable();
    resident_offsets.dedup();
    Some(Admission {
        program: host_program,
        plan: CpuReadbackPlan {
            sources,
            stack_len: layout.required_stack_operands,
            resident_offsets,
        },
        resources,
    })
}

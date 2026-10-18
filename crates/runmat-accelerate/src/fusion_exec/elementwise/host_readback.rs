//! Elementwise admission over shared physical readback and canonical host
//! allocation contracts. No policy or numerical semantics live here.
use super::*;
use crate::fusion::FusionOp;
use crate::fusion_exec::host_readback::{CpuHostResources, CpuReadbackPlan, Source};
use crate::fusion_exec::VmHostInputUses;
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinOperandPreparation, BuiltinPreparationOperation,
};
use runmat_runtime::builtins::common::spec::FusionHostOutputDomain;
use runmat_runtime::builtins::common::spec::{FusionHostResourceInput, FusionHostResourceQuery};
use runmat_runtime::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use runmat_runtime::context::preparation::AdmittedHostResources;
use runmat_runtime::context::{
    VmFusionHostInstruction, VmFusionHostOperation, VmFusionHostOutputEntry, VmFusionHostOutputs,
};
use runmat_types::NumericDomain;

pub(super) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
    pub program: crate::fusion::semantics::ResolvedProgram,
    pub outputs: Vec<HostOutputSpec>,
}

pub(crate) struct HostOutputSpec {
    owner: VmFusionHostOperation,
    instruction: VmFusionHostInstruction,
    envelope: AdmittedHostResources,
}

impl HostOutputSpec {
    fn prepare(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<
        VmFusionHostOutputEntry,
        runmat_runtime::context::host_allocation::HostAllocationError,
    > {
        let output =
            runmat_runtime::context::preparation::PreparedHostNumericalOutput::prepare_admitted(
                context,
                &self.envelope,
            )?
            .ok_or(
                runmat_runtime::context::host_allocation::HostAllocationError::MissingManagedOwner,
            )?;
        Ok(match self.owner {
            VmFusionHostOperation::Builtin(binding) => {
                VmFusionHostOutputEntry::builtin(binding, self.instruction, output)
            }
            VmFusionHostOperation::Intrinsic(operator)
                if operator != runmat_types::OperatorKind::UnaryPlus =>
            {
                VmFusionHostOutputEntry::intrinsic(operator, self.instruction, output)
            }
            VmFusionHostOperation::Intrinsic(_) => return Err(
                runmat_runtime::context::host_allocation::HostAllocationError::MissingManagedOwner,
            ),
        })
    }
}

pub(crate) fn prepare_outputs(
    context: &runmat_runtime::context::RuntimeContext,
    specs: &[HostOutputSpec],
) -> Result<
    Option<VmFusionHostOutputs>,
    runmat_runtime::context::host_allocation::HostAllocationError,
> {
    VmFusionHostOutputs::prepare_with(context, specs.len(), |index| specs[index].prepare(context))
}

fn numerical_form(
    binding: BuiltinBindingIdentity,
    input_count: usize,
) -> Option<BuiltinPreparationOperation> {
    let entry = runmat_builtins::builtin_catalog_entry_by_name(binding.builtin.name)?;
    let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
        return None;
    };
    let mut matches = forms.iter().filter(|form| {
        form.binding_variant == binding.variant
            && form.data_arguments.len() == input_count
            && form.data_arguments.iter().copied().eq(0..input_count)
            && runmat_runtime::context::preparation::PreparedHostExecution::supports_form(form)
    });
    let form = matches.next()?;
    matches.next().is_none().then_some(form.operation)
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

    let managed = HostAllocationReceipt::policy(request.runtime.as_ref()).ok()?
        == HostAllocationPolicy::Managed;
    if managed
        && sources
            .iter()
            .any(|source| !matches!(&source.capabilities, std::borrow::Cow::Borrowed(_)))
    {
        return None;
    }
    let canonical = if managed {
        let canonical = bindings.host_numerical_operations(request.runtime.as_ref())?;
        if canonical.len() != request.plan.operations.len() {
            return None;
        }
        canonical
    } else {
        Vec::new()
    };
    let output_count = canonical
        .iter()
        .filter(|operation| {
            !matches!(
                operation,
                VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryPlus)
            )
        })
        .count();
    let mut outputs = Vec::with_capacity(output_count);
    let mut retained_outputs = if managed {
        VmFusionHostOutputs::retained_bytes(output_count).ok()?
    } else {
        0
    };
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
        let selected = if managed {
            match canonical[index] {
                VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryPlus) => {
                    // Managed readback attaches retained claims to the decoded
                    // payload and shape. Unary plus can therefore return that
                    // same owner beyond region cleanup without another output
                    // allocation, just as it can for a pre-existing host input.
                    if !matches!(
                        operation,
                        FusionOp::Primitive {
                            op: crate::graph::PrimitiveOp::UPlus,
                            ..
                        }
                    ) || inputs.len() != 1
                    {
                        return None;
                    }
                    None
                }
                VmFusionHostOperation::Builtin(binding) => {
                    let preparation = numerical_form(binding, inputs.len())?;
                    let instruction = match operation {
                        FusionOp::Primitive { .. } => VmFusionHostInstruction::ElementwiseBinary,
                        FusionOp::Builtin { .. } => match preparation {
                            BuiltinPreparationOperation::ElementwiseBinary
                            | BuiltinPreparationOperation::RealPower => {
                                VmFusionHostInstruction::ElementwiseBinary
                            }
                            _ => VmFusionHostInstruction::ElementwiseCall,
                        },
                    };
                    Some((canonical[index], preparation, instruction))
                }
                VmFusionHostOperation::Intrinsic(operator) => {
                    let (expected, input_count) = match operator {
                        runmat_types::OperatorKind::UnaryMinus => {
                            (crate::graph::PrimitiveOp::Neg, 1)
                        }
                        runmat_types::OperatorKind::MatrixMultiply => {
                            (crate::graph::PrimitiveOp::Mul, 2)
                        }
                        _ => return None,
                    };
                    if inputs.len() != input_count
                        || !matches!(operation, FusionOp::Primitive { op, .. } if *op == expected)
                    {
                        return None;
                    }
                    Some((
                        canonical[index],
                        BuiltinPreparationOperation::ElementwiseBinary,
                        VmFusionHostInstruction::Intrinsic(operator),
                    ))
                }
            }
        } else {
            None
        };
        let output_domain = match selected {
            Some((
                _,
                BuiltinPreparationOperation::RealUnary(_) | BuiltinPreparationOperation::RealPower,
                _,
            )) => FusionHostOutputDomain::RealOrComplex,
            _ => FusionHostOutputDomain::Exact(NumericDomain::Real),
        };
        let output_class = program.floating_class(output).ok()?;
        let bound = bindings.host_resource_bound(
            index,
            request.runtime.as_ref(),
            &FusionHostResourceQuery {
                inputs: &args,
                output_domain,
                output: FusionHostResourceInput {
                    shape: &output_shape,
                    class: output_class,
                },
            },
        )?;
        if let Some((owner, preparation, instruction)) = selected {
            let operand_shapes = if matches!(
                owner,
                VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryMinus)
            ) {
                operand_metadata = operand_metadata.checked_add(
                    u64::try_from(shapes[0].len().checked_add(2)?)
                        .ok()?
                        .checked_mul(std::mem::size_of::<usize>() as u64)?,
                )?;
                vec![vec![1, 1], shapes[0].clone()]
            } else {
                shapes
            };
            let identity = runmat_execution::Digest::sha256_serialized(&(
                "selected-fused-host-numerical-output-v1",
                owner,
                index,
                preparation,
                &operand_shapes,
                &output_shape,
                output_class,
                output_domain,
                bound.retained_bytes,
                bound.scratch_bytes,
            ))
            .ok()?;
            outputs.push(HostOutputSpec {
                owner,
                instruction,
                envelope: AdmittedHostResources {
                    bound,
                    output_shape,
                    output_class,
                    output_domain,
                    operand_metadata: 0,
                    operand_shapes,
                    identity,
                    operation: preparation,
                },
            });
        }
        retained_outputs = retained_outputs.checked_add(bound.retained_bytes)?;
        // Each output owner is constructed before source staging and retains
        // its own scratch reservation through this selected VM continuation.
        active_scratch = if managed {
            active_scratch.checked_add(bound.scratch_bytes)?
        } else {
            active_scratch.max(bound.scratch_bytes)
        };
    }
    let resources = crate::fusion_exec::host_readback::resources(
        &sources,
        retained_outputs,
        active_scratch,
        operand_metadata,
        managed,
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
        outputs,
        plan: CpuReadbackPlan {
            sources,
            stack_len: layout.required_stack_operands,
            resident_offsets,
        },
        resources,
    })
}

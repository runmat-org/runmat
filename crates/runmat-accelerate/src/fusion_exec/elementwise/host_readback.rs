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
    VmFusionHostRetirement,
};
use runmat_types::NumericDomain;

// ValueId is u32; this bounded transcript is at most 4096 bytes, including
// tuple padding, on supported 32/64-bit targets. This is not another plan cap.
const _: () = assert!(
    std::mem::size_of::<[(ValueId, usize); runmat_runtime::numeric_region::MAX_NODES]>() <= 4096
);

pub(in crate::fusion_exec) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
    pub program: crate::fusion::semantics::ResolvedProgram,
    pub outputs: HostOutputPlan,
    pub canonical: Vec<VmFusionHostOperation>,
}

enum InputIdentity<'a> {
    Vm(&'a VmHostInputUses),
    Native {
        program: &'a runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
        right: &'a runmat_runtime::numeric_region::service::NativeNumericRegionRight,
    },
}

pub(crate) struct HostOutputSpec {
    source_index: usize,
    owner: VmFusionHostOperation,
    instruction: VmFusionHostInstruction,
    envelope: AdmittedHostResources,
}

pub(crate) struct HostOutputPlan {
    specs: Vec<HostOutputSpec>,
    liveness: Option<super::host_liveness::HostLivenessPlan>,
}

impl HostOutputPlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.specs.is_empty() && self.liveness.is_none()
    }
}

fn recyclable_named_instruction(
    specs: &[HostOutputSpec],
    source_index: usize,
    binding: BuiltinBindingIdentity,
    input_count: usize,
) -> Option<VmFusionHostInstruction> {
    if !(1..=2).contains(&input_count) {
        return None;
    }
    let spec = specs
        .binary_search_by_key(&source_index, |spec| spec.source_index)
        .ok()
        .and_then(|index| specs.get(index))?;
    if spec.owner != VmFusionHostOperation::Builtin(binding)
        || !matches!(
            spec.instruction,
            VmFusionHostInstruction::ElementwiseCall | VmFusionHostInstruction::ElementwiseBinary
        )
        || spec.envelope.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real)
        || !matches!(
            spec.envelope.output_class,
            runmat_types::NumericClass::Single | runmat_types::NumericClass::Double
        )
        || spec.envelope.operand_shapes.len() != input_count
    {
        return None;
    }
    let dtype = match spec.envelope.output_class {
        runmat_types::NumericClass::Single => runmat_value::NumericDType::F32,
        runmat_types::NumericClass::Double => runmat_value::NumericDType::F64,
        _ => return None,
    };
    let retained =
        runmat_runtime::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
            dtype,
            &spec.envelope.output_shape,
        )
        .ok()?;
    (spec.envelope.bound.retained_bytes == retained).then_some(spec.instruction)
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
    plan: &HostOutputPlan,
) -> Result<
    Option<VmFusionHostOutputs>,
    runmat_runtime::context::host_allocation::HostAllocationError,
> {
    let Some(liveness) = plan.liveness.as_ref() else {
        return VmFusionHostOutputs::prepare_with(context, plan.specs.len(), |index| {
            plan.specs[index].prepare(context)
        });
    };
    let slots = liveness.slot_specs();
    VmFusionHostOutputs::prepare_recycled_with(
        context,
        &slots,
        liveness.operations.len(),
        |source_index, slots| {
            let operation = &liveness.operations[source_index];
            let retirements = operation.retirements.map(|retirement| {
                retirement.map(|retirement| VmFusionHostRetirement {
                    operand_position: retirement.operand_position,
                    slot: retirement.slot,
                })
            });
            let work = match operation.output_slot {
                Some(slot) => {
                    let spec = plan
                        .specs
                        .binary_search_by_key(&source_index, |spec| spec.source_index)
                        .ok()
                        .and_then(|index| plan.specs.get(index))
                        .ok_or(runmat_runtime::context::host_allocation::HostAllocationError::Numeric(
                            runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
                        ))?;
                    let slot = slots.get(slot).ok_or(
                        runmat_runtime::context::host_allocation::HostAllocationError::Numeric(
                            runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
                        ),
                    )?;
                    Some(
                        runmat_runtime::context::preparation::PreparedHostNumericalOutput::prepare_recycling_work(
                            context,
                            &spec.envelope,
                            slot.dtype,
                            slot.shape,
                        )?,
                    )
                }
                None => None,
            };
            match operation.operation {
                VmFusionHostOperation::Intrinsic(operator) => {
                    Ok(VmFusionHostOutputEntry::recycled_intrinsic(
                        operator,
                        operation.instruction,
                        operation.output_slot,
                        retirements,
                        work,
                    ))
                }
                VmFusionHostOperation::Builtin(binding) => {
                    let output_slot = operation.output_slot.ok_or(
                        runmat_runtime::context::host_allocation::HostAllocationError::Numeric(
                            runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
                        ),
                    )?;
                    let work = work.ok_or(
                        runmat_runtime::context::host_allocation::HostAllocationError::Numeric(
                            runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
                        ),
                    )?;
                    Ok(VmFusionHostOutputEntry::recycled_builtin(
                        binding,
                        operation.instruction,
                        output_slot,
                        retirements,
                        work,
                    ))
                }
            }
        },
    )
    .map(Some)
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
    admit_inputs(request, bindings, provider, InputIdentity::Vm(uses), false)
}

/// A shader-unavailable route also needs an exact bound for entirely host
/// inputs; the normal shader path keeps its existing host-only fast path.
pub(super) fn admit_host(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    uses: &VmHostInputUses,
) -> Option<Admission> {
    admit_inputs(request, bindings, provider, InputIdentity::Vm(uses), true)
}

pub(super) fn admit_native(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    program: &runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
    right: &runmat_runtime::numeric_region::service::NativeNumericRegionRight,
) -> Option<Admission> {
    let mut admission = admit_inputs(
        request,
        bindings,
        provider,
        InputIdentity::Native { program, right },
        true,
    )?;
    let control = runmat_runtime::context::NativeRegionHostRoute::retained_control_bytes()?;
    let slots = runmat_value::OutputListStorage::requested_bytes(program.input_count()).ok()?;
    let slots = u64::try_from(slots).ok()?;
    let slots =
        runmat_runtime::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            slots,
            usize::from(program.input_count() != 0),
        )
        .ok()?;
    let runmat_execution::ResourceByteBound::Known(retained) =
        admission.resources.resources.retained_bytes
    else {
        return None;
    };
    admission.resources.resources.retained_bytes = runmat_execution::ResourceByteBound::Known(
        retained
            .checked_add(control)?
            .checked_add(slots)?
            .checked_add(super::super::observation::native_host_admission_retained_bytes()?)?,
    );
    Some(admission)
}

fn admit_inputs(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    identity: InputIdentity<'_>,
    allow_host_only: bool,
) -> Option<Admission> {
    // This route executes ordinary host operators. Provider-effective facts
    // may narrow Double under an opted F32 policy and cannot bound its host
    // allocations or determine the classes of intermediate CPU results.
    let host_program = bindings.host_facts(request, provider.precision())?;
    let program = &host_program;
    // These exact compiler offsets can cover a proper subset of operation
    // inputs: later occurrences have separately certified ordinary load PCs.
    // Native program validation already bounds captured inputs by MAX_NODES.
    // Keep this request-local transcript inline, not in an unquoted Vec.
    let mut native_entry = [(0 as ValueId, 0usize); runmat_runtime::numeric_region::MAX_NODES];
    let (entry, loads, stack_len) = match &identity {
        InputIdentity::Vm(uses) => {
            let layout = request.plan.group.stack_layout.as_ref()?;
            if uses.entry.iter().any(|&(id, offset)| {
                !layout
                    .compiler_variable_bindings
                    .iter()
                    .any(|binding| binding.value_id == id && binding.stack_offset == offset)
            }) {
                return None;
            }
            (
                uses.entry.as_slice(),
                uses.loads.as_slice(),
                layout.required_stack_operands,
            )
        }
        InputIdentity::Native { program, right } => {
            let runtime = request.runtime.as_ref()?;
            if !right.is_current(runtime)
                || !program.has_native_operation_sites()
                || program.native_operation_count() != request.plan.operations.len()
            {
                return None;
            }
            // These are actual positions in the native producer's immutable
            // captured input vector, not fabricated bytecode stack/load PCs.
            let entry = native_entry.get_mut(..request.plan.inputs.len())?;
            for (index, (slot, &id)) in entry.iter_mut().zip(&request.plan.inputs).enumerate() {
                *slot = (id, index);
            }
            (&*entry, &[][..], request.inputs.len())
        }
    };
    let mut used_offsets = entry.iter().map(|&(_, offset)| offset).collect::<Vec<_>>();
    if used_offsets.iter().any(|&offset| offset >= stack_len) {
        return None;
    }
    used_offsets.sort_unstable();
    if used_offsets.windows(2).any(|pair| pair[0] == pair[1]) {
        return None;
    }
    let mut used_loads = loads.iter().map(|load| load.pc).collect::<Vec<_>>();
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
        let mut offsets = Vec::with_capacity(entry.len());
        offsets.extend(
            entry
                .iter()
                .filter_map(|&(value, offset)| (value == id).then_some(offset)),
        );
        let mut source_loads = Vec::with_capacity(loads.len());
        source_loads.extend(loads.iter().filter(|load| load.value_id == id).cloned());
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
        if matches!(identity, InputIdentity::Vm(_))
            && (occurrences == 0 || offsets.len().checked_add(source_loads.len())? != occurrences)
        {
            return None;
        }
        if let Some(source) = sources
            .iter_mut()
            .find(|source| source.matches_handle(handle))
        {
            source.offsets.extend(offsets);
            source.loads.extend(source_loads);
        } else {
            sources.push(Source::admit(handle, provider, offsets, source_loads)?);
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
    let canonical = if managed || matches!(identity, InputIdentity::Native { .. }) {
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
        let query = FusionHostResourceQuery {
            inputs: &args,
            output_domain,
            output: FusionHostResourceInput {
                shape: &output_shape,
                class: output_class,
            },
        };
        let bound = bindings.host_resource_bound(index, request.runtime.as_ref(), &query)?;
        let scalar_shape = [1usize, 1];
        let prepared_scratch_bytes = match selected {
            Some((
                VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryMinus),
                preparation,
                _,
            )) => {
                let virtual_inputs = [
                    FusionHostResourceInput {
                        shape: &scalar_shape,
                        class: runmat_types::NumericClass::Double,
                    },
                    *args.first()?,
                ];
                let virtual_query = FusionHostResourceQuery {
                    inputs: &virtual_inputs,
                    output: query.output,
                    output_domain,
                };
                runmat_runtime::builtins::math::host_resources::prepared_scratch_bytes(
                    preparation,
                    &virtual_query,
                )?
            }
            Some((_, preparation, _)) => {
                runmat_runtime::builtins::math::host_resources::prepared_scratch_bytes(
                    preparation,
                    &query,
                )?
            }
            None => bound.scratch_bytes,
        };
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
                prepared_scratch_bytes,
            ))
            .ok()?;
            outputs.push(HostOutputSpec {
                source_index: index,
                owner,
                instruction,
                envelope: AdmittedHostResources {
                    bound,
                    prepared_scratch_bytes,
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
            active_scratch.checked_add(prepared_scratch_bytes)?
        } else {
            active_scratch.max(bound.scratch_bytes)
        };
    }
    let liveness = if managed && matches!(identity, InputIdentity::Vm(_)) {
        super::host_liveness::plan(
            request.plan,
            program,
            &canonical,
            |index, binding, input_count| {
                recyclable_named_instruction(&outputs, index, binding, input_count)
            },
        )
    } else {
        None
    };
    if let Some(plan) = liveness.as_ref() {
        let slots = plan.slot_specs();
        retained_outputs =
            VmFusionHostOutputs::retained_bytes_with_slots(plan.operations.len(), &slots).ok()?;
    }
    let resources = crate::fusion_exec::host_readback::resources(
        &sources,
        retained_outputs,
        active_scratch,
        operand_metadata,
        managed,
    )?;
    let mut resident_offsets = Vec::with_capacity(entry.len());
    resident_offsets.extend(
        sources
            .iter()
            .flat_map(|source| source.offsets.iter().copied()),
    );
    resident_offsets.sort_unstable();
    resident_offsets.dedup();
    Some(Admission {
        program: host_program,
        outputs: HostOutputPlan {
            specs: outputs,
            liveness,
        },
        plan: CpuReadbackPlan {
            sources,
            stack_len,
            resident_offsets,
        },
        resources,
        canonical,
    })
}

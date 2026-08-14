use std::collections::BTreeSet;

use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_types::{CapabilitySet, EffectSet, ProgramPointId};

pub(super) struct OperationSite {
    pub point: ProgramPointId,
    pub output_point: ProgramPointId,
    pub phase: NativeSitePhase,
    pub ordinal: u32,
    pub source: NativeSourceLocation,
}

#[derive(Clone)]
pub(super) struct OperationRequirements {
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_rvalue(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    site: OperationSite,
    value: &runmat_mir::MirRvalue,
    locals: &[NativeValueId],
    output_locals: &[Option<runmat_mir::MirLocalId>],
    result: NativeRvalueResult,
    requirements: OperationRequirements,
    side_effect_epoch: &mut NativeValueId,
) -> NativeCodegenResult<NativeInstruction> {
    let construct = runmat_mir::rvalue_construct_kind(value);
    if construct.native_lowering_class() == runmat_mir::NativeLoweringClass::CapabilityRejection {
        return Err(NativeCodegenError::new(
            "native.capability.distributed_core_pending",
            "construct requires the R25 distributed core",
        )
        .at_point(site.point)
        .for_construct(construct));
    }
    let inputs = rvalue_inputs(value, locals)?;
    build_instruction(
        ctx,
        site,
        construct,
        NativeOperation::Rvalue {
            value: value.clone(),
            result,
        },
        inputs,
        output_locals,
        requirements,
        side_effect_epoch,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_statement(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    site: OperationSite,
    statement: &runmat_mir::MirStmtKind,
    locals: &[NativeValueId],
    inputs: Vec<NativeValueId>,
    output_locals: &[runmat_mir::MirLocalId],
    requirements: OperationRequirements,
    side_effect_epoch: &mut NativeValueId,
) -> NativeCodegenResult<NativeInstruction> {
    let construct = runmat_mir::statement_construct_kind(statement);
    let mut all_inputs = inputs;
    all_inputs.extend(statement_inputs(statement, locals)?);
    deduplicate(&mut all_inputs);
    build_instruction(
        ctx,
        site,
        construct,
        NativeOperation::Statement(statement.clone()),
        all_inputs,
        &output_locals.iter().copied().map(Some).collect::<Vec<_>>(),
        requirements,
        side_effect_epoch,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_instruction(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    site: OperationSite,
    construct: runmat_mir::MirConstructKind,
    operation: NativeOperation,
    inputs: Vec<NativeValueId>,
    output_locals: &[Option<runmat_mir::MirLocalId>],
    requirements: OperationRequirements,
    side_effect_epoch: &mut NativeValueId,
) -> NativeCodegenResult<NativeInstruction> {
    let class = construct.native_lowering_class();
    let outputs = output_locals
        .iter()
        .map(|local| {
            Ok(NativeOutput {
                value: ctx.value(),
                value_type: local.map_or(NativeValueType::Generic, |local| {
                    ctx.value_type(site.output_point, local)
                }),
                local: local
                    .map(|local| super::function::checked_local(local, ctx.function))
                    .transpose()?,
            })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    let requires_safepoint = class != runmat_mir::NativeLoweringClass::NativeOperation
        || !requirements.effects.0.is_empty();
    let frame_state = requires_safepoint.then(|| {
        super::context::frame_state(
            site.point,
            site.source.clone(),
            &[],
            inputs.clone(),
            *side_effect_epoch,
        )
    });
    // The caller installs ordered locals after construction; keeping this
    // helper independent avoids hidden mutation of the SSA environment.
    let effect_epoch_output = (!requirements.effects.0.is_empty()).then(|| ctx.value());
    let embedded_constructs = match &operation {
        NativeOperation::Rvalue { value, .. } => runmat_mir::rvalue_construct_inventory(value)
            .into_iter()
            .skip(1)
            .collect(),
        NativeOperation::Statement(_) => Vec::new(),
    };
    let instruction = NativeInstruction {
        id: ctx.instruction(),
        site: NativeMirSite {
            point: site.point,
            phase: site.phase,
            ordinal: site.ordinal,
            construct,
        },
        source: site.source,
        class,
        effects: requirements.effects,
        capabilities: requirements.capabilities,
        inputs,
        outputs,
        effect_epoch_output,
        embedded_constructs,
        operation,
        safepoint: requires_safepoint.then(|| ctx.safepoint()),
        frame_state,
    };
    if let Some(output) = instruction.effect_epoch_output {
        *side_effect_epoch = output;
    }
    Ok(instruction)
}

pub(super) fn install_frame_locals(instruction: &mut NativeInstruction, locals: &[NativeValueId]) {
    if let Some(state) = &mut instruction.frame_state {
        state.locals = locals
            .iter()
            .enumerate()
            .map(|(local, value)| NativeFrameLocal {
                local: NativeLocalId(local as u32),
                value: *value,
            })
            .collect();
    }
}

pub(super) fn declared_requirements(value: &runmat_mir::MirRvalue) -> OperationRequirements {
    let (effects, capabilities) = runmat_mir::rvalue_declared_requirements(value);
    OperationRequirements {
        effects,
        capabilities,
    }
}

pub(super) fn declared_statement_requirements(
    statement: &runmat_mir::MirStmtKind,
) -> OperationRequirements {
    let effects = runmat_mir::statement_declared_effects(statement);
    OperationRequirements {
        effects,
        capabilities: CapabilitySet::default(),
    }
}

pub(super) fn merge_requirements(
    mut left: OperationRequirements,
    right: OperationRequirements,
) -> OperationRequirements {
    left.effects.0.extend(right.effects.0);
    left.capabilities.0.extend(right.capabilities.0);
    left
}

pub(super) fn root_local(place: &runmat_mir::MirPlace) -> Option<runmat_mir::MirLocalId> {
    match place {
        runmat_mir::MirPlace::Local(local) => Some(*local),
        runmat_mir::MirPlace::Binding(_) => None,
        runmat_mir::MirPlace::Member(base, _)
        | runmat_mir::MirPlace::DynamicMember(base, _)
        | runmat_mir::MirPlace::Index(base, _) => root_local(base),
    }
}

pub(super) fn rvalue_inputs(
    value: &runmat_mir::MirRvalue,
    locals: &[NativeValueId],
) -> NativeCodegenResult<Vec<NativeValueId>> {
    let mut inputs = Vec::new();
    collect_rvalue_inputs(value, locals, &mut inputs)?;
    deduplicate(&mut inputs);
    Ok(inputs)
}

fn statement_inputs(
    statement: &runmat_mir::MirStmtKind,
    locals: &[NativeValueId],
) -> NativeCodegenResult<Vec<NativeValueId>> {
    let mut inputs = Vec::new();
    match statement {
        runmat_mir::MirStmtKind::Assign { place, .. } => {
            collect_place_inputs(place, locals, &mut inputs)?;
        }
        runmat_mir::MirStmtKind::MultiAssign { targets, .. } => {
            for target in &targets.targets {
                if let runmat_mir::MirOutputTarget::Place(place) = target {
                    collect_place_inputs(place, locals, &mut inputs)?;
                }
            }
        }
        runmat_mir::MirStmtKind::PlaceMutation(mutation) => {
            collect_place_inputs(&mutation.place, locals, &mut inputs)?;
        }
        runmat_mir::MirStmtKind::Expr(_)
        | runmat_mir::MirStmtKind::WorkspaceEffect { .. }
        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => {}
    }
    Ok(inputs)
}

fn collect_rvalue_inputs(
    value: &runmat_mir::MirRvalue,
    locals: &[NativeValueId],
    output: &mut Vec<NativeValueId>,
) -> NativeCodegenResult<()> {
    use runmat_mir::MirRvalue as R;
    match value {
        R::Use(operand) | R::Unary(_, operand) | R::Spawn(operand) => {
            collect_operand(operand, locals, output)?;
        }
        R::Binary(left, _, right) => {
            collect_operand(left, locals, output)?;
            collect_operand(right, locals, output)?;
        }
        R::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            collect_operand(left, locals, output)?;
            for statement in right_temps {
                match &statement.kind {
                    runmat_mir::MirStmtKind::Assign { place, value } => {
                        collect_rvalue_inputs(value, locals, output)?;
                        collect_place_inputs(place, locals, output)?;
                    }
                    runmat_mir::MirStmtKind::MultiAssign { targets, value } => {
                        collect_rvalue_inputs(value, locals, output)?;
                        for target in &targets.targets {
                            if let runmat_mir::MirOutputTarget::Place(place) = target {
                                collect_place_inputs(place, locals, output)?;
                            }
                        }
                    }
                    runmat_mir::MirStmtKind::Expr(value) => {
                        collect_rvalue_inputs(value, locals, output)?;
                    }
                    runmat_mir::MirStmtKind::PlaceMutation(mutation) => {
                        collect_place_inputs(&mutation.place, locals, output)?;
                    }
                    _ => {}
                }
            }
            collect_operand(right, locals, output)?;
        }
        R::Range { start, step, end } => {
            collect_operand(start, locals, output)?;
            if let Some(step) = step {
                collect_operand(step, locals, output)?;
            }
            collect_operand(end, locals, output)?;
        }
        R::Call(call) => {
            if let runmat_mir::MirCallee::Dynamic(operand) = &call.callee {
                collect_operand(operand, locals, output)?;
            }
            for argument in &call.args {
                match argument {
                    runmat_mir::MirCallArg::Single(operand) => {
                        collect_operand(operand, locals, output)?;
                    }
                    runmat_mir::MirCallArg::Expansion { base, indices, .. } => {
                        collect_operand(base, locals, output)?;
                        for index in indices {
                            collect_operand(index, locals, output)?;
                        }
                    }
                }
            }
        }
        R::Aggregate { elements, .. } => {
            for operand in elements {
                collect_operand(operand, locals, output)?;
            }
        }
        R::StructLiteral { fields } | R::ObjectLiteral { fields, .. } => {
            for (_, operand) in fields {
                collect_operand(operand, locals, output)?;
            }
        }
        R::Index { base, indexing } => {
            collect_operand(base, locals, output)?;
            collect_indexing_inputs(indexing, locals, output)?;
        }
        R::Member { base, .. } => collect_operand(base, locals, output)?,
        R::DynamicMember { base, member } => {
            collect_operand(base, locals, output)?;
            collect_operand(member, locals, output)?;
        }
        R::Future { args, .. } => {
            for argument in args {
                match argument {
                    runmat_mir::MirCallArg::Single(operand) => {
                        collect_operand(operand, locals, output)?;
                    }
                    runmat_mir::MirCallArg::Expansion { base, indices, .. } => {
                        collect_operand(base, locals, output)?;
                        for index in indices {
                            collect_operand(index, locals, output)?;
                        }
                    }
                }
            }
        }
        R::Distributed(operation) => {
            use runmat_mir::parallel::MirDistributedOp as D;
            if let D::Create { input, .. } = operation {
                collect_operand(input, locals, output)?;
            }
        }
        R::Collective(operation) => {
            use runmat_mir::parallel::MirCollectiveOp as C;
            match operation {
                C::Broadcast { input, .. }
                | C::Gather { input, .. }
                | C::Scatter { input, .. }
                | C::AllGather { input, .. }
                | C::Reduce { input, .. }
                | C::AllReduce { input, .. }
                | C::Send { input, .. } => collect_operand(input, locals, output)?,
                C::Barrier { .. } | C::Receive { .. } => {}
            }
        }
        R::WorkspaceFirstStaticProperty { .. } | R::MetaClass(_) | R::Colon | R::End => {}
    }
    Ok(())
}

fn collect_place_inputs(
    place: &runmat_mir::MirPlace,
    locals: &[NativeValueId],
    output: &mut Vec<NativeValueId>,
) -> NativeCodegenResult<()> {
    match place {
        runmat_mir::MirPlace::Local(local) => push_local(*local, locals, output)?,
        runmat_mir::MirPlace::Binding(_) => {}
        runmat_mir::MirPlace::Member(base, _) => collect_place_inputs(base, locals, output)?,
        runmat_mir::MirPlace::DynamicMember(base, member) => {
            collect_place_inputs(base, locals, output)?;
            collect_operand(member, locals, output)?;
        }
        runmat_mir::MirPlace::Index(base, indexing) => {
            collect_place_inputs(base, locals, output)?;
            collect_indexing_inputs(indexing, locals, output)?;
        }
    }
    Ok(())
}

fn collect_indexing_inputs(
    indexing: &runmat_mir::MirIndexing,
    locals: &[NativeValueId],
    output: &mut Vec<NativeValueId>,
) -> NativeCodegenResult<()> {
    for component in &indexing.components {
        if let runmat_mir::MirIndexComponent::Expr(operand) = component {
            collect_operand(operand, locals, output)?;
        }
    }
    Ok(())
}

fn collect_operand(
    operand: &runmat_mir::MirOperand,
    locals: &[NativeValueId],
    output: &mut Vec<NativeValueId>,
) -> NativeCodegenResult<()> {
    if let runmat_mir::MirOperand::Local(local) = operand {
        push_local(*local, locals, output)?;
    }
    Ok(())
}

fn push_local(
    local: runmat_mir::MirLocalId,
    locals: &[NativeValueId],
    output: &mut Vec<NativeValueId>,
) -> NativeCodegenResult<()> {
    let value = locals.get(local.0).copied().ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.local_identity",
            "MIR operation references a local outside its function",
        )
    })?;
    output.push(value);
    Ok(())
}

fn deduplicate(values: &mut Vec<NativeValueId>) {
    let mut seen = BTreeSet::new();
    values.retain(|value| seen.insert(*value));
}

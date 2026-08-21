use std::collections::BTreeMap;

use crate::ir::*;
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::{BasicBlock, BasicBlockId, MirOperand, MirRvalue, MirTerminatorKind};

pub(super) fn lower_terminator(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    block: &BasicBlock,
    locals: &[NativeValueId],
    mut side_effect_epoch: NativeValueId,
    parameter_sets: &BTreeMap<BasicBlockId, Vec<NativeBlockParameter>>,
) -> NativeCodegenResult<(NativeTerminator, Vec<NativeInstruction>, Vec<NativeMirSite>)> {
    let point = ctx.point(block.id, block.statements.len())?;
    let source = ctx.source_location(block.terminator.span)?;
    let mut instructions = Vec::new();
    let mut sites = Vec::new();
    let mut ordinal = 0;

    let kind = match &block.terminator.kind {
        MirTerminatorKind::Goto(target) => NativeTerminatorKind::Goto {
            edge: edge(
                ctx,
                *target,
                locals,
                side_effect_epoch,
                parameter_sets,
                None,
            )?,
        },
        MirTerminatorKind::Branch {
            cond,
            then_block,
            else_block,
        } => {
            let condition = lower_operand(
                ctx,
                point,
                source.clone(),
                &mut ordinal,
                cond,
                locals,
                &mut side_effect_epoch,
                &mut instructions,
                &mut sites,
            )?;
            NativeTerminatorKind::Branch {
                condition,
                then_edge: edge(
                    ctx,
                    *then_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
                else_edge: edge(
                    ctx,
                    *else_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
            }
        }
        MirTerminatorKind::Switch {
            discr,
            cases,
            otherwise,
        } => {
            let discriminant = lower_operand(
                ctx,
                point,
                source.clone(),
                &mut ordinal,
                discr,
                locals,
                &mut side_effect_epoch,
                &mut instructions,
                &mut sites,
            )?;
            let cases = cases
                .iter()
                .map(|(case, target)| {
                    let value = lower_operand(
                        ctx,
                        point,
                        source.clone(),
                        &mut ordinal,
                        case,
                        locals,
                        &mut side_effect_epoch,
                        &mut instructions,
                        &mut sites,
                    )?;
                    Ok((
                        value,
                        edge(
                            ctx,
                            *target,
                            locals,
                            side_effect_epoch,
                            parameter_sets,
                            None,
                        )?,
                    ))
                })
                .collect::<NativeCodegenResult<Vec<_>>>()?;
            NativeTerminatorKind::Switch {
                discriminant,
                cases,
                otherwise: edge(
                    ctx,
                    *otherwise,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
            }
        }
        MirTerminatorKind::For {
            binding,
            iterable,
            body_block,
            exit_block,
        } => {
            let iterable = lower_rvalue(
                ctx,
                point,
                source.clone(),
                &mut ordinal,
                iterable,
                locals,
                &mut side_effect_epoch,
                &mut instructions,
                &mut sites,
            )?;
            let binding = super::function::checked_local(*binding, ctx.function)?;
            NativeTerminatorKind::For {
                iterable,
                binding,
                body: edge(
                    ctx,
                    *body_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    Some((binding, EdgeOverride::LoopIteration)),
                )?,
                exit: edge(
                    ctx,
                    *exit_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
            }
        }
        MirTerminatorKind::ParFor {
            region,
            binding,
            iterable,
            maximum_workers,
            body_block,
            exit_block,
        } => {
            let iterable = lower_rvalue(
                ctx,
                point,
                source.clone(),
                &mut ordinal,
                iterable,
                locals,
                &mut side_effect_epoch,
                &mut instructions,
                &mut sites,
            )?;
            let maximum_workers = maximum_workers
                .as_deref()
                .map(|value| {
                    lower_rvalue(
                        ctx,
                        point,
                        source.clone(),
                        &mut ordinal,
                        value,
                        locals,
                        &mut side_effect_epoch,
                        &mut instructions,
                        &mut sites,
                    )
                })
                .transpose()?;
            let binding = super::function::checked_local(*binding, ctx.function)?;
            NativeTerminatorKind::ParFor {
                region: *region,
                iterable,
                maximum_workers,
                binding,
                body: edge(
                    ctx,
                    *body_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    Some((binding, EdgeOverride::LoopIteration)),
                )?,
                exit: edge(
                    ctx,
                    *exit_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
            }
        }
        MirTerminatorKind::Spmd {
            region,
            header,
            body_block,
            exit_block,
        } => {
            let values = match header.as_ref() {
                runmat_mir::parallel::MirSpmdHeader::Default => Vec::new(),
                runmat_mir::parallel::MirSpmdHeader::One(a) => vec![a],
                runmat_mir::parallel::MirSpmdHeader::Two(a, b) => vec![a, b],
                runmat_mir::parallel::MirSpmdHeader::Three(a, b, c) => vec![a, b, c],
            };
            let header = values
                .into_iter()
                .map(|value| {
                    lower_rvalue(
                        ctx,
                        point,
                        source.clone(),
                        &mut ordinal,
                        value,
                        locals,
                        &mut side_effect_epoch,
                        &mut instructions,
                        &mut sites,
                    )
                })
                .collect::<NativeCodegenResult<Vec<_>>>()?;
            NativeTerminatorKind::Spmd {
                region: *region,
                header,
                body: edge(
                    ctx,
                    *body_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
                exit: edge(
                    ctx,
                    *exit_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
            }
        }
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            catch_binding,
        } => {
            let catch_binding = catch_binding
                .map(|binding| super::function::checked_local(binding, ctx.function))
                .transpose()?;
            NativeTerminatorKind::TryCatch {
                try_edge: edge(
                    ctx,
                    *try_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    None,
                )?,
                catch_edge: edge(
                    ctx,
                    *catch_block,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    catch_binding.map(|binding| (binding, EdgeOverride::CaughtException)),
                )?,
                catch_binding,
            }
        }
        MirTerminatorKind::Return(values) => NativeTerminatorKind::Return {
            values: values
                .iter()
                .map(|value| {
                    lower_operand(
                        ctx,
                        point,
                        source.clone(),
                        &mut ordinal,
                        value,
                        locals,
                        &mut side_effect_epoch,
                        &mut instructions,
                        &mut sites,
                    )
                })
                .collect::<NativeCodegenResult<Vec<_>>>()?,
        },
        MirTerminatorKind::Await {
            future,
            result,
            resume,
        } => {
            let future = lower_operand(
                ctx,
                point,
                source.clone(),
                &mut ordinal,
                future,
                locals,
                &mut side_effect_epoch,
                &mut instructions,
                &mut sites,
            )?;
            let result_local = result
                .as_ref()
                .and_then(super::operation::root_local)
                .map(|local| super::function::checked_local(local, ctx.function))
                .transpose()?;
            NativeTerminatorKind::Await {
                future,
                result: result.clone(),
                resume: edge(
                    ctx,
                    *resume,
                    locals,
                    side_effect_epoch,
                    parameter_sets,
                    result_local.map(|local| (local, EdgeOverride::AwaitResult)),
                )?,
            }
        }
        MirTerminatorKind::Unreachable => NativeTerminatorKind::Unreachable,
    };

    let construct = runmat_mir::terminator_construct_kind(&block.terminator.kind);
    let class = construct.native_lowering_class();
    let terminator = NativeTerminator {
        site: NativeMirSite {
            point,
            phase: NativeSitePhase::Terminator,
            ordinal: 0,
            construct,
        },
        source: source.clone(),
        class,
        safepoint: (class != runmat_mir::NativeLoweringClass::NativeOperation
            && class != runmat_mir::NativeLoweringClass::ProvenUnreachable)
            .then(|| ctx.safepoint()),
        kind,
        frame_state: super::context::frame_state(
            point,
            source,
            locals,
            Vec::new(),
            side_effect_epoch,
        ),
    };
    Ok((terminator, instructions, sites))
}

#[allow(clippy::too_many_arguments)]
fn lower_rvalue(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    point: runmat_types::ProgramPointId,
    source: NativeSourceLocation,
    ordinal: &mut u32,
    value: &MirRvalue,
    locals: &[NativeValueId],
    side_effect_epoch: &mut NativeValueId,
    instructions: &mut Vec<NativeInstruction>,
    sites: &mut Vec<NativeMirSite>,
) -> NativeCodegenResult<NativeValueId> {
    let current_ordinal = *ordinal;
    *ordinal = ordinal.checked_add(1).ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.terminator_ordinal",
            "terminator operation count exceeds the Native IR schema",
        )
    })?;
    let mut instruction = super::operation::lower_rvalue(
        ctx,
        super::operation::OperationSite {
            point,
            output_point: point,
            phase: NativeSitePhase::TerminatorRvalue,
            ordinal: current_ordinal,
            source,
        },
        value,
        locals,
        &[None],
        NativeRvalueResult::Terminator,
        super::operation::declared_requirements(value),
        side_effect_epoch,
    )?;
    super::operation::install_frame_locals(&mut instruction, locals);
    let output = instruction.outputs[0].value;
    sites.push(instruction.site.clone());
    instructions.push(instruction);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn lower_operand(
    ctx: &mut super::context::FunctionLoweringContext<'_>,
    point: runmat_types::ProgramPointId,
    source: NativeSourceLocation,
    ordinal: &mut u32,
    operand: &MirOperand,
    locals: &[NativeValueId],
    side_effect_epoch: &mut NativeValueId,
    instructions: &mut Vec<NativeInstruction>,
    sites: &mut Vec<NativeMirSite>,
) -> NativeCodegenResult<NativeValueId> {
    if let MirOperand::Local(local) = operand {
        return locals.get(local.0).copied().ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.local_identity",
                "terminator references a local outside its function",
            )
            .at_point(point)
        });
    }
    lower_rvalue(
        ctx,
        point,
        source,
        ordinal,
        &MirRvalue::Use(operand.clone()),
        locals,
        side_effect_epoch,
        instructions,
        sites,
    )
}

#[derive(Clone, Copy)]
enum EdgeOverride {
    LoopIteration,
    CaughtException,
    AwaitResult,
}

fn edge(
    ctx: &super::context::FunctionLoweringContext<'_>,
    target: BasicBlockId,
    locals: &[NativeValueId],
    side_effect_epoch: NativeValueId,
    parameter_sets: &BTreeMap<BasicBlockId, Vec<NativeBlockParameter>>,
    override_local: Option<(NativeLocalId, EdgeOverride)>,
) -> NativeCodegenResult<NativeEdge> {
    let parameters = parameter_sets.get(&target).ok_or_else(|| {
        NativeCodegenError::new(
            "native.lowering.edge_target",
            "MIR terminator references a missing block",
        )
        .at_function(ctx.function)
    })?;
    let arguments = parameters
        .iter()
        .map(|parameter| {
            if let Some((local, kind)) = override_local {
                if parameter.local == local {
                    return Ok(match kind {
                        EdgeOverride::LoopIteration => NativeEdgeArgument::LoopIteration { local },
                        EdgeOverride::CaughtException => {
                            NativeEdgeArgument::CaughtException { local }
                        }
                        EdgeOverride::AwaitResult => NativeEdgeArgument::AwaitResult { local },
                    });
                }
            }
            locals
                .get(parameter.local.0 as usize)
                .copied()
                .map(NativeEdgeArgument::Value)
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.edge_argument",
                        "target block parameter has no source local",
                    )
                    .at_function(ctx.function)
                })
        })
        .collect::<NativeCodegenResult<Vec<_>>>()?;
    Ok(NativeEdge {
        target: super::function::checked_block(target, ctx.function)?,
        arguments,
        side_effect_epoch,
    })
}

use crate::{
    MirAggregateKind, MirCall, MirCallArg, MirCallee, MirConstant, MirExpansionSource,
    MirIndexComponent, MirIndexPlan, MirIndexing, MirOperand, MirPlace, MirRvalue,
    MirShortCircuitOp, MirStmt, MirStmtKind,
};
use runmat_builtins::{BuiltinAsyncBehavior, BuiltinSemantics};
use runmat_hir::{
    CallableIdentity, CommandArgument, ExprId, HirCallableRef, HirCommandCall, HirError, HirExpr,
    HirExprKind, IndexComponent, IndexKind, IndexResultContext, IndexingSemantics, OperatorKind,
    RequestedOutputCount, StringLiteral,
};
use std::collections::HashMap;

use super::MirLoweringContext;

pub(crate) fn lower_expr_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirRvalue, HirError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(MirRvalue::Use(operand.clone()));
    }

    Ok(match &expr.kind {
        HirExprKind::Number(value) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::Number(value.clone())))
        }
        HirExprKind::IntegerLiteral(value) => MirRvalue::Use(MirOperand::Constant(
            MirConstant::IntegerLiteral(value.clone()),
        )),
        HirExprKind::String(value) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::String(value.clone())))
        }
        HirExprKind::Constant(name) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::Symbol(name.clone())))
        }
        HirExprKind::Binding(binding) => {
            MirRvalue::Use(MirOperand::Local(ctx.local_for_binding(*binding)?))
        }
        HirExprKind::Unary(op, inner) => MirRvalue::Unary(
            *op,
            lower_operand_with_replacements(ctx, inner, temps, await_replacements)?,
        ),
        HirExprKind::Binary(left, op, right) => match op {
            OperatorKind::ShortCircuitAnd | OperatorKind::ShortCircuitOr => {
                let left = lower_operand_with_replacements(ctx, left, temps, await_replacements)?;
                let mut right_temps = Vec::new();
                let right = lower_operand_with_replacements(
                    ctx,
                    right,
                    &mut right_temps,
                    await_replacements,
                )?;
                MirRvalue::ShortCircuit {
                    left,
                    op: if matches!(op, OperatorKind::ShortCircuitAnd) {
                        MirShortCircuitOp::And
                    } else {
                        MirShortCircuitOp::Or
                    },
                    right_temps,
                    right,
                }
            }
            _ => MirRvalue::Binary(
                lower_operand_with_replacements(ctx, left, temps, await_replacements)?,
                *op,
                lower_operand_with_replacements(ctx, right, temps, await_replacements)?,
            ),
        },
        HirExprKind::Range(start, step, end) => MirRvalue::Range {
            start: lower_operand_with_replacements(ctx, start, temps, await_replacements)?,
            step: step
                .as_ref()
                .map(|step| lower_operand_with_replacements(ctx, step, temps, await_replacements))
                .transpose()?,
            end: lower_operand_with_replacements(ctx, end, temps, await_replacements)?,
        },
        HirExprKind::Tensor(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Tensor,
            row_lengths: rows.iter().map(Vec::len).collect(),
            elements: lower_aggregate_elements(ctx, rows, temps, await_replacements)?,
        },
        HirExprKind::Cell(rows) => MirRvalue::Aggregate {
            kind: MirAggregateKind::Cell,
            row_lengths: rows.iter().map(Vec::len).collect(),
            elements: lower_aggregate_elements(ctx, rows, temps, await_replacements)?,
        },
        HirExprKind::StructLiteral(fields) => MirRvalue::StructLiteral {
            fields: fields
                .iter()
                .map(|(name, expr)| {
                    Ok((
                        name.clone(),
                        lower_operand_with_replacements(ctx, expr, temps, await_replacements)?,
                    ))
                })
                .collect::<Result<_, HirError>>()?,
        },
        HirExprKind::ObjectLiteral { class_name, fields } => MirRvalue::ObjectLiteral {
            class_name: class_name.clone(),
            fields: fields
                .iter()
                .map(|(name, expr)| {
                    Ok((
                        name.clone(),
                        lower_operand_with_replacements(ctx, expr, temps, await_replacements)?,
                    ))
                })
                .collect::<Result<_, HirError>>()?,
        },
        HirExprKind::Call(call) => {
            let mut arg_spans: Vec<runmat_hir::Span> =
                call.args.iter().map(|arg| arg.span).collect();
            let mut args: Vec<MirCallArg> = call
                .args
                .iter()
                .map(|arg| lower_call_arg(ctx, arg, temps, await_replacements))
                .collect::<Result<_, _>>()?;
            if let Some(value) = lower_parallel_intrinsic(ctx, call, &args)? {
                value
            } else if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                dynamic_call_rvalue(
                    call,
                    lower_operand_with_replacements(ctx, callee, temps, await_replacements)?,
                    args,
                    arg_spans,
                )?
            } else if call.callee.is_feval_builtin_like() {
                if args.is_empty() {
                    return Err(HirError::new("feval: missing function argument"));
                }
                let MirCallArg::Single(callee) = args.remove(0) else {
                    return Err(HirError::new(
                        "feval: function argument cannot be a comma-list expansion",
                    ));
                };
                arg_spans.remove(0);
                dynamic_call_rvalue(call, callee, args, arg_spans)?
            } else if let HirCallableRef::Function(function) = call.callee {
                if ctx.is_async_function(function) {
                    MirRvalue::Future {
                        function,
                        args,
                        syntax: call.syntax.clone(),
                        requested_outputs: call.requested_outputs,
                    }
                } else {
                    call_rvalue(call, args, arg_spans)?
                }
            } else {
                call_rvalue(call, args, arg_spans)?
            }
        }
        HirExprKind::CommandCall(call) => lower_command_call(call)?,
        HirExprKind::Index(base, indexing) => MirRvalue::Index {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            indexing: lower_indexing_with_replacements(ctx, indexing, temps, await_replacements)?,
        },
        HirExprKind::SubscriptChain(chain) => MirRvalue::SubscriptChain(lower_subscript_chain(
            ctx,
            chain,
            temps,
            await_replacements,
        )?),
        HirExprKind::Member {
            base,
            member,
            sequence_use,
        } => MirRvalue::Member {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            member: member.clone(),
            sequence_use: *sequence_use,
        },
        HirExprKind::MemberDynamic {
            base,
            member,
            sequence_use,
        } => MirRvalue::DynamicMember {
            base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
            member: lower_operand_with_replacements(ctx, member, temps, await_replacements)?,
            sequence_use: *sequence_use,
        },
        HirExprKind::WorkspaceFirstStaticProperty {
            workspace_name,
            class_name,
            property,
        } => MirRvalue::WorkspaceFirstStaticProperty {
            workspace_name: workspace_name.clone(),
            class_name: class_name.clone(),
            property: property.clone(),
        },
        HirExprKind::MetaClass(name) => MirRvalue::MetaClass(name.clone()),
        HirExprKind::Colon => MirRvalue::Colon,
        HirExprKind::End => MirRvalue::End,
        HirExprKind::Spawn(inner) => MirRvalue::Spawn(lower_operand_with_replacements(
            ctx,
            inner,
            temps,
            await_replacements,
        )?),
        HirExprKind::FunctionHandle(target) => {
            MirRvalue::Use(MirOperand::FunctionHandle(target.identity()))
        }
        HirExprKind::AnonymousFunction(function) => MirRvalue::Use(MirOperand::FunctionHandle(
            CallableIdentity::AnonymousFunction(*function),
        )),
        HirExprKind::Await(_) => {
            return Err(HirError::new(
                "await expression was not lowered through an await terminator",
            ))
        }
    })
}

fn lower_subscript_chain(
    ctx: &MirLoweringContext,
    chain: &runmat_hir::HirSubscriptChain,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<crate::MirSubscriptChain, HirError> {
    let root = lower_operand_with_replacements(ctx, &chain.root, temps, await_replacements)?;
    let mut steps = Vec::with_capacity(chain.steps.len());
    for step in &chain.steps {
        steps.push(match step {
            runmat_hir::HirSubscriptStep::Index(indexing) => crate::MirSubscriptStep::Index(
                lower_indexing_with_replacements(ctx, indexing, temps, await_replacements)?,
            ),
            runmat_hir::HirSubscriptStep::Member(member) => {
                crate::MirSubscriptStep::Member(member.clone())
            }
            runmat_hir::HirSubscriptStep::DynamicMember(member) => {
                crate::MirSubscriptStep::DynamicMember(lower_operand_with_replacements(
                    ctx,
                    member,
                    temps,
                    await_replacements,
                )?)
            }
            runmat_hir::HirSubscriptStep::DottedInvoke { member, indexing } => {
                crate::MirSubscriptStep::DottedInvoke {
                    member: member.clone(),
                    indexing: lower_indexing_with_replacements(
                        ctx,
                        indexing,
                        temps,
                        await_replacements,
                    )?,
                }
            }
        });
    }
    Ok(crate::MirSubscriptChain {
        root,
        steps,
        sequence_use: chain.sequence_use,
        context: chain.context,
    })
}

fn lower_parallel_intrinsic(
    ctx: &MirLoweringContext,
    call: &runmat_hir::HirCall,
    args: &[MirCallArg],
) -> Result<Option<MirRvalue>, HirError> {
    use crate::parallel::{
        MirCodistributedOverload, MirCollectiveOp as Collective, MirDistributedBuildValidation,
        MirDistributedOp as Distributed, ParallelIntrinsic,
    };

    let Some(intrinsic) = call
        .callee
        .identity()
        .and_then(|identity| ParallelIntrinsic::resolve(&identity))
    else {
        return Ok(None);
    };
    let name = intrinsic.name();
    let operands = || {
        args.iter()
            .map(|argument| match argument {
                MirCallArg::Single(value) => Ok(value.clone()),
                MirCallArg::Expansion(_) => Err(HirError::new(format!(
                    "{name}: comma-list expansion is not valid for a parallel primitive"
                ))),
                MirCallArg::CapturedSequence(_) => Err(HirError::new(format!(
                    "{name}: comma-list expansion is not valid for a parallel primitive"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let arity = |values: &[MirOperand], expected: &[usize]| {
        if expected.contains(&values.len()) {
            Ok(())
        } else {
            Err(HirError::new(format!(
                "{name}: expected {} argument(s), received {}",
                expected
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(" or "),
                values.len()
            )))
        }
    };
    let value = match intrinsic {
        ParallelIntrinsic::Distributed => {
            let values = operands()?;
            arity(&values, &[1])?;
            let (id, owner) = ctx.distributed_identity();
            Some(MirRvalue::Distributed(Distributed::Create {
                id,
                owner,
                input: values[0].clone(),
                scheme: runmat_types::DistributionScheme::Block { dimension: 1 },
            }))
        }
        ParallelIntrinsic::Codistributed => {
            let values = operands()?;
            arity(&values, &[1, 2, 3])?;
            let (id, owner) = ctx.distributed_identity();
            let coordination = ctx
                .in_spmd_region()
                .then(|| ctx.collective_identity())
                .transpose()?;
            let overload = match values.as_slice() {
                [_] => MirCodistributedOverload::ReplicatedInputDefault,
                [_, operand] => MirCodistributedOverload::CodistributorOrDesignatedWorker {
                    operand: operand.clone(),
                },
                [_, worker, codistributor] => {
                    MirCodistributedOverload::DesignatedWorkerWithCodistributor {
                        worker: worker.clone(),
                        codistributor: codistributor.clone(),
                    }
                }
                _ => unreachable!("arity was validated above"),
            };
            Some(MirRvalue::Distributed(Distributed::Codistributed {
                id,
                owner,
                input: values[0].clone(),
                overload,
                coordination,
            }))
        }
        ParallelIntrinsic::CodistributedBuild => {
            if !ctx.in_spmd_region() {
                return Err(HirError::new(
                    "codistributed.build: local-part construction requires an SPMD region",
                ));
            }
            let values = operands()?;
            arity(&values, &[1, 2, 3])?;
            let (id, owner) = ctx.distributed_identity();
            let coordination = ctx.collective_identity()?;
            let (codistributor, validation) = match values.as_slice() {
                [_] => (None, MirDistributedBuildValidation::ValidateAcrossWorkers),
                [_, codistributor] => (
                    Some(codistributor.clone()),
                    MirDistributedBuildValidation::ValidateAcrossWorkers,
                ),
                [_, codistributor, option] => {
                    let validation = match option {
                        MirOperand::Constant(MirConstant::String(value))
                            if value.0.eq_ignore_ascii_case("noCommunication") =>
                        {
                            MirDistributedBuildValidation::NoCommunication
                        }
                        operand => MirDistributedBuildValidation::RuntimeOption(operand.clone()),
                    };
                    (Some(codistributor.clone()), validation)
                }
                _ => unreachable!("arity was validated above"),
            };
            Some(MirRvalue::Distributed(Distributed::Build {
                id,
                owner,
                local_part: values[0].clone(),
                codistributor,
                validation,
                coordination,
            }))
        }
        ParallelIntrinsic::GetLocalPart => {
            let values = operands()?;
            arity(&values, &[1])?;
            Some(MirRvalue::Distributed(Distributed::LocalPart {
                value: values[0].clone(),
            }))
        }
        ParallelIntrinsic::GetCodistributor => {
            let values = operands()?;
            arity(&values, &[1])?;
            Some(MirRvalue::Distributed(Distributed::Codistributor {
                value: values[0].clone(),
            }))
        }
        ParallelIntrinsic::GlobalIndices => {
            let values = operands()?;
            arity(&values, &[2, 3])?;
            let requested_outputs =
                u8::try_from(call.requested_outputs.known_count().ok_or_else(|| {
                    HirError::new("globalIndices requires a statically known output count")
                })?)
                .map_err(|_| HirError::new("globalIndices: output count exceeds u8"))?;
            if !(1..=2).contains(&requested_outputs) {
                return Err(HirError::new("globalIndices: expected one or two outputs"));
            }
            Some(MirRvalue::Distributed(Distributed::GlobalIndices {
                value: values[0].clone(),
                dimension: values[1].clone(),
                lab: values.get(2).cloned(),
                requested_outputs,
            }))
        }
        ParallelIntrinsic::Redistribute => {
            let values = operands()?;
            arity(&values, &[2])?;
            Some(MirRvalue::Distributed(Distributed::Redistribute {
                value: values[0].clone(),
                codistributor: values[1].clone(),
            }))
        }
        _ if !ctx.in_spmd_region() => None,
        ParallelIntrinsic::LabBarrier => {
            let values = operands()?;
            arity(&values, &[0])?;
            Some(MirRvalue::Collective(Collective::Barrier {
                id: ctx.collective_identity()?,
            }))
        }
        ParallelIntrinsic::LabBroadcast => {
            let values = operands()?;
            arity(&values, &[1, 2])?;
            Some(MirRvalue::Collective(Collective::Broadcast {
                id: ctx.collective_identity()?,
                root: values[0].clone(),
                input: values.get(1).cloned(),
            }))
        }
        ParallelIntrinsic::LabSend => {
            let values = operands()?;
            arity(&values, &[2, 3])?;
            Some(MirRvalue::Collective(Collective::Send {
                id: ctx.collective_identity()?,
                input: values[0].clone(),
                destination: values[1].clone(),
                tag: values.get(2).cloned(),
            }))
        }
        ParallelIntrinsic::LabReceive | ParallelIntrinsic::LabProbe => {
            let values = operands()?;
            arity(&values, &[0, 1, 2])?;
            let id = ctx.collective_identity()?;
            let source = values.first().cloned();
            let tag = values.get(1).cloned();
            Some(MirRvalue::Collective(
                if intrinsic == ParallelIntrinsic::LabReceive {
                    let requested_outputs =
                        u8::try_from(call.requested_outputs.known_count().ok_or_else(|| {
                            HirError::new(format!(
                                "{name}: requires a statically known output count"
                            ))
                        })?)
                        .map_err(|_| HirError::new(format!("{name}: output count exceeds u8")))?;
                    if !(1..=3).contains(&requested_outputs) {
                        return Err(HirError::new(format!(
                            "{name}: expected between one and three outputs"
                        )));
                    }
                    Collective::Receive {
                        id,
                        source,
                        tag,
                        requested_outputs,
                    }
                } else {
                    Collective::Probe { id, source, tag }
                },
            ))
        }
        ParallelIntrinsic::LabSendReceive => {
            let values = operands()?;
            arity(&values, &[3, 4])?;
            Some(MirRvalue::Collective(Collective::SendReceive {
                id: ctx.collective_identity()?,
                destination: values[0].clone(),
                source: values[1].clone(),
                input: values[2].clone(),
                tag: values.get(3).cloned(),
            }))
        }
        ParallelIntrinsic::Gplus => {
            let values = operands()?;
            arity(&values, &[1, 2])?;
            let id = ctx.collective_identity()?;
            Some(MirRvalue::Collective(if let Some(root) = values.get(1) {
                Collective::Reduce {
                    id,
                    input: values[0].clone(),
                    root: root.clone(),
                    operator: runmat_types::OperatorKind::Add,
                }
            } else {
                Collective::AllReduce {
                    id,
                    input: values[0].clone(),
                    operator: runmat_types::OperatorKind::Add,
                }
            }))
        }
        ParallelIntrinsic::Gcat => {
            let values = operands()?;
            arity(&values, &[1, 2, 3])?;
            Some(MirRvalue::Collective(Collective::Cat {
                id: ctx.collective_identity()?,
                input: values[0].clone(),
                dimension: values
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| MirOperand::Constant(MirConstant::Number("2".into()))),
                root: values.get(2).cloned(),
            }))
        }
        ParallelIntrinsic::Gop => {
            let values = operands()?;
            arity(&values, &[2, 3])?;
            Some(MirRvalue::Collective(Collective::FunctionalReduce {
                id: ctx.collective_identity()?,
                reducer: values[0].clone(),
                input: values[1].clone(),
                root: values.get(2).cloned(),
            }))
        }
    };
    Ok(value)
}

fn lower_call_arg(
    ctx: &MirLoweringContext,
    arg: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirCallArg, HirError> {
    match &arg.kind {
        HirExprKind::Member {
            base,
            member,
            sequence_use: runmat_types::SequenceUse::ExpandAll,
        } => {
            let source = MirExpansionSource::Member {
                base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
                member: member.clone(),
            };
            return Ok(capture_call_sequence(ctx, source, temps, arg.span));
        }
        HirExprKind::SubscriptChain(chain)
            if chain.sequence_use == runmat_types::SequenceUse::ExpandAll =>
        {
            let source = MirExpansionSource::SubscriptChain(lower_subscript_chain(
                ctx,
                chain,
                temps,
                await_replacements,
            )?);
            return Ok(capture_call_sequence(ctx, source, temps, arg.span));
        }
        HirExprKind::MemberDynamic {
            base,
            member,
            sequence_use: runmat_types::SequenceUse::ExpandAll,
        } => {
            let source = MirExpansionSource::DynamicMember {
                base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
                member: lower_operand_with_replacements(ctx, member, temps, await_replacements)?,
            };
            return Ok(capture_call_sequence(ctx, source, temps, arg.span));
        }
        _ => {}
    }
    if let HirExprKind::Call(call) = &arg.kind {
        let requested_count = requested_output_count_for_arg_expansion(&call.requested_outputs)?;
        if requested_count > 1 {
            let operand = lower_operand_with_replacements(ctx, arg, temps, await_replacements)?;
            return Ok(capture_call_sequence(
                ctx,
                MirExpansionSource::ReturnedOutputs(operand),
                temps,
                arg.span,
            ));
        }
    }

    if matches!(
        &arg.kind,
        HirExprKind::Index(_, indexing)
            if matches!(indexing.result_context, IndexResultContext::FunctionArgumentExpansion)
    ) {
        let HirExprKind::Index(base, indexing) = &arg.kind else {
            unreachable!()
        };
        if indexing.kind != IndexKind::Brace {
            return Err(HirError::new(
                "comma-list expansion requires cell/content indexing",
            ));
        }
        let base = lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
        let indexing = lower_indexing_with_replacements(ctx, indexing, temps, await_replacements)?;
        Ok(capture_call_sequence(
            ctx,
            MirExpansionSource::CellContents { base, indexing },
            temps,
            arg.span,
        ))
    } else {
        let operand = lower_operand_with_replacements(ctx, arg, temps, await_replacements)?;
        Ok(MirCallArg::Single(operand))
    }
}

fn capture_call_sequence(
    ctx: &MirLoweringContext,
    source: MirExpansionSource,
    temps: &mut Vec<MirStmt>,
    span: runmat_hir::Span,
) -> MirCallArg {
    let destination = ctx.fresh_sequence_local();
    temps.push(MirStmt {
        kind: MirStmtKind::CaptureSequence {
            destination,
            source,
        },
        span,
    });
    MirCallArg::CapturedSequence(destination)
}

fn lower_aggregate_elements(
    ctx: &MirLoweringContext,
    rows: &[Vec<HirExpr>],
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<Vec<crate::MirAggregateElement>, HirError> {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|element| {
            let source = match &element.kind {
                HirExprKind::Member {
                    base,
                    member,
                    sequence_use: runmat_types::SequenceUse::ExpandAll,
                } => Some(MirExpansionSource::Member {
                    base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
                    member: member.clone(),
                }),
                HirExprKind::MemberDynamic {
                    base,
                    member,
                    sequence_use: runmat_types::SequenceUse::ExpandAll,
                } => Some(MirExpansionSource::DynamicMember {
                    base: lower_operand_with_replacements(ctx, base, temps, await_replacements)?,
                    member: lower_operand_with_replacements(
                        ctx,
                        member,
                        temps,
                        await_replacements,
                    )?,
                }),
                HirExprKind::Index(base, indexing)
                    if indexing.kind == IndexKind::Brace
                        && matches!(indexing.result_context, IndexResultContext::ReadCommaList) =>
                {
                    let base =
                        lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
                    let indexing =
                        lower_indexing_with_replacements(ctx, indexing, temps, await_replacements)?;
                    Some(MirExpansionSource::CellContents { base, indexing })
                }
                HirExprKind::SubscriptChain(chain)
                    if chain.sequence_use == runmat_types::SequenceUse::ExpandAll =>
                {
                    Some(MirExpansionSource::SubscriptChain(lower_subscript_chain(
                        ctx,
                        chain,
                        temps,
                        await_replacements,
                    )?))
                }
                _ => None,
            };
            if let Some(source) = source {
                let destination = ctx.fresh_sequence_local();
                temps.push(MirStmt {
                    kind: MirStmtKind::CaptureSequence {
                        destination,
                        source,
                    },
                    span: element.span,
                });
                Ok(crate::MirAggregateElement::CapturedSequence(destination))
            } else {
                lower_operand_with_replacements(ctx, element, temps, await_replacements)
                    .map(crate::MirAggregateElement::Single)
            }
        })
        .collect()
}

pub(crate) fn lower_indexing(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
) -> Result<MirIndexing, HirError> {
    lower_indexing_with_replacements(ctx, indexing, temps, &HashMap::new())
}

pub(crate) fn lower_indexing_with_replacements(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexing, HirError> {
    Ok(MirIndexing {
        kind: indexing.kind,
        plan: classify_mir_index_plan(indexing),
        components: indexing
            .components
            .iter()
            .enumerate()
            .map(|(dim, component)| {
                lower_index_component(ctx, dim, component, temps, await_replacements)
            })
            .collect::<Result<_, _>>()?,
        result_context: indexing.result_context,
        cell_expand_all: indexing.kind == IndexKind::Brace
            && indexing
                .components
                .iter()
                .all(|component| matches!(component, IndexComponent::Colon)),
    })
}

fn classify_mir_index_plan(indexing: &IndexingSemantics) -> MirIndexPlan {
    match indexing.kind {
        IndexKind::Brace => MirIndexPlan::Cell,
        IndexKind::Paren => {
            if indexing
                .components
                .iter()
                .all(index_component_is_definitely_scalar)
            {
                MirIndexPlan::Scalar
            } else {
                MirIndexPlan::Slice
            }
        }
    }
}

fn hir_expr_contains_end(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::End => true,
        HirExprKind::Unary(_, inner) => hir_expr_contains_end(inner),
        HirExprKind::Binary(left, _, right) => {
            hir_expr_contains_end(left) || hir_expr_contains_end(right)
        }
        HirExprKind::Range(start, step, end) => {
            hir_expr_contains_end(start)
                || step.as_deref().is_some_and(hir_expr_contains_end)
                || hir_expr_contains_end(end)
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            rows.iter().flatten().any(hir_expr_contains_end)
        }
        HirExprKind::Call(call) => call.args.iter().any(hir_expr_contains_end),
        HirExprKind::Index(base, indexing) => {
            hir_expr_contains_end(base)
                || indexing.components.iter().any(|component| match component {
                    IndexComponent::End { .. } => true,
                    IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => {
                        hir_expr_contains_end(expr)
                    }
                    IndexComponent::Colon => false,
                })
        }
        HirExprKind::SubscriptChain(chain) => {
            hir_expr_contains_end(&chain.root)
                || chain.steps.iter().any(|step| match step {
                    runmat_hir::HirSubscriptStep::Index(indexing)
                    | runmat_hir::HirSubscriptStep::DottedInvoke { indexing, .. } => {
                        indexing.components.iter().any(|component| match component {
                            IndexComponent::End { .. } => true,
                            IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => {
                                hir_expr_contains_end(expr)
                            }
                            IndexComponent::Colon => false,
                        })
                    }
                    runmat_hir::HirSubscriptStep::DynamicMember(member) => {
                        hir_expr_contains_end(member)
                    }
                    runmat_hir::HirSubscriptStep::Member(_) => false,
                })
        }
        HirExprKind::Member { base, .. } => hir_expr_contains_end(base),
        HirExprKind::MemberDynamic { base, member, .. } => {
            hir_expr_contains_end(base) || hir_expr_contains_end(member)
        }
        HirExprKind::Spawn(inner) | HirExprKind::Await(inner) => hir_expr_contains_end(inner),
        _ => false,
    }
}

fn index_component_is_definitely_scalar(component: &IndexComponent) -> bool {
    matches!(component, IndexComponent::Expr(expr) if hir_expr_is_definitely_scalar_index(expr))
}

fn hir_expr_is_definitely_scalar_index(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::Number(_) | HirExprKind::IntegerLiteral(_) => true,
        HirExprKind::Constant(name)
            if name.0.eq_ignore_ascii_case("true") || name.0.eq_ignore_ascii_case("false") =>
        {
            true
        }
        _ => false,
    }
}

fn lower_index_component(
    ctx: &MirLoweringContext,
    _dim: usize,
    component: &IndexComponent,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirIndexComponent, HirError> {
    Ok(match component {
        IndexComponent::Colon => MirIndexComponent::Colon,
        IndexComponent::End { offset, .. } => lower_contextual_end(ctx, *offset),
        IndexComponent::Expr(expr) if hir_expr_contains_end(expr) => {
            let mut statements = Vec::new();
            let value =
                lower_operand_with_replacements(ctx, expr, &mut statements, await_replacements)?;
            MirIndexComponent::ContextualExpr(crate::MirExpressionRegion::from_lowered(
                statements, value,
            )?)
        }
        IndexComponent::Expr(expr) => match expr.kind {
            HirExprKind::Colon => MirIndexComponent::Colon,
            HirExprKind::End => lower_contextual_end(ctx, 0),
            _ => MirIndexComponent::Expr(lower_operand_with_replacements(
                ctx,
                expr,
                temps,
                await_replacements,
            )?),
        },
        IndexComponent::Logical(expr) if hir_expr_contains_end(expr) => {
            let mut statements = Vec::new();
            let value =
                lower_operand_with_replacements(ctx, expr, &mut statements, await_replacements)?;
            MirIndexComponent::ContextualExpr(crate::MirExpressionRegion::from_lowered(
                statements, value,
            )?)
        }
        IndexComponent::Logical(expr) => MirIndexComponent::Expr(lower_operand_with_replacements(
            ctx,
            expr,
            temps,
            await_replacements,
        )?),
    })
}

fn lower_contextual_end(ctx: &MirLoweringContext, offset: isize) -> MirIndexComponent {
    let span = runmat_hir::Span::default();
    let end_local = ctx.fresh_temp(span);
    let mut statements = vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(end_local),
            value: MirRvalue::End,
        },
        span,
    }];
    let result = if offset == 0 {
        MirOperand::Local(end_local)
    } else {
        let result = ctx.fresh_temp(span);
        let operator = if offset.is_negative() {
            runmat_hir::OperatorKind::Subtract
        } else {
            runmat_hir::OperatorKind::Add
        };
        statements.push(MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(result),
                value: MirRvalue::Binary(
                    MirOperand::Local(end_local),
                    operator,
                    MirOperand::Constant(MirConstant::Number(offset.unsigned_abs().to_string())),
                ),
            },
            span,
        });
        MirOperand::Local(result)
    };
    MirIndexComponent::ContextualExpr(
        crate::MirExpressionRegion::from_lowered(statements, result)
            .expect("canonical end expression region is valid"),
    )
}

fn lower_command_call(call: &HirCommandCall) -> Result<MirRvalue, HirError> {
    let Some(identity) = call.command.identity() else {
        return Err(HirError::new(
            "command call requires a statically classified callee identity",
        ));
    };
    let callee = MirCallee::Static(identity);
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &runmat_hir::CallSyntax::Command);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args: call
            .args
            .iter()
            .map(|arg| MirCallArg::Single(MirOperand::Constant(command_arg_constant(arg))))
            .collect(),
        arg_spans: Vec::new(),
        syntax: runmat_hir::CallSyntax::Command,
        requested_outputs: RequestedOutputCount::Zero,
        fallback_policy,
        workspace_first_name: None,
        bare_identifier: false,
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn call_rvalue(
    call: &runmat_hir::HirCall,
    args: Vec<MirCallArg>,
    arg_spans: Vec<runmat_hir::Span>,
) -> Result<MirRvalue, HirError> {
    let callee = match &call.callee {
        HirCallableRef::SuperConstructor {
            current_class,
            super_class,
        } => MirCallee::SuperConstructor {
            current_class: runmat_types::ClassIdentity::new(current_class.0.clone())
                .map_err(|error| HirError::new(error.to_string()))?,
            super_class: runmat_types::ClassIdentity::from_qualified_name(super_class)
                .map_err(|error| HirError::new(error.to_string()))?,
        },
        HirCallableRef::SuperMethod {
            current_class,
            super_class,
            method,
        } => MirCallee::SuperMethod {
            current_class: runmat_types::ClassIdentity::new(current_class.0.clone())
                .map_err(|error| HirError::new(error.to_string()))?,
            super_class: runmat_types::ClassIdentity::from_qualified_name(super_class)
                .map_err(|error| HirError::new(error.to_string()))?,
            method: method.0.clone().into(),
        },
        _ => {
            let Some(identity) = call.callee.identity() else {
                return Err(HirError::new(
                    "call requires either a static callable identity or a dynamic callee expression",
                ));
            };
            MirCallee::Static(identity)
        }
    };
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &call.syntax);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args,
        arg_spans,
        syntax: call.syntax.clone(),
        requested_outputs: call.requested_outputs,
        fallback_policy,
        workspace_first_name: call.workspace_first_name.clone(),
        bare_identifier: call.bare_identifier,
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn dynamic_call_rvalue(
    call: &runmat_hir::HirCall,
    callee: MirOperand,
    args: Vec<MirCallArg>,
    arg_spans: Vec<runmat_hir::Span>,
) -> Result<MirRvalue, HirError> {
    let callee = MirCallee::Dynamic(callee);
    let semantics = call_semantics(&callee);
    let fallback_policy = call_fallback_policy(&callee, &call.syntax);
    Ok(MirRvalue::Call(MirCall {
        callee,
        args,
        arg_spans,
        syntax: call.syntax.clone(),
        requested_outputs: call.requested_outputs,
        fallback_policy,
        workspace_first_name: call.workspace_first_name.clone(),
        bare_identifier: call.bare_identifier,
        async_behavior: map_async_behavior(semantics.async_behavior),
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }))
}

fn requested_output_count_for_arg_expansion(
    requested: &RequestedOutputCount,
) -> Result<usize, HirError> {
    requested.executor_carrier_count().ok_or_else(|| {
        HirError::new("destination output cardinality must be resolved by assignment lowering")
    })
}

fn call_semantics(callee: &MirCallee) -> BuiltinSemantics {
    match callee {
        MirCallee::Static(CallableIdentity::Builtin(id)) => {
            runmat_builtins::builtin_catalog_entry_by_name(&id.0)
                .map(|entry| entry.legacy_semantics())
                .or_else(|| {
                    runmat_builtins::builtin_function_by_name(&id.0)
                        .map(|builtin| builtin.semantics())
                })
                .or_else(|| runmat_builtins::builtin_semantics_for_name(&id.0))
                .unwrap_or_else(BuiltinSemantics::unknown)
        }
        MirCallee::Static(
            CallableIdentity::ExternalName(_)
            | CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::Method(_)
            | CallableIdentity::AnonymousFunction(_),
        )
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. }
        | MirCallee::Dynamic(_) => BuiltinSemantics::unknown(),
        MirCallee::Static(
            CallableIdentity::BoundFunction(_) | CallableIdentity::ExternalFunction { .. },
        ) => BuiltinSemantics {
            compatibility: runmat_builtins::BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            effects: runmat_builtins::BuiltinEffects::none(),
            workspace_effect: None,
            environment_effect: None,
            purity: runmat_builtins::BuiltinPurity::Impure,
            semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
        },
    }
}

fn call_fallback_policy(
    callee: &MirCallee,
    syntax: &runmat_hir::CallSyntax,
) -> runmat_hir::CallableFallbackPolicy {
    if matches!(
        syntax,
        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
    ) && !matches!(
        callee,
        MirCallee::Static(
            runmat_hir::CallableIdentity::BoundFunction(_)
                | runmat_hir::CallableIdentity::ExternalFunction { .. },
        ) | MirCallee::SuperMethod { .. }
    ) {
        return runmat_hir::CallableFallbackPolicy::ObjectDispatch;
    }
    match callee {
        MirCallee::Static(
            runmat_hir::CallableIdentity::BoundFunction(_)
            | runmat_hir::CallableIdentity::ExternalFunction { .. },
        )
        | MirCallee::Static(runmat_hir::CallableIdentity::Builtin(_))
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. } => runmat_hir::CallableFallbackPolicy::None,
        MirCallee::Static(runmat_hir::CallableIdentity::ExternalName(_)) => {
            runmat_hir::CallableFallbackPolicy::ExternalBoundary
        }
        MirCallee::Static(
            runmat_hir::CallableIdentity::DynamicName(_)
            | runmat_hir::CallableIdentity::Imported(_)
            | runmat_hir::CallableIdentity::Method(_)
            | runmat_hir::CallableIdentity::AnonymousFunction(_),
        )
        | MirCallee::Dynamic(_) => runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
    }
}

fn map_async_behavior(behavior: BuiltinAsyncBehavior) -> crate::AsyncBehaviorFact {
    match behavior {
        BuiltinAsyncBehavior::NeverSuspends => crate::AsyncBehaviorFact::NeverSuspends,
        BuiltinAsyncBehavior::MaySuspend => crate::AsyncBehaviorFact::MaySuspend,
        BuiltinAsyncBehavior::RequiresAsyncRuntime => {
            crate::AsyncBehaviorFact::RequiresAsyncRuntime
        }
    }
}

fn command_arg_constant(arg: &CommandArgument) -> MirConstant {
    match arg {
        CommandArgument::Word(word) => MirConstant::String(StringLiteral(word.0.clone())),
        CommandArgument::StringLiteral(value) => {
            MirConstant::String(StringLiteral(value.0.trim_matches('"').to_string()))
        }
    }
}

pub(crate) fn lower_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
) -> Result<MirOperand, HirError> {
    lower_operand_with_replacements(ctx, expr, temps, &HashMap::new())
}

pub(crate) fn lower_operand_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &HashMap<ExprId, MirOperand>,
) -> Result<MirOperand, HirError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(operand.clone());
    }
    if let Some(operand) = lower_simple_operand(ctx, expr)? {
        return Ok(operand);
    }

    let value = lower_expr_with_replacements(ctx, expr, temps, await_replacements)?;
    let local = ctx.fresh_temp(expr.span);
    temps.push(MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(local),
            value,
        },
        span: expr.span,
    });
    Ok(MirOperand::Local(local))
}

pub(crate) fn lower_simple_operand(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
) -> Result<Option<MirOperand>, HirError> {
    Ok(Some(match &expr.kind {
        HirExprKind::Number(value) => MirOperand::Constant(MirConstant::Number(value.clone())),
        HirExprKind::IntegerLiteral(value) => {
            MirOperand::Constant(MirConstant::IntegerLiteral(value.clone()))
        }
        HirExprKind::String(value) => MirOperand::Constant(MirConstant::String(value.clone())),
        HirExprKind::Constant(name) => MirOperand::Constant(MirConstant::Symbol(name.clone())),
        HirExprKind::Binding(binding) => MirOperand::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::FunctionHandle(target) => MirOperand::FunctionHandle(target.identity()),
        _ => return Ok(None),
    }))
}

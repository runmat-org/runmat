use crate::{
    MirAggregateKind, MirCall, MirCallArg, MirCallee, MirConstant, MirExpansionSource,
    MirIndexComponent, MirIndexPlan, MirIndexing, MirOperand, MirPlace, MirRvalue,
    MirShortCircuitOp, MirStmt, MirStmtKind,
};
use runmat_builtins::{BuiltinAsyncBehavior, BuiltinSemantics};
use runmat_hir::{
    CallableIdentity, CommandArgument, HirCallableRef, HirCommandCall, HirError, HirExpr,
    HirExprKind, IndexComponent, IndexKind, IndexResultContext, IndexingSemantics, OperatorKind,
    RequestedOutputCount, StringLiteral,
};

use super::MirLoweringContext;

#[path = "expr/ordering.rs"]
mod ordering;
use ordering::{append_after_prior, capture_binding_before_later_preparation, is_bound_operand};

pub(crate) fn lower_expr_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<MirRvalue, HirError> {
    if await_replacements.sequence(expr.id).is_some() {
        return Err(HirError::new(
            "captured argument sequence cannot be scalarized across await",
        ));
    }
    if let Some(operand) = await_replacements.get(&expr.id) {
        return Ok(MirRvalue::Use(operand.clone()));
    }

    Ok(match &expr.kind {
        HirExprKind::Number(value) => {
            MirRvalue::Use(MirOperand::Constant(MirConstant::Number(value.clone())))
        }
        HirExprKind::ImaginaryNumber(value) => MirRvalue::Use(MirOperand::Constant(
            MirConstant::ImaginaryNumber(value.clone()),
        )),
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
            _ => {
                let mut left_operand =
                    lower_operand_with_replacements(ctx, left, temps, await_replacements)?;
                let mut preparation = Vec::new();
                let right = lower_operand_with_replacements(
                    ctx,
                    right,
                    &mut preparation,
                    await_replacements,
                )?;
                append_after_prior(ctx, [(&mut left_operand, left.span)], preparation, temps);
                MirRvalue::Binary(left_operand, *op, right)
            }
        },
        HirExprKind::Range(start, step, end) => {
            let mut start_operand =
                lower_operand_with_replacements(ctx, start, temps, await_replacements)?;
            let mut step_operand = None;
            if let Some(step) = step {
                let mut preparation = Vec::new();
                let value = lower_operand_with_replacements(
                    ctx,
                    step,
                    &mut preparation,
                    await_replacements,
                )?;
                append_after_prior(ctx, [(&mut start_operand, start.span)], preparation, temps);
                step_operand = Some(value);
            }
            let mut preparation = Vec::new();
            let end_operand =
                lower_operand_with_replacements(ctx, end, &mut preparation, await_replacements)?;
            let preceding = std::iter::once((&mut start_operand, start.span)).chain(
                step_operand
                    .iter_mut()
                    .zip(step.iter())
                    .map(|(operand, source)| (operand, source.span)),
            );
            append_after_prior(ctx, preceding, preparation, temps);
            MirRvalue::Range {
                start: start_operand,
                step: step_operand,
                end: end_operand,
            }
        }
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
            fields: lower_ordered_fields(ctx, fields, temps, await_replacements)?,
        },
        HirExprKind::ObjectLiteral { class_name, fields } => MirRvalue::ObjectLiteral {
            class_name: class_name.clone(),
            fields: lower_ordered_fields(ctx, fields, temps, await_replacements)?,
        },
        HirExprKind::Call(call) => {
            let mut arg_spans: Vec<runmat_hir::Span> =
                call.args.iter().map(|arg| arg.span).collect();
            // A dynamic callee is evaluated before its arguments. In particular,
            // argument preparation may mutate the binding holding that callee.
            let mut dynamic_callee = match &call.callee {
                HirCallableRef::DynamicExpr(callee) => Some(lower_operand_with_replacements(
                    ctx,
                    callee,
                    temps,
                    await_replacements,
                )?),
                _ => None,
            };
            let mut args: Vec<MirCallArg> = Vec::with_capacity(call.args.len());
            let mut pending_bindings = Vec::new();
            let mut pending_callee = dynamic_callee
                .as_ref()
                .is_some_and(|callee| is_bound_operand(ctx, callee));
            for arg in &call.args {
                let mut preparation = Vec::new();
                let lowered = lower_call_arg(ctx, arg, &mut preparation, await_replacements)?;
                if !preparation.is_empty() {
                    if pending_callee {
                        let (Some(callee), HirCallableRef::DynamicExpr(source)) =
                            (dynamic_callee.as_mut(), &call.callee)
                        else {
                            unreachable!("pending callee is a dynamic binding");
                        };
                        capture_binding_before_later_preparation(ctx, callee, source.span, temps);
                        pending_callee = false;
                    }
                    for (index, span) in pending_bindings.drain(..) {
                        let MirCallArg::Single(operand) = &mut args[index] else {
                            unreachable!("only single arguments hold pending bindings");
                        };
                        capture_binding_before_later_preparation(ctx, operand, span, temps);
                    }
                    temps.extend(preparation);
                }
                if let MirCallArg::Single(operand) = &lowered {
                    if is_bound_operand(ctx, operand) {
                        pending_bindings.push((args.len(), arg.span));
                    }
                }
                args.push(lowered);
            }
            if let Some(value) = lower_parallel_intrinsic(ctx, call, &args)? {
                value
            } else if let Some(callee) = dynamic_callee {
                dynamic_call_rvalue(call, callee, args, arg_spans)?
            } else if call.callee.is_feval_builtin_like() && call.workspace_first_name.is_none() {
                if args.is_empty() {
                    return Err(HirError::new("feval: missing function argument"));
                }
                if matches!(args.first(), Some(MirCallArg::Single(_))) {
                    let MirCallArg::Single(callee) = args.remove(0) else {
                        unreachable!()
                    };
                    arg_spans.remove(0);
                    dynamic_call_rvalue(call, callee, args, arg_spans)?
                } else {
                    // A captured comma list may supply the callable and later
                    // arguments. Its width is only known after expansion, so
                    // retain the whole argument sequence for the registered
                    // callable dispatcher to split at the first value.
                    call_rvalue(call, args, arg_spans)?
                }
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
            } else if let HirCallableRef::ExternalFunction {
                function,
                display_name,
            } = &call.callee
            {
                if ctx.is_external_async_function(*function) {
                    MirRvalue::ExternalFuture {
                        function: *function,
                        display_name: display_name.clone(),
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
        HirExprKind::Index(base, indexing) => {
            let mut base_operand =
                lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
            let mut preparation = Vec::new();
            let indexing = lower_indexing_with_replacements(
                ctx,
                indexing,
                &mut preparation,
                await_replacements,
            )?;
            append_after_prior(ctx, [(&mut base_operand, base.span)], preparation, temps);
            MirRvalue::Index {
                base: base_operand,
                indexing,
            }
        }
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
        } => {
            let mut base_operand =
                lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
            let mut preparation = Vec::new();
            let member_operand =
                lower_operand_with_replacements(ctx, member, &mut preparation, await_replacements)?;
            append_after_prior(ctx, [(&mut base_operand, base.span)], preparation, temps);
            MirRvalue::DynamicMember {
                base: base_operand,
                member: member_operand,
                sequence_use: *sequence_use,
            }
        }
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
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<crate::MirSubscriptChain, HirError> {
    let root = lower_operand_with_replacements(ctx, &chain.root, temps, await_replacements)?;
    let mut steps = Vec::with_capacity(chain.steps.len());
    for step in &chain.steps {
        steps.push(match step {
            runmat_hir::HirSubscriptStep::Index(indexing) => crate::MirSubscriptStep::Index(
                lower_chain_indexing(ctx, indexing, await_replacements)?,
            ),
            runmat_hir::HirSubscriptStep::Member(member) => {
                crate::MirSubscriptStep::Member(member.clone())
            }
            runmat_hir::HirSubscriptStep::DynamicMember(member) => {
                let mut preparation = Vec::new();
                let operand = lower_operand_with_replacements(
                    ctx,
                    member,
                    &mut preparation,
                    await_replacements,
                )?;
                if preparation.is_empty() {
                    crate::MirSubscriptStep::DynamicMember(operand)
                } else {
                    crate::MirSubscriptStep::PreparedDynamicMember(
                        crate::MirPreparedExpressionRegion::from_lowered(preparation, operand)?,
                    )
                }
            }
            runmat_hir::HirSubscriptStep::DottedInvoke {
                member,
                arguments,
                semantic_fallback,
            } => crate::MirSubscriptStep::DottedInvoke {
                member: member.clone(),
                semantic_fallback: semantic_fallback
                    .as_ref()
                    .map(|identity| std::sync::Arc::new(identity.clone())),
                arguments: arguments
                    .iter()
                    .map(|argument| match argument {
                        runmat_hir::HirDottedInvokeArgument::Colon => {
                            Ok(crate::MirDottedInvokeArgument::Colon)
                        }
                        runmat_hir::HirDottedInvokeArgument::End { offset } => {
                            let MirIndexComponent::ContextualExpr(region) =
                                lower_contextual_end(ctx, *offset)
                            else {
                                unreachable!("contextual end always lowers to a region")
                            };
                            Ok(crate::MirDottedInvokeArgument::Contextual(region))
                        }
                        runmat_hir::HirDottedInvokeArgument::Value(value) => {
                            let mut preparation = Vec::new();
                            let argument =
                                lower_call_arg(ctx, value, &mut preparation, await_replacements)?;
                            Ok(crate::MirDottedInvokeArgument::Call(
                                crate::MirCallArgumentRegion {
                                    steps: crate::MirExpressionRegion::steps_from_lowered(
                                        preparation,
                                    )?,
                                    argument,
                                    span: value.span,
                                },
                            ))
                        }
                    })
                    .collect::<Result<_, HirError>>()?,
            },
        });
    }
    Ok(crate::MirSubscriptChain {
        root,
        steps,
        sequence_use: chain.sequence_use,
        context: chain.context,
    })
}

fn lower_chain_indexing(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<MirIndexing, HirError> {
    let mut components = Vec::with_capacity(indexing.components.len());
    for (dim, source) in indexing.components.iter().enumerate() {
        let mut preparation = Vec::new();
        let component =
            lower_index_component(ctx, dim, source, &mut preparation, await_replacements)?;
        if !preparation.is_empty()
            || matches!(&component, MirIndexComponent::Expansion(region) if !region.steps.is_empty())
        {
            // Earlier selectors must be read before this selector's preparation,
            // even though the enclosing path is dispatched only after all
            // selector values have been assembled.
            for (position, earlier) in components.iter_mut().enumerate() {
                if let MirIndexComponent::Expr(operand) = earlier {
                    let mut snapshot = Vec::new();
                    let span = match &indexing.components[position] {
                        IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => expr.span,
                        IndexComponent::Colon | IndexComponent::End { .. } => {
                            unreachable!("plain selector comes from an expression")
                        }
                    };
                    capture_binding_before_later_preparation(ctx, operand, span, &mut snapshot);
                    if !snapshot.is_empty() {
                        *earlier = MirIndexComponent::PreparedExpr(
                            crate::MirPreparedExpressionRegion::from_lowered(
                                snapshot,
                                operand.clone(),
                            )?,
                        );
                    }
                }
            }
        }
        components.push(match component {
            MirIndexComponent::Expr(operand) if !preparation.is_empty() => {
                MirIndexComponent::PreparedExpr(crate::MirPreparedExpressionRegion::from_lowered(
                    preparation,
                    operand,
                )?)
            }
            other => other,
        });
    }
    Ok(MirIndexing {
        argument_spans: index_argument_spans(indexing),
        kind: indexing.kind,
        plan: classify_mir_index_plan(indexing),
        components,
        result_context: indexing.result_context,
        cell_expand_all: indexing.kind == IndexKind::Brace
            && indexing
                .components
                .iter()
                .all(|component| matches!(component, IndexComponent::Colon)),
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

pub(super) fn lower_call_arg(
    ctx: &MirLoweringContext,
    arg: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<MirCallArg, HirError> {
    if let Some(sequence) = await_replacements.sequence(arg.id) {
        return Ok(MirCallArg::CapturedSequence(sequence));
    }
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
            let mut base_operand =
                lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
            let mut preparation = Vec::new();
            let member_operand =
                lower_operand_with_replacements(ctx, member, &mut preparation, await_replacements)?;
            append_after_prior(ctx, [(&mut base_operand, base.span)], preparation, temps);
            let source = MirExpansionSource::DynamicMember {
                base: base_operand,
                member: member_operand,
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
        let mut base_operand =
            lower_operand_with_replacements(ctx, base, temps, await_replacements)?;
        let mut preparation = Vec::new();
        let indexing =
            lower_indexing_with_replacements(ctx, indexing, &mut preparation, await_replacements)?;
        append_after_prior(ctx, [(&mut base_operand, base.span)], preparation, temps);
        Ok(capture_call_sequence(
            ctx,
            MirExpansionSource::CellContents {
                base: base_operand,
                indexing,
            },
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

fn lower_ordered_fields(
    ctx: &MirLoweringContext,
    fields: &[(runmat_types::MemberName, HirExpr)],
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<Vec<(runmat_types::MemberName, MirOperand)>, HirError> {
    let mut lowered: Vec<(runmat_types::MemberName, MirOperand)> = Vec::with_capacity(fields.len());
    let mut pending_bindings: Vec<(usize, runmat_hir::Span)> = Vec::new();
    for (name, source) in fields {
        let mut preparation = Vec::new();
        let operand =
            lower_operand_with_replacements(ctx, source, &mut preparation, await_replacements)?;
        if !preparation.is_empty() {
            for (index, span) in pending_bindings.drain(..) {
                capture_binding_before_later_preparation(ctx, &mut lowered[index].1, span, temps);
            }
            temps.extend(preparation);
        }
        if is_bound_operand(ctx, &operand) {
            pending_bindings.push((lowered.len(), source.span));
        }
        lowered.push((name.clone(), operand));
    }
    Ok(lowered)
}

fn lower_aggregate_elements(
    ctx: &MirLoweringContext,
    rows: &[Vec<HirExpr>],
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<Vec<crate::MirAggregateElement>, HirError> {
    let mut lowered: Vec<(crate::MirAggregateElement, runmat_hir::Span)> = Vec::new();
    let mut pending_bindings: Vec<(usize, runmat_hir::Span)> = Vec::new();
    for element in rows.iter().flatten() {
        let mut preparation = Vec::new();
        let source = match &element.kind {
            HirExprKind::Member {
                base,
                member,
                sequence_use: runmat_types::SequenceUse::ExpandAll,
            } => Some(MirExpansionSource::Member {
                base: lower_operand_with_replacements(
                    ctx,
                    base,
                    &mut preparation,
                    await_replacements,
                )?,
                member: member.clone(),
            }),
            HirExprKind::MemberDynamic {
                base,
                member,
                sequence_use: runmat_types::SequenceUse::ExpandAll,
            } => {
                let mut base_operand = lower_operand_with_replacements(
                    ctx,
                    base,
                    &mut preparation,
                    await_replacements,
                )?;
                let mut member_preparation = Vec::new();
                let member_operand = lower_operand_with_replacements(
                    ctx,
                    member,
                    &mut member_preparation,
                    await_replacements,
                )?;
                append_after_prior(
                    ctx,
                    [(&mut base_operand, base.span)],
                    member_preparation,
                    &mut preparation,
                );
                Some(MirExpansionSource::DynamicMember {
                    base: base_operand,
                    member: member_operand,
                })
            }
            HirExprKind::Index(base, indexing)
                if indexing.kind == IndexKind::Brace
                    && matches!(indexing.result_context, IndexResultContext::ReadCommaList) =>
            {
                let mut base_operand = lower_operand_with_replacements(
                    ctx,
                    base,
                    &mut preparation,
                    await_replacements,
                )?;
                let mut indexing_preparation = Vec::new();
                let indexing = lower_indexing_with_replacements(
                    ctx,
                    indexing,
                    &mut indexing_preparation,
                    await_replacements,
                )?;
                append_after_prior(
                    ctx,
                    [(&mut base_operand, base.span)],
                    indexing_preparation,
                    &mut preparation,
                );
                Some(MirExpansionSource::CellContents {
                    base: base_operand,
                    indexing,
                })
            }
            HirExprKind::SubscriptChain(chain)
                if chain.sequence_use == runmat_types::SequenceUse::ExpandAll =>
            {
                Some(MirExpansionSource::SubscriptChain(lower_subscript_chain(
                    ctx,
                    chain,
                    &mut preparation,
                    await_replacements,
                )?))
            }
            _ => None,
        };
        let value = if let Some(source) = source {
            let destination = ctx.fresh_sequence_local();
            preparation.push(MirStmt {
                kind: MirStmtKind::CaptureSequence {
                    destination,
                    source,
                },
                span: element.span,
            });
            crate::MirAggregateElement::CapturedSequence(destination)
        } else {
            crate::MirAggregateElement::Single(lower_operand_with_replacements(
                ctx,
                element,
                &mut preparation,
                await_replacements,
            )?)
        };
        if !preparation.is_empty() {
            for (index, span) in pending_bindings.drain(..) {
                let crate::MirAggregateElement::Single(operand) = &mut lowered[index].0 else {
                    unreachable!("only single elements hold pending bindings");
                };
                capture_binding_before_later_preparation(ctx, operand, span, temps);
            }
            temps.extend(preparation);
        }
        if let crate::MirAggregateElement::Single(operand) = &value {
            if is_bound_operand(ctx, operand) {
                pending_bindings.push((lowered.len(), element.span));
            }
        }
        lowered.push((value, element.span));
    }
    Ok(lowered.into_iter().map(|(value, _)| value).collect())
}

pub(crate) fn lower_indexing(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
) -> Result<MirIndexing, HirError> {
    lower_indexing_with_replacements(
        ctx,
        indexing,
        temps,
        &crate::lowering::AwaitReplacements::default(),
    )
}

pub(crate) fn lower_indexing_with_replacements(
    ctx: &MirLoweringContext,
    indexing: &IndexingSemantics,
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<MirIndexing, HirError> {
    if matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget | IndexResultContext::DeletionTarget
    ) && indexing.components.iter().any(|component| {
        matches!(component,
            IndexComponent::Expr(expression) if expression.expands_call_argument_sequence())
    }) {
        // Keep each captured sequence and its preparation in its source slot.
        // Lifting later selector work into statement temporaries would execute
        // it before an earlier private expansion region has been captured.
        return lower_chain_indexing(ctx, indexing, await_replacements);
    }
    let mut components: Vec<MirIndexComponent> = Vec::with_capacity(indexing.components.len());
    let mut pending_bindings = Vec::new();
    for (dim, component) in indexing.components.iter().enumerate() {
        let mut preparation = Vec::new();
        let lowered =
            lower_index_component(ctx, dim, component, &mut preparation, await_replacements)?;
        if !preparation.is_empty() {
            for (index, span) in pending_bindings.drain(..) {
                let MirIndexComponent::Expr(operand) = &mut components[index] else {
                    unreachable!("only value selectors hold pending bindings");
                };
                capture_binding_before_later_preparation(ctx, operand, span, temps);
            }
            temps.extend(preparation);
        }
        if let MirIndexComponent::Expr(operand) = &lowered {
            if is_bound_operand(ctx, operand) {
                let span = match component {
                    IndexComponent::Expr(source) | IndexComponent::Logical(source) => source.span,
                    IndexComponent::Colon | IndexComponent::End { .. } => {
                        unreachable!("value selector comes from an expression")
                    }
                };
                pending_bindings.push((components.len(), span));
            }
        }
        components.push(lowered);
    }
    Ok(MirIndexing {
        argument_spans: index_argument_spans(indexing),
        kind: indexing.kind,
        plan: classify_mir_index_plan(indexing),
        components,
        result_context: indexing.result_context,
        cell_expand_all: indexing.kind == IndexKind::Brace
            && indexing
                .components
                .iter()
                .all(|component| matches!(component, IndexComponent::Colon)),
    })
}

pub(super) fn index_argument_spans(
    indexing: &IndexingSemantics,
) -> Option<Vec<runmat_types::Span>> {
    indexing
        .components
        .iter()
        .map(|component| match component {
            IndexComponent::Expr(expression) | IndexComponent::Logical(expression) => {
                Some(expression.span)
            }
            IndexComponent::Colon | IndexComponent::End { .. } => None,
        })
        .collect()
}

pub(super) fn classify_mir_index_plan(indexing: &IndexingSemantics) -> MirIndexPlan {
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
                    runmat_hir::HirSubscriptStep::Index(indexing) => {
                        indexing.components.iter().any(|component| match component {
                            IndexComponent::End { .. } => true,
                            IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => {
                                hir_expr_contains_end(expr)
                            }
                            IndexComponent::Colon => false,
                        })
                    }
                    runmat_hir::HirSubscriptStep::DottedInvoke { arguments, .. } => {
                        arguments.iter().any(|argument| match argument {
                            runmat_hir::HirDottedInvokeArgument::End { .. } => true,
                            runmat_hir::HirDottedInvokeArgument::Value(value) => {
                                hir_expr_contains_end(value)
                            }
                            runmat_hir::HirDottedInvokeArgument::Colon => false,
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
    await_replacements: &crate::lowering::AwaitReplacements,
) -> Result<MirIndexComponent, HirError> {
    Ok(match component {
        IndexComponent::Colon => MirIndexComponent::Colon,
        IndexComponent::End { offset, .. } => lower_contextual_end(ctx, *offset),
        IndexComponent::Expr(expr) if expr.expands_call_argument_sequence() => {
            let mut preparation = Vec::new();
            let argument = lower_call_arg(ctx, expr, &mut preparation, await_replacements)?;
            MirIndexComponent::Expansion(crate::MirCallArgumentRegion {
                steps: crate::MirExpressionRegion::steps_from_lowered(preparation)?,
                argument,
                span: expr.span,
            })
        }
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
    lower_contextual_end_at(ctx, offset, runmat_hir::Span::default())
}

fn lower_contextual_end_at(
    ctx: &MirLoweringContext,
    offset: isize,
    span: runmat_hir::Span,
) -> MirIndexComponent {
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

pub(super) fn contextual_end_argument_region(
    ctx: &MirLoweringContext,
    offset: isize,
    span: runmat_hir::Span,
) -> crate::MirCallArgumentRegion {
    let MirIndexComponent::ContextualExpr(region) = lower_contextual_end_at(ctx, offset, span)
    else {
        unreachable!("canonical end lowering is contextual");
    };
    crate::MirCallArgumentRegion {
        steps: region.steps().to_vec(),
        argument: MirCallArg::Single(region.result().clone()),
        span,
    }
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
            current_class: current_class.clone(),
            super_class: runmat_types::ClassIdentity::from_qualified_name(super_class)
                .map_err(|error| HirError::new(error.to_string()))?,
        },
        HirCallableRef::SuperMethod {
            current_class,
            super_class,
            method,
        } => MirCallee::SuperMethod {
            current_class: current_class.clone(),
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
            | CallableIdentity::AnonymousFunction(_)
            | CallableIdentity::ExternalFunction { .. },
        )
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. }
        | MirCallee::Dynamic(_) => BuiltinSemantics::unknown(),
        MirCallee::Static(CallableIdentity::BoundFunction(_)) => BuiltinSemantics {
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
        BuiltinAsyncBehavior::MaySuspend | BuiltinAsyncBehavior::MaySuspendLocally => {
            crate::AsyncBehaviorFact::MaySuspend
        }
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
    lower_operand_with_replacements(
        ctx,
        expr,
        temps,
        &crate::lowering::AwaitReplacements::default(),
    )
}

pub(crate) fn lower_operand_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &crate::lowering::AwaitReplacements,
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
        HirExprKind::ImaginaryNumber(value) => {
            MirOperand::Constant(MirConstant::ImaginaryNumber(value.clone()))
        }
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

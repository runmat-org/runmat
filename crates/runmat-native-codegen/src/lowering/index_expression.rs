use std::collections::{BTreeMap, BTreeSet};

use runmat_mir::{MirCall, MirCallArg, MirCallee, MirConstant, MirLocalId, MirOperand, MirRvalue};
use runmat_runtime::indexing::EndExpr;
use runmat_types::{OperatorKind, ProgramFunctionId};

use crate::{
    NativeCodegenResult, NativeIndexBound, NativeIndexExpression, NativeIndexExpressionKind,
    NativeRangeExpression,
};

pub(super) fn derive(
    body: &runmat_mir::MirBody,
    function: ProgramFunctionId,
) -> NativeCodegenResult<Vec<NativeIndexExpression>> {
    let mut assignments = BTreeMap::<MirLocalId, Vec<&MirRvalue>>::new();
    for statement in body.blocks.iter().flat_map(|block| block.statements.iter()) {
        if let runmat_mir::MirStmtKind::Assign {
            place: runmat_mir::MirPlace::Local(local),
            value,
        } = &statement.kind
        {
            assignments.entry(*local).or_default().push(value);
        }
    }

    let mut expressions = Vec::new();
    for local in &body.locals {
        let Some(value) = single_assignment(local.id, &assignments) else {
            continue;
        };
        let kind = if let Some(range) = range_expression(value, &assignments) {
            Some(NativeIndexExpressionKind::Range(Box::new(range)))
        } else {
            let mut visited = BTreeSet::new();
            rvalue_end_expression(value, &assignments, &mut visited).and_then(
                |(expression, has_end)| {
                    has_end.then_some(NativeIndexExpressionKind::Scalar(expression))
                },
            )
        };
        if let Some(kind) = kind {
            expressions.push(NativeIndexExpression {
                local: super::function::checked_local(local.id, function)?,
                kind,
            });
        }
    }
    expressions.sort_by_key(|expression| expression.local);
    Ok(expressions)
}

fn range_expression(
    value: &MirRvalue,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
) -> Option<NativeRangeExpression> {
    let MirRvalue::Range { start, step, end } = value else {
        return None;
    };
    let mut visited = BTreeSet::new();
    let (start, start_has_end) = index_bound(start, assignments, &mut visited)?;
    let (step, step_has_end) = match step {
        Some(step) => {
            let (bound, has_end) = index_bound(step, assignments, &mut visited)?;
            (Some(bound), has_end)
        }
        None => (None, false),
    };
    let (end, end_has_end) = range_bound_expression(end, assignments, &mut visited)?;
    (start_has_end || step_has_end || end_has_end).then_some(NativeRangeExpression {
        start,
        step,
        end,
    })
}

fn index_bound(
    operand: &MirOperand,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
    visited: &mut BTreeSet<MirLocalId>,
) -> Option<(NativeIndexBound, bool)> {
    Some(
        match operand_end_expression(operand, assignments, visited) {
            Some((expression, true)) => (NativeIndexBound::Expression(expression), true),
            Some((_, false)) | None => (NativeIndexBound::Operand(operand.clone()), false),
        },
    )
}

fn range_bound_expression(
    operand: &MirOperand,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
    visited: &mut BTreeSet<MirLocalId>,
) -> Option<(EndExpr, bool)> {
    match operand {
        MirOperand::Constant(MirConstant::Number(value)) => value
            .parse()
            .ok()
            .map(|value| (EndExpr::Const(value), false)),
        MirOperand::Local(local) => {
            if !visited.insert(*local) {
                return None;
            }
            let derived = single_assignment(*local, assignments)
                .and_then(|value| rvalue_end_expression(value, assignments, visited));
            visited.remove(local);
            match derived {
                Some((expression, true)) => Some((expression, true)),
                Some((_, false)) | None => Some((EndExpr::Var(local.0), false)),
            }
        }
        MirOperand::Constant(_) | MirOperand::FunctionHandle(_) => None,
    }
}

fn operand_end_expression(
    operand: &MirOperand,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
    visited: &mut BTreeSet<MirLocalId>,
) -> Option<(EndExpr, bool)> {
    match operand {
        MirOperand::Local(local) => {
            if !visited.insert(*local) {
                return None;
            }
            let result = single_assignment(*local, assignments)
                .and_then(|value| rvalue_end_expression(value, assignments, visited));
            visited.remove(local);
            result
        }
        MirOperand::Constant(MirConstant::Number(value)) => value
            .parse()
            .ok()
            .map(|value| (EndExpr::Const(value), false)),
        MirOperand::Constant(_) | MirOperand::FunctionHandle(_) => None,
    }
}

fn rvalue_end_expression(
    value: &MirRvalue,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
    visited: &mut BTreeSet<MirLocalId>,
) -> Option<(EndExpr, bool)> {
    match value {
        MirRvalue::End => Some((EndExpr::End, true)),
        MirRvalue::Use(operand) => operand_end_expression(operand, assignments, visited),
        MirRvalue::Unary(operator, operand) => {
            let (operand, has_end) = operand_end_expression(operand, assignments, visited)?;
            let expression = match operator {
                OperatorKind::UnaryPlus => EndExpr::Pos(Box::new(operand)),
                OperatorKind::UnaryMinus => EndExpr::Neg(Box::new(operand)),
                _ => return None,
            };
            Some((expression, has_end))
        }
        MirRvalue::Binary(left, operator, right) => {
            let (left, left_has_end) = operand_end_expression(left, assignments, visited)?;
            let (right, right_has_end) = operand_end_expression(right, assignments, visited)?;
            let expression = match operator {
                OperatorKind::Add => EndExpr::Add(Box::new(left), Box::new(right)),
                OperatorKind::Subtract => EndExpr::Sub(Box::new(left), Box::new(right)),
                OperatorKind::MatrixMultiply | OperatorKind::ElementwiseMultiply => {
                    EndExpr::Mul(Box::new(left), Box::new(right))
                }
                OperatorKind::Mrdivide | OperatorKind::ElementwiseDivide => {
                    EndExpr::Div(Box::new(left), Box::new(right))
                }
                OperatorKind::Mldivide | OperatorKind::ElementwiseLeftDivide => {
                    EndExpr::LeftDiv(Box::new(left), Box::new(right))
                }
                OperatorKind::MatrixPower | OperatorKind::ElementwisePower => {
                    EndExpr::Pow(Box::new(left), Box::new(right))
                }
                _ => return None,
            };
            Some((expression, left_has_end || right_has_end))
        }
        MirRvalue::Call(call) => call_end_expression(call, assignments, visited),
        _ => None,
    }
}

fn call_end_expression(
    call: &MirCall,
    assignments: &BTreeMap<MirLocalId, Vec<&MirRvalue>>,
    visited: &mut BTreeSet<MirLocalId>,
) -> Option<(EndExpr, bool)> {
    let identity = match &call.callee {
        MirCallee::Static(identity) => identity.clone(),
        MirCallee::Dynamic(_)
        | MirCallee::SuperConstructor { .. }
        | MirCallee::SuperMethod { .. } => {
            return None;
        }
    };
    let mut args = Vec::with_capacity(call.args.len());
    let mut has_end = false;
    for argument in &call.args {
        let MirCallArg::Single(operand) = argument else {
            return None;
        };
        let (argument, argument_has_end) = operand_end_expression(operand, assignments, visited)?;
        args.push(argument);
        has_end |= argument_has_end;
    }
    Some((
        EndExpr::ResolvedCall {
            identity,
            fallback_policy: call.fallback_policy,
            args,
        },
        has_end,
    ))
}

fn single_assignment<'a>(
    local: MirLocalId,
    assignments: &'a BTreeMap<MirLocalId, Vec<&'a MirRvalue>>,
) -> Option<&'a MirRvalue> {
    let assignments = assignments.get(&local)?;
    (assignments.len() == 1).then_some(assignments[0])
}

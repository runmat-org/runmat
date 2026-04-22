//! Lowering support for `end` expressions used in indexing.

use crate::instr::EndExpr;
use runmat_hir::{HirExpr, HirExprKind};

pub(crate) fn expr_is_one(expr: &HirExpr) -> bool {
    parse_number(expr)
        .map(|v| (v - 1.0).abs() < 1e-9)
        .unwrap_or(false)
}

fn parse_number(expr: &HirExpr) -> Option<f64> {
    if let HirExprKind::Number(raw) = &expr.kind {
        raw.parse::<f64>().ok()
    } else {
        None
    }
}

pub(crate) fn end_numeric_expr(expr: &HirExpr) -> Option<EndExpr> {
    end_numeric_expr_from_expr(&expr.kind)
}

pub(crate) fn numeric_expr_any(expr: &HirExpr) -> Option<EndExpr> {
    end_numeric_expr_internal(&expr.kind).map(|(e, _)| e)
}

pub(crate) fn range_dynamic_end_spec(
    expr: &HirExpr,
) -> Option<(Option<EndExpr>, Option<EndExpr>, EndExpr)> {
    let HirExprKind::Range(start, step, end) = &expr.kind else {
        return None;
    };
    let start_end = end_numeric_expr(start);
    let step_end = step.as_ref().and_then(|s| end_numeric_expr(s));
    let end_end = end_numeric_expr(end);
    if start_end.is_none() && step_end.is_none() && end_end.is_none() {
        return None;
    }
    let resolved_end = if let Some(e) = end_end {
        e
    } else {
        numeric_expr_any(end)?
    };
    Some((start_end, step_end, resolved_end))
}

fn end_numeric_expr_from_expr(kind: &HirExprKind) -> Option<EndExpr> {
    let (expr, has_end) = end_numeric_expr_internal(kind)?;
    if has_end {
        Some(expr)
    } else {
        None
    }
}

fn end_numeric_expr_internal(kind: &HirExprKind) -> Option<(EndExpr, bool)> {
    match kind {
        HirExprKind::End => Some((EndExpr::End, true)),
        HirExprKind::Number(s) => s.parse::<f64>().ok().map(|v| (EndExpr::Const(v), false)),
        HirExprKind::Var(id) => Some((EndExpr::Var(id.0), false)),
        HirExprKind::Unary(op, inner) => {
            let (child, has_end) = end_numeric_expr_internal(&inner.kind)?;
            match op {
                runmat_parser::UnOp::Plus => Some((EndExpr::Pos(Box::new(child)), has_end)),
                runmat_parser::UnOp::Minus => Some((EndExpr::Neg(Box::new(child)), has_end)),
                _ => None,
            }
        }
        HirExprKind::FuncCall(name, args) => {
            let mut out_args = Vec::with_capacity(args.len());
            let mut has_end = false;
            for arg in args {
                let (e, h) = end_numeric_expr_internal(&arg.kind)?;
                out_args.push(e);
                has_end |= h;
            }
            let lname = name.to_ascii_lowercase();
            if args.len() == 1 {
                let single = out_args.into_iter().next().unwrap_or(EndExpr::Const(0.0));
                let out = match lname.as_str() {
                    "floor" => EndExpr::Floor(Box::new(single)),
                    "ceil" | "ceiling" => EndExpr::Ceil(Box::new(single)),
                    "round" => EndExpr::Round(Box::new(single)),
                    "fix" => EndExpr::Fix(Box::new(single)),
                    _ => EndExpr::Call(name.clone(), vec![single]),
                };
                Some((out, has_end))
            } else {
                Some((EndExpr::Call(name.clone(), out_args), has_end))
            }
        }
        HirExprKind::Binary(left, op, right) => {
            let (lhs, left_has_end) = end_numeric_expr_internal(&left.kind)?;
            let (rhs, right_has_end) = end_numeric_expr_internal(&right.kind)?;
            let has_end = left_has_end || right_has_end;
            match op {
                runmat_parser::BinOp::Add => {
                    Some((EndExpr::Add(Box::new(lhs), Box::new(rhs)), has_end))
                }
                runmat_parser::BinOp::Sub => {
                    Some((EndExpr::Sub(Box::new(lhs), Box::new(rhs)), has_end))
                }
                runmat_parser::BinOp::Mul | runmat_parser::BinOp::ElemMul => {
                    Some((EndExpr::Mul(Box::new(lhs), Box::new(rhs)), has_end))
                }
                runmat_parser::BinOp::RightDiv | runmat_parser::BinOp::ElemDiv => {
                    Some((EndExpr::Div(Box::new(lhs), Box::new(rhs)), has_end))
                }
                runmat_parser::BinOp::LeftDiv | runmat_parser::BinOp::ElemLeftDiv => {
                    Some((EndExpr::LeftDiv(Box::new(lhs), Box::new(rhs)), has_end))
                }
                runmat_parser::BinOp::Pow | runmat_parser::BinOp::ElemPow => {
                    Some((EndExpr::Pow(Box::new(lhs), Box::new(rhs)), has_end))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

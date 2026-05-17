use super::shared::{eval_const_num, logical_binary_result, resolve_context_from_args};
use crate::hir::{CompatibilityHirExpr as HirExpr, CompatibilityHirExprKind as HirExprKind};
use crate::{Type, VarId};
use runmat_parser as parser;
use std::collections::HashMap;

pub(crate) fn infer_expr_type_with_env(
    expr: &HirExpr,
    env: &HashMap<VarId, Type>,
    func_returns: &HashMap<String, Vec<Type>>,
) -> Type {
    fn unify_tensor(a: &Type, b: &Type) -> Type {
        match (a, b) {
            (Type::Tensor { shape: sa }, Type::Tensor { shape: sb }) => match (sa, sb) {
                (Some(sa), Some(sb)) => {
                    let maxr = sa.len().max(sb.len());
                    let mut out: Vec<Option<usize>> = Vec::with_capacity(maxr);
                    for i in 0..maxr {
                        let da = sa.get(i).cloned().unwrap_or(None);
                        let db = sb.get(i).cloned().unwrap_or(None);
                        let d = match (da, db) {
                            (Some(a), Some(b)) => {
                                if a == b {
                                    Some(a)
                                } else if a == 1 {
                                    Some(b)
                                } else if b == 1 {
                                    Some(a)
                                } else {
                                    None
                                }
                            }
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                        out.push(d);
                    }
                    Type::Tensor { shape: Some(out) }
                }
                _ => Type::tensor(),
            },
            (Type::Tensor { .. }, _) | (_, Type::Tensor { .. }) => Type::tensor(),
            _ => Type::tensor(),
        }
    }

    use HirExprKind as K;

    match &expr.kind {
        K::Number(_) => Type::Num,
        K::String(_) => Type::String,
        K::Constant(_) => Type::Num,
        K::Var(id) => env.get(id).cloned().unwrap_or(Type::Unknown),
        K::Unary(_, e) => infer_expr_type_with_env(e, env, func_returns),
        K::Binary(a, op, b) => {
            let ta = infer_expr_type_with_env(a, env, func_returns);
            let tb = infer_expr_type_with_env(b, env, func_returns);
            match op {
                parser::BinOp::Mul => runmat_builtins::shape_rules::matmul_output_type(&ta, &tb),
                parser::BinOp::LeftDiv => {
                    runmat_builtins::shape_rules::left_divide_output_type(&ta, &tb)
                }
                parser::BinOp::RightDiv => {
                    runmat_builtins::shape_rules::right_divide_output_type(&ta, &tb)
                }
                parser::BinOp::Add
                | parser::BinOp::Sub
                | parser::BinOp::Pow
                | parser::BinOp::ElemMul
                | parser::BinOp::ElemDiv
                | parser::BinOp::ElemPow
                | parser::BinOp::ElemLeftDiv => {
                    if matches!(ta, Type::Tensor { .. }) || matches!(tb, Type::Tensor { .. }) {
                        unify_tensor(&ta, &tb)
                    } else {
                        Type::Num
                    }
                }
                parser::BinOp::Equal
                | parser::BinOp::NotEqual
                | parser::BinOp::Less
                | parser::BinOp::LessEqual
                | parser::BinOp::Greater
                | parser::BinOp::GreaterEqual => logical_binary_result(&ta, &tb),
                parser::BinOp::AndAnd
                | parser::BinOp::OrOr
                | parser::BinOp::BitAnd
                | parser::BinOp::BitOr => logical_binary_result(&ta, &tb),
                parser::BinOp::Colon => runmat_builtins::shape_rules::infer_range_shape(
                    eval_const_num(a),
                    None,
                    eval_const_num(b),
                )
                .map(|shape| Type::Tensor { shape: Some(shape) })
                .unwrap_or_else(Type::tensor),
            }
        }
        K::Tensor(rows) => {
            let mut row_types: Vec<Vec<Type>> = Vec::new();
            for row in rows {
                let mut types = Vec::new();
                for e in row {
                    types.push(infer_expr_type_with_env(e, env, func_returns));
                }
                row_types.push(types);
            }
            if let Some(shape) = runmat_builtins::shape_rules::concat_shape(&row_types) {
                return Type::Tensor { shape: Some(shape) };
            }
            let r = rows.len();
            let c = rows.iter().map(|row| row.len()).max().unwrap_or(0);
            if r > 0 && rows.iter().all(|row| row.len() == c) {
                Type::tensor_with_shape(vec![r, c])
            } else {
                Type::tensor()
            }
        }
        K::Cell(rows) => {
            let mut elem_ty: Option<Type> = None;
            let mut len: usize = 0;
            for row in rows {
                for e in row {
                    let t = infer_expr_type_with_env(e, env, func_returns);
                    elem_ty = Some(match elem_ty {
                        Some(curr) => curr.unify(&t),
                        None => t,
                    });
                    len += 1;
                }
            }
            Type::Cell {
                element_type: elem_ty.map(Box::new),
                length: Some(len),
            }
        }
        K::Index(base, idxs) => {
            let bt = infer_expr_type_with_env(base, env, func_returns);
            let idx_types: Vec<Type> = idxs
                .iter()
                .map(|e| infer_expr_type_with_env(e, env, func_returns))
                .collect();
            runmat_builtins::shape_rules::index_output_type(&bt, &idx_types)
        }
        K::IndexCell(base, idxs) => {
            let bt = infer_expr_type_with_env(base, env, func_returns);
            if let Type::Cell {
                element_type: Some(t),
                ..
            } = bt
            {
                let scalar = idxs.len() == 1
                    && matches!(
                        infer_expr_type_with_env(&idxs[0], env, func_returns),
                        Type::Int | Type::Num | Type::Bool | Type::Tensor { .. }
                    );
                if scalar {
                    *t
                } else {
                    Type::Unknown
                }
            } else {
                Type::Unknown
            }
        }
        K::Range(start, step, end) => runmat_builtins::shape_rules::infer_range_shape(
            eval_const_num(start),
            step.as_ref().and_then(|s| eval_const_num(s)),
            eval_const_num(end),
        )
        .map(|shape| Type::Tensor { shape: Some(shape) })
        .unwrap_or_else(Type::tensor),
        K::FuncCall(name, args) => {
            if let Some(v) = func_returns.get(name) {
                v.first().cloned().unwrap_or(Type::Unknown)
            } else {
                let arg_types: Vec<Type> = args
                    .iter()
                    .map(|arg| infer_expr_type_with_env(arg, env, func_returns))
                    .collect();
                let ctx = resolve_context_from_args(args);
                let builtins = runmat_builtins::builtin_functions();
                if let Some(b) = builtins.iter().find(|b| b.name == *name) {
                    b.infer_return_type_with_context(&arg_types, &ctx)
                } else {
                    Type::Unknown
                }
            }
        }
        K::MethodCall(base, method, args) | K::DottedInvoke(base, method, args) => {
            let base_ty = infer_expr_type_with_env(base, env, func_returns);
            let mut arg_types = Vec::with_capacity(args.len() + 1);
            arg_types.push(base_ty);
            arg_types.extend(
                args.iter()
                    .map(|arg| infer_expr_type_with_env(arg, env, func_returns)),
            );
            let ctx = resolve_context_from_args(args);
            let builtins = runmat_builtins::builtin_functions();
            let suffix = format!(".{method}");
            let candidates = builtins
                .iter()
                .filter(|b| b.name.ends_with(&suffix))
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                Type::Unknown
            } else {
                let mut ty = candidates[0].infer_return_type_with_context(&arg_types, &ctx);
                for candidate in candidates.iter().skip(1) {
                    let next = candidate.infer_return_type_with_context(&arg_types, &ctx);
                    ty = ty.unify(&next);
                }
                ty
            }
        }
        K::Member(base, _) => {
            let _bt = infer_expr_type_with_env(base, env, func_returns);
            Type::Unknown
        }
        K::MemberDynamic(_, _) => Type::Unknown,
        K::AnonFunc { .. } => Type::Function {
            params: vec![Type::Unknown],
            returns: Box::new(Type::Unknown),
        },
        K::FuncHandle(_) => Type::Function {
            params: vec![Type::Unknown],
            returns: Box::new(Type::Unknown),
        },
        K::MetaClass(_) => Type::String,
        K::End => Type::Unknown,
        K::Colon => Type::tensor(),
    }
}

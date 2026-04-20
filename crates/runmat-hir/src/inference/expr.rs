use super::shared::{
    eval_const_num, infer_dataset_type_from_literal_path, infer_slice_result_shape,
    literal_path_arg, logical_binary_result, resolve_context_from_args, shape_rank,
};
use crate::{HirExpr, HirExprKind, Type, VarId};
use runmat_parser as parser;
use std::collections::HashMap;

pub fn infer_expr_type_with_env(
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
            if name == "data.open" {
                if let Some(path_expr) = args.first() {
                    if let Some(path) = literal_path_arg(path_expr) {
                        if let Some(dataset_ty) = infer_dataset_type_from_literal_path(&path) {
                            return dataset_ty;
                        }
                    }
                }
                return Type::DataDataset { arrays: None };
            }
            if name == "data.create" || name == "data.import" {
                return Type::DataDataset { arrays: None };
            }
            if name == "Dataset.array" {
                if let Some(base) = args.first() {
                    let base_ty = infer_expr_type_with_env(base, env, func_returns);
                    if let Type::DataDataset {
                        arrays: Some(arrays),
                    } = base_ty
                    {
                        if let Some(name_expr) = args.get(1) {
                            if let Some(name) = literal_path_arg(name_expr) {
                                if let Some(info) = arrays.get(&name) {
                                    return Type::DataArray {
                                        dtype: info.dtype.clone(),
                                        shape: info.shape.clone(),
                                        chunk_shape: info.chunk_shape.clone(),
                                        codec: info.codec.clone(),
                                    };
                                }
                            }
                        }
                    }
                }
                return Type::DataArray {
                    dtype: None,
                    shape: None,
                    chunk_shape: None,
                    codec: None,
                };
            }
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
            if let Type::DataDataset { arrays } = &base_ty {
                match method.as_str() {
                    "array" => {
                        if let Some(name_expr) = args.first() {
                            if let Some(name) = literal_path_arg(name_expr) {
                                if let Some(info) = arrays.as_ref().and_then(|m| m.get(&name)) {
                                    return Type::DataArray {
                                        dtype: info.dtype.clone(),
                                        shape: info.shape.clone(),
                                        chunk_shape: info.chunk_shape.clone(),
                                        codec: info.codec.clone(),
                                    };
                                }
                            }
                        }
                        return Type::DataArray {
                            dtype: None,
                            shape: None,
                            chunk_shape: None,
                            codec: None,
                        };
                    }
                    "begin" => return Type::DataTransaction,
                    "arrays" => return Type::cell_of(Type::String),
                    "has_array" => return Type::Bool,
                    "id" | "path" | "version" | "snapshot" => return Type::String,
                    "attrs" => return Type::Struct { known_fields: None },
                    "get_attr" => return Type::Unknown,
                    "set_attr" | "set_attrs" => return Type::Bool,
                    "refresh" => {
                        return Type::DataDataset {
                            arrays: arrays.clone(),
                        };
                    }
                    _ => {}
                }
            }

            if let Type::DataArray {
                dtype,
                shape,
                chunk_shape,
                codec,
            } = &base_ty
            {
                match method.as_str() {
                    "name" | "dtype" | "codec" => return Type::String,
                    "rank" => return Type::Int,
                    "shape" => {
                        let rank = shape_rank(shape);
                        return Type::Tensor {
                            shape: Some(vec![Some(1), rank]),
                        };
                    }
                    "chunk_shape" => {
                        let rank = shape_rank(chunk_shape);
                        return Type::Tensor {
                            shape: Some(vec![Some(1), rank]),
                        };
                    }
                    "read" => {
                        let out_shape = if let Some(slice_expr) = args.first() {
                            infer_slice_result_shape(shape, slice_expr)
                        } else {
                            shape.clone()
                        };
                        return Type::Tensor { shape: out_shape };
                    }
                    "write" | "resize" | "fill" => return Type::Bool,
                    _ => {
                        let _ = (dtype, codec);
                    }
                }
            }

            if let Type::DataTransaction = base_ty {
                match method.as_str() {
                    "id" | "status" => return Type::String,
                    "write" | "resize" | "fill" | "set_attr" | "set_attrs" | "delete_array"
                    | "create_array" | "commit" | "abort" => return Type::Bool,
                    _ => {}
                }
            }

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

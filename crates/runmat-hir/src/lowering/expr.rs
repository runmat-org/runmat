use super::ctx::Ctx;
use crate::error::error_namespace;
use crate::inference::shared::resolve_context_from_args;
use crate::{HirExpr, HirExprKind, HirLValue, HirStmt, SemanticError, Type, VarId};
use runmat_parser::{self as parser, BinOp, Expr as AstExpr};
use std::collections::HashMap;

impl Ctx {
    pub(crate) fn lower_colon_expr(
        &mut self,
        start: &AstExpr,
        end: &AstExpr,
    ) -> Result<(HirExprKind, Type), SemanticError> {
        use parser::Expr;

        let start_hir = self.lower_expr(start)?;

        match end {
            Expr::Binary(mid, parser::BinOp::Colon, final_end, _) => {
                let mid_hir = self.lower_expr(mid)?;
                let end_hir = self.lower_expr(final_end)?;
                Ok((
                    HirExprKind::Range(
                        Box::new(start_hir),
                        Some(Box::new(mid_hir)),
                        Box::new(end_hir),
                    ),
                    Type::tensor(),
                ))
            }
            Expr::Range(mid, step, final_end, _) if step.is_none() => {
                let mid_hir = self.lower_expr(mid)?;
                let end_hir = self.lower_expr(final_end)?;
                Ok((
                    HirExprKind::Range(
                        Box::new(start_hir),
                        Some(Box::new(mid_hir)),
                        Box::new(end_hir),
                    ),
                    Type::tensor(),
                ))
            }
            _ => {
                let end_hir = self.lower_expr(end)?;
                Ok((
                    HirExprKind::Range(Box::new(start_hir), None, Box::new(end_hir)),
                    Type::tensor(),
                ))
            }
        }
    }

    pub(crate) fn lower_expr(&mut self, expr: &AstExpr) -> Result<HirExpr, SemanticError> {
        use parser::Expr::*;
        let span = expr.span();
        let (kind, ty) = match expr {
            Number(n, _) => (HirExprKind::Number(n.clone()), Type::Num),
            String(s, _) => (HirExprKind::String(s.clone()), Type::String),
            Ident(name, _) => {
                if let Some(id) = self.lookup(name) {
                    let ty = if id.0 < self.var_types.len() {
                        self.var_types[id.0].clone()
                    } else {
                        Type::Unknown
                    };
                    (HirExprKind::Var(id), ty)
                } else if self.is_constant(name) {
                    (HirExprKind::Constant(name.clone()), Type::Num)
                } else if self.is_function(name) {
                    let return_type = self.infer_function_return_type(name, &[]);
                    (HirExprKind::FuncCall(name.clone(), vec![]), return_type)
                } else if self.allow_unqualified_statics {
                    (HirExprKind::Constant(name.clone()), Type::Unknown)
                } else {
                    let ident = format!("{}:UndefinedVariable", error_namespace());
                    return Err(SemanticError::new(format!("Undefined variable: {name}"))
                        .with_identifier(ident)
                        .with_span(span));
                }
            }
            Unary(op, e, _) => {
                let inner = self.lower_expr(e)?;
                let ty = inner.ty.clone();
                (HirExprKind::Unary(*op, Box::new(inner)), ty)
            }
            Binary(a, op, b, _) => {
                if matches!(op, BinOp::Colon) {
                    let (kind, ty) = self.lower_colon_expr(a, b)?;
                    return Ok(HirExpr { kind, ty, span });
                }
                let left = self.lower_expr(a)?;
                let left_ty = left.ty.clone();
                let right = self.lower_expr(b)?;
                let right_ty = right.ty.clone();
                let ty = match op {
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::RightDiv
                    | BinOp::Pow
                    | BinOp::LeftDiv => {
                        if matches!(left_ty, Type::Tensor { .. })
                            || matches!(right_ty, Type::Tensor { .. })
                        {
                            Type::tensor()
                        } else {
                            Type::Num
                        }
                    }
                    BinOp::ElemMul | BinOp::ElemDiv | BinOp::ElemPow | BinOp::ElemLeftDiv => {
                        if matches!(left_ty, Type::Tensor { .. })
                            || matches!(right_ty, Type::Tensor { .. })
                        {
                            Type::tensor()
                        } else {
                            Type::Num
                        }
                    }
                    BinOp::Equal
                    | BinOp::NotEqual
                    | BinOp::Less
                    | BinOp::LessEqual
                    | BinOp::Greater
                    | BinOp::GreaterEqual => Type::Bool,
                    BinOp::AndAnd | BinOp::OrOr | BinOp::BitAnd | BinOp::BitOr => Type::Bool,
                    BinOp::Colon => Type::tensor(),
                };
                (
                    HirExprKind::Binary(Box::new(left), *op, Box::new(right)),
                    ty,
                )
            }
            AnonFunc { params, body, .. } => {
                let saved_len = self.scopes.len();
                self.push_scope();
                let mut param_ids: Vec<VarId> = Vec::with_capacity(params.len());
                for p in params {
                    param_ids.push(self.define(p.clone()));
                }
                let lowered_body = self.lower_expr(body)?;
                while self.scopes.len() > saved_len {
                    self.pop_scope();
                }
                (
                    HirExprKind::AnonFunc {
                        params: param_ids,
                        body: Box::new(lowered_body),
                    },
                    Type::Unknown,
                )
            }
            FuncHandle(name, _) => (HirExprKind::FuncHandle(name.clone()), Type::Unknown),
            FuncCall(name, args, _) => {
                if name == "__register_test_classes" {
                    self.allow_unqualified_statics = true;
                }
                let arg_exprs: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                let arg_exprs = arg_exprs?;

                if let Some(var_id) = self.lookup(name) {
                    let var_ty = if var_id.0 < self.var_types.len() {
                        self.var_types[var_id.0].clone()
                    } else {
                        Type::Unknown
                    };
                    let var_expr = HirExpr {
                        kind: HirExprKind::Var(var_id),
                        ty: var_ty,
                        span,
                    };
                    (HirExprKind::Index(Box::new(var_expr), arg_exprs), Type::Num)
                } else {
                    let return_type = self.infer_function_return_type(name, &arg_exprs);
                    (HirExprKind::FuncCall(name.clone(), arg_exprs), return_type)
                }
            }
            Tensor(rows, _) => {
                let mut hir_rows = Vec::new();
                for row in rows {
                    let mut hir_row = Vec::new();
                    for expr in row {
                        hir_row.push(self.lower_expr(expr)?);
                    }
                    hir_rows.push(hir_row);
                }
                (HirExprKind::Tensor(hir_rows), Type::tensor())
            }
            Cell(rows, _) => {
                let mut hir_rows = Vec::new();
                for row in rows {
                    let mut hir_row = Vec::new();
                    for expr in row {
                        hir_row.push(self.lower_expr(expr)?);
                    }
                    hir_rows.push(hir_row);
                }
                (HirExprKind::Cell(hir_rows), Type::Unknown)
            }
            Index(expr, indices, _) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                let ty = base.ty.clone();
                (HirExprKind::Index(Box::new(base), idx_exprs), ty)
            }
            IndexCell(expr, indices, _) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                (
                    HirExprKind::IndexCell(Box::new(base), idx_exprs),
                    Type::Unknown,
                )
            }
            Range(start, step, end, _) => {
                let start_hir = self.lower_expr(start)?;
                let end_hir = self.lower_expr(end)?;
                let step_hir = step.as_ref().map(|s| self.lower_expr(s)).transpose()?;
                (
                    HirExprKind::Range(
                        Box::new(start_hir),
                        step_hir.map(Box::new),
                        Box::new(end_hir),
                    ),
                    Type::tensor(),
                )
            }
            Colon(_) => (HirExprKind::Colon, Type::tensor()),
            EndKeyword(_) => (HirExprKind::End, Type::Unknown),
            Member(base, name, _) => {
                if let Ident(class_name, _) = &**base {
                    let dotted_name = format!("{class_name}.{name}");
                    if self.is_builtin_function(&dotted_name) {
                        return Ok(HirExpr {
                            kind: HirExprKind::FuncCall(dotted_name, Vec::new()),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                    if self.is_static_method_class(class_name) {
                        let metaclass = HirExpr {
                            kind: HirExprKind::MetaClass(class_name.clone()),
                            ty: Type::String,
                            span: base.span(),
                        };
                        return Ok(HirExpr {
                            kind: HirExprKind::Member(Box::new(metaclass), name.clone()),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                }
                let b = self.lower_expr(base)?;
                (
                    HirExprKind::Member(Box::new(b), name.clone()),
                    Type::Unknown,
                )
            }
            MemberDynamic(base, name_expr, _) => {
                let b = self.lower_expr(base)?;
                let n = self.lower_expr(name_expr)?;
                (
                    HirExprKind::MemberDynamic(Box::new(b), Box::new(n)),
                    Type::Unknown,
                )
            }
            DottedInvoke(base, name, args, _) => {
                if let Ident(class_name, _) = &**base {
                    let dotted_name = format!("{class_name}.{name}");
                    if self.is_builtin_function(&dotted_name) {
                        let lowered_args: Result<Vec<_>, _> =
                            args.iter().map(|arg| self.lower_expr(arg)).collect();
                        let lowered_args = lowered_args?;
                        return Ok(HirExpr {
                            kind: HirExprKind::FuncCall(dotted_name, lowered_args),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                    if self.is_static_method_class(class_name) {
                        let metaclass = HirExpr {
                            kind: HirExprKind::MetaClass(class_name.clone()),
                            ty: Type::String,
                            span: base.span(),
                        };
                        let lowered_args: Result<Vec<_>, _> =
                            args.iter().map(|a| self.lower_expr(a)).collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::MethodCall(
                                Box::new(metaclass),
                                name.clone(),
                                lowered_args?,
                            ),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                }
                let b = self.lower_expr(base)?;
                let lowered_args: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                let lowered_args = lowered_args?;
                if matches!(b.ty, Type::Struct { .. }) {
                    (
                        HirExprKind::Index(
                            Box::new(HirExpr {
                                kind: HirExprKind::Member(Box::new(b), name.clone()),
                                ty: Type::Unknown,
                                span,
                            }),
                            lowered_args,
                        ),
                        Type::Unknown,
                    )
                } else {
                    (
                        HirExprKind::DottedInvoke(Box::new(b), name.clone(), lowered_args),
                        Type::Unknown,
                    )
                }
            }
            MethodCall(base, name, args, _) => {
                if let Ident(class_name, _) = &**base {
                    let dotted_name = format!("{class_name}.{name}");
                    if self.is_builtin_function(&dotted_name) {
                        let lowered_args: Result<Vec<_>, _> =
                            args.iter().map(|arg| self.lower_expr(arg)).collect();
                        let lowered_args = lowered_args?;
                        return Ok(HirExpr {
                            kind: HirExprKind::FuncCall(dotted_name, lowered_args),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                    if self.is_static_method_class(class_name) {
                        let metaclass = HirExpr {
                            kind: HirExprKind::MetaClass(class_name.clone()),
                            ty: Type::String,
                            span: base.span(),
                        };
                        let lowered_args: Result<Vec<_>, _> =
                            args.iter().map(|a| self.lower_expr(a)).collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::MethodCall(
                                Box::new(metaclass),
                                name.clone(),
                                lowered_args?,
                            ),
                            ty: Type::Unknown,
                            span,
                        });
                    }
                }
                let b = self.lower_expr(base)?;
                let lowered_args: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                (
                    HirExprKind::MethodCall(Box::new(b), name.clone(), lowered_args?),
                    Type::Unknown,
                )
            }
            MetaClass(name, _) => (HirExprKind::MetaClass(name.clone()), Type::String),
        };
        Ok(HirExpr { kind, ty, span })
    }

    pub(crate) fn lower_lvalue(&mut self, lv: &parser::LValue) -> Result<HirLValue, SemanticError> {
        use parser::LValue as ALV;
        Ok(match lv {
            ALV::Var(name) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                HirLValue::Var(id)
            }
            ALV::Member(base, name) => {
                if let parser::Expr::Ident(var_name, _) = &**base {
                    let id = match self.lookup(var_name) {
                        Some(id) => id,
                        None => self.define(var_name.clone()),
                    };
                    let ty = if id.0 < self.var_types.len() {
                        self.var_types[id.0].clone()
                    } else {
                        Type::Unknown
                    };
                    let b = HirExpr {
                        kind: HirExprKind::Var(id),
                        ty,
                        span: base.span(),
                    };
                    HirLValue::Member(Box::new(b), name.clone())
                } else {
                    let b = self.lower_expr(base)?;
                    HirLValue::Member(Box::new(b), name.clone())
                }
            }
            ALV::MemberDynamic(base, name_expr) => {
                let b = self.lower_expr(base)?;
                let n = self.lower_expr(name_expr)?;
                HirLValue::MemberDynamic(Box::new(b), Box::new(n))
            }
            ALV::Index(base, idxs) => {
                let b = self.lower_expr(base)?;
                let lowered: Result<Vec<_>, _> = idxs.iter().map(|e| self.lower_expr(e)).collect();
                HirLValue::Index(Box::new(b), lowered?)
            }
            ALV::IndexCell(base, idxs) => {
                let b = self.lower_expr(base)?;
                let lowered: Result<Vec<_>, _> = idxs.iter().map(|e| self.lower_expr(e)).collect();
                HirLValue::IndexCell(Box::new(b), lowered?)
            }
        })
    }

    pub(crate) fn infer_function_return_type(&self, func_name: &str, args: &[HirExpr]) -> Type {
        if let Some(HirStmt::Function { outputs, body, .. }) = self.functions.get(func_name) {
            return self.infer_user_function_return_type(outputs, body, args);
        }

        let builtin_functions = runmat_builtins::builtin_functions();
        for builtin in builtin_functions {
            if builtin.name == func_name {
                let arg_types: Vec<Type> = args.iter().map(|arg| arg.ty.clone()).collect();
                let ctx = resolve_context_from_args(args);
                return builtin.infer_return_type_with_context(&arg_types, &ctx);
            }
        }

        Type::Unknown
    }

    fn infer_user_function_return_type(
        &self,
        outputs: &[VarId],
        body: &[HirStmt],
        _args: &[HirExpr],
    ) -> Type {
        if outputs.is_empty() {
            return Type::Void;
        }
        let result_types = self.infer_outputs_types(outputs, body);
        result_types.first().cloned().unwrap_or(Type::Unknown)
    }

    fn infer_outputs_types(&self, outputs: &[VarId], body: &[HirStmt]) -> Vec<Type> {
        #[derive(Clone)]
        struct Analysis {
            exits: Vec<HashMap<VarId, Type>>,
            fallthrough: Option<HashMap<VarId, Type>>,
        }

        fn join_type(a: &Type, b: &Type) -> Type {
            if a == b {
                return a.clone();
            }
            if matches!(a, Type::Unknown) {
                return b.clone();
            }
            if matches!(b, Type::Unknown) {
                return a.clone();
            }
            Type::Unknown
        }

        fn join_env(a: &HashMap<VarId, Type>, b: &HashMap<VarId, Type>) -> HashMap<VarId, Type> {
            let mut out = a.clone();
            for (k, v) in b {
                out.entry(*k)
                    .and_modify(|t| *t = join_type(t, v))
                    .or_insert_with(|| v.clone());
            }
            out
        }

        #[allow(clippy::type_complexity)]
        fn analyze_stmts(stmts: &[HirStmt], mut env: HashMap<VarId, Type>) -> Analysis {
            let mut exits = Vec::new();
            let mut i = 0usize;
            while i < stmts.len() {
                match &stmts[i] {
                    HirStmt::Assign(var, expr, _, _) => {
                        env.insert(*var, expr.ty.clone());
                    }
                    HirStmt::MultiAssign(vars, expr, _, _) => {
                        for v in vars.iter().flatten() {
                            env.insert(*v, expr.ty.clone());
                        }
                    }
                    HirStmt::ExprStmt(_, _, _) | HirStmt::Break(_) | HirStmt::Continue(_) => {}
                    HirStmt::Return(_) => {
                        exits.push(env.clone());
                        return Analysis {
                            exits,
                            fallthrough: None,
                        };
                    }
                    HirStmt::If {
                        then_body,
                        elseif_blocks,
                        else_body,
                        ..
                    } => {
                        let then_a = analyze_stmts(then_body, env.clone());
                        let mut out_env = then_a.fallthrough.unwrap_or_else(|| env.clone());
                        let mut all_exits = then_a.exits;
                        for (c, b) in elseif_blocks {
                            let _ = c;
                            let a = analyze_stmts(b, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = join_env(&out_env, &f);
                            }
                            all_exits.extend(a.exits);
                        }
                        if let Some(else_body) = else_body {
                            let a = analyze_stmts(else_body, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = join_env(&out_env, &f);
                            }
                            all_exits.extend(a.exits);
                        } else {
                            out_env = join_env(&out_env, &env);
                        }
                        env = out_env;
                        exits.extend(all_exits);
                    }
                    HirStmt::While { body, .. } => {
                        let a = analyze_stmts(body, env.clone());
                        if let Some(f) = a.fallthrough {
                            env = join_env(&env, &f);
                        }
                        exits.extend(a.exits);
                    }
                    HirStmt::For {
                        var, expr, body, ..
                    } => {
                        env.insert(*var, expr.ty.clone());
                        let a = analyze_stmts(body, env.clone());
                        if let Some(f) = a.fallthrough {
                            env = join_env(&env, &f);
                        }
                        exits.extend(a.exits);
                    }
                    HirStmt::Switch {
                        cases, otherwise, ..
                    } => {
                        let mut out_env: Option<HashMap<VarId, Type>> = None;
                        for (_v, b) in cases {
                            let a = analyze_stmts(b, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = Some(match out_env {
                                    Some(curr) => join_env(&curr, &f),
                                    None => f,
                                });
                            }
                            exits.extend(a.exits);
                        }
                        if let Some(otherwise) = otherwise {
                            let a = analyze_stmts(otherwise, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = Some(match out_env {
                                    Some(curr) => join_env(&curr, &f),
                                    None => f,
                                });
                            }
                            exits.extend(a.exits);
                        } else {
                            out_env = Some(match out_env {
                                Some(curr) => join_env(&curr, &env),
                                None => env.clone(),
                            });
                        }
                        if let Some(f) = out_env {
                            env = f;
                        }
                    }
                    HirStmt::TryCatch {
                        try_body,
                        catch_body,
                        ..
                    } => {
                        let a_try = analyze_stmts(try_body, env.clone());
                        let a_catch = analyze_stmts(catch_body, env.clone());
                        let mut out_env = a_try.fallthrough.unwrap_or_else(|| env.clone());
                        if let Some(f) = a_catch.fallthrough {
                            out_env = join_env(&out_env, &f);
                        }
                        env = out_env;
                        exits.extend(a_try.exits);
                        exits.extend(a_catch.exits);
                    }
                    HirStmt::Global(_, _) | HirStmt::Persistent(_, _) => {}
                    HirStmt::Function { .. } => {}
                    HirStmt::ClassDef { .. } => {}
                    HirStmt::AssignLValue(_, expr, _, _) => {
                        let _ = &expr.ty;
                    }
                    HirStmt::Import { .. } => {}
                }
                i += 1;
            }
            Analysis {
                exits,
                fallthrough: Some(env),
            }
        }

        let initial_env: HashMap<VarId, Type> = HashMap::new();
        let analysis = analyze_stmts(body, initial_env);
        let mut per_output: Vec<Type> = vec![Type::Unknown; outputs.len()];
        let mut accumulate = |env: &HashMap<VarId, Type>| {
            for (i, out) in outputs.iter().enumerate() {
                if let Some(t) = env.get(out) {
                    per_output[i] = join_type(&per_output[i], t);
                }
            }
        };
        for e in &analysis.exits {
            accumulate(e);
        }
        if let Some(f) = &analysis.fallthrough {
            accumulate(f);
        }
        per_output
    }
}

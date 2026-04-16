use super::shared::{resolve_context_from_args, FuncDef};
use crate::inference::expr::infer_expr_type_with_env;
use crate::{HirClassMember, HirExprKind, HirProgram, HirStmt, Type, VarId};
use std::collections::HashMap;

pub fn infer_global_variable_types(
    prog: &HirProgram,
    returns: &HashMap<String, Vec<Type>>,
) -> HashMap<VarId, Type> {
    #[derive(Clone)]
    struct Analysis {
        exits: Vec<HashMap<VarId, Type>>,
        fallthrough: Option<HashMap<VarId, Type>>,
    }

    fn join_env(a: &HashMap<VarId, Type>, b: &HashMap<VarId, Type>) -> HashMap<VarId, Type> {
        let mut out = a.clone();
        for (k, v) in b {
            out.entry(*k)
                .and_modify(|t| *t = t.unify(v))
                .or_insert_with(|| v.clone());
        }
        out
    }

    #[allow(clippy::type_complexity, clippy::only_used_in_recursion)]
    fn analyze_stmts(
        stmts: &[HirStmt],
        mut env: HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
        func_defs: &HashMap<String, FuncDef>,
    ) -> Analysis {
        let mut exits = Vec::new();
        let mut i = 0usize;
        while i < stmts.len() {
            match &stmts[i] {
                HirStmt::Assign(var, expr, _, _) => {
                    let t = infer_expr_type_with_env(expr, &env, returns);
                    env.insert(*var, t);
                }
                HirStmt::MultiAssign(vars, expr, _, _) => {
                    if let HirExprKind::FuncCall(ref name, _) = expr.kind {
                        let mut per_out: Vec<Type> = returns.get(name).cloned().unwrap_or_default();
                        let needs_fallback = per_out.is_empty()
                            || per_out.iter().any(|t| matches!(t, Type::Unknown));
                        if needs_fallback {
                            if let HirExprKind::FuncCall(_, args) = &expr.kind {
                                if let Some(builtin) = runmat_builtins::builtin_functions()
                                    .iter()
                                    .find(|b| b.name.eq_ignore_ascii_case(name))
                                {
                                    let arg_types: Vec<Type> = args
                                        .iter()
                                        .map(|arg| infer_expr_type_with_env(arg, &env, returns))
                                        .collect();
                                    let ctx = resolve_context_from_args(args);
                                    let out_type =
                                        builtin.infer_return_type_with_context(&arg_types, &ctx);
                                    per_out = vec![out_type; vars.len()];
                                }
                            }
                            if let Some((params, outs, body)) = func_defs.get(name).cloned() {
                                let mut penv: HashMap<VarId, Type> = HashMap::new();
                                for p in params {
                                    penv.insert(p, Type::Num);
                                }
                                let mut out_types: Vec<Type> = vec![Type::Unknown; outs.len()];
                                for s in &body {
                                    if let HirStmt::Assign(var, rhs, _, _) = s {
                                        if let Some(pos) = outs.iter().position(|o| o == var) {
                                            let t = infer_expr_type_with_env(rhs, &penv, returns);
                                            out_types[pos] = out_types[pos].unify(&t);
                                        }
                                    }
                                }
                                if per_out.is_empty() {
                                    per_out = out_types;
                                } else {
                                    for (i, t) in out_types.into_iter().enumerate() {
                                        if matches!(per_out.get(i), Some(Type::Unknown)) {
                                            if let Some(slot) = per_out.get_mut(i) {
                                                *slot = t;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (i, v) in vars.iter().enumerate() {
                            if let Some(id) = v {
                                env.insert(*id, per_out.get(i).cloned().unwrap_or(Type::Unknown));
                            }
                        }
                    } else {
                        let t = infer_expr_type_with_env(expr, &env, returns);
                        for v in vars.iter().flatten() {
                            env.insert(*v, t.clone());
                        }
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
                    cond,
                    then_body,
                    elseif_blocks,
                    else_body,
                    ..
                } => {
                    let _ = infer_expr_type_with_env(cond, &env, returns);
                    let then_analysis = analyze_stmts(then_body, env.clone(), returns, func_defs);
                    let mut branch_envs = Vec::new();
                    if let Some(f) = &then_analysis.fallthrough {
                        branch_envs.push(f.clone());
                    }
                    for e in &then_analysis.exits {
                        branch_envs.push(e.clone());
                    }
                    for (c, b) in elseif_blocks {
                        let _ = infer_expr_type_with_env(c, &env, returns);
                        let analysis = analyze_stmts(b, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    }
                    if let Some(else_body) = else_body {
                        let analysis = analyze_stmts(else_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    } else {
                        branch_envs.push(env.clone());
                    }
                    if let Some(first) = branch_envs.first().cloned() {
                        env = branch_envs
                            .iter()
                            .skip(1)
                            .fold(first, |acc, e| join_env(&acc, e));
                    }
                }
                HirStmt::Switch {
                    expr,
                    cases,
                    otherwise,
                    ..
                } => {
                    let _ = infer_expr_type_with_env(expr, &env, returns);
                    let mut branch_envs = Vec::new();
                    for (case_expr, case_body) in cases {
                        let _ = infer_expr_type_with_env(case_expr, &env, returns);
                        let analysis = analyze_stmts(case_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    }
                    if let Some(otherwise_body) = otherwise {
                        let analysis =
                            analyze_stmts(otherwise_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    } else {
                        branch_envs.push(env.clone());
                    }
                    if let Some(first) = branch_envs.first().cloned() {
                        env = branch_envs
                            .iter()
                            .skip(1)
                            .fold(first, |acc, e| join_env(&acc, e));
                    }
                }
                HirStmt::While { cond, body, .. } => {
                    let _ = infer_expr_type_with_env(cond, &env, returns);
                    let body_analysis = analyze_stmts(body, env.clone(), returns, func_defs);
                    if let Some(f) = &body_analysis.fallthrough {
                        env = join_env(&env, f);
                    }
                    for e in &body_analysis.exits {
                        env = join_env(&env, e);
                    }
                }
                HirStmt::For { expr, body, .. } => {
                    let range_ty = infer_expr_type_with_env(expr, &env, returns);
                    if let HirStmt::For { var, .. } = &stmts[i] {
                        let iter_ty = match &range_ty {
                            Type::Tensor { .. } => Type::Num,
                            Type::Logical { .. } => Type::Bool,
                            other => other.clone(),
                        };
                        env.insert(*var, iter_ty);
                    }
                    let body_analysis = analyze_stmts(body, env.clone(), returns, func_defs);
                    if let Some(f) = &body_analysis.fallthrough {
                        env = join_env(&env, f);
                    }
                    for e in &body_analysis.exits {
                        env = join_env(&env, e);
                    }
                }
                HirStmt::Function { .. } | HirStmt::ClassDef { .. } => {}
                HirStmt::Global(_, _)
                | HirStmt::Persistent(_, _)
                | HirStmt::Import { .. }
                | HirStmt::TryCatch { .. }
                | HirStmt::AssignLValue(_, _, _, _) => {}
            }
            i += 1;
        }

        Analysis {
            exits,
            fallthrough: Some(env),
        }
    }

    let mut func_defs: HashMap<String, FuncDef> = HashMap::new();
    for stmt in &prog.body {
        if let HirStmt::Function {
            name,
            params,
            outputs,
            body,
            ..
        } = stmt
        {
            func_defs.insert(
                name.clone(),
                (params.clone(), outputs.clone(), body.clone()),
            );
        } else if let HirStmt::ClassDef { members, .. } = stmt {
            for m in members {
                if let HirClassMember::Methods { body, .. } = m {
                    for s in body {
                        if let HirStmt::Function {
                            name,
                            params,
                            outputs,
                            body,
                            ..
                        } = s
                        {
                            func_defs.insert(
                                name.clone(),
                                (params.clone(), outputs.clone(), body.clone()),
                            );
                        }
                    }
                }
            }
        }
    }

    let analysis = analyze_stmts(&prog.body, HashMap::new(), returns, &func_defs);
    let mut out: HashMap<VarId, Type> = HashMap::new();
    let mut accumulate = |env: &HashMap<VarId, Type>| {
        for (k, v) in env {
            out.entry(*k)
                .and_modify(|t| *t = t.unify(v))
                .or_insert_with(|| v.clone());
        }
    };
    if let Some(f) = &analysis.fallthrough {
        accumulate(f);
    }
    for e in &analysis.exits {
        accumulate(e);
    }

    out
}

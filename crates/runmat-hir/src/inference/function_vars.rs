use super::expr::infer_expr_type_with_env;
use super::function_outputs::infer_function_output_types;
use super::shared::{
    apply_lvalue_type_effects, apply_struct_field_assertions, collect_function_defs,
    collect_struct_field_assertions, join_env, refine_multi_assign_outputs_from_func, Analysis,
    FuncDef,
};
use crate::{HirClassMember, HirExpr, HirExprKind, HirProgram, HirStmt, Type, VarId};
use std::collections::HashMap;

#[allow(clippy::type_complexity)]
pub fn infer_function_variable_types(prog: &HirProgram) -> HashMap<String, HashMap<VarId, Type>> {
    let returns_map = infer_function_output_types(prog);

    let func_defs = collect_function_defs(prog);

    fn infer_expr_type(
        expr: &HirExpr,
        env: &HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
    ) -> Type {
        infer_expr_type_with_env(expr, env, returns)
    }

    #[allow(clippy::type_complexity)]
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
                    let t = infer_expr_type(expr, &env, returns);
                    env.insert(*var, t);
                }
                HirStmt::MultiAssign(vars, expr, _, _) => {
                    if let HirExprKind::FuncCall(ref name, _) = expr.kind {
                        let mut per_out: Vec<Type> = returns.get(name).cloned().unwrap_or_default();
                        refine_multi_assign_outputs_from_func(
                            name,
                            &mut per_out,
                            returns,
                            func_defs,
                            infer_expr_type,
                        );
                        for (i, v) in vars.iter().enumerate() {
                            if let Some(id) = v {
                                env.insert(*id, per_out.get(i).cloned().unwrap_or(Type::Unknown));
                            }
                        }
                    } else {
                        let t = infer_expr_type(expr, &env, returns);
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
                    let mut assertions: Vec<(VarId, String)> = Vec::new();
                    collect_struct_field_assertions(cond, &mut assertions);
                    let mut then_env = env.clone();
                    apply_struct_field_assertions(&mut then_env, assertions);
                    let then_a = analyze_stmts(then_body, then_env, returns, func_defs);
                    let mut out_env = then_a.fallthrough.clone().unwrap_or_else(|| env.clone());
                    let mut all_exits = then_a.exits.clone();
                    for (c, b) in elseif_blocks {
                        let mut elseif_env = env.clone();
                        let mut els_assertions: Vec<(VarId, String)> = Vec::new();
                        collect_struct_field_assertions(c, &mut els_assertions);
                        apply_struct_field_assertions(&mut elseif_env, els_assertions);
                        let a = analyze_stmts(b, elseif_env, returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = join_env(&out_env, &f);
                        }
                        all_exits.extend(a.exits);
                    }
                    if let Some(else_body) = else_body {
                        let a = analyze_stmts(else_body, env.clone(), returns, func_defs);
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
                    let a = analyze_stmts(body, env.clone(), returns, func_defs);
                    if let Some(f) = a.fallthrough {
                        env = join_env(&env, &f);
                    }
                    exits.extend(a.exits);
                }
                HirStmt::For {
                    var, expr, body, ..
                } => {
                    let t = infer_expr_type(expr, &env, returns);
                    env.insert(*var, t);
                    let a = analyze_stmts(body, env.clone(), returns, func_defs);
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
                        let a = analyze_stmts(b, env.clone(), returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = Some(match out_env {
                                Some(curr) => join_env(&curr, &f),
                                None => f,
                            });
                        }
                        exits.extend(a.exits);
                    }
                    if let Some(otherwise) = otherwise {
                        let a = analyze_stmts(otherwise, env.clone(), returns, func_defs);
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
                    let a_try = analyze_stmts(try_body, env.clone(), returns, func_defs);
                    let a_catch = analyze_stmts(catch_body, env.clone(), returns, func_defs);
                    let mut out_env = a_try.fallthrough.clone().unwrap_or_else(|| env.clone());
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
                HirStmt::AssignLValue(lv, expr, _, _) => {
                    apply_lvalue_type_effects(&mut env, lv);
                    let _ = infer_expr_type(expr, &env, returns);
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

    let mut out: HashMap<String, HashMap<VarId, Type>> = HashMap::new();
    for stmt in &prog.body {
        match stmt {
            HirStmt::Function { name, body, .. } => {
                let a = analyze_stmts(body, HashMap::new(), &returns_map, &func_defs);
                let mut env = HashMap::new();
                for e in &a.exits {
                    env = join_env(&env, e);
                }
                if let Some(f) = &a.fallthrough {
                    env = join_env(&env, f);
                }
                out.insert(name.clone(), env);
            }
            HirStmt::ClassDef { members, .. } => {
                for m in members {
                    if let HirClassMember::Methods { body, .. } = m {
                        for s in body {
                            if let HirStmt::Function { name, body, .. } = s {
                                let a =
                                    analyze_stmts(body, HashMap::new(), &returns_map, &func_defs);
                                let mut env = HashMap::new();
                                for e in &a.exits {
                                    env = join_env(&env, e);
                                }
                                if let Some(f) = &a.fallthrough {
                                    env = join_env(&env, f);
                                }
                                out.insert(name.clone(), env);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

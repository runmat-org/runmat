use crate::{HirClassMember, HirExpr, HirExprKind, HirLValue, HirStmt, VarId};
use std::collections::{HashMap, HashSet};

pub fn remap_function_body(body: &[HirStmt], var_map: &HashMap<VarId, VarId>) -> Vec<HirStmt> {
    body.iter().map(|stmt| remap_stmt(stmt, var_map)).collect()
}

pub fn remap_stmt(stmt: &HirStmt, var_map: &HashMap<VarId, VarId>) -> HirStmt {
    match stmt {
        HirStmt::ExprStmt(expr, suppressed, span) => {
            HirStmt::ExprStmt(remap_expr(expr, var_map), *suppressed, *span)
        }
        HirStmt::Assign(var_id, expr, suppressed, span) => {
            let new_var_id = var_map.get(var_id).copied().unwrap_or(*var_id);
            HirStmt::Assign(new_var_id, remap_expr(expr, var_map), *suppressed, *span)
        }
        HirStmt::MultiAssign(var_ids, expr, suppressed, span) => {
            let mapped: Vec<Option<VarId>> = var_ids
                .iter()
                .map(|v| v.and_then(|vv| var_map.get(&vv).copied().or(Some(vv))))
                .collect();
            HirStmt::MultiAssign(mapped, remap_expr(expr, var_map), *suppressed, *span)
        }
        HirStmt::AssignLValue(lv, expr, suppressed, span) => {
            let remapped_lv = match lv {
                HirLValue::Var(v) => HirLValue::Var(var_map.get(v).copied().unwrap_or(*v)),
                HirLValue::Member(b, n) => {
                    HirLValue::Member(Box::new(remap_expr(b, var_map)), n.clone())
                }
                HirLValue::MemberDynamic(b, n) => HirLValue::MemberDynamic(
                    Box::new(remap_expr(b, var_map)),
                    Box::new(remap_expr(n, var_map)),
                ),
                HirLValue::Index(b, idxs) => HirLValue::Index(
                    Box::new(remap_expr(b, var_map)),
                    idxs.iter().map(|e| remap_expr(e, var_map)).collect(),
                ),
                HirLValue::IndexCell(b, idxs) => HirLValue::IndexCell(
                    Box::new(remap_expr(b, var_map)),
                    idxs.iter().map(|e| remap_expr(e, var_map)).collect(),
                ),
            };
            HirStmt::AssignLValue(remapped_lv, remap_expr(expr, var_map), *suppressed, *span)
        }
        HirStmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            span,
        } => HirStmt::If {
            cond: remap_expr(cond, var_map),
            then_body: remap_function_body(then_body, var_map),
            elseif_blocks: elseif_blocks
                .iter()
                .map(|(c, b)| (remap_expr(c, var_map), remap_function_body(b, var_map)))
                .collect(),
            else_body: else_body.as_ref().map(|b| remap_function_body(b, var_map)),
            span: *span,
        },
        HirStmt::While { cond, body, span } => HirStmt::While {
            cond: remap_expr(cond, var_map),
            body: remap_function_body(body, var_map),
            span: *span,
        },
        HirStmt::For {
            var,
            expr,
            body,
            span,
        } => {
            let new_var = var_map.get(var).copied().unwrap_or(*var);
            HirStmt::For {
                var: new_var,
                expr: remap_expr(expr, var_map),
                body: remap_function_body(body, var_map),
                span: *span,
            }
        }
        HirStmt::Switch {
            expr,
            cases,
            otherwise,
            span,
        } => HirStmt::Switch {
            expr: remap_expr(expr, var_map),
            cases: cases
                .iter()
                .map(|(c, b)| (remap_expr(c, var_map), remap_function_body(b, var_map)))
                .collect(),
            otherwise: otherwise.as_ref().map(|b| remap_function_body(b, var_map)),
            span: *span,
        },
        HirStmt::TryCatch {
            try_body,
            catch_var,
            catch_body,
            span,
        } => HirStmt::TryCatch {
            try_body: remap_function_body(try_body, var_map),
            catch_var: catch_var
                .as_ref()
                .map(|v| var_map.get(v).copied().unwrap_or(*v)),
            catch_body: remap_function_body(catch_body, var_map),
            span: *span,
        },
        HirStmt::Global(vars, span) => HirStmt::Global(
            vars.iter()
                .map(|(v, name)| (var_map.get(v).copied().unwrap_or(*v), name.clone()))
                .collect(),
            *span,
        ),
        HirStmt::Persistent(vars, span) => HirStmt::Persistent(
            vars.iter()
                .map(|(v, name)| (var_map.get(v).copied().unwrap_or(*v), name.clone()))
                .collect(),
            *span,
        ),
        HirStmt::Break(span) => HirStmt::Break(*span),
        HirStmt::Continue(span) => HirStmt::Continue(*span),
        HirStmt::Return(span) => HirStmt::Return(*span),
        HirStmt::Function { .. } => stmt.clone(),
        HirStmt::ClassDef {
            name,
            super_class,
            members,
            span,
        } => HirStmt::ClassDef {
            name: name.clone(),
            super_class: super_class.clone(),
            members: members
                .iter()
                .map(|m| match m {
                    HirClassMember::Properties { attributes, names } => {
                        HirClassMember::Properties {
                            attributes: attributes.clone(),
                            names: names.clone(),
                        }
                    }
                    HirClassMember::Events { attributes, names } => HirClassMember::Events {
                        attributes: attributes.clone(),
                        names: names.clone(),
                    },
                    HirClassMember::Enumeration { attributes, names } => {
                        HirClassMember::Enumeration {
                            attributes: attributes.clone(),
                            names: names.clone(),
                        }
                    }
                    HirClassMember::Arguments { attributes, names } => HirClassMember::Arguments {
                        attributes: attributes.clone(),
                        names: names.clone(),
                    },
                    HirClassMember::Methods { attributes, body } => HirClassMember::Methods {
                        attributes: attributes.clone(),
                        body: remap_function_body(body, var_map),
                    },
                })
                .collect(),
            span: *span,
        },
        HirStmt::Import {
            path,
            wildcard,
            span,
        } => HirStmt::Import {
            path: path.clone(),
            wildcard: *wildcard,
            span: *span,
        },
    }
}

pub fn remap_expr(expr: &HirExpr, var_map: &HashMap<VarId, VarId>) -> HirExpr {
    let new_kind = match &expr.kind {
        HirExprKind::Var(var_id) => {
            let new_var_id = var_map.get(var_id).copied().unwrap_or(*var_id);
            HirExprKind::Var(new_var_id)
        }
        HirExprKind::Unary(op, e) => HirExprKind::Unary(*op, Box::new(remap_expr(e, var_map))),
        HirExprKind::Binary(left, op, right) => HirExprKind::Binary(
            Box::new(remap_expr(left, var_map)),
            *op,
            Box::new(remap_expr(right, var_map)),
        ),
        HirExprKind::Tensor(rows) => HirExprKind::Tensor(
            rows.iter()
                .map(|row| row.iter().map(|e| remap_expr(e, var_map)).collect())
                .collect(),
        ),
        HirExprKind::Cell(rows) => HirExprKind::Cell(
            rows.iter()
                .map(|row| row.iter().map(|e| remap_expr(e, var_map)).collect())
                .collect(),
        ),
        HirExprKind::Index(base, indices) => HirExprKind::Index(
            Box::new(remap_expr(base, var_map)),
            indices.iter().map(|i| remap_expr(i, var_map)).collect(),
        ),
        HirExprKind::IndexCell(base, indices) => HirExprKind::IndexCell(
            Box::new(remap_expr(base, var_map)),
            indices.iter().map(|i| remap_expr(i, var_map)).collect(),
        ),
        HirExprKind::Range(start, step, end) => HirExprKind::Range(
            Box::new(remap_expr(start, var_map)),
            step.as_ref().map(|s| Box::new(remap_expr(s, var_map))),
            Box::new(remap_expr(end, var_map)),
        ),
        HirExprKind::Member(base, name) => {
            HirExprKind::Member(Box::new(remap_expr(base, var_map)), name.clone())
        }
        HirExprKind::MemberDynamic(base, name) => HirExprKind::MemberDynamic(
            Box::new(remap_expr(base, var_map)),
            Box::new(remap_expr(name, var_map)),
        ),
        HirExprKind::DottedInvoke(base, name, args) => HirExprKind::DottedInvoke(
            Box::new(remap_expr(base, var_map)),
            name.clone(),
            args.iter().map(|a| remap_expr(a, var_map)).collect(),
        ),
        HirExprKind::MethodCall(base, name, args) => HirExprKind::MethodCall(
            Box::new(remap_expr(base, var_map)),
            name.clone(),
            args.iter().map(|a| remap_expr(a, var_map)).collect(),
        ),
        HirExprKind::AnonFunc { params, body } => HirExprKind::AnonFunc {
            params: params.clone(),
            body: Box::new(remap_expr(body, var_map)),
        },
        HirExprKind::FuncHandle(name) => HirExprKind::FuncHandle(name.clone()),
        HirExprKind::FuncCall(name, args) => HirExprKind::FuncCall(
            name.clone(),
            args.iter().map(|a| remap_expr(a, var_map)).collect(),
        ),
        HirExprKind::Number(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::MetaClass(_) => expr.kind.clone(),
    };
    HirExpr {
        kind: new_kind,
        ty: expr.ty.clone(),
        span: expr.span,
    }
}

pub fn collect_function_variables(body: &[HirStmt]) -> HashSet<VarId> {
    let mut vars = HashSet::new();
    for stmt in body {
        collect_stmt_variables(stmt, &mut vars);
    }
    vars
}

fn collect_stmt_variables(stmt: &HirStmt, vars: &mut HashSet<VarId>) {
    match stmt {
        HirStmt::ExprStmt(expr, _, _) => collect_expr_variables(expr, vars),
        HirStmt::Assign(var_id, expr, _, _) => {
            vars.insert(*var_id);
            collect_expr_variables(expr, vars);
        }
        HirStmt::MultiAssign(var_ids, expr, _, _) => {
            for v in var_ids.iter().flatten() {
                vars.insert(*v);
            }
            collect_expr_variables(expr, vars);
        }
        HirStmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            ..
        } => {
            collect_expr_variables(cond, vars);
            for stmt in then_body {
                collect_stmt_variables(stmt, vars);
            }
            for (cond_expr, body) in elseif_blocks {
                collect_expr_variables(cond_expr, vars);
                for stmt in body {
                    collect_stmt_variables(stmt, vars);
                }
            }
            if let Some(body) = else_body {
                for stmt in body {
                    collect_stmt_variables(stmt, vars);
                }
            }
        }
        HirStmt::While { cond, body, .. } => {
            collect_expr_variables(cond, vars);
            for stmt in body {
                collect_stmt_variables(stmt, vars);
            }
        }
        HirStmt::For {
            var, expr, body, ..
        } => {
            vars.insert(*var);
            collect_expr_variables(expr, vars);
            for stmt in body {
                collect_stmt_variables(stmt, vars);
            }
        }
        HirStmt::Switch {
            expr,
            cases,
            otherwise,
            ..
        } => {
            collect_expr_variables(expr, vars);
            for (v, b) in cases {
                collect_expr_variables(v, vars);
                for s in b {
                    collect_stmt_variables(s, vars);
                }
            }
            if let Some(b) = otherwise {
                for s in b {
                    collect_stmt_variables(s, vars);
                }
            }
        }
        HirStmt::TryCatch {
            try_body,
            catch_var,
            catch_body,
            ..
        } => {
            if let Some(v) = catch_var {
                vars.insert(*v);
            }
            for s in try_body {
                collect_stmt_variables(s, vars);
            }
            for s in catch_body {
                collect_stmt_variables(s, vars);
            }
        }
        HirStmt::Global(vs, _) | HirStmt::Persistent(vs, _) => {
            for (v, _name) in vs {
                vars.insert(*v);
            }
        }
        HirStmt::AssignLValue(lv, expr, _, _) => {
            match lv {
                HirLValue::Var(v) => {
                    vars.insert(*v);
                }
                HirLValue::Member(base, _) => collect_expr_variables(base, vars),
                HirLValue::MemberDynamic(base, name) => {
                    collect_expr_variables(base, vars);
                    collect_expr_variables(name, vars);
                }
                HirLValue::Index(base, idxs) | HirLValue::IndexCell(base, idxs) => {
                    collect_expr_variables(base, vars);
                    for i in idxs {
                        collect_expr_variables(i, vars);
                    }
                }
            }
            collect_expr_variables(expr, vars);
        }
        HirStmt::Break(_) | HirStmt::Continue(_) | HirStmt::Return(_) => {}
        HirStmt::Function { .. } => {}
        HirStmt::ClassDef { .. } => {}
        HirStmt::Import { .. } => {}
    }
}

fn collect_expr_variables(expr: &HirExpr, vars: &mut HashSet<VarId>) {
    match &expr.kind {
        HirExprKind::Var(var_id) => {
            vars.insert(*var_id);
        }
        HirExprKind::Unary(_, e) => collect_expr_variables(e, vars),
        HirExprKind::Binary(left, _, right) => {
            collect_expr_variables(left, vars);
            collect_expr_variables(right, vars);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for e in row {
                    collect_expr_variables(e, vars);
                }
            }
        }
        HirExprKind::Index(base, indices) | HirExprKind::IndexCell(base, indices) => {
            collect_expr_variables(base, vars);
            for idx in indices {
                collect_expr_variables(idx, vars);
            }
        }
        HirExprKind::Range(start, step, end) => {
            collect_expr_variables(start, vars);
            if let Some(step_expr) = step {
                collect_expr_variables(step_expr, vars);
            }
            collect_expr_variables(end, vars);
        }
        HirExprKind::Member(base, _) => collect_expr_variables(base, vars),
        HirExprKind::MemberDynamic(base, name) => {
            collect_expr_variables(base, vars);
            collect_expr_variables(name, vars);
        }
        HirExprKind::MethodCall(base, _, args) | HirExprKind::DottedInvoke(base, _, args) => {
            collect_expr_variables(base, vars);
            for a in args {
                collect_expr_variables(a, vars);
            }
        }
        HirExprKind::AnonFunc { body, .. } => collect_expr_variables(body, vars),
        HirExprKind::FuncHandle(_) => {}
        HirExprKind::FuncCall(_, args) => {
            for arg in args {
                collect_expr_variables(arg, vars);
            }
        }
        HirExprKind::Number(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::MetaClass(_) => {}
    }
}

pub fn create_function_var_map(params: &[VarId], outputs: &[VarId]) -> HashMap<VarId, VarId> {
    let mut var_map = HashMap::new();
    let mut local_var_index = 0;

    for param_id in params {
        var_map.insert(*param_id, VarId(local_var_index));
        local_var_index += 1;
    }

    for output_id in outputs {
        if !var_map.contains_key(output_id) {
            var_map.insert(*output_id, VarId(local_var_index));
            local_var_index += 1;
        }
    }

    var_map
}

pub fn create_complete_function_var_map(
    params: &[VarId],
    outputs: &[VarId],
    body: &[HirStmt],
) -> HashMap<VarId, VarId> {
    let mut var_map = HashMap::new();
    let mut local_var_index = 0;
    let all_vars = collect_function_variables(body);

    for param_id in params {
        var_map.insert(*param_id, VarId(local_var_index));
        local_var_index += 1;
    }

    for output_id in outputs {
        if !var_map.contains_key(output_id) {
            var_map.insert(*output_id, VarId(local_var_index));
            local_var_index += 1;
        }
    }

    for var_id in &all_vars {
        if !var_map.contains_key(var_id) {
            var_map.insert(*var_id, VarId(local_var_index));
            local_var_index += 1;
        }
    }

    var_map
}

use super::expr::infer_expr_type_with_env;
use super::function_outputs::infer_function_output_types;
use super::shared::FuncDef;
use crate::{HirClassMember, HirExpr, HirExprKind, HirProgram, HirStmt, Type, VarId};
use runmat_parser as parser;
use std::collections::HashMap;

#[allow(clippy::type_complexity)]
pub fn infer_function_variable_types(prog: &HirProgram) -> HashMap<String, HashMap<VarId, Type>> {
    let returns_map = infer_function_output_types(prog);

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

    fn infer_expr_type(
        expr: &HirExpr,
        env: &HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
    ) -> Type {
        infer_expr_type_with_env(expr, env, returns)
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

    #[derive(Clone)]
    struct Analysis {
        exits: Vec<HashMap<VarId, Type>>,
        fallthrough: Option<HashMap<VarId, Type>>,
    }

    #[allow(clippy::type_complexity, clippy::only_used_in_recursion)]
    fn analyze_stmts(
        #[allow(clippy::only_used_in_recursion)] outputs: &[VarId],
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
                        let needs_fallback = per_out.is_empty()
                            || per_out.iter().any(|t| matches!(t, Type::Unknown));
                        if needs_fallback {
                            if let Some((params, outs, body)) = func_defs.get(name).cloned() {
                                let mut penv: HashMap<VarId, Type> = HashMap::new();
                                for p in params {
                                    penv.insert(p, Type::Num);
                                }
                                let mut out_types: Vec<Type> = vec![Type::Unknown; outs.len()];
                                for s in &body {
                                    if let HirStmt::Assign(var, rhs, _, _) = s {
                                        if let Some(pos) = outs.iter().position(|o| o == var) {
                                            let t = infer_expr_type(rhs, &penv, returns);
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
                    fn trim_quotes(s: &str) -> String {
                        let t = s.trim();
                        t.trim_matches('\'').to_string()
                    }
                    fn extract_field_literal(e: &HirExpr) -> Option<String> {
                        match &e.kind {
                            HirExprKind::String(s) => Some(trim_quotes(s)),
                            _ => None,
                        }
                    }
                    fn extract_field_list(e: &HirExpr) -> Vec<String> {
                        match &e.kind {
                            HirExprKind::String(s) => vec![trim_quotes(s)],
                            HirExprKind::Cell(rows) => {
                                let mut out = Vec::new();
                                for row in rows {
                                    for it in row {
                                        if let Some(v) = extract_field_literal(it) {
                                            out.push(v);
                                        }
                                    }
                                }
                                out
                            }
                            _ => Vec::new(),
                        }
                    }
                    fn collect_assertions(e: &HirExpr, out: &mut Vec<(VarId, String)>) {
                        use HirExprKind as K;
                        match &e.kind {
                            K::Unary(parser::UnOp::Not, _inner) => {}
                            K::Binary(left, parser::BinOp::AndAnd, right)
                            | K::Binary(left, parser::BinOp::BitAnd, right) => {
                                collect_assertions(left, out);
                                collect_assertions(right, out);
                            }
                            K::FuncCall(name, args) => {
                                let lname = name.as_str();
                                if lname.eq_ignore_ascii_case("isfield") && args.len() >= 2 {
                                    if let HirExprKind::Var(vid) = args[0].kind {
                                        if let Some(f) = extract_field_literal(&args[1]) {
                                            out.push((vid, f));
                                        }
                                    }
                                }
                                if lname.eq_ignore_ascii_case("ismember") && args.len() >= 2 {
                                    let mut fields: Vec<String> = Vec::new();
                                    let mut target: Option<VarId> = None;
                                    if let HirExprKind::FuncCall(ref n0, ref a0) = args[0].kind {
                                        if n0.eq_ignore_ascii_case("fieldnames") && a0.len() == 1 {
                                            if let HirExprKind::Var(vid) = a0[0].kind {
                                                target = Some(vid);
                                            }
                                        }
                                    }
                                    if let HirExprKind::FuncCall(ref n1, ref a1) = args[1].kind {
                                        if n1.eq_ignore_ascii_case("fieldnames") && a1.len() == 1 {
                                            if let HirExprKind::Var(vid) = a1[0].kind {
                                                target = Some(vid);
                                            }
                                        }
                                    }
                                    if fields.is_empty() {
                                        fields.extend(extract_field_list(&args[0]));
                                    }
                                    if fields.is_empty() {
                                        fields.extend(extract_field_list(&args[1]));
                                    }
                                    if let Some(vid) = target {
                                        for f in fields {
                                            out.push((vid, f));
                                        }
                                    }
                                }
                                if (lname.eq_ignore_ascii_case("any")
                                    || lname.eq_ignore_ascii_case("all"))
                                    && !args.is_empty()
                                {
                                    collect_assertions(&args[0], out);
                                }
                                if (lname.eq_ignore_ascii_case("strcmp")
                                    || lname.eq_ignore_ascii_case("strcmpi"))
                                    && args.len() >= 2
                                {
                                    let mut target: Option<VarId> = None;
                                    if let HirExprKind::FuncCall(ref n0, ref a0) = args[0].kind {
                                        if n0.eq_ignore_ascii_case("fieldnames") && a0.len() == 1 {
                                            if let HirExprKind::Var(vid) = a0[0].kind {
                                                target = Some(vid);
                                            }
                                        }
                                    }
                                    if let HirExprKind::FuncCall(ref n1, ref a1) = args[1].kind {
                                        if n1.eq_ignore_ascii_case("fieldnames") && a1.len() == 1 {
                                            if let HirExprKind::Var(vid) = a1[0].kind {
                                                target = Some(vid);
                                            }
                                        }
                                    }
                                    let mut fields = Vec::new();
                                    fields.extend(extract_field_list(&args[0]));
                                    fields.extend(extract_field_list(&args[1]));
                                    if let Some(vid) = target {
                                        for f in fields {
                                            out.push((vid, f));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    let mut assertions: Vec<(VarId, String)> = Vec::new();
                    collect_assertions(cond, &mut assertions);
                    let mut then_env = env.clone();
                    if !assertions.is_empty() {
                        for (vid, field) in assertions {
                            let mut known = match then_env.get(&vid) {
                                Some(Type::Struct { known_fields }) => known_fields.clone(),
                                _ => Some(Vec::new()),
                            };
                            if let Some(list) = &mut known {
                                if !list.iter().any(|f| f == &field) {
                                    list.push(field);
                                    list.sort();
                                    list.dedup();
                                }
                            }
                            then_env.insert(
                                vid,
                                Type::Struct {
                                    known_fields: known,
                                },
                            );
                        }
                    }
                    let then_a = analyze_stmts(outputs, then_body, then_env, returns, func_defs);
                    let mut out_env = then_a.fallthrough.clone().unwrap_or_else(|| env.clone());
                    let mut all_exits = then_a.exits.clone();
                    for (c, b) in elseif_blocks {
                        let mut elseif_env = env.clone();
                        let mut els_assertions: Vec<(VarId, String)> = Vec::new();
                        collect_assertions(c, &mut els_assertions);
                        if !els_assertions.is_empty() {
                            for (vid, field) in els_assertions {
                                let mut known = match elseif_env.get(&vid) {
                                    Some(Type::Struct { known_fields }) => known_fields.clone(),
                                    _ => Some(Vec::new()),
                                };
                                if let Some(list) = &mut known {
                                    if !list.iter().any(|f| f == &field) {
                                        list.push(field);
                                        list.sort();
                                        list.dedup();
                                    }
                                }
                                elseif_env.insert(
                                    vid,
                                    Type::Struct {
                                        known_fields: known,
                                    },
                                );
                            }
                        }
                        let a = analyze_stmts(outputs, b, elseif_env, returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = join_env(&out_env, &f);
                        }
                        all_exits.extend(a.exits);
                    }
                    if let Some(else_body) = else_body {
                        let a = analyze_stmts(outputs, else_body, env.clone(), returns, func_defs);
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
                    let a = analyze_stmts(outputs, body, env.clone(), returns, func_defs);
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
                    let a = analyze_stmts(outputs, body, env.clone(), returns, func_defs);
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
                        let a = analyze_stmts(outputs, b, env.clone(), returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = Some(match out_env {
                                Some(curr) => join_env(&curr, &f),
                                None => f,
                            });
                        }
                        exits.extend(a.exits);
                    }
                    if let Some(otherwise) = otherwise {
                        let a = analyze_stmts(outputs, otherwise, env.clone(), returns, func_defs);
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
                    let a_try = analyze_stmts(outputs, try_body, env.clone(), returns, func_defs);
                    let a_catch =
                        analyze_stmts(outputs, catch_body, env.clone(), returns, func_defs);
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
                HirStmt::AssignLValue(_, expr, _, _) => {
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
                let empty: &[VarId] = &[];
                let a = analyze_stmts(empty, body, HashMap::new(), &returns_map, &func_defs);
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
                                let empty: &[VarId] = &[];
                                let a = analyze_stmts(
                                    empty,
                                    body,
                                    HashMap::new(),
                                    &returns_map,
                                    &func_defs,
                                );
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

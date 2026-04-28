//! Function, nested-function, and closure lowering.

use crate::compiler::core::Compiler;
use crate::compiler::CompileError;
use crate::functions::UserFunction;
use crate::instr::Instr;
use runmat_builtins::Type;
use runmat_hir::{HirExpr, HirExprKind, HirStmt, VarId};
use std::collections::{HashMap, HashSet};

impl Compiler {
    #[allow(clippy::only_used_in_recursion)]
    pub(crate) fn collect_free_vars(
        &self,
        expr: &HirExpr,
        bound: &HashSet<VarId>,
        seen: &mut HashSet<VarId>,
        out: &mut Vec<VarId>,
    ) {
        use runmat_hir::HirExprKind as K;

        match &expr.kind {
            K::Var(id) => {
                if !bound.contains(id) && seen.insert(*id) {
                    out.push(*id);
                }
            }
            K::Unary(_, e) => self.collect_free_vars(e, bound, seen, out),
            K::Binary(a, _, b) => {
                self.collect_free_vars(a, bound, seen, out);
                self.collect_free_vars(b, bound, seen, out);
            }
            K::Tensor(rows) | K::Cell(rows) => {
                for row in rows {
                    for e in row {
                        self.collect_free_vars(e, bound, seen, out);
                    }
                }
            }
            K::Index(base, idxs) | K::IndexCell(base, idxs) => {
                self.collect_free_vars(base, bound, seen, out);
                for i in idxs {
                    self.collect_free_vars(i, bound, seen, out);
                }
            }
            K::Range(s, st, e) => {
                self.collect_free_vars(s, bound, seen, out);
                if let Some(st) = st {
                    self.collect_free_vars(st, bound, seen, out);
                }
                self.collect_free_vars(e, bound, seen, out);
            }
            K::FuncCall(_, args) => {
                for a in args {
                    self.collect_free_vars(a, bound, seen, out);
                }
            }
            K::MethodCall(base, _, args) | K::DottedInvoke(base, _, args) => {
                self.collect_free_vars(base, bound, seen, out);
                for a in args {
                    self.collect_free_vars(a, bound, seen, out);
                }
            }
            K::Member(base, _) => self.collect_free_vars(base, bound, seen, out),
            K::MemberDynamic(base, name) => {
                self.collect_free_vars(base, bound, seen, out);
                self.collect_free_vars(name, bound, seen, out);
            }
            K::AnonFunc { params, body } => {
                let mut new_bound = bound.clone();
                for p in params {
                    new_bound.insert(*p);
                }
                self.collect_free_vars(body, &new_bound, seen, out);
            }
            _ => {}
        }
    }

    pub(crate) fn compile_function_stmt(
        &mut self,
        name: &str,
        params: &[VarId],
        outputs: &[VarId],
        body: &[HirStmt],
        has_varargin: bool,
        has_varargout: bool,
    ) -> Result<(), CompileError> {
        let mut max_local_var = 0;

        fn visit_expr_for_vars(expr: &HirExpr, max: &mut usize) {
            match &expr.kind {
                HirExprKind::Var(id) => {
                    if id.0 + 1 > *max {
                        *max = id.0 + 1;
                    }
                }
                HirExprKind::Unary(_, e) => visit_expr_for_vars(e, max),
                HirExprKind::Binary(a, _, b) => {
                    visit_expr_for_vars(a, max);
                    visit_expr_for_vars(b, max);
                }
                HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
                    for row in rows {
                        for elem in row {
                            visit_expr_for_vars(elem, max);
                        }
                    }
                }
                HirExprKind::Index(base, indices) | HirExprKind::IndexCell(base, indices) => {
                    visit_expr_for_vars(base, max);
                    for idx in indices {
                        visit_expr_for_vars(idx, max);
                    }
                }
                HirExprKind::Range(start, step, end) => {
                    visit_expr_for_vars(start, max);
                    if let Some(step) = step {
                        visit_expr_for_vars(step, max);
                    }
                    visit_expr_for_vars(end, max);
                }
                HirExprKind::FuncCall(_, args) | HirExprKind::MethodCall(_, _, args) => {
                    for arg in args {
                        visit_expr_for_vars(arg, max);
                    }
                }
                _ => {}
            }
        }

        fn visit_stmt_for_vars(stmt: &HirStmt, max: &mut usize) {
            match stmt {
                HirStmt::ExprStmt(expr, _, _) => visit_expr_for_vars(expr, max),
                HirStmt::Assign(id, expr, _, _) => {
                    if id.0 + 1 > *max {
                        *max = id.0 + 1;
                    }
                    visit_expr_for_vars(expr, max);
                }
                HirStmt::If {
                    cond,
                    then_body,
                    elseif_blocks,
                    else_body,
                    ..
                } => {
                    visit_expr_for_vars(cond, max);
                    for stmt in then_body {
                        visit_stmt_for_vars(stmt, max);
                    }
                    for (cond, body) in elseif_blocks {
                        visit_expr_for_vars(cond, max);
                        for stmt in body {
                            visit_stmt_for_vars(stmt, max);
                        }
                    }
                    if let Some(body) = else_body {
                        for stmt in body {
                            visit_stmt_for_vars(stmt, max);
                        }
                    }
                }
                HirStmt::While { cond, body, .. } => {
                    visit_expr_for_vars(cond, max);
                    for stmt in body {
                        visit_stmt_for_vars(stmt, max);
                    }
                }
                HirStmt::For {
                    var, expr, body, ..
                } => {
                    if var.0 + 1 > *max {
                        *max = var.0 + 1;
                    }
                    visit_expr_for_vars(expr, max);
                    for stmt in body {
                        visit_stmt_for_vars(stmt, max);
                    }
                }
                HirStmt::Break(_) | HirStmt::Continue(_) | HirStmt::Return(_) => {}
                HirStmt::Switch {
                    expr,
                    cases,
                    otherwise,
                    ..
                } => {
                    visit_expr_for_vars(expr, max);
                    for (c, b) in cases {
                        visit_expr_for_vars(c, max);
                        for s in b {
                            visit_stmt_for_vars(s, max);
                        }
                    }
                    if let Some(b) = otherwise {
                        for s in b {
                            visit_stmt_for_vars(s, max);
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
                        if v.0 + 1 > *max {
                            *max = v.0 + 1;
                        }
                    }
                    for s in try_body {
                        visit_stmt_for_vars(s, max);
                    }
                    for s in catch_body {
                        visit_stmt_for_vars(s, max);
                    }
                }
                HirStmt::Global(vars, _) | HirStmt::Persistent(vars, _) => {
                    for (v, _name) in vars {
                        if v.0 + 1 > *max {
                            *max = v.0 + 1;
                        }
                    }
                }
                HirStmt::AssignLValue(_, expr, _, _) => visit_expr_for_vars(expr, max),
                HirStmt::MultiAssign(vars, expr, _, _) => {
                    for v in vars.iter().flatten() {
                        if v.0 + 1 > *max {
                            *max = v.0 + 1;
                        }
                    }
                    visit_expr_for_vars(expr, max);
                }
                HirStmt::Function { .. } | HirStmt::ClassDef { .. } | HirStmt::Import { .. } => {}
            }
        }

        for stmt in body {
            visit_stmt_for_vars(stmt, &mut max_local_var);
        }
        let var_map =
            runmat_hir::remapping::create_complete_function_var_map(params, outputs, body);
        let local_var_count = var_map.len();
        if local_var_count > max_local_var {
            max_local_var = local_var_count;
        }
        let mut func_var_types = vec![Type::Unknown; local_var_count];
        for (orig, local) in &var_map {
            if let Some(ty) = self.var_types.get(orig.0) {
                if let Some(slot) = func_var_types.get_mut(local.0) {
                    *slot = ty.clone();
                }
            }
        }
        let user_func = UserFunction {
            name: name.to_string(),
            params: params.to_vec(),
            outputs: outputs.to_vec(),
            body: body.to_vec(),
            local_var_count: max_local_var,
            has_varargin,
            has_varargout,
            var_types: func_var_types,
            source_id: None,
        };
        self.functions.insert(name.to_string(), user_func);
        Ok(())
    }

    pub(crate) fn compile_anon_func(
        &mut self,
        params: &[VarId],
        body: &HirExpr,
    ) -> Result<(), CompileError> {
        let mut seen: HashSet<VarId> = HashSet::new();
        let mut captures_order: Vec<VarId> = Vec::new();
        let bound: HashSet<VarId> = params.iter().cloned().collect();
        self.collect_free_vars(body, &bound, &mut seen, &mut captures_order);

        let capture_count = captures_order.len();
        let mut placeholder_params: Vec<VarId> = Vec::with_capacity(capture_count + params.len());
        for i in 0..capture_count {
            placeholder_params.push(VarId(i));
        }
        for j in 0..params.len() {
            placeholder_params.push(VarId(capture_count + j));
        }
        let output_id = VarId(capture_count + params.len());

        let mut var_map: HashMap<VarId, VarId> = HashMap::new();
        for (i, old) in captures_order.iter().enumerate() {
            var_map.insert(*old, VarId(i));
        }
        for (j, old) in params.iter().enumerate() {
            var_map.insert(*old, VarId(capture_count + j));
        }
        let remapped_body = runmat_hir::remapping::remap_expr(body, &var_map);
        let func_body = vec![HirStmt::Assign(
            output_id,
            remapped_body,
            true,
            runmat_hir::Span::default(),
        )];

        let synthesized = format!("__anon_{}", self.functions.len());
        let user_func = UserFunction {
            name: synthesized.clone(),
            params: placeholder_params,
            outputs: vec![output_id],
            body: func_body,
            local_var_count: capture_count + params.len() + 1,
            has_varargin: false,
            has_varargout: false,
            var_types: vec![Type::Unknown; capture_count + params.len() + 1],
            source_id: None,
        };
        self.functions.insert(synthesized.clone(), user_func);

        for old in &captures_order {
            self.emit(Instr::LoadVar(old.0));
        }
        self.emit(Instr::CreateClosure(synthesized, capture_count));
        Ok(())
    }
}

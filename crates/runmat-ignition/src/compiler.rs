use crate::functions::UserFunction;
use crate::instr::Instr;
use runmat_hir::{HirExpr, HirExprKind, HirProgram, HirStmt};
use std::collections::HashMap;

pub struct LoopLabels {
    pub break_jumps: Vec<usize>,
    pub continue_jumps: Vec<usize>,
}

pub struct Compiler {
    pub instructions: Vec<Instr>,
    pub var_count: usize,
    pub loop_stack: Vec<LoopLabels>,
    pub functions: HashMap<String, UserFunction>,
    pub imports: Vec<(Vec<String>, bool)>,
}

impl Compiler {
    fn attr_access_from_str(s: &str) -> runmat_builtins::Access {
        match s.to_ascii_lowercase().as_str() { "private" => runmat_builtins::Access::Private, _ => runmat_builtins::Access::Public }
    }
    fn parse_prop_attrs(attrs: &Vec<runmat_parser::Attr>) -> (bool, String, String) {
        // Defaults
        let mut is_static = false;
        let mut get_acc = runmat_builtins::Access::Public;
        let mut set_acc = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") { is_static = true; }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value { let acc = Self::attr_access_from_str(v.trim_matches('\'').trim()); get_acc = acc.clone(); set_acc = acc; }
            }
            if a.name.eq_ignore_ascii_case("GetAccess") {
                if let Some(v) = &a.value { get_acc = Self::attr_access_from_str(v.trim_matches('\'').trim()); }
            }
            if a.name.eq_ignore_ascii_case("SetAccess") {
                if let Some(v) = &a.value { set_acc = Self::attr_access_from_str(v.trim_matches('\'').trim()); }
            }
        }
        let gs = match get_acc { runmat_builtins::Access::Private => "private".to_string(), _ => "public".to_string() };
        let ss = match set_acc { runmat_builtins::Access::Private => "private".to_string(), _ => "public".to_string() };
        (is_static, gs, ss)
    }
    fn parse_method_attrs(attrs: &Vec<runmat_parser::Attr>) -> (bool, String) {
        let mut is_static = false;
        let mut access = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") { is_static = true; }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value { access = Self::attr_access_from_str(v.trim_matches('\'').trim()); }
            }
        }
        let acc_str = match access { runmat_builtins::Access::Private => "private".to_string(), _ => "public".to_string() };
        (is_static, acc_str)
    }
    fn collect_free_vars(
        &self,
        expr: &runmat_hir::HirExpr,
        bound: &std::collections::HashSet<runmat_hir::VarId>,
        seen: &mut std::collections::HashSet<runmat_hir::VarId>,
        out: &mut Vec<runmat_hir::VarId>,
    ) {
        use runmat_hir::HirExprKind as K;
        match &expr.kind {
            K::Var(id) => {
                if !bound.contains(id) && !seen.contains(id) {
                    seen.insert(*id);
                    out.push(*id);
                }
            }
            K::Unary(_, e) => self.collect_free_vars(e, bound, seen, out),
            K::Binary(a, _, b) => { self.collect_free_vars(a, bound, seen, out); self.collect_free_vars(b, bound, seen, out); }
            K::Tensor(rows) | K::Cell(rows) => { for row in rows { for e in row { self.collect_free_vars(e, bound, seen, out); } } }
            K::Index(base, idxs) | K::IndexCell(base, idxs) => { self.collect_free_vars(base, bound, seen, out); for i in idxs { self.collect_free_vars(i, bound, seen, out); } }
            K::Range(s, st, e) => { self.collect_free_vars(s, bound, seen, out); if let Some(st) = st { self.collect_free_vars(st, bound, seen, out); } self.collect_free_vars(e, bound, seen, out); }
            K::FuncCall(_, args) | K::MethodCall(_, _, args) => { for a in args { self.collect_free_vars(a, bound, seen, out); } }
            K::Member(base, _) => self.collect_free_vars(base, bound, seen, out),
            K::AnonFunc { params, body } => {
                let mut new_bound = bound.clone();
                for p in params { new_bound.insert(*p); }
                self.collect_free_vars(body, &new_bound, seen, out);
            }
            _ => {}
        }
    }
    pub fn new(prog: &HirProgram) -> Self {
        let mut max_var = 0;
        fn visit_expr(expr: &HirExpr, max: &mut usize) {
            match &expr.kind {
                HirExprKind::Var(id) => {
                    if id.0 + 1 > *max {
                        *max = id.0 + 1;
                    }
                }
                HirExprKind::Unary(_, e) => visit_expr(e, max),
                HirExprKind::Binary(left, _, right) => {
                    visit_expr(left, max);
                    visit_expr(right, max);
                }
                HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
                    for row in rows {
                        for expr in row {
                            visit_expr(expr, max);
                        }
                    }
                }
                HirExprKind::Index(expr, indices) | HirExprKind::IndexCell(expr, indices) => {
                    visit_expr(expr, max);
                    for idx in indices {
                        visit_expr(idx, max);
                    }
                }
                HirExprKind::Range(start, step, end) => {
                    visit_expr(start, max);
                    if let Some(step) = step {
                        visit_expr(step, max);
                    }
                    visit_expr(end, max);
                }
                HirExprKind::FuncCall(_, args)
                | HirExprKind::MethodCall(_, _, args) => {
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::Member(base, _) => visit_expr(base, max),
                HirExprKind::AnonFunc { body, .. } => visit_expr(body, max),
                HirExprKind::Number(_)
                | HirExprKind::String(_)
                | HirExprKind::Constant(_)
                | HirExprKind::Colon
                | HirExprKind::End
                | HirExprKind::FuncHandle(_)
                | HirExprKind::MetaClass(_) => {}
            }
        }

        fn visit_stmts(stmts: &[HirStmt], max: &mut usize) {
            for s in stmts {
                match s {
                    HirStmt::Assign(id, expr, _) => {
                        if id.0 + 1 > *max {
                            *max = id.0 + 1;
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::ExprStmt(expr, _) => visit_expr(expr, max),
                    HirStmt::Return => {}
                    HirStmt::If { cond, then_body, elseif_blocks, else_body } => {
                        visit_expr(cond, max);
                        visit_stmts(then_body, max);
                        for (cond, body) in elseif_blocks {
                            visit_expr(cond, max);
                            visit_stmts(body, max);
                        }
                        if let Some(body) = else_body { visit_stmts(body, max); }
                    }
                    HirStmt::While { cond, body } => {
                        visit_expr(cond, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::For { var, expr, body } => {
                        if var.0 + 1 > *max { *max = var.0 + 1; }
                        visit_expr(expr, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::Switch { expr, cases, otherwise } => {
                        visit_expr(expr, max);
                        for (c, b) in cases { visit_expr(c, max); visit_stmts(b, max); }
                        if let Some(b) = otherwise { visit_stmts(b, max); }
                    }
                    HirStmt::TryCatch { try_body, catch_var, catch_body } => {
                        if let Some(v) = catch_var { if v.0 + 1 > *max { *max = v.0 + 1; } }
                        visit_stmts(try_body, max);
                        visit_stmts(catch_body, max);
                    }
                    HirStmt::Global(vars) | HirStmt::Persistent(vars) => {
                        for v in vars { if v.0 + 1 > *max { *max = v.0 + 1; } }
                    }
                    HirStmt::AssignLValue(_, expr, _) => visit_expr(expr, max),
                    HirStmt::MultiAssign(vars, expr, _) => {
                        for v in vars { if let Some(v) = v { if v.0 + 1 > *max { *max = v.0 + 1; } } }
                        visit_expr(expr, max);
                    }
                    HirStmt::Function { .. } | HirStmt::ClassDef { .. } | HirStmt::Import { .. }
                    | HirStmt::Break | HirStmt::Continue => {}
                }
            }
        }

        visit_stmts(&prog.body, &mut max_var);
        Self { instructions: Vec::new(), var_count: max_var, loop_stack: Vec::new(), functions: HashMap::new(), imports: Vec::new() }
    }

    pub fn emit(&mut self, instr: Instr) -> usize {
        let pc = self.instructions.len();
        self.instructions.push(instr);
        pc
    }

    pub fn patch(&mut self, idx: usize, instr: Instr) { self.instructions[idx] = instr; }

    pub fn compile_program(&mut self, prog: &HirProgram) -> Result<(), String> {
        // Pre-collect imports
        for stmt in &prog.body {
            if let HirStmt::Import { path, wildcard } = stmt {
                self.imports.push((path.clone(), *wildcard));
                self.emit(Instr::RegisterImport { path: path.clone(), wildcard: *wildcard });
            }
        }
        for stmt in &prog.body { if !matches!(stmt, HirStmt::Import { .. }) { self.compile_stmt(stmt)?; } }
        Ok(())
    }


    pub fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), String> {
        match stmt {
            HirStmt::ExprStmt(expr, _) => { self.compile_expr(expr)?; self.emit(Instr::Pop); }
            HirStmt::Assign(id, expr, _) => { self.compile_expr(expr)?; self.emit(Instr::StoreVar(id.0)); }
            HirStmt::If { cond, then_body, elseif_blocks, else_body } => {
                self.compile_expr(cond)?;
                let mut last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                for s in then_body { self.compile_stmt(s)?; }
                let mut end_jumps = Vec::new();
                end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                for (c, b) in elseif_blocks {
                    self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                    self.compile_expr(c)?;
                    last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in b { self.compile_stmt(s)?; }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                }
                self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                if let Some(body) = else_body { for s in body { self.compile_stmt(s)?; } }
                let end = self.instructions.len();
                for j in end_jumps { self.patch(j, Instr::Jump(end)); }
            }
            HirStmt::While { cond, body } => {
                let start = self.instructions.len();
                self.compile_expr(cond)?;
                let jump_end = self.emit(Instr::JumpIfFalse(usize::MAX));
                let labels = LoopLabels { break_jumps: Vec::new(), continue_jumps: Vec::new() };
                self.loop_stack.push(labels);
                for s in body { self.compile_stmt(s)?; }
                let labels = self.loop_stack.pop().unwrap();
                for j in labels.continue_jumps { self.patch(j, Instr::Jump(start)); }
                self.emit(Instr::Jump(start));
                let end = self.instructions.len();
                self.patch(jump_end, Instr::JumpIfFalse(end));
                for j in labels.break_jumps { self.patch(j, Instr::Jump(end)); }
            }
            HirStmt::For { var, expr, body } => {
                if let HirExprKind::Range(start, step, end) = &expr.kind {
                    // Initialize loop variable, end, and step
                    self.compile_expr(start)?; self.emit(Instr::StoreVar(var.0));
                    self.compile_expr(end)?; let end_var = self.var_count; self.var_count += 1; self.emit(Instr::StoreVar(end_var));
                    let step_var = self.var_count; self.var_count += 1;
                    if let Some(step_expr) = step {
                        self.compile_expr(step_expr)?; self.emit(Instr::StoreVar(step_var));
                    } else {
                        self.emit(Instr::LoadConst(1.0)); self.emit(Instr::StoreVar(step_var));
                    }

                    let loop_start = self.instructions.len();

                    // If step == 0 -> terminate loop immediately
                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::LoadConst(0.0));
                    self.emit(Instr::Equal);
                    let j_step_zero_skip = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let jump_end = self.emit(Instr::Jump(usize::MAX));
                    let after_step_zero_check = self.instructions.len();
                    self.patch(j_step_zero_skip, Instr::JumpIfFalse(after_step_zero_check));

                    // Determine condition based on sign(step)
                    // if step >= 0: cond = var <= end else cond = var >= end
                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::LoadConst(0.0));
                    self.emit(Instr::GreaterEqual);
                    let j_neg_branch = self.emit(Instr::JumpIfFalse(usize::MAX));
                    // positive step: var <= end
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(end_var));
                    self.emit(Instr::LessEqual);
                    let j_exit_pos = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let j_cond_done = self.emit(Instr::Jump(usize::MAX));
                    let neg_branch = self.instructions.len();
                    self.patch(j_neg_branch, Instr::JumpIfFalse(neg_branch));
                    // negative step: var >= end
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(end_var));
                    self.emit(Instr::GreaterEqual);
                    let j_exit_neg = self.emit(Instr::JumpIfFalse(usize::MAX));
                    let cond_done = self.instructions.len();
                    self.patch(j_cond_done, Instr::Jump(cond_done));

                    // At this point, loop condition is satisfied; execute body
                    self.loop_stack.push(LoopLabels { break_jumps: Vec::new(), continue_jumps: Vec::new() });
                    for s in body { self.compile_stmt(s)?; }
                    let labels = self.loop_stack.pop().unwrap();
                    for j in labels.continue_jumps { self.patch(j, Instr::Jump(self.instructions.len())); }
                    // increment: var = var + step
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(step_var));
                    self.emit(Instr::Add);
                    self.emit(Instr::StoreVar(var.0));
                    self.emit(Instr::Jump(loop_start));

                    // Exit jump targets
                    let end_pc = self.instructions.len();
                    self.patch(jump_end, Instr::Jump(end_pc));
                    self.patch(j_exit_pos, Instr::JumpIfFalse(end_pc));
                    self.patch(j_exit_neg, Instr::JumpIfFalse(end_pc));
                    for j in labels.break_jumps { self.patch(j, Instr::Jump(end_pc)); }
                } else { return Err("for loop expects range".into()); }
            }
            HirStmt::Break => {
                if let Some(labels) = self.loop_stack.last_mut() { let idx = self.instructions.len(); self.instructions.push(Instr::Jump(usize::MAX)); labels.break_jumps.push(idx); } else { return Err("break outside loop".into()); }
            }
            HirStmt::Continue => {
                if let Some(labels) = self.loop_stack.last_mut() { let idx = self.instructions.len(); self.instructions.push(Instr::Jump(usize::MAX)); labels.continue_jumps.push(idx); } else { return Err("continue outside loop".into()); }
            }
            HirStmt::Return => { self.emit(Instr::Return); }
            HirStmt::Function { name, params, outputs, body } => {
                let mut max_local_var = 0;
                fn visit_expr_for_vars(expr: &HirExpr, max: &mut usize) {
                    match &expr.kind {
                        HirExprKind::Var(id) => { if id.0 + 1 > *max { *max = id.0 + 1; } }
                        HirExprKind::Unary(_, e) => visit_expr_for_vars(e, max),
                        HirExprKind::Binary(a, _, b) => { visit_expr_for_vars(a, max); visit_expr_for_vars(b, max); }
                        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
                            for row in rows { for elem in row { visit_expr_for_vars(elem, max); } }
                        }
                        HirExprKind::Index(base, indices) | HirExprKind::IndexCell(base, indices) => {
                            visit_expr_for_vars(base, max); for idx in indices { visit_expr_for_vars(idx, max); }
                        }
                        HirExprKind::Range(start, step, end) => {
                            visit_expr_for_vars(start, max); if let Some(step) = step { visit_expr_for_vars(step, max); } visit_expr_for_vars(end, max);
                        }
                        HirExprKind::FuncCall(_, args) | HirExprKind::MethodCall(_, _, args) => {
                            for arg in args { visit_expr_for_vars(arg, max); }
                        }
                        _ => {}
                    }
                }
                fn visit_stmt_for_vars(stmt: &HirStmt, max: &mut usize) {
                    match stmt {
                        HirStmt::ExprStmt(expr, _) => visit_expr_for_vars(expr, max),
                        HirStmt::Assign(id, expr, _) => { if id.0 + 1 > *max { *max = id.0 + 1; } visit_expr_for_vars(expr, max); }
                        HirStmt::If { cond, then_body, elseif_blocks, else_body } => {
                            visit_expr_for_vars(cond, max);
                            for stmt in then_body { visit_stmt_for_vars(stmt, max); }
                            for (cond, body) in elseif_blocks { visit_expr_for_vars(cond, max); for stmt in body { visit_stmt_for_vars(stmt, max); } }
                            if let Some(body) = else_body { for stmt in body { visit_stmt_for_vars(stmt, max); } }
                        }
                        HirStmt::While { cond, body } => { visit_expr_for_vars(cond, max); for stmt in body { visit_stmt_for_vars(stmt, max); } }
                        HirStmt::For { var, expr, body } => { if var.0 + 1 > *max { *max = var.0 + 1; } visit_expr_for_vars(expr, max); for stmt in body { visit_stmt_for_vars(stmt, max); } }
                        HirStmt::Break | HirStmt::Continue | HirStmt::Return => {}
                        HirStmt::Switch { expr, cases, otherwise } => { visit_expr_for_vars(expr, max); for (c, b) in cases { visit_expr_for_vars(c, max); for s in b { visit_stmt_for_vars(s, max); } } if let Some(b) = otherwise { for s in b { visit_stmt_for_vars(s, max); } } }
                        HirStmt::TryCatch { try_body, catch_var, catch_body } => { if let Some(v) = catch_var { if v.0 + 1 > *max { *max = v.0 + 1; } } for s in try_body { visit_stmt_for_vars(s, max); } for s in catch_body { visit_stmt_for_vars(s, max); } }
                        HirStmt::Global(vars) | HirStmt::Persistent(vars) => { for v in vars { if v.0 + 1 > *max { *max = v.0 + 1; } } }
                        HirStmt::AssignLValue(_, expr, _) => visit_expr_for_vars(expr, max),
                        HirStmt::MultiAssign(vars, expr, _) => { for v in vars { if let Some(v) = v { if v.0 + 1 > *max { *max = v.0 + 1; } } } visit_expr_for_vars(expr, max); }
                        HirStmt::Function { .. } | HirStmt::ClassDef { .. } | HirStmt::Import { .. } => {}
                    }
                }
                for stmt in body { visit_stmt_for_vars(stmt, &mut max_local_var); }
                let user_func = UserFunction { name: name.clone(), params: params.clone(), outputs: outputs.clone(), body: body.clone(), local_var_count: max_local_var };
                self.functions.insert(name.clone(), user_func);
            }
            HirStmt::Switch { expr, cases, otherwise } => {
                let temp_id = self.var_count; self.var_count += 1;
                self.compile_expr(expr)?; self.emit(Instr::StoreVar(temp_id));
                let mut end_jumps: Vec<usize> = Vec::new();
                let mut next_case_jump_to_here: Option<usize> = None;
                for (case_expr, body) in cases {
                    if let Some(j) = next_case_jump_to_here.take() { self.patch(j, Instr::JumpIfFalse(self.instructions.len())); }
                    self.emit(Instr::LoadVar(temp_id));
                    self.compile_expr(case_expr)?; self.emit(Instr::Equal);
                    let jmp = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in body { self.compile_stmt(s)?; }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                    next_case_jump_to_here = Some(jmp);
                }
                let otherwise_start = self.instructions.len();
                if let Some(j) = next_case_jump_to_here.take() { self.patch(j, Instr::JumpIfFalse(otherwise_start)); }
                if let Some(body) = otherwise { for s in body { self.compile_stmt(s)?; } }
                let end = self.instructions.len();
                for j in end_jumps { self.patch(j, Instr::Jump(end)); }
            }
            HirStmt::TryCatch { try_body, catch_var, catch_body } => {
                // Reserve slot for EnterTry with placeholder
                let enter_idx = self.emit(Instr::EnterTry(usize::MAX, catch_var.map(|v| v.0)));
                // Compile try body
                for s in try_body { self.compile_stmt(s)?; }
                // On normal completion, pop try frame and jump past catch
                self.emit(Instr::PopTry);
                let jmp_end = self.emit(Instr::Jump(usize::MAX));
                // Catch block starts here
                let catch_pc = self.instructions.len();
                // Patch EnterTry with catch_pc
                self.patch(enter_idx, Instr::EnterTry(catch_pc, catch_var.map(|v| v.0)));
                // Compile catch body
                for s in catch_body { self.compile_stmt(s)?; }
                let end_pc = self.instructions.len();
                self.patch(jmp_end, Instr::Jump(end_pc));
            }
            HirStmt::AssignLValue(lv, rhs, _) => {
                match lv {
                    runmat_hir::HirLValue::Index(base, indices) => {
                        if let runmat_hir::HirExprKind::Var(var_id) = base.kind {
                            // Load base variable first
                            self.emit(Instr::LoadVar(var_id.0));
                            // Compute masks and numeric indices as in IndexSlice
                            let has_colon = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::Colon));
                            let has_end = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::End));
                            let has_vector = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::Range(_,_,_) | runmat_hir::HirExprKind::Tensor(_)) || matches!(e.ty, runmat_hir::Type::Matrix{..}));
                            if has_colon || has_end || has_vector || indices.len() > 2 {
                                let mut colon_mask: u32 = 0;
                                let mut end_mask: u32 = 0;
                                let mut numeric_count = 0usize;
                                for (dim, index) in indices.iter().enumerate() {
                                    if matches!(index.kind, runmat_hir::HirExprKind::Colon) {
                                        colon_mask |= 1u32 << dim;
                                    } else if matches!(index.kind, runmat_hir::HirExprKind::End) {
                                        end_mask |= 1u32 << dim;
                                    } else {
                                        self.compile_expr(index)?;
                                        numeric_count += 1;
                                    }
                                }
                                // Push RHS last so VM pops it first
                                self.compile_expr(rhs)?;
                                self.emit(Instr::StoreSlice(indices.len(), numeric_count, colon_mask, end_mask));
                                // Store updated base back to variable
                                self.emit(Instr::StoreVar(var_id.0));
                            } else {
                                // Pure numeric indexing
                                for index in indices { self.compile_expr(index)?; }
                                self.compile_expr(rhs)?;
                                self.emit(Instr::StoreIndex(indices.len()));
                                self.emit(Instr::StoreVar(var_id.0));
                            }
                        } else if let runmat_hir::HirExprKind::Member(member_base, field) = &base.kind {
                            // Chain: base is a member access. Evaluate object, load member, update via index, then write member back.
                            // Evaluate object and get member value
                            self.compile_expr(member_base)?;
                            self.emit(Instr::LoadMember(field.clone()));
                            // Decide slice vs numeric
                            let has_colon = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::Colon));
                            let has_end = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::End));
                            let has_vector = indices.iter().any(|e| matches!(e.kind, runmat_hir::HirExprKind::Range(_,_,_) | runmat_hir::HirExprKind::Tensor(_)) || matches!(e.ty, runmat_hir::Type::Matrix{..}));
                            if has_colon || has_end || has_vector || indices.len() > 2 {
                                let mut colon_mask: u32 = 0;
                                let mut end_mask: u32 = 0;
                                let mut numeric_count = 0usize;
                                for (dim, index) in indices.iter().enumerate() {
                                    if matches!(index.kind, runmat_hir::HirExprKind::Colon) { colon_mask |= 1u32 << dim; }
                                    else if matches!(index.kind, runmat_hir::HirExprKind::End) { end_mask |= 1u32 << dim; }
                                    else { self.compile_expr(index)?; numeric_count += 1; }
                                }
                                self.compile_expr(rhs)?;
                                self.emit(Instr::StoreSlice(indices.len(), numeric_count, colon_mask, end_mask));
                            } else {
                                for index in indices { self.compile_expr(index)?; }
                                self.compile_expr(rhs)?;
                                self.emit(Instr::StoreIndex(indices.len()));
                            }
                            // Now updated member is on stack. Re-evaluate object, swap, and StoreMember
                            self.compile_expr(member_base)?;
                            self.emit(Instr::Swap);
                            self.emit(Instr::StoreMember(field.clone()));
                            // If object is a variable, also store back to var
                            if let runmat_hir::HirExprKind::Var(root_var) = member_base.kind {
                                self.emit(Instr::StoreVar(root_var.0));
                            }
                        } else {
                            return Err("unsupported lvalue target (index on non-variable/non-member)".into());
                        }
                    }
                    runmat_hir::HirLValue::IndexCell(base, indices) => {
                        if let runmat_hir::HirExprKind::Var(var_id) = base.kind {
                            self.emit(Instr::LoadVar(var_id.0));
                            for index in indices { self.compile_expr(index)?; }
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreIndexCell(indices.len()));
                            self.emit(Instr::StoreVar(var_id.0));
                        } else if let runmat_hir::HirExprKind::Member(member_base, field) = &base.kind {
                            // Load object, load member, perform cell index store, then write member back to object
                            self.compile_expr(member_base)?;
                            self.emit(Instr::LoadMember(field.clone()));
                            for index in indices { self.compile_expr(index)?; }
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreIndexCell(indices.len()));
                            // Updated member on stack; re-evaluate object, swap, store member
                            self.compile_expr(member_base)?;
                            self.emit(Instr::Swap);
                            self.emit(Instr::StoreMember(field.clone()));
                            if let runmat_hir::HirExprKind::Var(root_var) = member_base.kind { self.emit(Instr::StoreVar(root_var.0)); }
                        } else {
                            // Fallback: evaluate base, indices, rhs, and store (for object chains via subsasgn)
                            self.compile_expr(base)?;
                            for index in indices { self.compile_expr(index)?; }
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreIndexCell(indices.len()));
                        }
                    }
                    runmat_hir::HirLValue::Member(base, field) => {
                        // Member assignment. If base is a variable, ensure we store updated object back.
                        if let runmat_hir::HirExprKind::Var(var_id) = base.kind.clone() {
                            self.emit(Instr::LoadVar(var_id.0));
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreMember(field.clone()));
                            self.emit(Instr::StoreVar(var_id.0));
                        } else {
                            // Complex base: evaluate to a value, then store member; updated object remains on stack
                            self.compile_expr(base)?;
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreMember(field.clone()));
                        }
                    }
                    _ => return Err("unsupported lvalue target".into()),
                }
            }
            HirStmt::Global(_) | HirStmt::Persistent(_) => {}
            HirStmt::Import { path, wildcard } => {
                self.emit(Instr::RegisterImport { path: path.clone(), wildcard: *wildcard });
            }
            HirStmt::ClassDef { name, super_class, members } => {
                // Synthesize a minimal RegisterClass instruction by extracting property names and method names
                let mut props: Vec<(String, bool, String, String)> = Vec::new();
                let mut methods: Vec<(String, String, bool, String)> = Vec::new();
                for m in members {
                    match m {
                        runmat_hir::HirClassMember::Properties { names, attributes } => {
                            let (is_static, get_access, set_access) = Self::parse_prop_attrs(attributes);
                            for n in names { props.push((n.clone(), is_static, get_access.clone(), set_access.clone())); }
                        }
                        runmat_hir::HirClassMember::Methods { body, attributes } => {
                            let (is_static, access) = Self::parse_method_attrs(attributes);
                            for s in body {
                                if let runmat_hir::HirStmt::Function { name: mname, .. } = s { methods.push((mname.clone(), mname.clone(), is_static, access.clone())); }
                            }
                        }
                        _ => {}
                    }
                }
                self.emit(Instr::RegisterClass { name: name.clone(), super_class: super_class.clone(), properties: props, methods });
            }
            HirStmt::MultiAssign(vars, expr, _) => {
                // Compile RHS once; if function call or value, arrange to extract multiple
                match &expr.kind {
                    HirExprKind::FuncCall(name, args) => {
                        if self.functions.contains_key(name) {
                            for arg in args { self.compile_expr(arg)?; }
                            // Emit multi-call to request N outputs
                            self.emit(Instr::CallFunctionMulti(name.clone(), args.len(), vars.len()));
                            // Store outputs in order
                            for (_i, var) in vars.iter().enumerate().rev() {
                                if let Some(v) = var { self.emit(Instr::StoreVar(v.0)); } else { self.emit(Instr::Pop); }
                            }
                        } else {
                            // Builtin or unknown: treat as single return value
                            for arg in args { self.compile_expr(arg)?; }
                            self.emit(Instr::CallBuiltinMulti(name.clone(), args.len(), vars.len()));
                            for (_i, var) in vars.iter().enumerate().rev() { if let Some(v) = var { self.emit(Instr::StoreVar(v.0)); } else { self.emit(Instr::Pop); } }
                        }
                    }
                    HirExprKind::IndexCell(base, indices) => {
                        // Support comma-list expansion from cell indexing: [a,b,...] = C{idx}
                        self.compile_expr(base)?;
                        for index in indices { self.compile_expr(index)?; }
                        // Expand into N outputs
                        self.emit(Instr::IndexCellExpand(indices.len(), vars.len()));
                        for (_i, var) in vars.iter().enumerate().rev() { if let Some(v) = var { self.emit(Instr::StoreVar(v.0)); } else { self.emit(Instr::Pop); } }
                    }
                    _ => {
                        // Non-call: assign expr to first non-placeholder, zeros to remaining non-placeholders
                        let first_real = vars.iter().position(|v| v.is_some());
                        if let Some(first_idx) = first_real {
                            self.compile_expr(expr)?;
                            if let Some(Some(first_var)) = vars.get(first_idx) { self.emit(Instr::StoreVar(first_var.0)); }
                        }
                        for (i, var) in vars.iter().enumerate() {
                            if Some(i) == first_real { continue; }
                            if let Some(v) = var { self.emit(Instr::LoadConst(0.0)); self.emit(Instr::StoreVar(v.0)); }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn compile_expr(&mut self, expr: &HirExpr) -> Result<(), String> {
        match &expr.kind {
            HirExprKind::Number(n) => {
                let val: f64 = n.parse().map_err(|_| "invalid number")?;
                self.emit(Instr::LoadConst(val));
            }
            HirExprKind::String(s) => {
                let clean_string = if s.starts_with('\'') && s.ends_with('\'') { s[1..s.len() - 1].to_string() } else { s.clone() };
                self.emit(Instr::LoadString(clean_string));
            }
            HirExprKind::Var(id) => { self.emit(Instr::LoadVar(id.0)); }
            HirExprKind::Constant(name) => {
                let constants = runmat_builtins::constants();
                if let Some(constant) = constants.iter().find(|c| c.name == name) {
                    if let runmat_builtins::Value::Num(val) = &constant.value { self.emit(Instr::LoadConst(*val)); } else { return Err(format!("Constant {name} is not a number")); }
                } else { return Err(format!("Unknown constant: {name}")); }
            }
            HirExprKind::Unary(op, e) => {
                self.compile_expr(e)?;
                match op {
                    runmat_parser::UnOp::Plus => {}
                    runmat_parser::UnOp::Minus => { self.emit(Instr::Neg); }
                    runmat_parser::UnOp::Transpose => { self.emit(Instr::Transpose); }
                    runmat_parser::UnOp::NonConjugateTranspose => { self.emit(Instr::Transpose); }
                    runmat_parser::UnOp::Not => {
                        // Simple lowering: x -> (x == 0)
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::Equal);
                    }
                }
            }
            HirExprKind::Binary(a, op, b) => {
                use runmat_parser::BinOp;
                match op {
                    BinOp::AndAnd => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let j_false = self.emit(Instr::JumpIfFalse(usize::MAX));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let end = self.emit(Instr::Jump(usize::MAX));
                        let after_cond = self.instructions.len();
                        self.patch(j_false, Instr::JumpIfFalse(after_cond));
                        self.emit(Instr::LoadConst(0.0));
                        let end_pc = self.instructions.len();
                        self.patch(end, Instr::Jump(end_pc));
                    }
                    BinOp::OrOr => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let j_true = self.emit(Instr::JumpIfFalse(usize::MAX));
                        self.emit(Instr::LoadConst(1.0));
                        let end = self.emit(Instr::Jump(usize::MAX));
                        let after_check = self.instructions.len();
                        self.patch(j_true, Instr::JumpIfFalse(after_check));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::NotEqual);
                        let end_pc = self.instructions.len();
                        self.patch(end, Instr::Jump(end_pc));
                    }
                    BinOp::BitAnd => {
                        self.compile_expr(a)?; self.emit(Instr::LoadConst(0.0)); self.emit(Instr::NotEqual);
                        self.compile_expr(b)?; self.emit(Instr::LoadConst(0.0)); self.emit(Instr::NotEqual);
                        self.emit(Instr::Mul);
                    }
                    BinOp::BitOr => {
                        self.compile_expr(a)?; self.emit(Instr::LoadConst(0.0)); self.emit(Instr::NotEqual);
                        self.compile_expr(b)?; self.emit(Instr::LoadConst(0.0)); self.emit(Instr::NotEqual);
                        self.emit(Instr::Add); self.emit(Instr::LoadConst(0.0)); self.emit(Instr::NotEqual);
                    }
                    _ => {
                        self.compile_expr(a)?; self.compile_expr(b)?;
                        match op {
                            BinOp::Add => { self.emit(Instr::Add); }
                            BinOp::Sub => { self.emit(Instr::Sub); }
                            BinOp::Mul => { self.emit(Instr::Mul); }
                            BinOp::Div | BinOp::LeftDiv => { self.emit(Instr::Div); }
                            BinOp::Pow => { self.emit(Instr::Pow); }
                            BinOp::ElemMul => { self.emit(Instr::ElemMul); }
                            BinOp::ElemDiv | BinOp::ElemLeftDiv => { self.emit(Instr::ElemDiv); }
                            BinOp::ElemPow => { self.emit(Instr::ElemPow); }
                            BinOp::Equal => { self.emit(Instr::Equal); }
                            BinOp::NotEqual => { self.emit(Instr::NotEqual); }
                            BinOp::Less => { self.emit(Instr::Less); }
                            BinOp::LessEqual => { self.emit(Instr::LessEqual); }
                            BinOp::Greater => { self.emit(Instr::Greater); }
                            BinOp::GreaterEqual => { self.emit(Instr::GreaterEqual); }
                            BinOp::Colon => { return Err("colon operator not supported".into()); }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            HirExprKind::Range(start, step, end) => {
                self.compile_expr(start)?;
                if let Some(step) = step { self.compile_expr(step)?; self.compile_expr(end)?; self.emit(Instr::CreateRange(true)); } else { self.compile_expr(end)?; self.emit(Instr::CreateRange(false)); }
            }
            HirExprKind::FuncCall(name, args) => {
                // Special lowering for getmethod(obj, 'name') into LoadMethod
                if name == "getmethod" && args.len() == 2 {
                    self.compile_expr(&args[0])?;
                    if let HirExprKind::String(s) = &args[1].kind {
                        let method = if s.starts_with('\'') && s.ends_with('\'') { s[1..s.len()-1].to_string() } else { s.clone() };
                        self.emit(Instr::LoadMethod(method));
                        return Ok(());
                    }
                }
                // Lower feval(f, args...) specially when name == "feval"
                if name == "feval" && !args.is_empty() {
                    // Compile f then each arg
                    for arg in args { self.compile_expr(arg)?; }
                    self.emit(Instr::CallFeval(args.len()-1));
                    return Ok(());
                }
                // Detect all arguments that are cell indexing expressions for expansion
                let has_any_expand = args.iter().any(|e| matches!(e.kind, HirExprKind::IndexCell(_, _)));
                if self.functions.contains_key(name) {
                    if has_any_expand {
                        // Build ArgSpec list and push stack values in proper order
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1 && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec { is_expand: true, num_indices: 0, expand_all: true });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec { is_expand: true, num_indices: indices.len(), expand_all: false });
                                    self.compile_expr(base)?; for i in indices { self.compile_expr(i)?; }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec { is_expand: false, num_indices: 0, expand_all: false });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit(Instr::CallFunctionExpandMulti(name.clone(), specs));
                    } else {
                        for arg in args { self.compile_expr(arg)?; }
                        self.emit(Instr::CallFunction(name.clone(), args.len()));
                    }
                } else {
                    // Attempt compile-time import resolution for builtins
                    let mut resolved = name.clone();
                    if !runmat_builtins::builtin_functions().iter().any(|b| b.name == resolved) {
                        for (path, wildcard) in &self.imports {
                            if !*wildcard { continue; }
                            if path.is_empty() { continue; }
                            let mut qual = String::new();
                            for (i, part) in path.iter().enumerate() { if i>0 { qual.push('.'); } qual.push_str(part); }
                            qual.push('.'); qual.push_str(name);
                            if runmat_builtins::builtin_functions().iter().any(|b| b.name == qual) {
                                resolved = qual;
                                break;
                            }
                        }
                    }
                    if has_any_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1 && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec { is_expand: true, num_indices: 0, expand_all: true });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec { is_expand: true, num_indices: indices.len(), expand_all: false });
                                    self.compile_expr(base)?; for i in indices { self.compile_expr(i)?; }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec { is_expand: false, num_indices: 0, expand_all: false });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit(Instr::CallBuiltinExpandMulti(resolved, specs));
                    } else {
                        for arg in args { self.compile_expr(arg)?; }
                        self.emit(Instr::CallBuiltin(resolved, args.len()));
                    }
                }
            }
            HirExprKind::Tensor(matrix_data) | HirExprKind::Cell(matrix_data) => {
                let rows = matrix_data.len();
                let has_non_literals = matrix_data.iter().any(|row| row.iter().any(|expr| {
                    !matches!(expr.kind, HirExprKind::Number(_) | HirExprKind::String(_) | HirExprKind::Constant(_))
                }));
                if has_non_literals {
                    for row in matrix_data { for element in row { self.compile_expr(element)?; } }
                    let row_lengths: Vec<usize> = matrix_data.iter().map(|row| row.len()).collect();
                    for &row_len in &row_lengths { self.emit(Instr::LoadConst(row_len as f64)); }
                    if matches!(expr.kind, HirExprKind::Cell(_)) {
                        // For 2D cells, we know rows and row lengths; emit 2D version when rectangular
                        let rectangular = row_lengths.iter().all(|&c| c == row_lengths[0]);
                        if rectangular {
                            let cols = if rows > 0 { row_lengths[0] } else { 0 };
                            self.emit(Instr::CreateCell2D(rows, cols));
                        } else {
                            // Ragged cells: fall back to 1D create with total count and row-major ordering
                            let total: usize = row_lengths.iter().sum();
                            self.emit(Instr::CreateCell2D(1, total));
                        }
                    } else {
                        self.emit(Instr::CreateMatrixDynamic(rows));
                    }
                } else {
                    let cols = if rows > 0 { matrix_data[0].len() } else { 0 };
                    for row in matrix_data { for element in row { self.compile_expr(element)?; } }
                    if matches!(expr.kind, HirExprKind::Cell(_)) {
                        self.emit(Instr::CreateCell2D(rows, cols));
                    } else {
                        self.emit(Instr::CreateMatrix(rows, cols));
                    }
                }
            }
            HirExprKind::Index(base, indices) => {
                let has_colon = indices.iter().any(|e| matches!(e.kind, HirExprKind::Colon));
                let has_end = indices.iter().any(|e| matches!(e.kind, HirExprKind::End));
                let has_vector = indices.iter().any(|e| matches!(e.kind, HirExprKind::Range(_,_,_) | HirExprKind::Tensor(_)) || matches!(e.ty, runmat_hir::Type::Matrix{..}));
                if has_colon || has_vector || has_end || indices.len() > 2 {
                    // Push base first, then numeric indices in order; compute colon mask
                    self.compile_expr(base)?;
                    let mut colon_mask: u32 = 0;
                    let mut end_mask: u32 = 0;
                    let mut numeric_count = 0usize;
                    for (dim, index) in indices.iter().enumerate() {
                        if matches!(index.kind, HirExprKind::Colon) {
                            colon_mask |= 1u32 << dim;
                        } else if matches!(index.kind, HirExprKind::End) {
                            end_mask |= 1u32 << dim;
                        } else {
                            self.compile_expr(index)?;
                            numeric_count += 1;
                        }
                    }
                    self.emit(Instr::IndexSlice(indices.len(), numeric_count, colon_mask, end_mask));
                } else {
                    self.compile_expr(base)?;
                    for index in indices { self.compile_expr(index)?; }
                    self.emit(Instr::Index(indices.len()));
                }
            }
            HirExprKind::IndexCell(base, indices) => {
                self.compile_expr(base)?;
                for index in indices { self.compile_expr(index)?; }
                self.emit(Instr::IndexCell(indices.len()));
            }
            HirExprKind::Colon => { return Err("colon expression not supported".into()); }
            HirExprKind::End => { self.emit(Instr::LoadConst(-0.0)); /* placeholder, resolved via end_mask in IndexSlice */ }
            HirExprKind::Member(base, field) => {
                // If base is a known class ref literal (string via classref builtin), static access
                // Otherwise, instance member
                match &base.kind {
                    HirExprKind::FuncCall(name, args) if name == "classref" && args.len() == 1 => {
                        if let HirExprKind::String(cls) = &args[0].kind {
                            let cls_name = if cls.starts_with('\'') && cls.ends_with('\'') { cls[1..cls.len()-1].to_string() } else { cls.clone() };
                            self.emit(Instr::LoadStaticProperty(cls_name, field.clone()));
                        } else {
                            self.compile_expr(base)?;
                            self.emit(Instr::LoadMember(field.clone()));
                        }
                    }
                    _ => {
                        // Default to instance property access; subsref overloading is handled at runtime via call_method if needed
                        self.compile_expr(base)?;
                        self.emit(Instr::LoadMember(field.clone()));
                    }
                }
            }
            HirExprKind::MethodCall(base, method, args) => {
                match &base.kind {
                    HirExprKind::FuncCall(name, bargs) if name == "classref" && bargs.len() == 1 => {
                        if let HirExprKind::String(cls) = &bargs[0].kind {
                            let cls_name = if cls.starts_with('\'') && cls.ends_with('\'') { cls[1..cls.len()-1].to_string() } else { cls.clone() };
                            for arg in args { self.compile_expr(arg)?; }
                            self.emit(Instr::CallStaticMethod(cls_name, method.clone(), args.len()));
                        } else {
                            self.compile_expr(base)?;
                            for arg in args { self.compile_expr(arg)?; }
                            self.emit(Instr::CallMethod(method.clone(), args.len()));
                        }
                    }
                    _ => {
                        self.compile_expr(base)?;
                        for arg in args { self.compile_expr(arg)?; }
                        self.emit(Instr::CallMethod(method.clone(), args.len()));
                    }
                }
            }
            HirExprKind::AnonFunc { params, body } => {
                // Collect free variables in body (in order of first appearance)
                use std::collections::{HashSet, HashMap};
                let mut seen: HashSet<runmat_hir::VarId> = HashSet::new();
                let mut captures_order: Vec<runmat_hir::VarId> = Vec::new();
                let bound: HashSet<runmat_hir::VarId> = params.iter().cloned().collect();
                self.collect_free_vars(body, &bound, &mut seen, &mut captures_order);

                // Build placeholder VarIds for captures and parameters
                let capture_count = captures_order.len();
                let mut placeholder_params: Vec<runmat_hir::VarId> = Vec::with_capacity(capture_count + params.len());
                for i in 0..capture_count { placeholder_params.push(runmat_hir::VarId(i)); }
                for j in 0..params.len() { placeholder_params.push(runmat_hir::VarId(capture_count + j)); }
                let output_id = runmat_hir::VarId(capture_count + params.len());

                // Remap body vars: free vars -> capture placeholders; param vars -> shifted placeholders
                let mut var_map: HashMap<runmat_hir::VarId, runmat_hir::VarId> = HashMap::new();
                for (i, old) in captures_order.iter().enumerate() { var_map.insert(*old, runmat_hir::VarId(i)); }
                for (j, old) in params.iter().enumerate() { var_map.insert(*old, runmat_hir::VarId(capture_count + j)); }
                let remapped_body = runmat_hir::remapping::remap_expr(body, &var_map);
                let func_body = vec![ runmat_hir::HirStmt::Assign(output_id, remapped_body, true) ];

                // Synthesize function name and register
                let synthesized = format!("__anon_{}", self.functions.len());
                let user_func = UserFunction { name: synthesized.clone(), params: placeholder_params, outputs: vec![output_id], body: func_body, local_var_count: capture_count + params.len() + 1 };
                self.functions.insert(synthesized.clone(), user_func);

                // Emit capture values on stack then create closure
                for old in &captures_order { self.emit(Instr::LoadVar(old.0)); }
                self.emit(Instr::CreateClosure(synthesized, capture_count));
            }
            HirExprKind::FuncHandle(name) => {
                self.emit(Instr::LoadString(name.clone()));
                self.emit(Instr::CallBuiltin("make_handle".to_string(), 1));
            }
            HirExprKind::MetaClass(name) => { self.emit(Instr::LoadString(name.clone())); }
        }
        Ok(())
    }
}



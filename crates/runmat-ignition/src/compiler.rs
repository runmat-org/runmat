use crate::functions::UserFunction;
use crate::instr::Instr;
use once_cell::sync::OnceCell;
use runmat_builtins::Type;
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
    pub var_types: Vec<Type>,
}

struct StochasticEvolutionPlan<'a> {
    state: runmat_hir::VarId,
    drift: &'a HirExpr,
    scale: &'a HirExpr,
    steps: &'a HirExpr,
}

fn expr_is_one(expr: &HirExpr) -> bool {
    parse_number(expr)
        .map(|v| (v - 1.0).abs() < 1e-9)
        .unwrap_or(false)
}

fn parse_number(expr: &HirExpr) -> Option<f64> {
    if let runmat_hir::HirExprKind::Number(raw) = &expr.kind {
        raw.parse::<f64>().ok()
    } else {
        None
    }
}

fn stochastic_evolution_disabled() -> bool {
    static DISABLE: OnceCell<bool> = OnceCell::new();
    *DISABLE.get_or_init(|| {
        std::env::var("RUNMAT_DISABLE_STOCHASTIC_EVOLUTION")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

fn is_randn_call(expr: &HirExpr) -> bool {
    match &expr.kind {
        runmat_hir::HirExprKind::FuncCall(name, _) => name.eq_ignore_ascii_case("randn"),
        _ => false,
    }
}

fn matches_var(expr: &HirExpr, var: runmat_hir::VarId) -> bool {
    matches!(expr.kind, runmat_hir::HirExprKind::Var(id) if id == var)
}

fn extract_drift_and_scale(
    expr: &HirExpr,
    state_var: runmat_hir::VarId,
    z_var: runmat_hir::VarId,
) -> Option<(&HirExpr, &HirExpr)> {
    use runmat_hir::HirExprKind as EK;
    use runmat_parser::BinOp;

    let (maybe_state_side, maybe_exp_side) = match &expr.kind {
        EK::Binary(lhs, BinOp::ElemMul, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    let exp_side = if matches_var(maybe_state_side, state_var) && is_exp_call(maybe_exp_side) {
        maybe_exp_side
    } else if matches_var(maybe_exp_side, state_var) && is_exp_call(maybe_state_side) {
        maybe_state_side
    } else {
        return None;
    };

    let exp_arg = match &exp_side.kind {
        EK::FuncCall(name, args) if name.eq_ignore_ascii_case("exp") && args.len() == 1 => &args[0],
        _ => return None,
    };

    match &exp_arg.kind {
        EK::Binary(lhs, BinOp::Add, rhs) => {
            if let Some(scale_expr) = extract_scale_term(lhs, z_var) {
                Some((rhs.as_ref(), scale_expr))
            } else if let Some(scale_expr) = extract_scale_term(rhs, z_var) {
                Some((lhs.as_ref(), scale_expr))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_scale_term(expr: &HirExpr, z_var: runmat_hir::VarId) -> Option<&HirExpr> {
    use runmat_hir::HirExprKind as EK;
    use runmat_parser::BinOp;

    match &expr.kind {
        EK::Binary(lhs, BinOp::ElemMul, rhs) => {
            if matches_var(lhs, z_var) {
                Some(rhs.as_ref())
            } else if matches_var(rhs, z_var) {
                Some(lhs.as_ref())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_exp_call(expr: &HirExpr) -> bool {
    matches!(
        &expr.kind,
        runmat_hir::HirExprKind::FuncCall(name, _) if name.eq_ignore_ascii_case("exp")
    )
}

impl Compiler {
    fn compile_stochastic_evolution(
        &mut self,
        plan: StochasticEvolutionPlan<'_>,
    ) -> Result<(), String> {
        self.emit(Instr::LoadVar(plan.state.0));
        self.compile_expr(plan.drift)?;
        self.compile_expr(plan.scale)?;
        self.compile_expr(plan.steps)?;
        self.emit(Instr::StochasticEvolution);
        self.emit(Instr::StoreVar(plan.state.0));
        Ok(())
    }

    fn detect_stochastic_evolution<'a>(
        &self,
        expr: &'a HirExpr,
        body: &'a [HirStmt],
    ) -> Option<StochasticEvolutionPlan<'a>> {
        if stochastic_evolution_disabled() {
            return None;
        }
        use runmat_hir::HirExprKind as EK;
        match &expr.kind {
            EK::Range(start, step, end) => {
                if !expr_is_one(start) {
                    return None;
                }
                if let Some(step_expr) = step {
                    if !expr_is_one(step_expr) {
                        return None;
                    }
                }
                if body.len() != 2 {
                    return None;
                }
                let (z_var, randn_expr) = match &body[0] {
                    HirStmt::Assign(var, expr, _) => (*var, expr),
                    _ => return None,
                };
                if !is_randn_call(randn_expr) {
                    return None;
                }
                let (state_var, update_expr) = match &body[1] {
                    HirStmt::Assign(var, expr, _) => (*var, expr),
                    _ => return None,
                };
                let (drift, scale) = extract_drift_and_scale(update_expr, state_var, z_var)?;
                Some(StochasticEvolutionPlan {
                    state: state_var,
                    drift,
                    scale,
                    steps: end,
                })
            }
            _ => None,
        }
    }
    fn attr_access_from_str(s: &str) -> runmat_builtins::Access {
        match s.to_ascii_lowercase().as_str() {
            "private" => runmat_builtins::Access::Private,
            _ => runmat_builtins::Access::Public,
        }
    }
    fn parse_prop_attrs(attrs: &Vec<runmat_parser::Attr>) -> (bool, bool, String, String) {
        // Defaults
        let mut is_static = false;
        let mut is_dependent = false;
        let mut get_acc = runmat_builtins::Access::Public;
        let mut set_acc = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") {
                is_static = true;
            }
            if a.name.eq_ignore_ascii_case("Dependent") {
                is_dependent = true;
            }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value {
                    let acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                    get_acc = acc.clone();
                    set_acc = acc;
                }
            }
            if a.name.eq_ignore_ascii_case("GetAccess") {
                if let Some(v) = &a.value {
                    get_acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
            if a.name.eq_ignore_ascii_case("SetAccess") {
                if let Some(v) = &a.value {
                    set_acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
        }
        let gs = match get_acc {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        let ss = match set_acc {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        (is_static, is_dependent, gs, ss)
    }
    fn parse_method_attrs(attrs: &Vec<runmat_parser::Attr>) -> (bool, String) {
        let mut is_static = false;
        let mut access = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") {
                is_static = true;
            }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value {
                    access = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
        }
        let acc_str = match access {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        (is_static, acc_str)
    }
    fn expr_contains_end(expr: &runmat_hir::HirExpr) -> bool {
        use runmat_hir::HirExprKind as K;
        match &expr.kind {
            K::End => true,
            K::Unary(_, e) => Self::expr_contains_end(e),
            K::Binary(a, _, b) => Self::expr_contains_end(a) || Self::expr_contains_end(b),
            K::Tensor(rows) | K::Cell(rows) => rows
                .iter()
                .flat_map(|r| r.iter())
                .any(Self::expr_contains_end),
            K::Index(base, idxs) | K::IndexCell(base, idxs) => {
                if Self::expr_contains_end(base) {
                    return true;
                }
                idxs.iter().any(Self::expr_contains_end)
            }
            _ => false,
        }
    }
    #[allow(clippy::only_used_in_recursion)]
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
            K::FuncCall(_, args) | K::MethodCall(_, _, args) => {
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
                HirExprKind::FuncCall(_, args) | HirExprKind::MethodCall(_, _, args) => {
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::Member(base, _) => visit_expr(base, max),
                HirExprKind::MemberDynamic(base, name) => {
                    visit_expr(base, max);
                    visit_expr(name, max);
                }
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
                    HirStmt::If {
                        cond,
                        then_body,
                        elseif_blocks,
                        else_body,
                    } => {
                        visit_expr(cond, max);
                        visit_stmts(then_body, max);
                        for (cond, body) in elseif_blocks {
                            visit_expr(cond, max);
                            visit_stmts(body, max);
                        }
                        if let Some(body) = else_body {
                            visit_stmts(body, max);
                        }
                    }
                    HirStmt::While { cond, body } => {
                        visit_expr(cond, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::For { var, expr, body } => {
                        if var.0 + 1 > *max {
                            *max = var.0 + 1;
                        }
                        visit_expr(expr, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::Switch {
                        expr,
                        cases,
                        otherwise,
                    } => {
                        visit_expr(expr, max);
                        for (c, b) in cases {
                            visit_expr(c, max);
                            visit_stmts(b, max);
                        }
                        if let Some(b) = otherwise {
                            visit_stmts(b, max);
                        }
                    }
                    HirStmt::TryCatch {
                        try_body,
                        catch_var,
                        catch_body,
                    } => {
                        if let Some(v) = catch_var {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                        visit_stmts(try_body, max);
                        visit_stmts(catch_body, max);
                    }
                    HirStmt::Global(vars) | HirStmt::Persistent(vars) => {
                        for (v, _name) in vars {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                    }
                    HirStmt::AssignLValue(_, expr, _) => visit_expr(expr, max),
                    HirStmt::MultiAssign(vars, expr, _) => {
                        for v in vars.iter().flatten() {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::Function { .. }
                    | HirStmt::ClassDef { .. }
                    | HirStmt::Import { .. }
                    | HirStmt::Break
                    | HirStmt::Continue => {}
                }
            }
        }

        visit_stmts(&prog.body, &mut max_var);
        let mut var_types = prog.var_types.clone();
        if var_types.len() < max_var {
            var_types.resize(max_var, Type::Unknown);
        }
        Self {
            instructions: Vec::new(),
            var_count: max_var,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
            imports: Vec::new(),
            var_types,
        }
    }

    fn ensure_var(&mut self, id: usize) {
        if id + 1 > self.var_count {
            self.var_count = id + 1;
        }
        while self.var_types.len() <= id {
            self.var_types.push(Type::Unknown);
        }
    }

    fn alloc_temp(&mut self) -> usize {
        let id = self.var_count;
        self.var_count += 1;
        if self.var_types.len() <= id {
            self.var_types.push(Type::Unknown);
        }
        id
    }

    pub fn emit(&mut self, instr: Instr) -> usize {
        match &instr {
            Instr::LoadVar(id) | Instr::StoreVar(id) => self.ensure_var(*id),
            _ => {}
        }
        let pc = self.instructions.len();
        self.instructions.push(instr);
        pc
    }

    pub fn patch(&mut self, idx: usize, instr: Instr) {
        self.instructions[idx] = instr;
    }

    pub fn compile_program(&mut self, prog: &HirProgram) -> Result<(), String> {
        // Validate imports early for duplicate/specific-name ambiguities
        runmat_hir::validate_imports(prog)?;
        // Validate class definitions for attribute correctness and name conflicts
        runmat_hir::validate_classdefs(prog)?;
        // Pre-collect imports (both wildcard and specific) for name resolution
        for stmt in &prog.body {
            if let HirStmt::Import { path, wildcard } = stmt {
                self.imports.push((path.clone(), *wildcard));
                self.emit(Instr::RegisterImport {
                    path: path.clone(),
                    wildcard: *wildcard,
                });
            }
            if let HirStmt::Global(vars) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            if let HirStmt::Persistent(vars) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
        }
        for stmt in &prog.body {
            if !matches!(
                stmt,
                HirStmt::Import { .. } | HirStmt::Global(_) | HirStmt::Persistent(_)
            ) {
                self.compile_stmt(stmt)?;
            }
        }
        Ok(())
    }

    pub fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), String> {
        match stmt {
            HirStmt::ExprStmt(expr, _) => {
                self.compile_expr(expr)?;
                self.emit(Instr::Pop);
            }
            HirStmt::Assign(id, expr, _) => {
                self.compile_expr(expr)?;
                self.emit(Instr::StoreVar(id.0));
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                self.compile_expr(cond)?;
                let mut last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                for s in then_body {
                    self.compile_stmt(s)?;
                }
                let mut end_jumps = Vec::new();
                end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                for (c, b) in elseif_blocks {
                    self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                    self.compile_expr(c)?;
                    last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in b {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                }
                self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                if let Some(body) = else_body {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }
                let end = self.instructions.len();
                for j in end_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::While { cond, body } => {
                let start = self.instructions.len();
                self.compile_expr(cond)?;
                let jump_end = self.emit(Instr::JumpIfFalse(usize::MAX));
                let labels = LoopLabels {
                    break_jumps: Vec::new(),
                    continue_jumps: Vec::new(),
                };
                self.loop_stack.push(labels);
                for s in body {
                    self.compile_stmt(s)?;
                }
                let labels = self.loop_stack.pop().unwrap();
                for j in labels.continue_jumps {
                    self.patch(j, Instr::Jump(start));
                }
                self.emit(Instr::Jump(start));
                let end = self.instructions.len();
                self.patch(jump_end, Instr::JumpIfFalse(end));
                for j in labels.break_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::For { var, expr, body } => {
                if let Some(plan) = self.detect_stochastic_evolution(expr, body) {
                    self.compile_stochastic_evolution(plan)?;
                    return Ok(());
                }
                if let HirExprKind::Range(start, step, end) = &expr.kind {
                    // Initialize loop variable, end, and step
                    self.compile_expr(start)?;
                    self.emit(Instr::StoreVar(var.0));
                    self.compile_expr(end)?;
                    let end_var = self.alloc_temp();
                    self.emit(Instr::StoreVar(end_var));
                    let step_var = self.alloc_temp();
                    if let Some(step_expr) = step {
                        self.compile_expr(step_expr)?;
                        self.emit(Instr::StoreVar(step_var));
                    } else {
                        self.emit(Instr::LoadConst(1.0));
                        self.emit(Instr::StoreVar(step_var));
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
                    self.loop_stack.push(LoopLabels {
                        break_jumps: Vec::new(),
                        continue_jumps: Vec::new(),
                    });
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    let labels = self.loop_stack.pop().unwrap();
                    for j in labels.continue_jumps {
                        self.patch(j, Instr::Jump(self.instructions.len()));
                    }
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
                    for j in labels.break_jumps {
                        self.patch(j, Instr::Jump(end_pc));
                    }
                } else {
                    return Err("for loop expects range".into());
                }
            }
            HirStmt::Break => {
                if let Some(labels) = self.loop_stack.last_mut() {
                    let idx = self.instructions.len();
                    self.instructions.push(Instr::Jump(usize::MAX));
                    labels.break_jumps.push(idx);
                } else {
                    return Err("break outside loop".into());
                }
            }
            HirStmt::Continue => {
                if let Some(labels) = self.loop_stack.last_mut() {
                    let idx = self.instructions.len();
                    self.instructions.push(Instr::Jump(usize::MAX));
                    labels.continue_jumps.push(idx);
                } else {
                    return Err("continue outside loop".into());
                }
            }
            HirStmt::Return => {
                self.emit(Instr::Return);
            }
            HirStmt::Function {
                name,
                params,
                outputs,
                body,
                has_varargin,
                has_varargout,
            } => {
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
                        HirExprKind::Index(base, indices)
                        | HirExprKind::IndexCell(base, indices) => {
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
                        HirStmt::ExprStmt(expr, _) => visit_expr_for_vars(expr, max),
                        HirStmt::Assign(id, expr, _) => {
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
                        HirStmt::While { cond, body } => {
                            visit_expr_for_vars(cond, max);
                            for stmt in body {
                                visit_stmt_for_vars(stmt, max);
                            }
                        }
                        HirStmt::For { var, expr, body } => {
                            if var.0 + 1 > *max {
                                *max = var.0 + 1;
                            }
                            visit_expr_for_vars(expr, max);
                            for stmt in body {
                                visit_stmt_for_vars(stmt, max);
                            }
                        }
                        HirStmt::Break | HirStmt::Continue | HirStmt::Return => {}
                        HirStmt::Switch {
                            expr,
                            cases,
                            otherwise,
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
                        HirStmt::Global(vars) | HirStmt::Persistent(vars) => {
                            for (v, _name) in vars {
                                if v.0 + 1 > *max {
                                    *max = v.0 + 1;
                                }
                            }
                        }
                        HirStmt::AssignLValue(_, expr, _) => visit_expr_for_vars(expr, max),
                        HirStmt::MultiAssign(vars, expr, _) => {
                            for v in vars.iter().flatten() {
                                if v.0 + 1 > *max {
                                    *max = v.0 + 1;
                                }
                            }
                            visit_expr_for_vars(expr, max);
                        }
                        HirStmt::Function { .. }
                        | HirStmt::ClassDef { .. }
                        | HirStmt::Import { .. } => {}
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
                    name: name.clone(),
                    params: params.clone(),
                    outputs: outputs.clone(),
                    body: body.clone(),
                    local_var_count: max_local_var,
                    has_varargin: *has_varargin,
                    has_varargout: *has_varargout,
                    var_types: func_var_types,
                };
                self.functions.insert(name.clone(), user_func);
            }
            HirStmt::Switch {
                expr,
                cases,
                otherwise,
            } => {
                let temp_id = self.alloc_temp();
                self.compile_expr(expr)?;
                self.emit(Instr::StoreVar(temp_id));
                let mut end_jumps: Vec<usize> = Vec::new();
                let mut next_case_jump_to_here: Option<usize> = None;
                for (case_expr, body) in cases {
                    if let Some(j) = next_case_jump_to_here.take() {
                        self.patch(j, Instr::JumpIfFalse(self.instructions.len()));
                    }
                    self.emit(Instr::LoadVar(temp_id));
                    self.compile_expr(case_expr)?;
                    self.emit(Instr::Equal);
                    let jmp = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                    next_case_jump_to_here = Some(jmp);
                }
                let otherwise_start = self.instructions.len();
                if let Some(j) = next_case_jump_to_here.take() {
                    self.patch(j, Instr::JumpIfFalse(otherwise_start));
                }
                if let Some(body) = otherwise {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }
                let end = self.instructions.len();
                for j in end_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
            } => {
                // Reserve slot for EnterTry with placeholder
                let enter_idx = self.emit(Instr::EnterTry(usize::MAX, catch_var.map(|v| v.0)));
                // Compile try body
                for s in try_body {
                    self.compile_stmt(s)?;
                }
                // On normal completion, pop try frame and jump past catch
                self.emit(Instr::PopTry);
                let jmp_end = self.emit(Instr::Jump(usize::MAX));
                // Catch block starts here
                let catch_pc = self.instructions.len();
                // Patch EnterTry with catch_pc
                self.patch(enter_idx, Instr::EnterTry(catch_pc, catch_var.map(|v| v.0)));
                // Compile catch body
                for s in catch_body {
                    self.compile_stmt(s)?;
                }
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
                            let has_colon = indices
                                .iter()
                                .any(|e| matches!(e.kind, runmat_hir::HirExprKind::Colon));
                            let has_end = indices
                                .iter()
                                .any(|e| matches!(e.kind, runmat_hir::HirExprKind::End));
                            let has_vector = indices.iter().any(|e| {
                                matches!(
                                    e.kind,
                                    HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_)
                                ) || matches!(e.ty, runmat_hir::Type::Tensor { .. })
                            });
                            if has_colon || has_end || has_vector || indices.len() > 2 {
                                let mut colon_mask: u32 = 0;
                                let mut end_mask: u32 = 0;
                                let mut numeric_count = 0usize;
                                let mut end_offsets: Vec<(usize, i64)> = Vec::new();
                                let mut lowered_range_end = false;
                                for (dim, index) in indices.iter().enumerate() {
                                    if matches!(index.kind, runmat_hir::HirExprKind::Colon) {
                                        colon_mask |= 1u32 << dim;
                                    } else if matches!(index.kind, runmat_hir::HirExprKind::End) {
                                        end_mask |= 1u32 << dim;
                                    } else {
                                        // If this index is a Range whose end references End (with or without offset),
                                        // skip compiling it here; it will be handled by StoreRangeEnd.
                                        if indices.len() > 1 {
                                            if let runmat_hir::HirExprKind::Range(
                                                _start,
                                                _step,
                                                end,
                                            ) = &index.kind
                                            {
                                                match &end.kind {
                                                    runmat_hir::HirExprKind::End => {
                                                        // offset 0
                                                        end_offsets.push((numeric_count, 0));
                                                        continue;
                                                    }
                                                    runmat_hir::HirExprKind::Binary(
                                                        left,
                                                        op,
                                                        right,
                                                    ) => {
                                                        if matches!(op, runmat_parser::BinOp::Sub)
                                                            && matches!(
                                                                left.kind,
                                                                runmat_hir::HirExprKind::End
                                                            )
                                                        {
                                                            if let runmat_hir::HirExprKind::Number(
                                                                ref s,
                                                            ) = right.kind
                                                            {
                                                                if let Ok(k) = s.parse::<i64>() {
                                                                    end_offsets
                                                                        .push((numeric_count, k));
                                                                } else {
                                                                    end_offsets
                                                                        .push((numeric_count, 0));
                                                                }
                                                            } else {
                                                                end_offsets
                                                                    .push((numeric_count, 0));
                                                            }
                                                            continue;
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                        // Special-case 1-D range with end arithmetic when dims == 1
                                        if indices.len() == 1 {
                                            if let runmat_hir::HirExprKind::Range(
                                                start,
                                                step,
                                                end,
                                            ) = &index.kind
                                            {
                                                if let runmat_hir::HirExprKind::Binary(
                                                    left,
                                                    op,
                                                    right,
                                                ) = &end.kind
                                                {
                                                    if matches!(op, runmat_parser::BinOp::Sub)
                                                        && matches!(
                                                            left.kind,
                                                            runmat_hir::HirExprKind::End
                                                        )
                                                    {
                                                        // Emit StoreSlice1DRangeEnd: base is already on stack
                                                        self.compile_expr(start)?;
                                                        if let Some(st) = step {
                                                            self.compile_expr(st)?;
                                                            self.compile_expr(rhs)?;
                                                            self.emit(Instr::StoreSlice1DRangeEnd { has_step: true, offset: match right.kind { runmat_hir::HirExprKind::Number(ref s) => s.parse::<i64>().unwrap_or(0), _ => 0 } });
                                                        } else {
                                                            self.compile_expr(rhs)?;
                                                            self.emit(Instr::StoreSlice1DRangeEnd { has_step: false, offset: match right.kind { runmat_hir::HirExprKind::Number(ref s) => s.parse::<i64>().unwrap_or(0), _ => 0 } });
                                                        }
                                                        lowered_range_end = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if !lowered_range_end {
                                            self.compile_expr(index)?;
                                            numeric_count += 1;
                                        }
                                    }
                                }
                                if lowered_range_end {
                                    // VM already pushed updated tensor; store back to var
                                    self.emit(Instr::StoreVar(var_id.0));
                                } else {
                                    // Push RHS last so VM pops it first
                                    // Detect any ranges with end arithmetic across dims
                                    let mut has_any_range_end = false;
                                    let mut range_dims: Vec<usize> = Vec::new();
                                    let mut range_has_step: Vec<bool> = Vec::new();
                                    let mut end_offs: Vec<i64> = Vec::new();
                                    for (dim, index) in indices.iter().enumerate() {
                                        if let runmat_hir::HirExprKind::Range(_start, step, end) =
                                            &index.kind
                                        {
                                            match &end.kind {
                                                runmat_hir::HirExprKind::End => {
                                                    has_any_range_end = true;
                                                    range_dims.push(dim);
                                                    range_has_step.push(step.is_some());
                                                    end_offs.push(0);
                                                }
                                                runmat_hir::HirExprKind::Binary(
                                                    left,
                                                    op,
                                                    right,
                                                ) => {
                                                    if matches!(op, runmat_parser::BinOp::Sub)
                                                        && matches!(
                                                            left.kind,
                                                            runmat_hir::HirExprKind::End
                                                        )
                                                    {
                                                        has_any_range_end = true;
                                                        range_dims.push(dim);
                                                        range_has_step.push(step.is_some());
                                                        if let runmat_hir::HirExprKind::Number(
                                                            ref s,
                                                        ) = right.kind
                                                        {
                                                            end_offs.push(
                                                                s.parse::<i64>().unwrap_or(0),
                                                            );
                                                        } else {
                                                            end_offs.push(0);
                                                        }
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    if has_any_range_end {
                                        // Push start[, step] for each range in dim order
                                        for &dim in &range_dims {
                                            if let runmat_hir::HirExprKind::Range(
                                                start,
                                                step,
                                                _end,
                                            ) = &indices[dim].kind
                                            {
                                                self.compile_expr(start)?;
                                                if let Some(st) = step {
                                                    self.compile_expr(st)?;
                                                }
                                            }
                                        }
                                        self.compile_expr(rhs)?;
                                        self.emit(Instr::StoreRangeEnd {
                                            dims: indices.len(),
                                            numeric_count,
                                            colon_mask,
                                            end_mask,
                                            range_dims,
                                            range_has_step,
                                            end_offsets: end_offs,
                                        });
                                    } else {
                                        // Attempt packing of function returns or cell expansion for 1-D slices
                                        let dims_len = indices.len();
                                        let idx_is_scalar = |e: &HirExpr| -> bool {
                                            matches!(
                                                e.kind,
                                                HirExprKind::Number(_) | HirExprKind::End
                                            )
                                        };
                                        let idx_is_vector = |e: &HirExpr| -> bool {
                                            matches!(
                                                e.kind,
                                                HirExprKind::Colon
                                                    | HirExprKind::Range(_, _, _)
                                                    | HirExprKind::Tensor(_)
                                            )
                                        };
                                        let (is_row_slice, is_col_slice) = if dims_len == 2 {
                                            (
                                                idx_is_scalar(&indices[0])
                                                    && idx_is_vector(&indices[1]),
                                                idx_is_vector(&indices[0])
                                                    && idx_is_scalar(&indices[1]),
                                            )
                                        } else {
                                            (false, false)
                                        };
                                        fn const_vec_len(e: &HirExpr) -> Option<usize> {
                                            match &e.kind {
                                                HirExprKind::Number(_) | HirExprKind::End => {
                                                    Some(1)
                                                }
                                                HirExprKind::Tensor(rows) => {
                                                    Some(rows.iter().map(|r| r.len()).sum())
                                                }
                                                HirExprKind::Range(start, step, end) => {
                                                    if let (
                                                        HirExprKind::Number(sa),
                                                        HirExprKind::Number(ea),
                                                    ) = (&start.kind, &end.kind)
                                                    {
                                                        let s: f64 = sa.parse().ok()?;
                                                        let en: f64 = ea.parse().ok()?;
                                                        let st: f64 = if let Some(st) = step {
                                                            if let HirExprKind::Number(x) = &st.kind
                                                            {
                                                                x.parse().ok()?
                                                            } else {
                                                                return None;
                                                            }
                                                        } else {
                                                            1.0
                                                        };
                                                        if st == 0.0 {
                                                            return None;
                                                        }
                                                        let n =
                                                            ((en - s) / st).floor() as isize + 1;
                                                        if n <= 0 {
                                                            Some(0)
                                                        } else {
                                                            Some(n as usize)
                                                        }
                                                    } else {
                                                        None
                                                    }
                                                }
                                                HirExprKind::Colon => None,
                                                _ => None,
                                            }
                                        }
                                        let mut packed = false;
                                        if let HirExprKind::FuncCall(fname, fargs) = &rhs.kind {
                                            if self.functions.contains_key(fname)
                                                && (dims_len == 1 || is_row_slice || is_col_slice)
                                            {
                                                for a in fargs {
                                                    self.compile_expr(a)?;
                                                }
                                                let outc = self
                                                    .functions
                                                    .get(fname)
                                                    .map(|f| f.outputs.len().max(1))
                                                    .unwrap_or(1);
                                                self.emit(Instr::CallFunctionMulti(
                                                    fname.clone(),
                                                    fargs.len(),
                                                    outc,
                                                ));
                                                if dims_len == 1 || is_col_slice {
                                                    self.emit(Instr::PackToCol(outc));
                                                } else {
                                                    self.emit(Instr::PackToRow(outc));
                                                }
                                                packed = true;
                                            }
                                        } else if let HirExprKind::IndexCell(cbase, cidx) =
                                            &rhs.kind
                                        {
                                            // Expand cell into vector matching selected slice length if determinable
                                            let outc = if dims_len == 1 {
                                                const_vec_len(&indices[0])
                                            } else if is_row_slice {
                                                const_vec_len(&indices[1])
                                            } else if is_col_slice {
                                                const_vec_len(&indices[0])
                                            } else {
                                                None
                                            };
                                            if let Some(n) = outc {
                                                self.compile_expr(cbase)?;
                                                // Special case: C{:} => expand all; do not compile colon index
                                                let expand_all = cidx.len() == 1
                                                    && matches!(cidx[0].kind, HirExprKind::Colon);
                                                if expand_all {
                                                    self.emit(Instr::IndexCellExpand(0, n));
                                                } else {
                                                    for i in cidx {
                                                        self.compile_expr(i)?;
                                                    }
                                                    self.emit(Instr::IndexCellExpand(
                                                        cidx.len(),
                                                        n,
                                                    ));
                                                }
                                                if dims_len == 1 || is_col_slice {
                                                    self.emit(Instr::PackToCol(n));
                                                } else {
                                                    self.emit(Instr::PackToRow(n));
                                                }
                                                packed = true;
                                            }
                                        }
                                        if !packed {
                                            self.compile_expr(rhs)?;
                                        }
                                        if end_offsets.is_empty() {
                                            self.emit(Instr::StoreSlice(
                                                indices.len(),
                                                numeric_count,
                                                colon_mask,
                                                end_mask,
                                            ));
                                        } else {
                                            self.emit(Instr::StoreSliceEx(
                                                indices.len(),
                                                numeric_count,
                                                colon_mask,
                                                end_mask,
                                                end_offsets,
                                            ));
                                        }
                                    }
                                    // Store updated base back to variable
                                    self.emit(Instr::StoreVar(var_id.0));
                                }
                            } else {
                                // Pure numeric indexing
                                for index in indices {
                                    self.compile_expr(index)?;
                                }
                                // If RHS is a user function call, request multiple outputs and pack to column for linear targets
                                if let HirExprKind::FuncCall(fname, fargs) = &rhs.kind {
                                    if self.functions.contains_key(fname) && indices.len() == 1 {
                                        for a in fargs {
                                            self.compile_expr(a)?;
                                        }
                                        let outc = self
                                            .functions
                                            .get(fname)
                                            .map(|f| f.outputs.len().max(1))
                                            .unwrap_or(1);
                                        self.emit(Instr::CallFunctionMulti(
                                            fname.clone(),
                                            fargs.len(),
                                            outc,
                                        ));
                                        self.emit(Instr::PackToCol(outc));
                                    } else {
                                        self.compile_expr(rhs)?;
                                    }
                                } else {
                                    self.compile_expr(rhs)?;
                                }
                                self.emit(Instr::StoreIndex(indices.len()));
                                self.emit(Instr::StoreVar(var_id.0));
                            }
                        } else if let runmat_hir::HirExprKind::Member(member_base, field) =
                            &base.kind
                        {
                            // Chain: base is a member access. Evaluate object, load member, update via index, then write member back.
                            // Evaluate object and get member value
                            self.compile_expr(member_base)?;
                            self.emit(Instr::LoadMember(field.clone()));
                            // Decide slice vs numeric
                            let has_colon = indices
                                .iter()
                                .any(|e| matches!(e.kind, runmat_hir::HirExprKind::Colon));
                            let has_end = indices
                                .iter()
                                .any(|e| matches!(e.kind, runmat_hir::HirExprKind::End));
                            let has_vector = indices.iter().any(|e| {
                                matches!(
                                    e.kind,
                                    HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_)
                                ) || matches!(e.ty, runmat_hir::Type::Tensor { .. })
                            });
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
                                self.compile_expr(rhs)?;
                                self.emit(Instr::StoreSlice(
                                    indices.len(),
                                    numeric_count,
                                    colon_mask,
                                    end_mask,
                                ));
                            } else {
                                for index in indices {
                                    self.compile_expr(index)?;
                                }
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
                            return Err(
                                "unsupported lvalue target (index on non-variable/non-member)"
                                    .into(),
                            );
                        }
                    }
                    runmat_hir::HirLValue::IndexCell(base, indices) => {
                        if let runmat_hir::HirExprKind::Var(var_id) = base.kind {
                            self.emit(Instr::LoadVar(var_id.0));
                            for index in indices {
                                self.compile_expr(index)?;
                            }
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreIndexCell(indices.len()));
                            self.emit(Instr::StoreVar(var_id.0));
                        } else if let runmat_hir::HirExprKind::Member(member_base, field) =
                            &base.kind
                        {
                            // Load object, load member, perform cell index store, then write member back to object
                            self.compile_expr(member_base)?;
                            self.emit(Instr::LoadMember(field.clone()));
                            for index in indices {
                                self.compile_expr(index)?;
                            }
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreIndexCell(indices.len()));
                            // Updated member on stack; re-evaluate object, swap, store member
                            self.compile_expr(member_base)?;
                            self.emit(Instr::Swap);
                            self.emit(Instr::StoreMember(field.clone()));
                            if let runmat_hir::HirExprKind::Var(root_var) = member_base.kind {
                                self.emit(Instr::StoreVar(root_var.0));
                            }
                        } else {
                            // Fallback: evaluate base, indices, rhs, and store (for object chains via subsasgn)
                            self.compile_expr(base)?;
                            for index in indices {
                                self.compile_expr(index)?;
                            }
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
                    runmat_hir::HirLValue::MemberDynamic(base, name_expr) => {
                        if let runmat_hir::HirExprKind::Var(var_id) = base.kind.clone() {
                            self.emit(Instr::LoadVar(var_id.0));
                            self.compile_expr(name_expr)?;
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreMemberDynamic);
                            self.emit(Instr::StoreVar(var_id.0));
                        } else {
                            self.compile_expr(base)?;
                            self.compile_expr(name_expr)?;
                            self.compile_expr(rhs)?;
                            self.emit(Instr::StoreMemberDynamic);
                        }
                    }
                    _ => return Err("unsupported lvalue target".into()),
                }
            }
            HirStmt::Global(vars) => {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            HirStmt::Persistent(vars) => {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
            HirStmt::Import { path, wildcard } => {
                self.emit(Instr::RegisterImport {
                    path: path.clone(),
                    wildcard: *wildcard,
                });
            }
            HirStmt::ClassDef {
                name,
                super_class,
                members,
            } => {
                // Synthesize a minimal RegisterClass instruction by extracting property names and method names
                let mut props: Vec<(String, bool, String, String)> = Vec::new();
                let mut methods: Vec<(String, String, bool, String)> = Vec::new();
                for m in members {
                    match m {
                        runmat_hir::HirClassMember::Properties { names, attributes } => {
                            let (is_static, is_dependent, get_access, set_access) =
                                Self::parse_prop_attrs(attributes);
                            // Encode dependent flag by prefixing name with "@dep:"; VM will strip and set flag.
                            for n in names {
                                let enc = if is_dependent {
                                    format!("@dep:{n}")
                                } else {
                                    n.clone()
                                };
                                props.push((
                                    enc,
                                    is_static,
                                    get_access.clone(),
                                    set_access.clone(),
                                ));
                            }
                        }
                        runmat_hir::HirClassMember::Methods { body, attributes } => {
                            let (is_static, access) = Self::parse_method_attrs(attributes);
                            for s in body {
                                if let runmat_hir::HirStmt::Function { name: mname, .. } = s {
                                    methods.push((
                                        mname.clone(),
                                        mname.clone(),
                                        is_static,
                                        access.clone(),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                self.emit(Instr::RegisterClass {
                    name: name.clone(),
                    super_class: super_class.clone(),
                    properties: props,
                    methods,
                });
            }
            HirStmt::MultiAssign(vars, expr, _) => {
                // Compile RHS once; if function call or value, arrange to extract multiple
                match &expr.kind {
                    HirExprKind::FuncCall(name, args) => {
                        if self.functions.contains_key(name) {
                            for arg in args {
                                self.compile_expr(arg)?;
                            }
                            // Emit multi-call to request N outputs
                            self.emit(Instr::CallFunctionMulti(
                                name.clone(),
                                args.len(),
                                vars.len(),
                            ));
                            // Store outputs in order
                            for (_i, var) in vars.iter().enumerate().rev() {
                                if let Some(v) = var {
                                    self.emit(Instr::StoreVar(v.0));
                                } else {
                                    self.emit(Instr::Pop);
                                }
                            }
                        } else {
                            // Builtin or unknown: treat as single return value
                            for arg in args {
                                self.compile_expr(arg)?;
                            }
                            self.emit(Instr::CallBuiltinMulti(
                                name.clone(),
                                args.len(),
                                vars.len(),
                            ));
                            for (_i, var) in vars.iter().enumerate().rev() {
                                if let Some(v) = var {
                                    self.emit(Instr::StoreVar(v.0));
                                } else {
                                    self.emit(Instr::Pop);
                                }
                            }
                        }
                    }
                    HirExprKind::IndexCell(base, indices) => {
                        // Support comma-list expansion from cell indexing: [a,b,...] = C{idx}
                        self.compile_expr(base)?;
                        for index in indices {
                            self.compile_expr(index)?;
                        }
                        // Expand into N outputs
                        self.emit(Instr::IndexCellExpand(indices.len(), vars.len()));
                        for (_i, var) in vars.iter().enumerate().rev() {
                            if let Some(v) = var {
                                self.emit(Instr::StoreVar(v.0));
                            } else {
                                self.emit(Instr::Pop);
                            }
                        }
                    }
                    _ => {
                        // Non-call: assign expr to first non-placeholder, zeros to remaining non-placeholders
                        let first_real = vars.iter().position(|v| v.is_some());
                        if let Some(first_idx) = first_real {
                            self.compile_expr(expr)?;
                            if let Some(Some(first_var)) = vars.get(first_idx) {
                                self.emit(Instr::StoreVar(first_var.0));
                            }
                        }
                        for (i, var) in vars.iter().enumerate() {
                            if Some(i) == first_real {
                                continue;
                            }
                            if let Some(v) = var {
                                self.emit(Instr::LoadConst(0.0));
                                self.emit(Instr::StoreVar(v.0));
                            }
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
                if s.starts_with('"') && s.ends_with('"') {
                    // String scalar
                    let inner = &s[1..s.len() - 1];
                    let clean = inner.replace("\"\"", "\"");
                    self.emit(Instr::LoadString(clean));
                } else if s.starts_with('\'') && s.ends_with('\'') {
                    // Char vector -> CharArray row
                    let inner = &s[1..s.len() - 1];
                    let clean = inner.replace("''", "'");
                    // Encode as CharArray(1, len)
                    self.emit(Instr::LoadCharRow(clean));
                } else {
                    self.emit(Instr::LoadString(s.clone()));
                }
            }
            HirExprKind::Var(id) => {
                self.emit(Instr::LoadVar(id.0));
            }
            // Fallback path for unqualified static property via Class.* imports: treat bare identifier as Class.prop
            // Only when not a known var/const/func; handled earlier in HIR, so we also handle here for robustness via Member lowering
            // Note: HIR would have errored on undefined variable; we resolve at compile time before that for function calls,
            // but for expressions like `v = staticValue;` under `import Point.*`, we handle here by probing imports.
            HirExprKind::Constant(name) => {
                let constants = runmat_builtins::constants();
                if let Some(constant) = constants.iter().find(|c| c.name == name) {
                    match &constant.value {
                        runmat_builtins::Value::Num(val) => {
                            self.emit(Instr::LoadConst(*val));
                        }
                        runmat_builtins::Value::Complex(re, im) => {
                            self.emit(Instr::LoadComplex(*re, *im));
                        }
                        runmat_builtins::Value::Bool(b) => {
                            self.emit(Instr::LoadBool(*b));
                        }
                        _ => {
                            return Err(format!("Constant {name} is not a number or boolean"));
                        }
                    }
                } else {
                    // Try resolving as unqualified static property via Class.* imports (multi-segment)
                    let mut classes: Vec<String> = Vec::new();
                    for (path, wildcard) in &self.imports {
                        if !*wildcard {
                            continue;
                        }
                        if path.is_empty() {
                            continue;
                        }
                        let mut cls = String::new();
                        for (i, part) in path.iter().enumerate() {
                            if i > 0 {
                                cls.push('.');
                            }
                            cls.push_str(part);
                        }
                        if let Some((p, _owner)) = runmat_builtins::lookup_property(&cls, name) {
                            if p.is_static {
                                classes.push(cls.clone());
                            }
                        }
                    }
                    if classes.len() > 1 {
                        return Err(format!(
                            "ambiguous unqualified static property '{}' via Class.* imports: {}",
                            name,
                            classes.join(", ")
                        ));
                    }
                    if classes.len() == 1 {
                        self.emit(Instr::LoadStaticProperty(classes.remove(0), name.clone()));
                        return Ok(());
                    }
                    return Err(format!("Unknown constant or static property: {name}"));
                }
            }
            HirExprKind::Unary(op, e) => {
                self.compile_expr(e)?;
                match op {
                    runmat_parser::UnOp::Plus => {
                        self.emit(Instr::UPlus);
                    }
                    runmat_parser::UnOp::Minus => {
                        self.emit(Instr::Neg);
                    }
                    runmat_parser::UnOp::Transpose => {
                        self.emit(Instr::ConjugateTranspose);
                    }
                    runmat_parser::UnOp::NonConjugateTranspose => {
                        self.emit(Instr::Transpose);
                    }
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
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.emit(Instr::ElemMul);
                    }
                    BinOp::BitOr => {
                        self.compile_expr(a)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.compile_expr(b)?;
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        self.emit(Instr::Add);
                        self.emit(Instr::LoadConst(0.0));
                        self.emit(Instr::CallBuiltin("ne".to_string(), 2));
                        // Stay element-wise by folding through elementwise ops
                    }
                    _ => {
                        self.compile_expr(a)?;
                        self.compile_expr(b)?;
                        match op {
                            BinOp::Add => {
                                self.emit(Instr::Add);
                            }
                            BinOp::Sub => {
                                self.emit(Instr::Sub);
                            }
                            BinOp::Mul => {
                                self.emit(Instr::Mul);
                            }
                            BinOp::Div | BinOp::LeftDiv => {
                                self.emit(Instr::Div);
                            }
                            BinOp::Pow => {
                                self.emit(Instr::Pow);
                            }
                            BinOp::ElemMul => {
                                self.emit(Instr::ElemMul);
                            }
                            BinOp::ElemDiv | BinOp::ElemLeftDiv => {
                                self.emit(Instr::ElemDiv);
                            }
                            BinOp::ElemPow => {
                                self.emit(Instr::ElemPow);
                            }
                            BinOp::Equal => {
                                self.emit(Instr::Equal);
                            }
                            BinOp::NotEqual => {
                                self.emit(Instr::NotEqual);
                            }
                            BinOp::Less => {
                                self.emit(Instr::Less);
                            }
                            BinOp::LessEqual => {
                                self.emit(Instr::LessEqual);
                            }
                            BinOp::Greater => {
                                self.emit(Instr::Greater);
                            }
                            BinOp::GreaterEqual => {
                                self.emit(Instr::GreaterEqual);
                            }
                            BinOp::Colon => {
                                return Err("colon operator not supported".into());
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            HirExprKind::Range(start, step, end) => {
                self.compile_expr(start)?;
                if let Some(step) = step {
                    self.compile_expr(step)?;
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(true));
                } else {
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(false));
                }
            }
            HirExprKind::FuncCall(name, args) => {
                // Special-case: feval(f, a1, a2, ...) compiles to VM feval to access user functions/closures
                if name == "feval" {
                    if args.is_empty() {
                        return Err("feval: missing function argument".into());
                    }
                    // Push function value first
                    self.compile_expr(&args[0])?;
                    let rest = &args[1..];
                    let has_expand = rest
                        .iter()
                        .any(|a| matches!(a.kind, HirExprKind::IndexCell(_, _)));
                    if has_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(rest.len());
                        for arg in rest {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit(Instr::CallFevalExpandMulti(specs));
                    } else {
                        for arg in rest {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallFeval(rest.len()));
                    }
                    return Ok(());
                }
                let has_any_expand = args
                    .iter()
                    .any(|a| matches!(a.kind, HirExprKind::IndexCell(_, _)));
                if self.functions.contains_key(name) {
                    // existing path
                    if has_any_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit(Instr::CallFunctionExpandMulti(name.clone(), specs));
                    } else {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallFunction(name.clone(), args.len()));
                    }
                } else {
                    // Existing import-based function/builtin resolution, extended with static method via Class.*
                    // Attempt compile-time import resolution for builtins and user functions with ambiguity checks
                    // Precedence for unqualified resolution:
                    // locals > user functions in scope > specific imports > wildcard imports > Class.* static methods
                    // 1) Specific imports: import pkg.foo => resolve 'foo' (takes precedence over wildcard)
                    // 2) Wildcard imports: import pkg.* => resolve 'pkg.foo'
                    // 3) Class.* static methods: import Class.* (or pkg.Class.*) => resolve static methods if unambiguous
                    let mut resolved = name.clone();
                    let mut static_candidates: Vec<(String, String)> = Vec::new();
                    if !runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == resolved)
                    {
                        // Specific candidates
                        let mut specific_candidates: Vec<String> = Vec::new();
                        for (path, wildcard) in &self.imports {
                            if *wildcard {
                                continue;
                            }
                            if path.last().map(|s| s.as_str()) == Some(name.as_str()) {
                                let qual = path.join(".");
                                if runmat_builtins::builtin_functions()
                                    .iter()
                                    .any(|b| b.name == qual)
                                    || self.functions.contains_key(&qual)
                                {
                                    specific_candidates.push(qual);
                                }
                            }
                        }
                        if specific_candidates.len() > 1 {
                            return Err(format!(
                                "ambiguous unqualified reference '{}' via imports: {}",
                                name,
                                specific_candidates.join(", ")
                            ));
                        }
                        if specific_candidates.len() == 1 {
                            resolved = specific_candidates.remove(0);
                        } else {
                            // Wildcard candidates for functions
                            let mut wildcard_candidates: Vec<String> = Vec::new();
                            for (path, wildcard) in &self.imports {
                                if !*wildcard {
                                    continue;
                                }
                                if path.is_empty() {
                                    continue;
                                }
                                let mut qual = String::new();
                                for (i, part) in path.iter().enumerate() {
                                    if i > 0 {
                                        qual.push('.');
                                    }
                                    qual.push_str(part);
                                }
                                qual.push('.');
                                qual.push_str(name);
                                if runmat_builtins::builtin_functions()
                                    .iter()
                                    .any(|b| b.name == qual)
                                    || self.functions.contains_key(&qual)
                                {
                                    wildcard_candidates.push(qual);
                                }
                                // Accumulate Class.* static method candidates for any class path
                                let mut cls = String::new();
                                for (i, part) in path.iter().enumerate() {
                                    if i > 0 {
                                        cls.push('.');
                                    }
                                    cls.push_str(part);
                                }
                                if let Some((m, _owner)) =
                                    runmat_builtins::lookup_method(&cls, name)
                                {
                                    if m.is_static {
                                        static_candidates.push((cls.clone(), name.clone()));
                                    }
                                }
                            }
                            if wildcard_candidates.len() > 1 {
                                return Err(format!(
                                    "ambiguous unqualified reference '{}' via wildcard imports: {}",
                                    name,
                                    wildcard_candidates.join(", ")
                                ));
                            }
                            if wildcard_candidates.len() == 1 {
                                resolved = wildcard_candidates.remove(0);
                            }
                        }
                    }
                    // If resolved maps to a user function, compile it now
                    if self.functions.contains_key(&resolved) {
                        if has_any_expand {
                            let mut specs: Vec<crate::instr::ArgSpec> =
                                Vec::with_capacity(args.len());
                            for arg in args {
                                if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                    let is_expand_all = indices.len() == 1
                                        && matches!(indices[0].kind, HirExprKind::Colon);
                                    if is_expand_all {
                                        specs.push(crate::instr::ArgSpec {
                                            is_expand: true,
                                            num_indices: 0,
                                            expand_all: true,
                                        });
                                        self.compile_expr(base)?;
                                    } else {
                                        specs.push(crate::instr::ArgSpec {
                                            is_expand: true,
                                            num_indices: indices.len(),
                                            expand_all: false,
                                        });
                                        self.compile_expr(base)?;
                                        for i in indices {
                                            self.compile_expr(i)?;
                                        }
                                    }
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: false,
                                        num_indices: 0,
                                        expand_all: false,
                                    });
                                    self.compile_expr(arg)?;
                                }
                            }
                            self.emit(Instr::CallFunctionExpandMulti(resolved.clone(), specs));
                            return Ok(());
                        } else {
                            // Flatten inner user-function returns into argument list for the call
                            let mut total_argc: usize = 0;
                            for arg in args {
                                if let HirExprKind::FuncCall(inner, inner_args) = &arg.kind {
                                    if self.functions.contains_key(inner) {
                                        for a in inner_args {
                                            self.compile_expr(a)?;
                                        }
                                        let outc = self
                                            .functions
                                            .get(inner)
                                            .map(|f| f.outputs.len().max(1))
                                            .unwrap_or(1);
                                        self.emit(Instr::CallFunctionMulti(
                                            inner.clone(),
                                            inner_args.len(),
                                            outc,
                                        ));
                                        total_argc += outc;
                                        continue;
                                    }
                                }
                                self.compile_expr(arg)?;
                                total_argc += 1;
                            }
                            self.emit(Instr::CallFunction(resolved.clone(), total_argc));
                            return Ok(());
                        }
                    }
                    // If still no function, and exactly one static candidate, call it
                    if !runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == resolved)
                        && static_candidates.len() == 1
                    {
                        let (cls, method) = static_candidates.remove(0);
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallStaticMethod(cls, method, args.len()));
                        return Ok(());
                    }
                    // If multiple static candidates and no function resolved, report ambiguity
                    if !runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == resolved)
                        && static_candidates.len() > 1
                    {
                        return Err(format!(
                            "ambiguous unqualified static method '{}' via Class.* imports: {}",
                            name,
                            static_candidates
                                .iter()
                                .map(|(c, _)| c.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                    // Existing propagation path and builtin call
                    if !has_any_expand {
                        // First scan for user-defined inner function calls to expand; avoid compiling
                        // simple args here to prevent duplicating them on the stack. If any inner
                        // user functions are found, compile them and then compile the remaining
                        // simple args once, emit the builtin call, and return. Otherwise, fall
                        // through to the normal (single-pass) compilation below.
                        let mut total_argc = 0usize;
                        let mut did_expand_inner = false;
                        let mut pending_simple: Vec<&runmat_hir::HirExpr> = Vec::new();
                        for arg in args {
                            if let HirExprKind::FuncCall(inner, inner_args) = &arg.kind {
                                if self.functions.contains_key(inner) {
                                    for a in inner_args {
                                        self.compile_expr(a)?;
                                    }
                                    let outc = self
                                        .functions
                                        .get(inner)
                                        .map(|f| f.outputs.len().max(1))
                                        .unwrap_or(1);
                                    self.emit(Instr::CallFunctionMulti(
                                        inner.clone(),
                                        inner_args.len(),
                                        outc,
                                    ));
                                    total_argc += outc;
                                    did_expand_inner = true;
                                } else {
                                    pending_simple.push(arg);
                                }
                            } else {
                                pending_simple.push(arg);
                            }
                        }
                        if did_expand_inner {
                            for arg in pending_simple {
                                self.compile_expr(arg)?;
                                total_argc += 1;
                            }
                            self.emit(Instr::CallBuiltin(resolved, total_argc));
                            return Ok(());
                        }
                    }
                    if has_any_expand {
                        let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(args.len());
                        for arg in args {
                            if let HirExprKind::IndexCell(base, indices) = &arg.kind {
                                let is_expand_all = indices.len() == 1
                                    && matches!(indices[0].kind, HirExprKind::Colon);
                                if is_expand_all {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: 0,
                                        expand_all: true,
                                    });
                                    self.compile_expr(base)?;
                                } else {
                                    specs.push(crate::instr::ArgSpec {
                                        is_expand: true,
                                        num_indices: indices.len(),
                                        expand_all: false,
                                    });
                                    self.compile_expr(base)?;
                                    for i in indices {
                                        self.compile_expr(i)?;
                                    }
                                }
                            } else {
                                specs.push(crate::instr::ArgSpec {
                                    is_expand: false,
                                    num_indices: 0,
                                    expand_all: false,
                                });
                                self.compile_expr(arg)?;
                            }
                        }
                        self.emit(Instr::CallBuiltinExpandMulti(resolved, specs));
                    } else {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallBuiltin(resolved, args.len()));
                    }
                    return Ok(());
                }
                return Ok(());
            }
            HirExprKind::Tensor(matrix_data) | HirExprKind::Cell(matrix_data) => {
                let rows = matrix_data.len();
                // Special case: 1-row tensor literal with a single element that is IndexCell(base, {:})
                // Lower "[C{:}]" into cat(2, C{:}) so downstream expansion works without colon compilation
                if matches!(expr.kind, HirExprKind::Tensor(_))
                    && rows == 1
                    && matrix_data.first().map(|r| r.len()).unwrap_or(0) == 1
                {
                    if let HirExprKind::IndexCell(base, indices) = &matrix_data[0][0].kind {
                        if indices.len() == 1 && matches!(indices[0].kind, HirExprKind::Colon) {
                            // Build specs: first fixed dim=2, then expand_all for base
                            let mut specs: Vec<crate::instr::ArgSpec> = Vec::with_capacity(2);
                            // Fixed dimension 2
                            specs.push(crate::instr::ArgSpec {
                                is_expand: false,
                                num_indices: 0,
                                expand_all: false,
                            });
                            self.emit(Instr::LoadConst(2.0));
                            // Expand all from base cell
                            specs.push(crate::instr::ArgSpec {
                                is_expand: true,
                                num_indices: 0,
                                expand_all: true,
                            });
                            self.compile_expr(base)?;
                            self.emit(Instr::CallBuiltinExpandMulti("cat".to_string(), specs));
                            return Ok(());
                        }
                    }
                }
                let has_non_literals = matrix_data.iter().any(|row| {
                    row.iter()
                        .any(|expr| !matches!(expr.kind, HirExprKind::Number(_)))
                });
                if has_non_literals {
                    for row in matrix_data {
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }
                    let row_lengths: Vec<usize> = matrix_data.iter().map(|row| row.len()).collect();
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
                        for &row_len in &row_lengths {
                            self.emit(Instr::LoadConst(row_len as f64));
                        }
                        self.emit(Instr::CreateMatrixDynamic(rows));
                    }
                } else {
                    let cols = if rows > 0 { matrix_data[0].len() } else { 0 };
                    for row in matrix_data {
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }
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
                let has_vector = indices.iter().any(|e| {
                    matches!(e.kind, HirExprKind::Range(_, _, _) | HirExprKind::Tensor(_))
                        || matches!(e.ty, runmat_hir::Type::Tensor { .. })
                });
                // General case: any-dimension ranges with end arithmetic (e.g., A(:,2:2:end-1,...))
                // We lower into IndexRangeEnd: push base, then per-range start[, step] in increasing dimension order,
                // then any numeric scalar indices (in order). Colon and plain end dims are marked in masks.
                {
                    let mut has_any_range_end = false;
                    let mut range_dims: Vec<usize> = Vec::new();
                    let mut range_has_step: Vec<bool> = Vec::new();
                    let mut end_offsets: Vec<i64> = Vec::new();
                    // First pass: detect any Range with End-Sub on end expression
                    for (dim, index) in indices.iter().enumerate() {
                        if let HirExprKind::Range(_start, step, end) = &index.kind {
                            if let HirExprKind::Binary(left, op, right) = &end.kind {
                                if matches!(op, runmat_parser::BinOp::Sub)
                                    && matches!(left.kind, HirExprKind::End)
                                {
                                    has_any_range_end = true;
                                    range_dims.push(dim);
                                    range_has_step.push(step.is_some());
                                    let off = if let HirExprKind::Number(ref s) = right.kind {
                                        s.parse::<i64>().unwrap_or(0)
                                    } else {
                                        0
                                    };
                                    end_offsets.push(off);
                                }
                            }
                        }
                    }
                    if has_any_range_end {
                        self.compile_expr(base)?;
                        // Push per-range start and optional step in dimension order
                        for &dim in &range_dims {
                            if let HirExprKind::Range(start, step, _end) = &indices[dim].kind {
                                self.compile_expr(start)?;
                                if let Some(st) = step {
                                    self.compile_expr(st)?;
                                }
                            }
                        }
                        // Count numeric scalar indices and push them
                        let mut colon_mask: u32 = 0;
                        let mut end_mask: u32 = 0;
                        let mut numeric_count = 0usize;
                        for (dim, index) in indices.iter().enumerate() {
                            match &index.kind {
                                HirExprKind::Colon => {
                                    colon_mask |= 1u32 << dim;
                                }
                                HirExprKind::End => {
                                    end_mask |= 1u32 << dim;
                                }
                                HirExprKind::Range(_, _, end) => {
                                    // If this range used end arithmetic, we already handled; otherwise treat as plain numeric idx vector at runtime via VM
                                    if let HirExprKind::Binary(left, op, _right) = &end.kind {
                                        if matches!(op, runmat_parser::BinOp::Sub)
                                            && matches!(left.kind, HirExprKind::End)
                                        {
                                            // skip pushing numeric for this dim
                                            continue;
                                        }
                                    }
                                    // For non-end ranges, we will resolve via VM range gather; push placeholders via numeric_count == 0 (no need to push indices)
                                }
                                _ => {
                                    self.compile_expr(index)?;
                                    numeric_count += 1;
                                }
                            }
                        }
                        // For pure 1-D case with a single range_dim, degrade to legacy Index1DRangeEnd to keep existing test stable
                        if indices.len() == 1 && range_dims.len() == 1 {
                            // We pushed start[, step] already. Emit Index1DRangeEnd.
                            self.emit(Instr::Index1DRangeEnd {
                                has_step: range_has_step[0],
                                offset: end_offsets[0],
                            });
                        } else {
                            self.emit(Instr::IndexRangeEnd {
                                dims: indices.len(),
                                numeric_count,
                                colon_mask,
                                end_mask,
                                range_dims,
                                range_has_step,
                                end_offsets,
                            });
                        }
                        return Ok(());
                    }
                }
                if has_colon
                    || has_vector
                    || has_end
                    || indices.len() > 2
                    || Self::expr_contains_end(base)
                {
                    // Push base first, then numeric indices in order; compute colon mask
                    self.compile_expr(base)?;
                    let mut colon_mask: u32 = 0;
                    let mut end_mask: u32 = 0;
                    let mut numeric_count = 0usize;
                    let mut end_offsets: Vec<(usize, i64)> = Vec::new();
                    for (dim, index) in indices.iter().enumerate() {
                        if matches!(index.kind, HirExprKind::Colon) {
                            colon_mask |= 1u32 << dim;
                        } else if matches!(index.kind, HirExprKind::End) {
                            end_mask |= 1u32 << dim;
                        } else {
                            // Detect simple end arithmetic forms: end-1, end-2 etc.
                            if let HirExprKind::Binary(left, op, right) = &index.kind {
                                if matches!(op, runmat_parser::BinOp::Sub)
                                    && matches!(left.kind, HirExprKind::End)
                                {
                                    // Right should be number literal string; parse as integer offset if possible
                                    if let HirExprKind::Number(ref s) = right.kind {
                                        if let Ok(k) = s.parse::<i64>() {
                                            // Reserve a numeric slot: push placeholder and count it
                                            self.emit(Instr::LoadConst(0.0));
                                            end_offsets.push((numeric_count, k));
                                            numeric_count += 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                            self.compile_expr(index)?;
                            numeric_count += 1;
                        }
                    }
                    if end_offsets.is_empty() {
                        self.emit(Instr::IndexSlice(
                            indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                        ));
                    } else {
                        self.emit(Instr::IndexSliceEx(
                            indices.len(),
                            numeric_count,
                            colon_mask,
                            end_mask,
                            end_offsets,
                        ));
                    }
                } else {
                    self.compile_expr(base)?;
                    for index in indices {
                        self.compile_expr(index)?;
                    }
                    self.emit(Instr::Index(indices.len()));
                }
            }
            HirExprKind::Colon => {
                // Placeholder for contexts where colon appeared in RHS expansion; real colon handling occurs in indexing logic
                // Emit a benign constant to keep stack discipline when mistakenly compiled
                self.emit(Instr::LoadConst(0.0));
            }
            HirExprKind::End => {
                self.emit(Instr::LoadConst(-0.0)); /* placeholder, resolved via end_mask in IndexSlice */
            }
            HirExprKind::Member(base, field) => {
                // If base is a known class ref literal (string via classref builtin), static access
                // Or if base is MetaClass (string literal), treat as class name for static access
                // Otherwise, instance member
                match &base.kind {
                    HirExprKind::MetaClass(cls_name) => {
                        self.emit(Instr::LoadStaticProperty(cls_name.clone(), field.clone()));
                    }
                    HirExprKind::FuncCall(name, args) if name == "classref" && args.len() == 1 => {
                        if let HirExprKind::String(cls) = &args[0].kind {
                            let cls_name = if cls.starts_with('\'') && cls.ends_with('\'') {
                                cls[1..cls.len() - 1].to_string()
                            } else {
                                cls.clone()
                            };
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
            HirExprKind::MemberDynamic(base, name_expr) => {
                self.compile_expr(base)?;
                self.compile_expr(name_expr)?;
                self.emit(Instr::LoadMemberDynamic);
            }
            // Dynamic member s.(expr)
            HirExprKind::MethodCall(b, m, a) if m == &"()".to_string() && a.len() == 1 => {
                // Note: parser currently doesn't produce this form; placeholder for dynamic
                self.compile_expr(b)?;
                self.compile_expr(&a[0])?;
                self.emit(Instr::LoadMemberDynamic);
            }
            HirExprKind::MethodCall(base, method, args) => match &base.kind {
                HirExprKind::MetaClass(cls_name) => {
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    self.emit(Instr::CallStaticMethod(
                        cls_name.clone(),
                        method.clone(),
                        args.len(),
                    ));
                }
                HirExprKind::FuncCall(name, bargs) if name == "classref" && bargs.len() == 1 => {
                    if let HirExprKind::String(cls) = &bargs[0].kind {
                        let cls_name = if cls.starts_with('\'') && cls.ends_with('\'') {
                            cls[1..cls.len() - 1].to_string()
                        } else {
                            cls.clone()
                        };
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallStaticMethod(
                            cls_name,
                            method.clone(),
                            args.len(),
                        ));
                    } else {
                        self.compile_expr(base)?;
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Instr::CallMethod(method.clone(), args.len()));
                    }
                }
                _ => {
                    self.compile_expr(base)?;
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    self.emit(Instr::CallMethod(method.clone(), args.len()));
                }
            },
            HirExprKind::AnonFunc { params, body } => {
                // Collect free variables in body (in order of first appearance)
                use std::collections::{HashMap, HashSet};
                let mut seen: HashSet<runmat_hir::VarId> = HashSet::new();
                let mut captures_order: Vec<runmat_hir::VarId> = Vec::new();
                let bound: HashSet<runmat_hir::VarId> = params.iter().cloned().collect();
                self.collect_free_vars(body, &bound, &mut seen, &mut captures_order);

                // Build placeholder VarIds for captures and parameters
                let capture_count = captures_order.len();
                let mut placeholder_params: Vec<runmat_hir::VarId> =
                    Vec::with_capacity(capture_count + params.len());
                for i in 0..capture_count {
                    placeholder_params.push(runmat_hir::VarId(i));
                }
                for j in 0..params.len() {
                    placeholder_params.push(runmat_hir::VarId(capture_count + j));
                }
                let output_id = runmat_hir::VarId(capture_count + params.len());

                // Remap body vars: free vars -> capture placeholders; param vars -> shifted placeholders
                let mut var_map: HashMap<runmat_hir::VarId, runmat_hir::VarId> = HashMap::new();
                for (i, old) in captures_order.iter().enumerate() {
                    var_map.insert(*old, runmat_hir::VarId(i));
                }
                for (j, old) in params.iter().enumerate() {
                    var_map.insert(*old, runmat_hir::VarId(capture_count + j));
                }
                let remapped_body = runmat_hir::remapping::remap_expr(body, &var_map);
                let func_body = vec![runmat_hir::HirStmt::Assign(output_id, remapped_body, true)];

                // Synthesize function name and register
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
                };
                self.functions.insert(synthesized.clone(), user_func);

                // Emit capture values on stack then create closure
                for old in &captures_order {
                    self.emit(Instr::LoadVar(old.0));
                }
                self.emit(Instr::CreateClosure(synthesized, capture_count));
            }
            HirExprKind::FuncHandle(name) => {
                self.emit(Instr::LoadString(name.clone()));
                self.emit(Instr::CallBuiltin("make_handle".to_string(), 1));
            }
            HirExprKind::MetaClass(name) => {
                self.emit(Instr::LoadString(name.clone()));
            }
            // Member/Method on metaclass (string on stack) will be handled by runtime as static property/method via classref
            // We lower MetaClass to a string (class name) and then member/method code paths remain unchanged.
            HirExprKind::IndexCell(base, indices) => {
                self.compile_expr(base)?;
                for index in indices {
                    self.compile_expr(index)?;
                }
                self.emit(Instr::IndexCell(indices.len()));
            }
        }
        Ok(())
    }
}

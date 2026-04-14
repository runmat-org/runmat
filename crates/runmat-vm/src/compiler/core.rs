use crate::compiler::CompileError;
use crate::functions::UserFunction;
use crate::instr::{EmitLabel, Instr};
use runmat_builtins::{self, Type};
use runmat_hir::{HirExpr, HirExprKind, HirProgram, HirStmt};
use std::collections::HashMap;

pub struct LoopLabels {
    pub break_jumps: Vec<usize>,
    pub continue_jumps: Vec<usize>,
}

pub struct Compiler {
    pub instructions: Vec<Instr>,
    pub instr_spans: Vec<runmat_hir::Span>,
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    pub var_count: usize,
    pub loop_stack: Vec<LoopLabels>,
    pub functions: HashMap<String, UserFunction>,
    pub imports: Vec<(Vec<String>, bool)>,
    pub var_types: Vec<Type>,
    current_span: Option<runmat_hir::Span>,
}

struct SpanGuard {
    compiler: *mut Compiler,
    prev: Option<runmat_hir::Span>,
}

impl SpanGuard {
    fn new(compiler: &mut Compiler, span: runmat_hir::Span) -> Self {
        let prev = compiler.current_span;
        compiler.current_span = Some(span);
        Self {
            compiler: compiler as *mut Compiler,
            prev,
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(compiler) = self.compiler.as_mut() {
                compiler.current_span = self.prev;
            }
        }
    }
}

impl Compiler {
    pub(crate) fn normalize_class_literal_name(raw: &str) -> String {
        if raw.len() >= 2 {
            let bytes = raw.as_bytes();
            let first = bytes[0] as char;
            let last = bytes[raw.len() - 1] as char;
            if (first == '\'' || first == '"') && first == last {
                return raw[1..raw.len() - 1].to_string();
            }
        }
        raw.to_string()
    }

    pub(crate) fn emit_multiassign_outputs(&mut self, vars: &[Option<runmat_hir::VarId>]) {
        for v in vars.iter().flatten() {
            self.emit(Instr::EmitVar {
                var_index: v.0,
                label: EmitLabel::Var(v.0),
            });
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
                HirExprKind::FuncCall(_, args) => {
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::MethodCall(base, _, args)
                | HirExprKind::DottedInvoke(base, _, args) => {
                    visit_expr(base, max);
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
                    HirStmt::Assign(id, expr, _, _) => {
                        if id.0 + 1 > *max {
                            *max = id.0 + 1;
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::ExprStmt(expr, _, _) => visit_expr(expr, max),
                    HirStmt::Return(_) => {}
                    HirStmt::If {
                        cond,
                        then_body,
                        elseif_blocks,
                        else_body,
                        ..
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
                    HirStmt::While { cond, body, .. } => {
                        visit_expr(cond, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::For {
                        var, expr, body, ..
                    } => {
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
                        ..
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
                        ..
                    } => {
                        if let Some(v) = catch_var {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                        visit_stmts(try_body, max);
                        visit_stmts(catch_body, max);
                    }
                    HirStmt::Global(vars, _) | HirStmt::Persistent(vars, _) => {
                        for (v, _name) in vars {
                            if v.0 + 1 > *max {
                                *max = v.0 + 1;
                            }
                        }
                    }
                    HirStmt::AssignLValue(_, expr, _, _) => visit_expr(expr, max),
                    HirStmt::MultiAssign(vars, expr, _, _) => {
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
                    | HirStmt::Break(_)
                    | HirStmt::Continue(_) => {}
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
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count: max_var,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
            imports: Vec::new(),
            var_types,
            current_span: None,
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

    pub(crate) fn alloc_temp(&mut self) -> usize {
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
        let span = self.current_span.unwrap_or_default();
        self.instr_spans.push(span);
        self.call_arg_spans.push(None);
        pc
    }

    pub(crate) fn emit_call_with_arg_spans(
        &mut self,
        instr: Instr,
        arg_spans: &[runmat_hir::Span],
    ) -> usize {
        let pc = self.emit(instr);
        if !arg_spans.is_empty() {
            if let Some(slot) = self.call_arg_spans.get_mut(pc) {
                *slot = Some(arg_spans.to_vec());
            }
        }
        pc
    }

    pub fn patch(&mut self, idx: usize, instr: Instr) {
        self.instructions[idx] = instr;
    }

    pub(crate) fn compile_error(&self, message: impl Into<String>) -> CompileError {
        let mut err = CompileError::new(message);
        if let Some(span) = self.current_span {
            err = err.with_span(span);
        }
        err
    }

    pub fn compile_program(&mut self, prog: &HirProgram) -> Result<(), CompileError> {
        // Validate imports early for duplicate/specific-name ambiguities
        runmat_hir::validate_imports(prog)?;
        // Validate class definitions for attribute correctness and name conflicts
        runmat_hir::validate_classdefs(prog)?;
        // Pre-collect imports (both wildcard and specific) for name resolution
        for stmt in &prog.body {
            let _span_guard = SpanGuard::new(self, stmt.span());
            if let HirStmt::Import { path, wildcard, .. } = stmt {
                self.imports.push((path.clone(), *wildcard));
                self.emit(Instr::RegisterImport {
                    path: path.clone(),
                    wildcard: *wildcard,
                });
            }
            if let HirStmt::Global(vars, _) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclareGlobalNamed(ids, names));
            }
            if let HirStmt::Persistent(vars, _) = stmt {
                let ids: Vec<usize> = vars.iter().map(|(v, _n)| v.0).collect();
                let names: Vec<String> = vars.iter().map(|(_v, n)| n.clone()).collect();
                self.emit(Instr::DeclarePersistentNamed(ids, names));
            }
        }
        for stmt in &prog.body {
            if !matches!(
                stmt,
                HirStmt::Import { .. } | HirStmt::Global(_, _) | HirStmt::Persistent(_, _)
            ) {
                self.compile_stmt(stmt)?;
            }
        }
        Ok(())
    }

    pub fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, stmt.span());
        self.compile_stmt_impl(stmt)
    }

    pub fn compile_expr(&mut self, expr: &HirExpr) -> Result<(), CompileError> {
        let _span_guard = SpanGuard::new(self, expr.span);
        self.compile_expr_impl(expr)
    }
}

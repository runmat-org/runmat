use runmat_parser::{
    self as parser, BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export Type from builtins for consistency
pub use runmat_builtins::Type;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub usize);

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirExprKind {
    Number(String),
    String(String),
    Var(VarId),
    Constant(String), // For built-in constants like pi, e, etc.
    Unary(UnOp, Box<HirExpr>),
    Binary(Box<HirExpr>, BinOp, Box<HirExpr>),
    Tensor(Vec<Vec<HirExpr>>),
    Cell(Vec<Vec<HirExpr>>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    End,
    Member(Box<HirExpr>, String),
    MethodCall(Box<HirExpr>, String, Vec<HirExpr>),
    AnonFunc { params: Vec<VarId>, body: Box<HirExpr> },
    FuncHandle(String),
    FuncCall(String, Vec<HirExpr>),
    MetaClass(String),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirStmt {
    ExprStmt(HirExpr, bool), // Expression and whether it's semicolon-terminated (suppressed)
    Assign(VarId, HirExpr, bool), // Variable, Expression, and whether it's semicolon-terminated (suppressed)
    MultiAssign(Vec<VarId>, HirExpr, bool),
    AssignLValue(HirLValue, HirExpr, bool),
    If {
        cond: HirExpr,
        then_body: Vec<HirStmt>,
        elseif_blocks: Vec<(HirExpr, Vec<HirStmt>)>,
        else_body: Option<Vec<HirStmt>>,
    },
    While {
        cond: HirExpr,
        body: Vec<HirStmt>,
    },
    For {
        var: VarId,
        expr: HirExpr,
        body: Vec<HirStmt>,
    },
    Switch {
        expr: HirExpr,
        cases: Vec<(HirExpr, Vec<HirStmt>)>,
        otherwise: Option<Vec<HirStmt>>,
    },
    TryCatch {
        try_body: Vec<HirStmt>,
        catch_var: Option<VarId>,
        catch_body: Vec<HirStmt>,
    },
    Global(Vec<VarId>),
    Persistent(Vec<VarId>),
    Break,
    Continue,
    Return,
    Function {
        name: String,
        params: Vec<VarId>,
        outputs: Vec<VarId>,
        body: Vec<HirStmt>,
    },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<HirClassMember>,
    },
    Import { path: Vec<String>, wildcard: bool },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirClassMember {
    Properties(Vec<String>),
    Methods(Vec<HirStmt>),
    Events(Vec<String>),
    Enumeration(Vec<String>),
    Arguments(Vec<String>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirLValue {
    Var(VarId),
    Member(Box<HirExpr>, String),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirProgram {
    pub body: Vec<HirStmt>,
}

/// Result of lowering AST to HIR with full context tracking
#[derive(Debug, Clone)]
pub struct LoweringResult {
    pub hir: HirProgram,
    pub variables: HashMap<String, usize>,
    pub functions: HashMap<String, HirStmt>,
}

pub fn lower(prog: &AstProgram) -> Result<HirProgram, String> {
    let mut ctx = Ctx::new();
    let body = ctx.lower_stmts(&prog.body)?;
    Ok(HirProgram { body })
}

/// Lower AST to HIR with existing variable context for REPL
pub fn lower_with_context(
    prog: &AstProgram,
    existing_vars: &HashMap<String, usize>,
) -> Result<(HirProgram, HashMap<String, usize>), String> {
    let empty_functions = HashMap::new();
    let result = lower_with_full_context(prog, existing_vars, &empty_functions)?;
    Ok((result.hir, result.variables))
}

/// Lower AST to HIR with existing variable and function context for REPL
pub fn lower_with_full_context(
    prog: &AstProgram,
    existing_vars: &HashMap<String, usize>,
    existing_functions: &HashMap<String, HirStmt>,
) -> Result<LoweringResult, String> {
    let mut ctx = Ctx::new();

    // Pre-populate the context with existing variables
    for (name, var_id) in existing_vars {
        ctx.scopes[0].bindings.insert(name.clone(), VarId(*var_id));
        // Ensure var_types has enough capacity
        while ctx.var_types.len() <= *var_id {
            ctx.var_types.push(Type::Unknown);
        }
        // Update next_var to be at least one more than the highest existing var
        if *var_id >= ctx.next_var {
            ctx.next_var = var_id + 1;
        }
    }

    // Pre-populate the context with existing functions
    for (name, func_stmt) in existing_functions {
        ctx.functions.insert(name.clone(), func_stmt.clone());
    }

    let body = ctx.lower_stmts(&prog.body)?;

    // Extract all variable bindings (both existing and newly defined)
    let mut all_vars = HashMap::new();
    for (name, var_id) in &ctx.scopes[0].bindings {
        all_vars.insert(name.clone(), var_id.0);
    }

    Ok(LoweringResult {
        hir: HirProgram { body },
        variables: all_vars,
        functions: ctx.functions,
    })
}

/// Variable remapping utilities for function execution
/// These functions remap VarIds in HIR to create local variable contexts
pub mod remapping {
    use super::*;
    use std::collections::HashMap;

    /// Remap VarIds in HIR statements to create a local variable context for function execution
    pub fn remap_function_body(body: &[HirStmt], var_map: &HashMap<VarId, VarId>) -> Vec<HirStmt> {
        body.iter().map(|stmt| remap_stmt(stmt, var_map)).collect()
    }

    /// Remap VarIds in a single HIR statement
    pub fn remap_stmt(stmt: &HirStmt, var_map: &HashMap<VarId, VarId>) -> HirStmt {
        match stmt {
            HirStmt::ExprStmt(expr, suppressed) => {
                HirStmt::ExprStmt(remap_expr(expr, var_map), *suppressed)
            }
            HirStmt::Assign(var_id, expr, suppressed) => {
                let new_var_id = var_map.get(var_id).copied().unwrap_or(*var_id);
                HirStmt::Assign(new_var_id, remap_expr(expr, var_map), *suppressed)
            }
            HirStmt::MultiAssign(var_ids, expr, suppressed) => {
                let mapped: Vec<VarId> = var_ids
                    .iter()
                    .map(|v| var_map.get(v).copied().unwrap_or(*v))
                    .collect();
                HirStmt::MultiAssign(mapped, remap_expr(expr, var_map), *suppressed)
            }
            HirStmt::AssignLValue(lv, expr, suppressed) => {
                let remapped_lv = match lv {
                    super::HirLValue::Var(v) => super::HirLValue::Var(var_map.get(v).copied().unwrap_or(*v)),
                    super::HirLValue::Member(b, n) => super::HirLValue::Member(Box::new(remap_expr(b, var_map)), n.clone()),
                    super::HirLValue::Index(b, idxs) => super::HirLValue::Index(
                        Box::new(remap_expr(b, var_map)),
                        idxs.iter().map(|e| remap_expr(e, var_map)).collect(),
                    ),
                    super::HirLValue::IndexCell(b, idxs) => super::HirLValue::IndexCell(
                        Box::new(remap_expr(b, var_map)),
                        idxs.iter().map(|e| remap_expr(e, var_map)).collect(),
                    ),
                };
                HirStmt::AssignLValue(remapped_lv, remap_expr(expr, var_map), *suppressed)
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => HirStmt::If {
                cond: remap_expr(cond, var_map),
                then_body: remap_function_body(then_body, var_map),
                elseif_blocks: elseif_blocks
                    .iter()
                    .map(|(c, b)| (remap_expr(c, var_map), remap_function_body(b, var_map)))
                    .collect(),
                else_body: else_body.as_ref().map(|b| remap_function_body(b, var_map)),
            },
            HirStmt::While { cond, body } => HirStmt::While {
                cond: remap_expr(cond, var_map),
                body: remap_function_body(body, var_map),
            },
            HirStmt::For { var, expr, body } => {
                let new_var = var_map.get(var).copied().unwrap_or(*var);
                HirStmt::For {
                    var: new_var,
                    expr: remap_expr(expr, var_map),
                    body: remap_function_body(body, var_map),
                }
            }
            HirStmt::Switch { expr, cases, otherwise } => HirStmt::Switch {
                expr: remap_expr(expr, var_map),
                cases: cases
                    .iter()
                    .map(|(c, b)| (remap_expr(c, var_map), remap_function_body(b, var_map)))
                    .collect(),
                otherwise: otherwise.as_ref().map(|b| remap_function_body(b, var_map)),
            },
            HirStmt::TryCatch { try_body, catch_var, catch_body } => HirStmt::TryCatch {
                try_body: remap_function_body(try_body, var_map),
                catch_var: catch_var.map(|v| var_map.get(&v).copied().unwrap_or(v)),
                catch_body: remap_function_body(catch_body, var_map),
            },
            HirStmt::Global(vars) => HirStmt::Global(
                vars.iter()
                    .map(|v| var_map.get(v).copied().unwrap_or(*v))
                    .collect(),
            ),
            HirStmt::Persistent(vars) => HirStmt::Persistent(
                vars.iter()
                    .map(|v| var_map.get(v).copied().unwrap_or(*v))
                    .collect(),
            ),
            HirStmt::Break | HirStmt::Continue | HirStmt::Return => stmt.clone(),
            HirStmt::Function { .. } => stmt.clone(), // Functions shouldn't be nested in our current implementation
            HirStmt::ClassDef { name, super_class, members } => HirStmt::ClassDef {
                name: name.clone(),
                super_class: super_class.clone(),
                members: members
                    .iter()
                    .map(|m| match m {
                        HirClassMember::Properties(p) => HirClassMember::Properties(p.clone()),
                        HirClassMember::Events(e) => HirClassMember::Events(e.clone()),
                        HirClassMember::Enumeration(e) => HirClassMember::Enumeration(e.clone()),
                        HirClassMember::Arguments(a) => HirClassMember::Arguments(a.clone()),
                        HirClassMember::Methods(body) =>
                            HirClassMember::Methods(remap_function_body(body, var_map)),
                    })
                    .collect(),
            },
            HirStmt::Import { path, wildcard } => HirStmt::Import { path: path.clone(), wildcard: *wildcard },
        }
    }

    /// Remap VarIds in a HIR expression
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
            HirExprKind::Member(base, name) =>
                HirExprKind::Member(Box::new(remap_expr(base, var_map)), name.clone()),
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
        }
    }

    /// Collect all variable IDs referenced in a function body
    pub fn collect_function_variables(body: &[HirStmt]) -> std::collections::HashSet<VarId> {
        let mut vars = std::collections::HashSet::new();

        for stmt in body {
            collect_stmt_variables(stmt, &mut vars);
        }

        vars
    }

    fn collect_stmt_variables(stmt: &HirStmt, vars: &mut std::collections::HashSet<VarId>) {
        match stmt {
            HirStmt::ExprStmt(expr, _) => collect_expr_variables(expr, vars),
            HirStmt::Assign(var_id, expr, _) => {
                vars.insert(*var_id);
                collect_expr_variables(expr, vars);
            }
            HirStmt::MultiAssign(var_ids, expr, _) => {
                for v in var_ids { vars.insert(*v); }
                collect_expr_variables(expr, vars);
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
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
            HirStmt::While { cond, body } => {
                collect_expr_variables(cond, vars);
                for stmt in body {
                    collect_stmt_variables(stmt, vars);
                }
            }
            HirStmt::For { var, expr, body } => {
                vars.insert(*var);
                collect_expr_variables(expr, vars);
                for stmt in body {
                    collect_stmt_variables(stmt, vars);
                }
            }
            HirStmt::Switch { expr, cases, otherwise } => {
                collect_expr_variables(expr, vars);
                for (v, b) in cases {
                    collect_expr_variables(v, vars);
                    for s in b { collect_stmt_variables(s, vars); }
                }
                if let Some(b) = otherwise {
                    for s in b { collect_stmt_variables(s, vars); }
                }
            }
            HirStmt::TryCatch { try_body, catch_var, catch_body } => {
                if let Some(v) = catch_var { vars.insert(*v); }
                for s in try_body { collect_stmt_variables(s, vars); }
                for s in catch_body { collect_stmt_variables(s, vars); }
            }
            HirStmt::Global(vs) | HirStmt::Persistent(vs) => {
                for v in vs { vars.insert(*v); }
            }
            HirStmt::AssignLValue(lv, expr, _) => {
                match lv {
                    HirLValue::Var(v) => { vars.insert(*v); }
                    HirLValue::Member(base, _) => collect_expr_variables(base, vars),
                    HirLValue::Index(base, idxs) | HirLValue::IndexCell(base, idxs) => {
                        collect_expr_variables(base, vars);
                        for i in idxs { collect_expr_variables(i, vars); }
                    }
                }
                collect_expr_variables(expr, vars);
            }
            HirStmt::Break | HirStmt::Continue | HirStmt::Return => {}
            HirStmt::Function { .. } => {} // Nested functions not supported
            HirStmt::ClassDef { .. } => {}
            HirStmt::Import { .. } => {}
        }
    }

    fn collect_expr_variables(expr: &HirExpr, vars: &mut std::collections::HashSet<VarId>) {
        match &expr.kind {
            HirExprKind::Var(var_id) => {
                vars.insert(*var_id);
            }
            HirExprKind::Unary(_, e) => collect_expr_variables(e, vars),
            HirExprKind::Binary(left, _, right) => {
                collect_expr_variables(left, vars);
                collect_expr_variables(right, vars);
            }
            HirExprKind::Tensor(rows) => {
                for row in rows {
                    for e in row {
                        collect_expr_variables(e, vars);
                    }
                }
            }
            HirExprKind::Cell(rows) => {
                for row in rows {
                    for e in row { collect_expr_variables(e, vars); }
                }
            }
            HirExprKind::Index(base, indices) => {
                collect_expr_variables(base, vars);
                for idx in indices {
                    collect_expr_variables(idx, vars);
                }
            }
            HirExprKind::IndexCell(base, indices) => {
                collect_expr_variables(base, vars);
                for idx in indices { collect_expr_variables(idx, vars); }
            }
            HirExprKind::Range(start, step, end) => {
                collect_expr_variables(start, vars);
                if let Some(step_expr) = step {
                    collect_expr_variables(step_expr, vars);
                }
                collect_expr_variables(end, vars);
            }
            HirExprKind::Member(base, _) => collect_expr_variables(base, vars),
            HirExprKind::MethodCall(base, _, args) => {
                collect_expr_variables(base, vars);
                for a in args { collect_expr_variables(a, vars); }
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

    /// Create a variable mapping for function execution
    /// Maps function parameters, outputs, and all referenced variables to local indices starting from 0
    pub fn create_function_var_map(params: &[VarId], outputs: &[VarId]) -> HashMap<VarId, VarId> {
        let mut var_map = HashMap::new();
        let mut local_var_index = 0;

        // Map parameters to local indices first (they have priority)
        for param_id in params {
            var_map.insert(*param_id, VarId(local_var_index));
            local_var_index += 1;
        }

        // Map output variables to local indices
        for output_id in outputs {
            if !var_map.contains_key(output_id) {
                var_map.insert(*output_id, VarId(local_var_index));
                local_var_index += 1;
            }
        }

        var_map
    }

    /// Create a variable mapping for function execution that includes all variables referenced in the body
    pub fn create_complete_function_var_map(
        params: &[VarId],
        outputs: &[VarId],
        body: &[HirStmt],
    ) -> HashMap<VarId, VarId> {
        let mut var_map = HashMap::new();
        let mut local_var_index = 0;

        // Collect all variables referenced in the function body
        let all_vars = collect_function_variables(body);

        // Map parameters to local indices first (they have priority)
        for param_id in params {
            var_map.insert(*param_id, VarId(local_var_index));
            local_var_index += 1;
        }

        // Map output variables to local indices
        for output_id in outputs {
            if !var_map.contains_key(output_id) {
                var_map.insert(*output_id, VarId(local_var_index));
                local_var_index += 1;
            }
        }

        // Map any other variables referenced in the body
        for var_id in &all_vars {
            if !var_map.contains_key(var_id) {
                var_map.insert(*var_id, VarId(local_var_index));
                local_var_index += 1;
            }
        }

        var_map
    }
}

struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, VarId>,
}

struct Ctx {
    scopes: Vec<Scope>,
    var_types: Vec<Type>,
    next_var: usize,
    functions: HashMap<String, HirStmt>, // Track user-defined functions
}

impl Ctx {
    fn new() -> Self {
        Self {
            scopes: vec![Scope {
                parent: None,
                bindings: HashMap::new(),
            }],
            var_types: Vec::new(),
            next_var: 0,
            functions: HashMap::new(),
        }
    }

    fn push_scope(&mut self) -> usize {
        let parent = Some(self.scopes.len() - 1);
        self.scopes.push(Scope {
            parent,
            bindings: HashMap::new(),
        });
        self.scopes.len() - 1
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: String) -> VarId {
        let id = VarId(self.next_var);
        self.next_var += 1;
        let current = self.scopes.len() - 1;
        self.scopes[current].bindings.insert(name, id);
        self.var_types.push(Type::Unknown);
        id
    }

    fn lookup(&self, name: &str) -> Option<VarId> {
        let mut scope_idx = Some(self.scopes.len() - 1);
        while let Some(idx) = scope_idx {
            if let Some(id) = self.scopes[idx].bindings.get(name) {
                return Some(*id);
            }
            scope_idx = self.scopes[idx].parent;
        }
        None
    }

    fn is_constant(&self, name: &str) -> bool {
        // Check if name is a registered constant
        runmat_builtins::constants().iter().any(|c| c.name == name)
    }

    fn is_builtin_function(&self, name: &str) -> bool {
        // Check if name is a registered builtin function
        runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == name)
    }

    fn is_user_defined_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    fn is_function(&self, name: &str) -> bool {
        self.is_user_defined_function(name) || self.is_builtin_function(name)
    }

    fn lower_stmts(&mut self, stmts: &[AstStmt]) -> Result<Vec<HirStmt>, String> {
        stmts.iter().map(|s| self.lower_stmt(s)).collect()
    }

    fn lower_stmt(&mut self, stmt: &AstStmt) -> Result<HirStmt, String> {
        match stmt {
            AstStmt::ExprStmt(e, semicolon_terminated) => Ok(HirStmt::ExprStmt(
                self.lower_expr(e)?,
                *semicolon_terminated,
            )),
            AstStmt::Assign(name, expr, semicolon_terminated) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                let value = self.lower_expr(expr)?;
                if id.0 < self.var_types.len() {
                    self.var_types[id.0] = value.ty.clone();
                }
                Ok(HirStmt::Assign(id, value, *semicolon_terminated))
            }
            AstStmt::MultiAssign(names, expr, semicolon_terminated) => {
                let ids: Vec<VarId> = names
                    .iter()
                    .map(|n| match self.lookup(n) { Some(id) => id, None => self.define(n.clone()) })
                    .collect();
                let value = self.lower_expr(expr)?;
                Ok(HirStmt::MultiAssign(ids, value, *semicolon_terminated))
            }
            AstStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                let cond = self.lower_expr(cond)?;
                let then_body = self.lower_stmts(then_body)?;
                let mut elseif_vec = Vec::new();
                for (c, b) in elseif_blocks {
                    elseif_vec.push((self.lower_expr(c)?, self.lower_stmts(b)?));
                }
                let else_body = match else_body {
                    Some(b) => Some(self.lower_stmts(b)?),
                    None => None,
                };
                Ok(HirStmt::If {
                    cond,
                    then_body,
                    elseif_blocks: elseif_vec,
                    else_body,
                })
            }
            AstStmt::While { cond, body } => Ok(HirStmt::While {
                cond: self.lower_expr(cond)?,
                body: self.lower_stmts(body)?,
            }),
            AstStmt::For { var, expr, body } => {
                let id = match self.lookup(var) {
                    Some(id) => id,
                    None => self.define(var.clone()),
                };
                let expr = self.lower_expr(expr)?;
                let body = self.lower_stmts(body)?;
                Ok(HirStmt::For {
                    var: id,
                    expr,
                    body,
                })
            }
            AstStmt::Switch { expr, cases, otherwise } => {
                let control = self.lower_expr(expr)?;
                let mut cases_hir: Vec<(HirExpr, Vec<HirStmt>)> = Vec::new();
                for (v, b) in cases {
                    let ve = self.lower_expr(v)?;
                    let vb = self.lower_stmts(b)?;
                    cases_hir.push((ve, vb));
                }
                let otherwise_hir = otherwise
                    .as_ref()
                    .map(|b| self.lower_stmts(b))
                    .transpose()?;
                Ok(HirStmt::Switch { expr: control, cases: cases_hir, otherwise: otherwise_hir })
            }
            AstStmt::TryCatch { try_body, catch_var, catch_body } => {
                let try_hir = self.lower_stmts(try_body)?;
                let catch_var_id = match catch_var {
                    Some(name) => Some(match self.lookup(name) { Some(id) => id, None => self.define(name.clone()) }),
                    None => None,
                };
                let catch_hir = self.lower_stmts(catch_body)?;
                Ok(HirStmt::TryCatch { try_body: try_hir, catch_var: catch_var_id, catch_body: catch_hir })
            }
            AstStmt::Global(names) => {
                let ids: Vec<VarId> = names
                    .iter()
                    .map(|n| match self.lookup(n) { Some(id) => id, None => self.define(n.clone()) })
                    .collect();
                Ok(HirStmt::Global(ids))
            }
            AstStmt::Persistent(names) => {
                let ids: Vec<VarId> = names
                    .iter()
                    .map(|n| match self.lookup(n) { Some(id) => id, None => self.define(n.clone()) })
                    .collect();
                Ok(HirStmt::Persistent(ids))
            }
            AstStmt::Break => Ok(HirStmt::Break),
            AstStmt::Continue => Ok(HirStmt::Continue),
            AstStmt::Return => Ok(HirStmt::Return),
            AstStmt::Function {
                name,
                params,
                outputs,
                body,
            } => {
                self.push_scope();
                let param_ids: Vec<VarId> = params.iter().map(|p| self.define(p.clone())).collect();
                let output_ids: Vec<VarId> =
                    outputs.iter().map(|o| self.define(o.clone())).collect();
                let body_hir = self.lower_stmts(body)?;
                self.pop_scope();

                let func_stmt = HirStmt::Function {
                    name: name.clone(),
                    params: param_ids,
                    outputs: output_ids,
                    body: body_hir,
                };

                // Register the function in the context for future calls
                self.functions.insert(name.clone(), func_stmt.clone());

                Ok(func_stmt)
            }
            AstStmt::ClassDef { name, super_class, members } => {
                // Lightweight lowering of class blocks into HIR without deep semantic checks
                let members_hir = members
                    .iter()
                    .map(|m| match m {
                        parser::ClassMember::Properties(p) => HirClassMember::Properties(p.clone()),
                        parser::ClassMember::Events(e) => HirClassMember::Events(e.clone()),
                        parser::ClassMember::Enumeration(e) => HirClassMember::Enumeration(e.clone()),
                        parser::ClassMember::Arguments(a) => HirClassMember::Arguments(a.clone()),
                        parser::ClassMember::Methods(stmts) => {
                            match self.lower_stmts(stmts) {
                                Ok(s) => HirClassMember::Methods(s),
                                Err(_) => HirClassMember::Methods(Vec::new()),
                            }
                        }
                    })
                    .collect();
                Ok(HirStmt::ClassDef { name: name.clone(), super_class: super_class.clone(), members: members_hir })
            }
            AstStmt::AssignLValue(lv, rhs, suppressed) => {
                // Lower true lvalue assignment into HirStmt::AssignLValue
                let hir_lv = self.lower_lvalue(lv)?;
                let value = self.lower_expr(rhs)?;
                // If target is a plain variable, update its type from RHS
                if let HirLValue::Var(var_id) = hir_lv {
                    if var_id.0 < self.var_types.len() { self.var_types[var_id.0] = value.ty.clone(); }
                    return Ok(HirStmt::Assign(var_id, value, *suppressed));
                }
                Ok(HirStmt::AssignLValue(hir_lv, value, *suppressed))
            }
            AstStmt::Import { .. } => {
                // Import statements have no runtime effect in HIR
                if let AstStmt::Import { path, wildcard } = stmt {
                    Ok(HirStmt::Import { path: path.clone(), wildcard: *wildcard })
                } else { unreachable!() }
            }
        }
    }

    fn lower_expr(&mut self, expr: &AstExpr) -> Result<HirExpr, String> {
        use parser::Expr::*;
        let (kind, ty) = match expr {
            Number(n) => (HirExprKind::Number(n.clone()), Type::Num),
            String(s) => (HirExprKind::String(s.clone()), Type::String),
            Ident(name) => {
                // First check if it's a variable in scope; variables shadow constants
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
                    // Treat bare identifier as function call with no arguments (MATLAB style)
                    let return_type = self.infer_function_return_type(name, &[]);
                    (HirExprKind::FuncCall(name.clone(), vec![]), return_type)
                } else {
                    return Err(format!("undefined variable `{name}`"));
                }
            }
            Unary(op, e) => {
                let inner = self.lower_expr(e)?;
                let ty = inner.ty.clone();
                (HirExprKind::Unary(*op, Box::new(inner)), ty)
            }
            Binary(a, op, b) => {
                let left = self.lower_expr(a)?;
                let left_ty = left.ty.clone();
                let right = self.lower_expr(b)?;
                let right_ty = right.ty.clone();
                let ty = match op {
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Pow
                    | BinOp::LeftDiv => {
                        if matches!(left_ty, Type::Matrix { .. })
                            || matches!(right_ty, Type::Matrix { .. })
                        {
                            Type::matrix()
                        } else {
                            Type::Num
                        }
                    }
                    // Element-wise operations preserve the matrix type if either operand is a matrix
                    BinOp::ElemMul | BinOp::ElemDiv | BinOp::ElemPow | BinOp::ElemLeftDiv => {
                        if matches!(left_ty, Type::Matrix { .. })
                            || matches!(right_ty, Type::Matrix { .. })
                        {
                            Type::matrix()
                        } else {
                            Type::Num
                        }
                    }
                    // Comparison operations always return boolean
                    BinOp::Equal
                    | BinOp::NotEqual
                    | BinOp::Less
                    | BinOp::LessEqual
                    | BinOp::Greater
                    | BinOp::GreaterEqual => Type::Bool,
                    // Logical
                    BinOp::AndAnd | BinOp::OrOr | BinOp::BitAnd | BinOp::BitOr => Type::Bool,
                    BinOp::Colon => Type::matrix(),
                };
                (
                    HirExprKind::Binary(Box::new(left), *op, Box::new(right)),
                    ty,
                )
            }
            AnonFunc { params, body } => {
                // Lower body in a fresh scope with parameters bound to local VarIds
                let saved_len = self.scopes.len();
                self.push_scope();
                let mut param_ids: Vec<VarId> = Vec::with_capacity(params.len());
                for p in params {
                    param_ids.push(self.define(p.clone()));
                }
                let lowered_body = self.lower_expr(body)?;
                // restore scope
                while self.scopes.len() > saved_len { self.pop_scope(); }
                (HirExprKind::AnonFunc { params: param_ids, body: Box::new(lowered_body) }, Type::Unknown)
            }
            FuncHandle(name) => (HirExprKind::FuncHandle(name.clone()), Type::Unknown),
            FuncCall(name, args) => {
                let arg_exprs: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                let arg_exprs = arg_exprs?;

                // Check if 'name' refers to a variable in scope
                // If so, this is array indexing, not a function call
                if let Some(var_id) = self.lookup(name) {
                    // This is array indexing: variable(indices)
                    let var_ty = if var_id.0 < self.var_types.len() {
                        self.var_types[var_id.0].clone()
                    } else {
                        Type::Unknown
                    };
                    let var_expr = HirExpr {
                        kind: HirExprKind::Var(var_id),
                        ty: var_ty,
                    };
                    // Array indexing returns scalar for single element, matrix for slices
                    let index_result_type = Type::Num; // Both A(i) and A(i,j) return scalar
                    (
                        HirExprKind::Index(Box::new(var_expr), arg_exprs),
                        index_result_type,
                    )
                } else {
                    // This is a function call - determine return type based on function
                    let return_type = self.infer_function_return_type(name, &arg_exprs);
                    (HirExprKind::FuncCall(name.clone(), arg_exprs), return_type)
                }
            }
            Tensor(rows) => {
                let mut hir_rows = Vec::new();
                for row in rows {
                    let mut hir_row = Vec::new();
                    for expr in row {
                        hir_row.push(self.lower_expr(expr)?);
                    }
                    hir_rows.push(hir_row);
                }
                (HirExprKind::Tensor(hir_rows), Type::matrix())
            }
            Cell(rows) => {
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
            Index(expr, indices) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                let ty = base.ty.clone(); // Indexing preserves base type for now
                (HirExprKind::Index(Box::new(base), idx_exprs), ty)
            }
            IndexCell(expr, indices) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                (HirExprKind::IndexCell(Box::new(base), idx_exprs), Type::Unknown)
            }
            Range(start, step, end) => {
                let start_hir = self.lower_expr(start)?;
                let end_hir = self.lower_expr(end)?;
                let step_hir = step.as_ref().map(|s| self.lower_expr(s)).transpose()?;
                (
                    HirExprKind::Range(
                        Box::new(start_hir),
                        step_hir.map(Box::new),
                        Box::new(end_hir),
                    ),
                    Type::matrix(),
                )
            }
            Colon => (HirExprKind::Colon, Type::matrix()),
            EndKeyword => (HirExprKind::End, Type::Unknown),
            Member(base, name) => {
                let b = self.lower_expr(base)?;
                (HirExprKind::Member(Box::new(b), name.clone()), Type::Unknown)
            }
            MethodCall(base, name, args) => {
                let b = self.lower_expr(base)?;
                let lowered_args: Result<Vec<_>, _> = args.iter().map(|a| self.lower_expr(a)).collect();
                (HirExprKind::MethodCall(Box::new(b), name.clone(), lowered_args?), Type::Unknown)
            }
            MetaClass(name) => (HirExprKind::MetaClass(name.clone()), Type::String),
        };
        Ok(HirExpr { kind, ty })
    }

    fn lower_lvalue(&mut self, lv: &parser::LValue) -> Result<HirLValue, String> {
        use parser::LValue as ALV;
        Ok(match lv {
            ALV::Var(name) => {
                let id = match self.lookup(name) { Some(id) => id, None => self.define(name.clone()) };
                HirLValue::Var(id)
            }
            ALV::Member(base, name) => {
                let b = self.lower_expr(base)?;
                HirLValue::Member(Box::new(b), name.clone())
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

    /// Infer the return type of a function call based on the function name and arguments
    fn infer_function_return_type(&self, func_name: &str, args: &[HirExpr]) -> Type {
        // First check if it's a user-defined function
        if let Some(HirStmt::Function { outputs, body, .. }) = self.functions.get(func_name) {
            // Analyze the function body to infer the output type
            return self.infer_user_function_return_type(outputs, body, args);
        }

        // Check builtin functions using the proper signature system
        let builtin_functions = runmat_builtins::builtin_functions();
        for builtin in builtin_functions {
            if builtin.name == func_name {
                return builtin.return_type.clone();
            }
        }

        // No builtin function found with proper signature
        Type::Unknown
    }

    /// Analyze user-defined function body to infer return type
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
        // If multiple outputs supported, pick the first for scalar function calls context
        result_types.get(0).cloned().unwrap_or(Type::Unknown)
    }

    fn infer_outputs_types(&self, outputs: &[VarId], body: &[HirStmt]) -> Vec<Type> {
        use std::collections::HashMap;

        #[derive(Clone)]
        struct Analysis {
            exits: Vec<HashMap<VarId, Type>>,           // envs at return points
            fallthrough: Option<HashMap<VarId, Type>>, // env after block if not returned
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

        fn analyze_stmts(
            outputs: &[VarId],
            stmts: &[HirStmt],
            mut env: HashMap<VarId, Type>,
        ) -> Analysis {
            let mut exits = Vec::new();
            let mut i = 0usize;
            while i < stmts.len() {
                match &stmts[i] {
                    HirStmt::Assign(var, expr, _) => {
                        env.insert(*var, expr.ty.clone());
                    }
                    HirStmt::MultiAssign(vars, expr, _) => {
                        for v in vars {
                            env.insert(*v, expr.ty.clone());
                        }
                    }
                    HirStmt::ExprStmt(_, _) | HirStmt::Break | HirStmt::Continue => {}
                    HirStmt::Return => {
                        exits.push(env.clone());
                        return Analysis { exits, fallthrough: None };
                    }
                    HirStmt::If { cond: _, then_body, elseif_blocks, else_body } => {
                        let then_a = analyze_stmts(outputs, then_body, env.clone());
                        let mut out_env = then_a.fallthrough.unwrap_or_else(|| env.clone());
                        let mut all_exits = then_a.exits;
                        for (c, b) in elseif_blocks {
                            let _ = c; // cond type unused in analysis
                            let a = analyze_stmts(outputs, b, env.clone());
                            if let Some(f) = a.fallthrough { out_env = join_env(&out_env, &f); }
                            all_exits.extend(a.exits);
                        }
                        if let Some(else_body) = else_body {
                            let a = analyze_stmts(outputs, else_body, env.clone());
                            if let Some(f) = a.fallthrough { out_env = join_env(&out_env, &f); }
                            all_exits.extend(a.exits);
                        } else {
                            // no else: join with incoming env
                            out_env = join_env(&out_env, &env);
                        }
                        env = out_env;
                        exits.extend(all_exits);
                    }
                    HirStmt::While { cond: _, body } => {
                        // Approximate: analyze once and join with incoming env
                        let a = analyze_stmts(outputs, body, env.clone());
                        if let Some(f) = a.fallthrough { env = join_env(&env, &f); }
                        exits.extend(a.exits);
                    }
                    HirStmt::For { var, expr, body } => {
                        // Assign loop var type from expr type
                        env.insert(*var, expr.ty.clone());
                        let a = analyze_stmts(outputs, body, env.clone());
                        if let Some(f) = a.fallthrough { env = join_env(&env, &f); }
                        exits.extend(a.exits);
                    }
                    HirStmt::Switch { expr: _, cases, otherwise } => {
                        let mut out_env: Option<HashMap<VarId, Type>> = None;
                        for (_v, b) in cases {
                            let a = analyze_stmts(outputs, b, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = Some(match out_env { Some(curr) => join_env(&curr, &f), None => f });
                            }
                            exits.extend(a.exits);
                        }
                        if let Some(otherwise) = otherwise {
                            let a = analyze_stmts(outputs, otherwise, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = Some(match out_env { Some(curr) => join_env(&curr, &f), None => f });
                            }
                            exits.extend(a.exits);
                        } else {
                            out_env = Some(match out_env { Some(curr) => join_env(&curr, &env), None => env.clone() });
                        }
                        if let Some(f) = out_env { env = f; }
                    }
                    HirStmt::TryCatch { try_body, catch_var: _, catch_body } => {
                        let a_try = analyze_stmts(outputs, try_body, env.clone());
                        let a_catch = analyze_stmts(outputs, catch_body, env.clone());
                        let mut out_env = a_try.fallthrough.unwrap_or_else(|| env.clone());
                        if let Some(f) = a_catch.fallthrough { out_env = join_env(&out_env, &f); }
                        env = out_env;
                        exits.extend(a_try.exits);
                        exits.extend(a_catch.exits);
                    }
                    HirStmt::Global(_) | HirStmt::Persistent(_) => {}
                    HirStmt::Function { .. } => {}
                    HirStmt::ClassDef { .. } => {}
                    HirStmt::AssignLValue(_, expr, _) => {
                        // Update env conservatively based on RHS type (specific lvalue target unknown)
                        // No binding updated unless it's a plain variable (handled elsewhere)
                        let _ = &expr.ty;
                    }
                    HirStmt::Import { .. } => {}
                }
                i += 1;
            }
            Analysis { exits, fallthrough: Some(env) }
        }

        let initial_env: HashMap<VarId, Type> = HashMap::new();
        let analysis = analyze_stmts(outputs, body, initial_env);
        let mut per_output: Vec<Type> = vec![Type::Unknown; outputs.len()];
        let mut accumulate = |env: &std::collections::HashMap<VarId, Type>| {
            for (i, out) in outputs.iter().enumerate() {
                if let Some(t) = env.get(out) {
                    per_output[i] = join_type(&per_output[i], t);
                }
            }
        };
        for e in &analysis.exits { accumulate(e); }
        if let Some(f) = &analysis.fallthrough { accumulate(f); }
        per_output
    }
}

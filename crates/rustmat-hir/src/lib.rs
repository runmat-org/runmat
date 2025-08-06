use rustmat_parser::{
    self as parser, BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Type {
    Scalar,
    String,
    Matrix,
    Void,
    Unknown,
}

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
    Matrix(Vec<Vec<HirExpr>>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    FuncCall(String, Vec<HirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirStmt {
    ExprStmt(HirExpr),
    Assign(VarId, HirExpr),
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
    Break,
    Continue,
    Return,
    Function {
        name: String,
        params: Vec<VarId>,
        outputs: Vec<VarId>,
        body: Vec<HirStmt>,
    },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirProgram {
    pub body: Vec<HirStmt>,
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

    let body = ctx.lower_stmts(&prog.body)?;

    // Extract all variable bindings (both existing and newly defined)
    let mut all_vars = HashMap::new();
    for (name, var_id) in &ctx.scopes[0].bindings {
        all_vars.insert(name.clone(), var_id.0);
    }

    Ok((HirProgram { body }, all_vars))
}

struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, VarId>,
}

struct Ctx {
    scopes: Vec<Scope>,
    var_types: Vec<Type>,
    next_var: usize,
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
        rustmat_builtins::constants().iter().any(|c| c.name == name)
    }

    fn is_builtin_function(&self, name: &str) -> bool {
        // Check if name is a registered builtin function
        rustmat_builtins::builtins().iter().any(|b| b.name == name)
    }

    fn lower_stmts(&mut self, stmts: &[AstStmt]) -> Result<Vec<HirStmt>, String> {
        stmts.iter().map(|s| self.lower_stmt(s)).collect()
    }

    fn lower_stmt(&mut self, stmt: &AstStmt) -> Result<HirStmt, String> {
        match stmt {
            AstStmt::ExprStmt(e) => Ok(HirStmt::ExprStmt(self.lower_expr(e)?)),
            AstStmt::Assign(name, expr) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                let value = self.lower_expr(expr)?;
                if id.0 < self.var_types.len() {
                    self.var_types[id.0] = value.ty;
                }
                Ok(HirStmt::Assign(id, value))
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
                Ok(HirStmt::Function {
                    name: name.clone(),
                    params: param_ids,
                    outputs: output_ids,
                    body: body_hir,
                })
            }
        }
    }

    fn lower_expr(&mut self, expr: &AstExpr) -> Result<HirExpr, String> {
        use parser::Expr::*;
        let (kind, ty) = match expr {
            Number(n) => (HirExprKind::Number(n.clone()), Type::Scalar),
            String(s) => (HirExprKind::String(s.clone()), Type::String),
            Ident(name) => {
                // First check if it's a built-in constant
                if self.is_constant(name) {
                    (HirExprKind::Constant(name.clone()), Type::Scalar)
                } else if let Some(id) = self.lookup(name) {
                    let ty = if id.0 < self.var_types.len() {
                        self.var_types[id.0]
                    } else {
                        Type::Unknown
                    };
                    (HirExprKind::Var(id), ty)
                } else if self.is_builtin_function(name) {
                    // Treat bare identifier as function call with no arguments (MATLAB style)
                    let return_type = Self::infer_function_return_type(name, &[]);
                    (HirExprKind::FuncCall(name.clone(), vec![]), return_type)
                } else {
                    return Err(format!("undefined variable `{name}`"));
                }
            }
            Unary(op, e) => {
                let inner = self.lower_expr(e)?;
                let ty = inner.ty;
                (HirExprKind::Unary(*op, Box::new(inner)), ty)
            }
            Binary(a, op, b) => {
                let left = self.lower_expr(a)?;
                let left_ty = left.ty;
                let right = self.lower_expr(b)?;
                let right_ty = right.ty;
                let ty = match op {
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Pow
                    | BinOp::LeftDiv => {
                        if matches!(left_ty, Type::Matrix) || matches!(right_ty, Type::Matrix) {
                            Type::Matrix
                        } else {
                            Type::Scalar
                        }
                    }
                    // Element-wise operations preserve the matrix type if either operand is a matrix
                    BinOp::ElemMul
                    | BinOp::ElemDiv
                    | BinOp::ElemPow
                    | BinOp::ElemLeftDiv => {
                        if matches!(left_ty, Type::Matrix) || matches!(right_ty, Type::Matrix) {
                            Type::Matrix
                        } else {
                            Type::Scalar
                        }
                    }
                    BinOp::Colon => Type::Matrix,
                };
                (
                    HirExprKind::Binary(Box::new(left), *op, Box::new(right)),
                    ty,
                )
            }
            FuncCall(name, args) => {
                let arg_exprs: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                let arg_exprs = arg_exprs?;
                
                // Check if 'name' refers to a variable in scope
                // If so, this is array indexing, not a function call
                if let Some(var_id) = self.lookup(name) {
                    // This is array indexing: variable(indices)
                    let var_ty = if var_id.0 < self.var_types.len() {
                        self.var_types[var_id.0]
                    } else {
                        Type::Unknown
                    };
                    let var_expr = HirExpr {
                        kind: HirExprKind::Var(var_id),
                        ty: var_ty,
                    };
                    // Array indexing returns scalar for single element, matrix for slices
                    let index_result_type = if arg_exprs.len() == 1 {
                        Type::Scalar // Single element access A(i) returns scalar
                    } else {
                        Type::Scalar // 2D access A(i,j) also returns scalar
                    };
                    (HirExprKind::Index(Box::new(var_expr), arg_exprs), index_result_type)
                } else {
                    // This is a function call - determine return type based on function
                    let return_type = Self::infer_function_return_type(name, &arg_exprs);
                    (HirExprKind::FuncCall(name.clone(), arg_exprs), return_type)
                }
            }
            Matrix(rows) => {
                let mut hir_rows = Vec::new();
                for row in rows {
                    let mut hir_row = Vec::new();
                    for expr in row {
                        hir_row.push(self.lower_expr(expr)?);
                    }
                    hir_rows.push(hir_row);
                }
                (HirExprKind::Matrix(hir_rows), Type::Matrix)
            }
            Index(expr, indices) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                let ty = base.ty; // Indexing preserves base type for now
                (HirExprKind::Index(Box::new(base), idx_exprs), ty)
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
                    Type::Matrix,
                )
            }
            Colon => (HirExprKind::Colon, Type::Matrix),
        };
        Ok(HirExpr { kind, ty })
    }

    /// Infer the return type of a function call based on the function name and arguments
    fn infer_function_return_type(func_name: &str, args: &[HirExpr]) -> Type {
        match func_name {
            // Array generation functions - always return Matrix
            "linspace" | "logspace" | "zeros" | "ones" | "eye" | "rand" | "range" | "meshgrid" => Type::Matrix,
            
            // Mathematical functions that can work element-wise
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | 
            "asinh" | "acosh" | "atanh" | "ln" | "exp" | "exp2" | "exp10" |
            "round" | "floor" | "ceil" | "trunc" | "fract" | "sign" | "gamma" | "factorial" |
            "real" | "imag" | "angle" => {
                // If any argument is a matrix, return matrix (element-wise operation)
                if args.iter().any(|arg| matches!(arg.ty, Type::Matrix)) {
                    Type::Matrix
                } else {
                    Type::Scalar
                }
            }
            
            // Statistical functions - return scalar for matrices, preserve type for scalars
            "sum" | "mean" | "std" | "var" | "min" | "max" => {
                if args.len() == 1 && matches!(args[0].ty, Type::Matrix) {
                    Type::Scalar // sum([1,2,3]) = 6
                } else if args.len() >= 2 {
                    Type::Scalar // min(a,b) = scalar
                } else {
                    Type::Scalar // Default to scalar
                }
            }
            
            // Binary operations between scalars
            "pow" | "atan2" if args.len() == 2 => {
                if args.iter().any(|arg| matches!(arg.ty, Type::Matrix)) {
                    Type::Matrix // Element-wise if any operand is matrix
                } else {
                    Type::Scalar
                }
            }
            
            // Fill function creates matrix of specified size
            "fill" => Type::Matrix,
            
            // Constants and special functions are scalars
            "pi" | "e" | "inf" | "nan" | "eps" | "sqrt2" | "log2" | "log10" => Type::Scalar,
            
            // Default: assume scalar return
            _ => Type::Scalar,
        }
    }
}

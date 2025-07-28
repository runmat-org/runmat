use rustmat_parser::{
    self as parser, BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp,
};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    Scalar,
    Matrix,
    Void,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

#[derive(Debug, PartialEq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
}

#[derive(Debug, PartialEq)]
pub enum HirExprKind {
    Number(String),
    Var(VarId),
    Unary(UnOp, Box<HirExpr>),
    Binary(Box<HirExpr>, BinOp, Box<HirExpr>),
    Matrix(Vec<Vec<HirExpr>>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
}

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub struct HirProgram {
    pub body: Vec<HirStmt>,
}

pub fn lower(prog: &AstProgram) -> Result<HirProgram, String> {
    let mut ctx = Ctx::new();
    let body = ctx.lower_stmts(&prog.body)?;
    Ok(HirProgram { body })
}

struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, VarId>,
}

struct Ctx {
    scopes: Vec<Scope>,
    next_var: usize,
}

impl Ctx {
    fn new() -> Self {
        Self {
            scopes: vec![Scope {
                parent: None,
                bindings: HashMap::new(),
            }],
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
                let scope_idx = self.push_scope();
                let param_ids: Vec<VarId> = params.iter().map(|p| self.define(p.clone())).collect();
                let output_ids: Vec<VarId> =
                    outputs.iter().map(|o| self.define(o.clone())).collect();
                let body_hir = self.lower_stmts(body)?;
                self.scopes.truncate(scope_idx); // pop to previous
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
        let kind = match expr {
            Number(n) => HirExprKind::Number(n.clone()),
            Ident(name) => {
                let id = self
                    .lookup(name)
                    .ok_or_else(|| format!("undefined variable `{}`", name))?;
                HirExprKind::Var(id)
            }
            Unary(op, e) => HirExprKind::Unary(*op, Box::new(self.lower_expr(e)?)),
            Binary(a, op, b) => HirExprKind::Binary(
                Box::new(self.lower_expr(a)?),
                *op,
                Box::new(self.lower_expr(b)?),
            ),
            Matrix(rows) => {
                let rows_hir: Result<Vec<Vec<HirExpr>>, String> = rows
                    .iter()
                    .map(|r| r.iter().map(|e| self.lower_expr(e)).collect())
                    .collect();
                HirExprKind::Matrix(rows_hir?)
            }
            Index(base, idxs) => HirExprKind::Index(
                Box::new(self.lower_expr(base)?),
                idxs.iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_, _>>()?,
            ),
            Range(start, maybe_step, end) => HirExprKind::Range(
                Box::new(self.lower_expr(start)?),
                match maybe_step {
                    Some(e) => Some(Box::new(self.lower_expr(e)?)),
                    None => None,
                },
                Box::new(self.lower_expr(end)?),
            ),
            Colon => HirExprKind::Colon,
        };
        let ty = self.infer_type(expr);
        Ok(HirExpr { kind, ty })
    }

    fn infer_type(&self, expr: &AstExpr) -> Type {
        use parser::Expr::*;
        match expr {
            Number(_) => Type::Scalar,
            Matrix(_) => Type::Matrix,
            Range(_, _, _) => Type::Matrix,
            Unary(_, e) => self.infer_type(e),
            Binary(a, op, b) => {
                let left = self.infer_type(a);
                let right = self.infer_type(b);
                match op {
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Pow
                    | BinOp::LeftDiv => {
                        if matches!(left, Type::Matrix) || matches!(right, Type::Matrix) {
                            Type::Matrix
                        } else {
                            Type::Scalar
                        }
                    }
                    BinOp::Colon => Type::Matrix,
                }
            }
            Ident(_) => Type::Unknown,
            Index(_, _) => Type::Unknown,
            Colon => Type::Unknown,
        }
    }
}

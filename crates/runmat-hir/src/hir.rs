use crate::{Span, Type, VarId};
use runmat_parser as parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirExprKind {
    Number(String),
    String(String),
    Var(VarId),
    Constant(String),
    Unary(parser::UnOp, Box<HirExpr>),
    Binary(Box<HirExpr>, parser::BinOp, Box<HirExpr>),
    Tensor(Vec<Vec<HirExpr>>),
    Cell(Vec<Vec<HirExpr>>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    End,
    Member(Box<HirExpr>, String),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    DottedInvoke(Box<HirExpr>, String, Vec<HirExpr>),
    MethodCall(Box<HirExpr>, String, Vec<HirExpr>),
    AnonFunc {
        params: Vec<VarId>,
        body: Box<HirExpr>,
    },
    FuncHandle(String),
    FuncCall(String, Vec<HirExpr>),
    MetaClass(String),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirStmt {
    ExprStmt(HirExpr, bool, Span),
    Assign(VarId, HirExpr, bool, Span),
    MultiAssign(Vec<Option<VarId>>, HirExpr, bool, Span),
    AssignLValue(HirLValue, HirExpr, bool, Span),
    If {
        cond: HirExpr,
        then_body: Vec<HirStmt>,
        elseif_blocks: Vec<(HirExpr, Vec<HirStmt>)>,
        else_body: Option<Vec<HirStmt>>,
        span: Span,
    },
    While {
        cond: HirExpr,
        body: Vec<HirStmt>,
        span: Span,
    },
    For {
        var: VarId,
        expr: HirExpr,
        body: Vec<HirStmt>,
        span: Span,
    },
    Switch {
        expr: HirExpr,
        cases: Vec<(HirExpr, Vec<HirStmt>)>,
        otherwise: Option<Vec<HirStmt>>,
        span: Span,
    },
    TryCatch {
        try_body: Vec<HirStmt>,
        catch_var: Option<VarId>,
        catch_body: Vec<HirStmt>,
        span: Span,
    },
    Global(Vec<(VarId, String)>, Span),
    Persistent(Vec<(VarId, String)>, Span),
    Break(Span),
    Continue(Span),
    Return(Span),
    Function {
        name: String,
        params: Vec<VarId>,
        outputs: Vec<VarId>,
        body: Vec<HirStmt>,
        has_varargin: bool,
        has_varargout: bool,
        span: Span,
    },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<HirClassMember>,
        span: Span,
    },
    Import {
        path: Vec<String>,
        wildcard: bool,
        span: Span,
    },
}

impl HirExpr {
    pub fn span(&self) -> Span {
        self.span
    }
}

impl HirStmt {
    pub fn span(&self) -> Span {
        match self {
            HirStmt::ExprStmt(_, _, span)
            | HirStmt::Assign(_, _, _, span)
            | HirStmt::MultiAssign(_, _, _, span)
            | HirStmt::AssignLValue(_, _, _, span)
            | HirStmt::Global(_, span)
            | HirStmt::Persistent(_, span)
            | HirStmt::Break(span)
            | HirStmt::Continue(span)
            | HirStmt::Return(span) => *span,
            HirStmt::If { span, .. }
            | HirStmt::While { span, .. }
            | HirStmt::For { span, .. }
            | HirStmt::Switch { span, .. }
            | HirStmt::TryCatch { span, .. }
            | HirStmt::Function { span, .. }
            | HirStmt::ClassDef { span, .. }
            | HirStmt::Import { span, .. } => *span,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirClassMember {
    Properties {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Methods {
        attributes: Vec<parser::Attr>,
        body: Vec<HirStmt>,
    },
    Events {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Enumeration {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
    Arguments {
        attributes: Vec<parser::Attr>,
        names: Vec<String>,
    },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum HirLValue {
    Var(VarId),
    Member(Box<HirExpr>, String),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct HirProgram {
    pub body: Vec<HirStmt>,
    #[serde(default)]
    pub var_types: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct LoweringResult {
    pub hir: HirProgram,
    pub variables: HashMap<String, usize>,
    pub functions: HashMap<String, HirStmt>,
    pub var_types: Vec<Type>,
    pub var_names: HashMap<VarId, String>,
    pub inferred_globals: HashMap<VarId, Type>,
    pub inferred_function_envs: HashMap<String, HashMap<VarId, Type>>,
    pub inferred_function_returns: HashMap<String, Vec<Type>>,
}

use runmat_parser::{
    self as parser, BinOp, Expr as AstExpr, Program as AstProgram, Stmt as AstStmt, UnOp,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

// Re-export Type from builtins for consistency
pub use runmat_builtins::Type;

pub type Span = runmat_parser::Span;

const DEFAULT_ERROR_NAMESPACE: &str = "MATLAB";
static ERROR_NAMESPACE: OnceLock<RwLock<String>> = OnceLock::new();
static CONST_NUM_LOOKUP: OnceLock<HashMap<String, f64>> = OnceLock::new();

fn error_namespace_store() -> &'static RwLock<String> {
    ERROR_NAMESPACE.get_or_init(|| RwLock::new(DEFAULT_ERROR_NAMESPACE.to_string()))
}

pub fn set_error_namespace(namespace: &str) {
    let namespace = if namespace.trim().is_empty() {
        DEFAULT_ERROR_NAMESPACE.to_string()
    } else {
        namespace.to_string()
    };
    if let Ok(mut guard) = error_namespace_store().write() {
        *guard = namespace;
    }
}

fn error_namespace() -> String {
    error_namespace_store()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| DEFAULT_ERROR_NAMESPACE.to_string())
}

#[derive(Debug, Clone)]
pub struct SemanticError {
    pub message: String,
    pub span: Option<Span>,
    pub identifier: Option<String>,
}

impl SemanticError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            identifier: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = Some(identifier.into());
        self
    }
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SemanticError {}

impl From<String> for SemanticError {
    fn from(value: String) -> Self {
        SemanticError::new(value)
    }
}

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    pub functions: &'a HashMap<String, HirStmt>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(
        variables: &'a HashMap<String, usize>,
        functions: &'a HashMap<String, HirStmt>,
    ) -> Self {
        Self {
            variables,
            functions,
        }
    }

    pub fn empty() -> Self {
        static EMPTY_VARS: OnceLock<HashMap<String, usize>> = OnceLock::new();
        static EMPTY_FUNCS: OnceLock<HashMap<String, HirStmt>> = OnceLock::new();
        Self {
            variables: EMPTY_VARS.get_or_init(HashMap::new),
            functions: EMPTY_FUNCS.get_or_init(HashMap::new),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceId(pub usize);

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
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
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
    ExprStmt(HirExpr, bool, Span), // Expression and whether it's semicolon-terminated (suppressed)
    Assign(VarId, HirExpr, bool, Span), // Variable, Expression, and whether it's semicolon-terminated (suppressed)
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
    // Carry both VarId and its canonical source name for cross-unit/global binding
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

/// Result of lowering AST to HIR with full context tracking
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HirDiagnosticSeverity {
    Warning,
    Information,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirDiagnostic {
    pub message: String,
    pub span: Span,
    pub code: &'static str,
    pub severity: HirDiagnosticSeverity,
}

fn merge_span(lhs: Span, rhs: Span) -> Span {
    Span {
        start: lhs.start.min(rhs.start),
        end: lhs.end.max(rhs.end),
    }
}

pub fn lower(
    prog: &AstProgram,
    context: &LoweringContext<'_>,
) -> Result<LoweringResult, SemanticError> {
    let mut ctx = Ctx::new();

    for (name, var_id) in context.variables {
        ctx.scopes[0].bindings.insert(name.clone(), VarId(*var_id));
        while ctx.var_types.len() <= *var_id {
            ctx.var_types.push(Type::Unknown);
        }
        while ctx.var_names.len() <= *var_id {
            ctx.var_names.push(None);
        }
        ctx.var_names[*var_id] = Some(name.clone());
        if *var_id >= ctx.next_var {
            ctx.next_var = var_id + 1;
        }
    }

    for (name, func_stmt) in context.functions {
        ctx.functions.insert(name.clone(), func_stmt.clone());
    }

    let body = ctx.lower_stmts(&prog.body)?;
    let var_types = ctx.var_types.clone();
    let hir = HirProgram { body, var_types };
    let _ = infer_function_output_types(&hir);
    validate_classdefs(&hir)?;

    let mut variables: HashMap<String, usize> = HashMap::new();
    for (name, var_id) in ctx.scopes[0].bindings.iter() {
        variables.insert(name.clone(), var_id.0);
    }
    let mut var_names = HashMap::new();
    for (idx, name_opt) in ctx.var_names.iter().enumerate() {
        if let Some(name) = name_opt {
            var_names.insert(VarId(idx), name.clone());
        }
    }

    let inferred_function_returns = infer_function_output_types(&hir);
    let inferred_function_envs = infer_function_variable_types(&hir);
    let inferred_globals = infer_global_variable_types(&hir, &inferred_function_returns);

    Ok(LoweringResult {
        hir,
        variables,
        functions: ctx.functions,
        var_types: ctx.var_types,
        var_names,
        inferred_globals,
        inferred_function_envs,
        inferred_function_returns,
    })
}

fn logical_binary_result(lhs: &Type, rhs: &Type) -> Type {
    match (lhs, rhs) {
        (Type::Tensor { shape: Some(a) }, Type::Tensor { shape: Some(b) })
        | (Type::Logical { shape: Some(a) }, Type::Logical { shape: Some(b) })
        | (Type::Tensor { shape: Some(a) }, Type::Logical { shape: Some(b) })
        | (Type::Logical { shape: Some(a) }, Type::Tensor { shape: Some(b) }) => Type::Logical {
            shape: Some(runmat_builtins::shape_rules::broadcast_shapes(a, b)),
        },
        (Type::Tensor { shape: Some(a) }, Type::Num)
        | (Type::Tensor { shape: Some(a) }, Type::Int)
        | (Type::Tensor { shape: Some(a) }, Type::Bool)
        | (Type::Logical { shape: Some(a) }, Type::Num)
        | (Type::Logical { shape: Some(a) }, Type::Int)
        | (Type::Logical { shape: Some(a) }, Type::Bool)
        | (Type::Num, Type::Tensor { shape: Some(a) })
        | (Type::Int, Type::Tensor { shape: Some(a) })
        | (Type::Bool, Type::Tensor { shape: Some(a) })
        | (Type::Num, Type::Logical { shape: Some(a) })
        | (Type::Int, Type::Logical { shape: Some(a) })
        | (Type::Bool, Type::Logical { shape: Some(a) }) => Type::Logical {
            shape: Some(a.clone()),
        },
        (Type::Tensor { .. }, _)
        | (_, Type::Tensor { .. })
        | (Type::Logical { .. }, _)
        | (_, Type::Logical { .. }) => Type::logical(),
        _ => Type::Bool,
    }
}

fn eval_const_num(expr: &HirExpr) -> Option<f64> {
    fn numeric_value(value: &runmat_builtins::Value) -> Option<f64> {
        use runmat_builtins::IntValue;

        match value {
            runmat_builtins::Value::Num(v) => Some(*v),
            runmat_builtins::Value::Int(int_value) => match int_value {
                IntValue::I8(v) => Some(*v as f64),
                IntValue::I16(v) => Some(*v as f64),
                IntValue::I32(v) => Some(*v as f64),
                IntValue::I64(v) => Some(*v as f64),
                IntValue::U8(v) => Some(*v as f64),
                IntValue::U16(v) => Some(*v as f64),
                IntValue::U32(v) => Some(*v as f64),
                IntValue::U64(v) => Some(*v as f64),
            },
            runmat_builtins::Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    fn const_numeric_value(name: &str) -> Option<f64> {
        let map = CONST_NUM_LOOKUP.get_or_init(|| {
            let mut out: HashMap<String, f64> = HashMap::new();
            for constant in runmat_builtins::constants() {
                if let Some(value) = numeric_value(&constant.value) {
                    out.insert(constant.name.to_ascii_lowercase(), value);
                }
            }
            out
        });
        map.get(&name.to_ascii_lowercase()).copied()
    }

    match &expr.kind {
        HirExprKind::Number(text) => text.parse::<f64>().ok(),
        HirExprKind::Constant(name) => const_numeric_value(name),
        HirExprKind::Unary(op, inner) => {
            let value = eval_const_num(inner)?;
            match op {
                parser::UnOp::Plus => Some(value),
                parser::UnOp::Minus => Some(-value),
                _ => None,
            }
        }
        HirExprKind::Binary(lhs, op, rhs) => {
            let a = eval_const_num(lhs)?;
            let b = eval_const_num(rhs)?;
            match op {
                parser::BinOp::Add => Some(a + b),
                parser::BinOp::Sub => Some(a - b),
                parser::BinOp::Mul | parser::BinOp::ElemMul => Some(a * b),
                parser::BinOp::Div | parser::BinOp::ElemDiv => Some(a / b),
                parser::BinOp::LeftDiv | parser::BinOp::ElemLeftDiv => Some(b / a),
                parser::BinOp::Pow | parser::BinOp::ElemPow => Some(a.powf(b)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn infer_expr_type_with_env(
    expr: &HirExpr,
    env: &std::collections::HashMap<VarId, Type>,
    func_returns: &std::collections::HashMap<String, Vec<Type>>,
) -> Type {
    fn unify_tensor(a: &Type, b: &Type) -> Type {
        match (a, b) {
            (Type::Tensor { shape: sa }, Type::Tensor { shape: sb }) => match (sa, sb) {
                (Some(sa), Some(sb)) => {
                    let maxr = sa.len().max(sb.len());
                    let mut out: Vec<Option<usize>> = Vec::with_capacity(maxr);
                    for i in 0..maxr {
                        let da = sa.get(i).cloned().unwrap_or(None);
                        let db = sb.get(i).cloned().unwrap_or(None);
                        let d = match (da, db) {
                            (Some(a), Some(b)) => {
                                if a == b {
                                    Some(a)
                                } else if a == 1 {
                                    Some(b)
                                } else if b == 1 {
                                    Some(a)
                                } else {
                                    None
                                }
                            }
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                        out.push(d);
                    }
                    Type::Tensor { shape: Some(out) }
                }
                _ => Type::tensor(),
            },
            (Type::Tensor { .. }, _) | (_, Type::Tensor { .. }) => Type::tensor(),
            _ => Type::tensor(),
        }
    }

    use HirExprKind as K;

    match &expr.kind {
        K::Number(_) => Type::Num,
        K::String(_) => Type::String,
        K::Constant(_) => Type::Num,
        K::Var(id) => env.get(id).cloned().unwrap_or(Type::Unknown),
        K::Unary(_, e) => infer_expr_type_with_env(e, env, func_returns),
        K::Binary(a, op, b) => {
            let ta = infer_expr_type_with_env(a, env, func_returns);
            let tb = infer_expr_type_with_env(b, env, func_returns);
            match op {
                parser::BinOp::Mul => runmat_builtins::shape_rules::matmul_output_type(&ta, &tb),
                parser::BinOp::LeftDiv => {
                    runmat_builtins::shape_rules::left_divide_output_type(&ta, &tb)
                }
                parser::BinOp::Div => {
                    runmat_builtins::shape_rules::right_divide_output_type(&ta, &tb)
                }
                parser::BinOp::Add
                | parser::BinOp::Sub
                | parser::BinOp::Pow
                | parser::BinOp::ElemMul
                | parser::BinOp::ElemDiv
                | parser::BinOp::ElemPow
                | parser::BinOp::ElemLeftDiv => {
                    if matches!(ta, Type::Tensor { .. }) || matches!(tb, Type::Tensor { .. }) {
                        unify_tensor(&ta, &tb)
                    } else {
                        Type::Num
                    }
                }
                parser::BinOp::Equal
                | parser::BinOp::NotEqual
                | parser::BinOp::Less
                | parser::BinOp::LessEqual
                | parser::BinOp::Greater
                | parser::BinOp::GreaterEqual => logical_binary_result(&ta, &tb),
                parser::BinOp::AndAnd
                | parser::BinOp::OrOr
                | parser::BinOp::BitAnd
                | parser::BinOp::BitOr => logical_binary_result(&ta, &tb),
                parser::BinOp::Colon => runmat_builtins::shape_rules::infer_range_shape(
                    eval_const_num(a),
                    None,
                    eval_const_num(b),
                )
                .map(|shape| Type::Tensor { shape: Some(shape) })
                .unwrap_or_else(Type::tensor),
            }
        }
        K::Tensor(rows) => {
            let mut row_types: Vec<Vec<Type>> = Vec::new();
            for row in rows {
                let mut types = Vec::new();
                for e in row {
                    types.push(infer_expr_type_with_env(e, env, func_returns));
                }
                row_types.push(types);
            }
            if let Some(shape) = runmat_builtins::shape_rules::concat_shape(&row_types) {
                return Type::Tensor { shape: Some(shape) };
            }
            let r = rows.len();
            let c = rows.iter().map(|row| row.len()).max().unwrap_or(0);
            if r > 0 && rows.iter().all(|row| row.len() == c) {
                Type::tensor_with_shape(vec![r, c])
            } else {
                Type::tensor()
            }
        }
        K::Cell(rows) => {
            let mut elem_ty: Option<Type> = None;
            let mut len: usize = 0;
            for row in rows {
                for e in row {
                    let t = infer_expr_type_with_env(e, env, func_returns);
                    elem_ty = Some(match elem_ty {
                        Some(curr) => curr.unify(&t),
                        None => t,
                    });
                    len += 1;
                }
            }
            Type::Cell {
                element_type: elem_ty.map(Box::new),
                length: Some(len),
            }
        }
        K::Index(base, idxs) => {
            let bt = infer_expr_type_with_env(base, env, func_returns);
            let idx_types: Vec<Type> = idxs
                .iter()
                .map(|e| infer_expr_type_with_env(e, env, func_returns))
                .collect();
            runmat_builtins::shape_rules::index_output_type(&bt, &idx_types)
        }
        K::IndexCell(base, idxs) => {
            let bt = infer_expr_type_with_env(base, env, func_returns);
            if let Type::Cell {
                element_type: Some(t),
                ..
            } = bt
            {
                let scalar = idxs.len() == 1
                    && matches!(
                        infer_expr_type_with_env(&idxs[0], env, func_returns),
                        Type::Int | Type::Num | Type::Bool | Type::Tensor { .. }
                    );
                if scalar {
                    *t
                } else {
                    Type::Unknown
                }
            } else {
                Type::Unknown
            }
        }
        K::Range(start, step, end) => runmat_builtins::shape_rules::infer_range_shape(
            eval_const_num(start),
            step.as_ref().and_then(|s| eval_const_num(s)),
            eval_const_num(end),
        )
        .map(|shape| Type::Tensor { shape: Some(shape) })
        .unwrap_or_else(Type::tensor),
        K::FuncCall(name, _args) => {
            let lower_name = name.to_ascii_lowercase();
            if matches!(
                lower_name.as_str(),
                "zeros" | "ones" | "rand" | "randn" | "eye"
            ) {
                let mut dims: Vec<Option<usize>> = Vec::with_capacity(_args.len());
                for arg in _args {
                    if let Some(value) = eval_const_num(arg) {
                        if value.is_finite() {
                            let rounded = value.round();
                            if (value - rounded).abs() <= 1e-9 && rounded >= 0.0 {
                                dims.push(Some(rounded as usize));
                                continue;
                            }
                        }
                    }
                    dims.push(None);
                }
                if let Some(shape) =
                    runmat_builtins::shape_rules::constructor_shape_from_dims(&dims)
                {
                    return Type::Tensor { shape: Some(shape) };
                }
            }
            if lower_name == "repmat" && _args.len() >= 2 {
                let input_ty = infer_expr_type_with_env(&_args[0], env, func_returns);
                let input_shape = match input_ty {
                    Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                        Some(shape)
                    }
                    _ => None,
                };
                if let Some(input_shape) = input_shape {
                    let mut reps: Vec<Option<usize>> = Vec::new();
                    for arg in _args.iter().skip(1) {
                        if let Some(value) = eval_const_num(arg) {
                            if value.is_finite() {
                                let rounded = value.round();
                                if (value - rounded).abs() <= 1e-9 && rounded >= 0.0 {
                                    reps.push(Some(rounded as usize));
                                    continue;
                                }
                            }
                        }
                        reps.push(None);
                    }
                    if let Some(shape) =
                        runmat_builtins::shape_rules::repmat_shape(&input_shape, &reps)
                    {
                        return Type::Tensor { shape: Some(shape) };
                    }
                }
            }
            if let Some(v) = func_returns.get(name) {
                v.first().cloned().unwrap_or(Type::Unknown)
            } else {
                let arg_types: Vec<Type> = _args
                    .iter()
                    .map(|arg| infer_expr_type_with_env(arg, env, func_returns))
                    .collect();
                let builtins = runmat_builtins::builtin_functions();
                if let Some(b) = builtins.iter().find(|b| b.name == *name) {
                    b.infer_return_type(&arg_types)
                } else {
                    Type::Unknown
                }
            }
        }
        K::MethodCall(_, _, _) => Type::Unknown,
        K::Member(base, _) => {
            let _bt = infer_expr_type_with_env(base, env, func_returns);
            Type::Unknown
        }
        K::MemberDynamic(_, _) => Type::Unknown,
        K::AnonFunc { .. } => Type::Function {
            params: vec![Type::Unknown],
            returns: Box::new(Type::Unknown),
        },
        K::FuncHandle(_) => Type::Function {
            params: vec![Type::Unknown],
            returns: Box::new(Type::Unknown),
        },
        K::MetaClass(_) => Type::String,
        K::End => Type::Unknown,
        K::Colon => Type::tensor(),
    }
}

/// Infer output types for each function defined in the program using a flow-sensitive, block-structured
/// dataflow analysis over the function body. Returns a mapping from function name to per-output types.
pub fn infer_function_output_types(
    prog: &HirProgram,
) -> std::collections::HashMap<String, Vec<Type>> {
    use std::collections::HashMap;

    fn infer_expr_type(
        expr: &HirExpr,
        env: &HashMap<VarId, Type>,
        func_returns: &HashMap<String, Vec<Type>>,
    ) -> Type {
        infer_expr_type_with_env(expr, env, func_returns)
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

    #[allow(clippy::only_used_in_recursion)]
    #[allow(clippy::type_complexity, clippy::only_used_in_recursion)]
    fn analyze_stmts(
        #[allow(clippy::only_used_in_recursion)] _outputs: &[VarId],
        stmts: &[HirStmt],
        mut env: HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
        func_defs: &HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)>,
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
                        // Start from summary
                        let mut per_out: Vec<Type> = returns.get(name).cloned().unwrap_or_default();
                        // If summary missing/unknown, try simple callsite fallback using func_defs and argument types
                        let needs_fallback = per_out.is_empty()
                            || per_out.iter().any(|t| matches!(t, Type::Unknown));
                        if needs_fallback {
                            if let Some((params, outs, body)) = func_defs.get(name).cloned() {
                                // Seed param env with argument types at callsite by reusing current env typing
                                let mut penv: HashMap<VarId, Type> = HashMap::new();
                                // We don't have direct access to call args here (expr doesn't carry), so default to Num for simplicity when outputs are computed from params via arithmetic; otherwise Unknown
                                // Heuristic: assume params are Num when used in arithmetic contexts; conservative elsewhere
                                for p in params {
                                    penv.insert(p, Type::Num);
                                }
                                // Single pass: collect direct assignments to outputs
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
                    span: _,
                } => {
                    // Try to refine struct field knowledge from the condition for the then-branch
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
                            K::Unary(parser::UnOp::Not, _inner) => {
                                // Negative condition - do not refine
                            }
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
                                // ismember('f', fieldnames(s)) or ismember(fieldnames(s),'f')
                                if lname.eq_ignore_ascii_case("ismember") && args.len() >= 2 {
                                    let mut fields: Vec<String> = Vec::new();
                                    let mut target: Option<VarId> = None;
                                    // Extract fields from either arg
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
                                // any(strcmp(fieldnames(s), 'f')) and variants; also strcmpi
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
                    let then_a = analyze_stmts(_outputs, then_body, then_env, returns, func_defs);
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
                        let a = analyze_stmts(_outputs, b, elseif_env, returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = join_env(&out_env, &f);
                        }
                        all_exits.extend(a.exits);
                    }
                    if let Some(else_body) = else_body {
                        let a = analyze_stmts(_outputs, else_body, env.clone(), returns, func_defs);
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
                HirStmt::While {
                    cond: _,
                    body,
                    span: _,
                } => {
                    let a = analyze_stmts(_outputs, body, env.clone(), returns, func_defs);
                    if let Some(f) = a.fallthrough {
                        env = join_env(&env, &f);
                    }
                    exits.extend(a.exits);
                }
                HirStmt::For {
                    var,
                    expr,
                    body,
                    span: _,
                } => {
                    let t = infer_expr_type(expr, &env, returns);
                    env.insert(*var, t);
                    let a = analyze_stmts(_outputs, body, env.clone(), returns, func_defs);
                    if let Some(f) = a.fallthrough {
                        env = join_env(&env, &f);
                    }
                    exits.extend(a.exits);
                }
                HirStmt::Switch {
                    expr: _,
                    cases,
                    otherwise,
                    span: _,
                } => {
                    let mut out_env: Option<HashMap<VarId, Type>> = None;
                    for (_v, b) in cases {
                        let a = analyze_stmts(_outputs, b, env.clone(), returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = Some(match out_env {
                                Some(curr) => join_env(&curr, &f),
                                None => f,
                            });
                        }
                        exits.extend(a.exits);
                    }
                    if let Some(otherwise) = otherwise {
                        let a = analyze_stmts(_outputs, otherwise, env.clone(), returns, func_defs);
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
                    catch_var: _,
                    catch_body,
                    span: _,
                } => {
                    let a_try = analyze_stmts(_outputs, try_body, env.clone(), returns, func_defs);
                    let a_catch =
                        analyze_stmts(_outputs, catch_body, env.clone(), returns, func_defs);
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
                HirStmt::AssignLValue(lv, expr, _, _) => {
                    // Update struct field knowledge if we see s.field = expr
                    if let HirLValue::Member(base, field) = lv {
                        // If base is a variable, mark it as Struct with this field
                        if let HirExprKind::Var(vid) = base.kind {
                            let mut known = match env.get(&vid) {
                                Some(Type::Struct { known_fields }) => known_fields.clone(),
                                _ => Some(Vec::new()),
                            };
                            if let Some(list) = &mut known {
                                if !list.iter().any(|f| f == field) {
                                    list.push(field.clone());
                                    list.sort();
                                    list.dedup();
                                }
                            }
                            env.insert(
                                vid,
                                Type::Struct {
                                    known_fields: known,
                                },
                            );
                        }
                    }
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

    // Collect function names (top-level and class methods)
    fn collect_function_names(stmts: &[HirStmt], acc: &mut Vec<String>) {
        for s in stmts {
            match s {
                HirStmt::Function { name, .. } => acc.push(name.clone()),
                HirStmt::ClassDef { members, .. } => {
                    for m in members {
                        if let HirClassMember::Methods { body, .. } = m {
                            collect_function_names(body, acc);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut function_names: Vec<String> = Vec::new();
    collect_function_names(&prog.body, &mut function_names);
    let mut returns: HashMap<String, Vec<Type>> = function_names
        .iter()
        .map(|n| (n.clone(), Vec::new()))
        .collect();

    // Globals/persistents symbol table across units (basic wiring): collect names
    let mut globals: std::collections::HashSet<VarId> = std::collections::HashSet::new();
    let mut persistents: std::collections::HashSet<VarId> = std::collections::HashSet::new();
    for stmt in &prog.body {
        if let HirStmt::Global(vs, _) = stmt {
            for (v, _n) in vs {
                globals.insert(*v);
            }
        }
    }
    for stmt in &prog.body {
        if let HirStmt::Persistent(vs, _) = stmt {
            for (v, _n) in vs {
                persistents.insert(*v);
            }
        }
    }

    // Collect function defs for simple callsite fallback inference
    #[allow(clippy::type_complexity)]
    let mut func_defs: HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)> = HashMap::new();
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

    // Seed returns: per function, default outputs Unknown; if a function contains obvious numeric assignments to outputs, capture them on the first pass
    for stmt in &prog.body {
        if let HirStmt::Function {
            name,
            outputs,
            body,
            ..
        } = stmt
        {
            let mut per_output: Vec<Type> = vec![Type::Unknown; outputs.len()];
            let analysis = analyze_stmts(outputs, body, HashMap::new(), &returns, &func_defs);
            let mut accumulate = |env: &HashMap<VarId, Type>| {
                for (i, out_id) in outputs.iter().enumerate() {
                    if let Some(t) = env.get(out_id) {
                        per_output[i] = per_output[i].unify(t);
                    }
                }
            };
            if let Some(f) = &analysis.fallthrough {
                accumulate(f);
            }
            for e in &analysis.exits {
                accumulate(e);
            }
            returns.insert(name.clone(), per_output);
        }
    }

    let mut changed = true;
    let mut iter = 0usize;
    let max_iters = 3usize;
    while changed && iter < max_iters {
        changed = false;
        iter += 1;
        for stmt in &prog.body {
            match stmt {
                HirStmt::Function {
                    name,
                    outputs,
                    body,
                    ..
                } => {
                    let analysis =
                        analyze_stmts(outputs, body, HashMap::new(), &returns, &func_defs);
                    let mut per_output: Vec<Type> = vec![Type::Unknown; outputs.len()];
                    let mut accumulate = |env: &HashMap<VarId, Type>| {
                        for (i, out_id) in outputs.iter().enumerate() {
                            if let Some(t) = env.get(out_id) {
                                per_output[i] = per_output[i].unify(t);
                            }
                        }
                    };
                    for e in &analysis.exits {
                        accumulate(e);
                    }
                    if let Some(f) = &analysis.fallthrough {
                        accumulate(f);
                    }
                    if returns.get(name) != Some(&per_output) {
                        returns.insert(name.clone(), per_output);
                        changed = true;
                    }
                }
                HirStmt::ClassDef { members, .. } => {
                    for m in members {
                        if let HirClassMember::Methods { body, .. } = m {
                            for s in body {
                                if let HirStmt::Function {
                                    name,
                                    outputs,
                                    body,
                                    ..
                                } = s
                                {
                                    let analysis = analyze_stmts(
                                        outputs,
                                        body,
                                        HashMap::new(),
                                        &returns,
                                        &func_defs,
                                    );
                                    let mut per_output: Vec<Type> =
                                        vec![Type::Unknown; outputs.len()];
                                    let mut accumulate = |env: &HashMap<VarId, Type>| {
                                        for (i, out_id) in outputs.iter().enumerate() {
                                            if let Some(t) = env.get(out_id) {
                                                per_output[i] = per_output[i].unify(t);
                                            }
                                        }
                                    };
                                    for e in &analysis.exits {
                                        accumulate(e);
                                    }
                                    if let Some(f) = &analysis.fallthrough {
                                        accumulate(f);
                                    }
                                    if returns.get(name) != Some(&per_output) {
                                        returns.insert(name.clone(), per_output);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    returns
}

/// Infer variable types inside each function by performing a flow-sensitive analysis and
/// returning the joined environment (types per VarId) at function exits and fallthrough.
#[allow(clippy::type_complexity)]
pub fn infer_function_variable_types(
    prog: &HirProgram,
) -> std::collections::HashMap<String, std::collections::HashMap<VarId, Type>> {
    use std::collections::HashMap;

    // Reuse function return inference to improve call result typing
    let returns_map = infer_function_output_types(prog);

    // Collect function defs for simple callsite fallback inference
    #[allow(clippy::type_complexity)]
    let mut func_defs: HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)> = HashMap::new();
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
        #[allow(clippy::only_used_in_recursion)] _outputs: &[VarId],
        stmts: &[HirStmt],
        mut env: HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
        func_defs: &HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)>,
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
                        // Start from summary
                        let mut per_out: Vec<Type> = returns.get(name).cloned().unwrap_or_default();
                        // If summary missing/unknown, try simple callsite fallback using func_defs and argument types
                        let needs_fallback = per_out.is_empty()
                            || per_out.iter().any(|t| matches!(t, Type::Unknown));
                        if needs_fallback {
                            if let Some((params, outs, body)) = func_defs.get(name).cloned() {
                                // Seed param env with argument types at callsite by reusing current env typing
                                let mut penv: HashMap<VarId, Type> = HashMap::new();
                                // We don't have direct access to call args here (expr doesn't carry), so default to Num for simplicity when outputs are computed from params via arithmetic; otherwise Unknown
                                // Heuristic: assume params are Num when used in arithmetic contexts; conservative elsewhere
                                for p in params {
                                    penv.insert(p, Type::Num);
                                }
                                // Single pass: collect direct assignments to outputs
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
                    span: _,
                } => {
                    // Apply the same struct field refinement in the variable-type analysis
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
                                // ismember('f', fieldnames(s)) or ismember(fieldnames(s),'f')
                                if lname.eq_ignore_ascii_case("ismember") && args.len() >= 2 {
                                    let mut fields: Vec<String> = Vec::new();
                                    let mut target: Option<VarId> = None;
                                    // Extract fields from either arg
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
                                // any(strcmp(fieldnames(s), 'f')) and variants; also strcmpi
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
                    let then_a = analyze_stmts(_outputs, then_body, then_env, returns, func_defs);
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
                        let a = analyze_stmts(_outputs, b, elseif_env, returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = join_env(&out_env, &f);
                        }
                        all_exits.extend(a.exits);
                    }
                    if let Some(else_body) = else_body {
                        let a = analyze_stmts(_outputs, else_body, env.clone(), returns, func_defs);
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
                    let a = analyze_stmts(_outputs, body, env.clone(), returns, func_defs);
                    if let Some(f) = a.fallthrough {
                        env = join_env(&env, &f);
                    }
                    exits.extend(a.exits);
                }
                HirStmt::For {
                    var,
                    expr,
                    body,
                    span: _,
                } => {
                    let t = infer_expr_type(expr, &env, returns);
                    env.insert(*var, t);
                    let a = analyze_stmts(_outputs, body, env.clone(), returns, func_defs);
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
                        let a = analyze_stmts(_outputs, b, env.clone(), returns, func_defs);
                        if let Some(f) = a.fallthrough {
                            out_env = Some(match out_env {
                                Some(curr) => join_env(&curr, &f),
                                None => f,
                            });
                        }
                        exits.extend(a.exits);
                    }
                    if let Some(otherwise) = otherwise {
                        let a = analyze_stmts(_outputs, otherwise, env.clone(), returns, func_defs);
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
                    let a_try = analyze_stmts(_outputs, try_body, env.clone(), returns, func_defs);
                    let a_catch =
                        analyze_stmts(_outputs, catch_body, env.clone(), returns, func_defs);
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

/// Infer variable types for top-level (script) statements by performing a flow-sensitive analysis
/// over the program body. Returns a mapping from VarId to inferred Type.
pub fn infer_global_variable_types(
    prog: &HirProgram,
    returns: &std::collections::HashMap<String, Vec<Type>>,
) -> std::collections::HashMap<VarId, Type> {
    use std::collections::HashMap;

    #[derive(Clone)]
    struct Analysis {
        exits: Vec<HashMap<VarId, Type>>,
        fallthrough: Option<HashMap<VarId, Type>>,
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

    #[allow(clippy::type_complexity, clippy::only_used_in_recursion)]
    fn analyze_stmts(
        stmts: &[HirStmt],
        mut env: HashMap<VarId, Type>,
        returns: &HashMap<String, Vec<Type>>,
        func_defs: &HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)>,
    ) -> Analysis {
        let mut exits = Vec::new();
        let mut i = 0usize;
        while i < stmts.len() {
            match &stmts[i] {
                HirStmt::Assign(var, expr, _, _) => {
                    let t = infer_expr_type_with_env(expr, &env, returns);
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
                                            let t = infer_expr_type_with_env(rhs, &penv, returns);
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
                        let t = infer_expr_type_with_env(expr, &env, returns);
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
                    span: _,
                } => {
                    let _ = infer_expr_type_with_env(cond, &env, returns);
                    let then_analysis = analyze_stmts(then_body, env.clone(), returns, func_defs);
                    let mut branch_envs = Vec::new();
                    if let Some(f) = &then_analysis.fallthrough {
                        branch_envs.push(f.clone());
                    }
                    for e in &then_analysis.exits {
                        branch_envs.push(e.clone());
                    }
                    for (c, b) in elseif_blocks {
                        let _ = infer_expr_type_with_env(c, &env, returns);
                        let analysis = analyze_stmts(b, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    }
                    if let Some(else_body) = else_body {
                        let analysis = analyze_stmts(else_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    } else {
                        branch_envs.push(env.clone());
                    }
                    if let Some(first) = branch_envs.first().cloned() {
                        env = branch_envs
                            .iter()
                            .skip(1)
                            .fold(first, |acc, e| join_env(&acc, e));
                    }
                }
                HirStmt::Switch {
                    expr,
                    cases,
                    otherwise,
                    ..
                } => {
                    let _ = infer_expr_type_with_env(expr, &env, returns);
                    let mut branch_envs = Vec::new();
                    for (case_expr, case_body) in cases {
                        let _ = infer_expr_type_with_env(case_expr, &env, returns);
                        let analysis = analyze_stmts(case_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    }
                    if let Some(otherwise_body) = otherwise {
                        let analysis =
                            analyze_stmts(otherwise_body, env.clone(), returns, func_defs);
                        if let Some(f) = &analysis.fallthrough {
                            branch_envs.push(f.clone());
                        }
                        for e in &analysis.exits {
                            branch_envs.push(e.clone());
                        }
                    } else {
                        branch_envs.push(env.clone());
                    }
                    if let Some(first) = branch_envs.first().cloned() {
                        env = branch_envs
                            .iter()
                            .skip(1)
                            .fold(first, |acc, e| join_env(&acc, e));
                    }
                }
                HirStmt::While { cond, body, .. } => {
                    let _ = infer_expr_type_with_env(cond, &env, returns);
                    let body_analysis = analyze_stmts(body, env.clone(), returns, func_defs);
                    if let Some(f) = &body_analysis.fallthrough {
                        env = join_env(&env, f);
                    }
                    for e in &body_analysis.exits {
                        env = join_env(&env, e);
                    }
                }
                HirStmt::For { expr, body, .. } => {
                    let _ = infer_expr_type_with_env(expr, &env, returns);
                    let body_analysis = analyze_stmts(body, env.clone(), returns, func_defs);
                    if let Some(f) = &body_analysis.fallthrough {
                        env = join_env(&env, f);
                    }
                    for e in &body_analysis.exits {
                        env = join_env(&env, e);
                    }
                }
                HirStmt::Function { .. } | HirStmt::ClassDef { .. } => {}
                HirStmt::Global(_, _)
                | HirStmt::Persistent(_, _)
                | HirStmt::Import { .. }
                | HirStmt::TryCatch { .. }
                | HirStmt::AssignLValue(_, _, _, _) => {}
            }
            i += 1;
        }

        Analysis {
            exits,
            fallthrough: Some(env),
        }
    }

    let mut func_defs: HashMap<String, (Vec<VarId>, Vec<VarId>, Vec<HirStmt>)> = HashMap::new();
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

    let analysis = analyze_stmts(&prog.body, HashMap::new(), returns, &func_defs);
    let mut out: HashMap<VarId, Type> = HashMap::new();
    let mut accumulate = |env: &HashMap<VarId, Type>| {
        for (k, v) in env {
            out.entry(*k)
                .and_modify(|t| *t = t.unify(v))
                .or_insert_with(|| v.clone());
        }
    };
    if let Some(f) = &analysis.fallthrough {
        accumulate(f);
    }
    for e in &analysis.exits {
        accumulate(e);
    }

    out
}

pub fn lint_shapes(result: &LoweringResult) -> Vec<HirDiagnostic> {
    fn vector_literal_length(expr: &HirExpr) -> Option<usize> {
        let shape = tensor_literal_shape(expr)?;
        match (
            shape.get(0).cloned().unwrap_or(None),
            shape.get(1).cloned().unwrap_or(None),
        ) {
            (Some(r), Some(c)) => {
                if r == 1 {
                    Some(c)
                } else if c == 1 {
                    Some(r)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn concat_dims(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
        match ty {
            Type::Num | Type::Int | Type::Bool => Some((Some(1), Some(1))),
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(runmat_builtins::shape_rules::matrix_dims(shape))
            }
            _ => None,
        }
    }
    fn format_dim(dim: Option<usize>) -> String {
        dim.map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn format_shape(shape: &[Option<usize>]) -> String {
        if shape.len() == 2 {
            return format!("{} x {}", format_dim(shape[0]), format_dim(shape[1]));
        }
        let dims: Vec<String> = shape.iter().map(|d| format_dim(*d)).collect();
        format!("[{}]", dims.join(", "))
    }

    fn matrix_dims_from_type(ty: &Type) -> Option<(Option<usize>, Option<usize>)> {
        match ty {
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(runmat_builtins::shape_rules::matrix_dims(shape))
            }
            _ => None,
        }
    }

    fn element_count(shape: &[Option<usize>]) -> Option<usize> {
        runmat_builtins::shape_rules::element_count_if_known(shape)
    }

    fn vector_length(shape: &[Option<usize>]) -> Option<usize> {
        let count = element_count(shape)?;
        let is_vector = shape.len() == 1
            || (shape.len() == 2
                && (shape[0] == Some(1) || shape[1] == Some(1))
                && shape.iter().all(|d| d.is_some()));
        if is_vector {
            Some(count)
        } else {
            None
        }
    }

    fn tensor_literal_shape(expr: &HirExpr) -> Option<Vec<Option<usize>>> {
        let HirExprKind::Tensor(rows) = &expr.kind else {
            return None;
        };
        if rows.is_empty() {
            return Some(vec![Some(0), Some(0)]);
        }
        let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        Some(vec![Some(rows.len()), Some(cols)])
    }

    enum DimSpec {
        Known(usize),
        Unknown,
        Negative,
        NonInteger,
    }

    fn parse_dim(expr: &HirExpr) -> DimSpec {
        if let Some(value) = eval_const_num(expr) {
            if value.is_finite() {
                let rounded = value.round();
                if (value - rounded).abs() <= 1e-9 {
                    if rounded < 0.0 {
                        return DimSpec::Negative;
                    }
                    return DimSpec::Known(rounded as usize);
                }
                return DimSpec::NonInteger;
            }
        }
        DimSpec::Unknown
    }
    fn type_shape_for_broadcast(ty: &Type) -> Option<Vec<Option<usize>>> {
        match ty {
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                Some(shape.clone())
            }
            Type::Num | Type::Int | Type::Bool => Some(vec![Some(1), Some(1)]),
            _ => None,
        }
    }

    fn check_binary(
        op: &parser::BinOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        let lhs_ty = infer_expr_type_with_env(lhs, env, returns);
        let rhs_ty = infer_expr_type_with_env(rhs, env, returns);
        match op {
            parser::BinOp::Mul => {
                if let Some(false) =
                    runmat_builtins::shape_rules::matmul_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((lrows, lcols)), Some((rrows, rcols))) => format!(
                            "left is {} x {}, right is {} x {} (inner dimensions {} and {})",
                            format_dim(lrows),
                            format_dim(lcols),
                            format_dim(rrows),
                            format_dim(rcols),
                            format_dim(lcols),
                            format_dim(rrows)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Matrix multiply dimension mismatch: {detail} (inner dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.matmul",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::LeftDiv => {
                if let Some(false) =
                    runmat_builtins::shape_rules::left_divide_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((lrows, _)), Some((rrows, _))) => format!(
                            "left row dimension {}, right row dimension {}",
                            format_dim(lrows),
                            format_dim(rrows)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Left divide dimension mismatch: {detail} (row dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.ldivide",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::Div => {
                if let Some(false) =
                    runmat_builtins::shape_rules::right_divide_compatible(&lhs_ty, &rhs_ty)
                {
                    let detail = match (
                        matrix_dims_from_type(&lhs_ty),
                        matrix_dims_from_type(&rhs_ty),
                    ) {
                        (Some((_, lcols)), Some((_, rcols))) => format!(
                            "left column dimension {}, right column dimension {}",
                            format_dim(lcols),
                            format_dim(rcols)
                        ),
                        _ => "unknown shapes".to_string(),
                    };
                    diags.push(HirDiagnostic {
                        message: format!(
                            "Right divide dimension mismatch: {detail} (column dimensions must match)"
                        ),
                        span: merge_span(lhs.span, rhs.span),
                        code: "lint.shape.rdivide",
                        severity: HirDiagnosticSeverity::Warning,
                    });
                }
            }
            parser::BinOp::Add
            | parser::BinOp::Sub
            | parser::BinOp::ElemMul
            | parser::BinOp::ElemDiv
            | parser::BinOp::ElemPow
            | parser::BinOp::ElemLeftDiv
            | parser::BinOp::Equal
            | parser::BinOp::NotEqual
            | parser::BinOp::Less
            | parser::BinOp::LessEqual
            | parser::BinOp::Greater
            | parser::BinOp::GreaterEqual => {
                let lhs_shape = type_shape_for_broadcast(&lhs_ty);
                let rhs_shape = type_shape_for_broadcast(&rhs_ty);
                if let (Some(a), Some(b)) = (lhs_shape, rhs_shape) {
                    if let Some(false) = runmat_builtins::shape_rules::broadcast_compatible(&a, &b)
                    {
                        let detail = format!(
                            "left is {}, right is {}",
                            format_shape(&a),
                            format_shape(&b)
                        );
                        diags.push(HirDiagnostic {
                            message: format!(
                                "Elementwise/broadcast dimension mismatch: {detail} (broadcasting failed)"
                            ),
                            span: merge_span(lhs.span, rhs.span),
                            code: "lint.shape.broadcast",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn walk_expr(
        expr: &HirExpr,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        match &expr.kind {
            HirExprKind::Unary(_, inner) => walk_expr(inner, env, returns, diags),
            HirExprKind::Binary(lhs, op, rhs) => {
                check_binary(op, lhs, rhs, env, returns, diags);
                walk_expr(lhs, env, returns, diags);
                walk_expr(rhs, env, returns, diags);
            }
            HirExprKind::Tensor(rows) => {
                let mut col_constraint: Option<usize> = None;
                for row in rows {
                    let mut row_dim: Option<usize> = None;
                    let mut row_cols: Option<usize> = Some(0);
                    let mut first_span: Option<Span> = None;
                    for e in row {
                        if first_span.is_none() {
                            first_span = Some(e.span);
                        }
                        let ty = infer_expr_type_with_env(e, env, returns);
                        if let Some((rows_dim, cols_dim)) = concat_dims(&ty) {
                            if let (Some(prev), Some(curr)) = (row_dim, rows_dim) {
                                if prev != curr {
                                    diags.push(HirDiagnostic {
                                        message: format!(
                                            "Horizontal concatenation dimension mismatch: left row dimension {prev}, right row dimension {curr} (row dimensions must match)"
                                        ),
                                        span: merge_span(first_span.unwrap_or(e.span), e.span),
                                        code: "lint.shape.horzcat",
                                        severity: HirDiagnosticSeverity::Warning,
                                    });
                                }
                            }
                            if row_dim.is_none() {
                                row_dim = rows_dim;
                            }
                            match (row_cols, cols_dim) {
                                (Some(total), Some(value)) => row_cols = Some(total + value),
                                _ => row_cols = None,
                            }
                        } else {
                            row_dim = None;
                            row_cols = None;
                        }
                    }

                    if let (Some(prev_cols), Some(curr_cols)) = (col_constraint, row_cols) {
                        if prev_cols != curr_cols {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Vertical concatenation dimension mismatch: upper column dimension {prev_cols}, lower column dimension {curr_cols} (column dimensions must match)"
                                ),
                                span: expr.span,
                                code: "lint.shape.vertcat",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                    if col_constraint.is_none() {
                        col_constraint = row_cols;
                    }
                }

                for row in rows {
                    for e in row {
                        walk_expr(e, env, returns, diags);
                    }
                }
            }
            HirExprKind::Cell(rows) => {
                for row in rows {
                    for e in row {
                        walk_expr(e, env, returns, diags);
                    }
                }
            }
            HirExprKind::Index(base, idxs) | HirExprKind::IndexCell(base, idxs) => {
                walk_expr(base, env, returns, diags);
                for idx in idxs {
                    walk_expr(idx, env, returns, diags);
                }
                if matches!(expr.kind, HirExprKind::Index(_, _)) && idxs.len() == 1 {
                    let base_ty = infer_expr_type_with_env(base, env, returns);
                    let idx_ty = infer_expr_type_with_env(&idxs[0], env, returns);
                    let base_shape = match base_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    let mask_shape = match idx_ty {
                        Type::Logical { shape: Some(shape) }
                        | Type::Tensor { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    if let (Some(base_shape), Some(mask_shape)) = (base_shape, mask_shape) {
                        if let (Some(base_count), Some(mask_count)) =
                            (element_count(&base_shape), element_count(&mask_shape))
                        {
                            if base_count != mask_count {
                                diags.push(HirDiagnostic {
                                    message: format!(
                                        "Logical index size mismatch: mask has {mask_count}, array has {base_count} (must match)"
                                    ),
                                    span: merge_span(base.span, idxs[0].span),
                                    code: "lint.shape.logical_index",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }
            }
            HirExprKind::Range(start, step, end) => {
                walk_expr(start, env, returns, diags);
                if let Some(step) = step.as_ref() {
                    walk_expr(step, env, returns, diags);
                }
                walk_expr(end, env, returns, diags);
            }
            HirExprKind::FuncCall(name, args) => {
                if name.eq_ignore_ascii_case("dot") && args.len() >= 2 {
                    let lhs_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let rhs_ty = infer_expr_type_with_env(&args[1], env, returns);
                    let lhs_len = match lhs_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => vector_length(&shape),
                        _ => None,
                    };
                    let rhs_len = match rhs_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => vector_length(&shape),
                        _ => None,
                    };
                    if let (Some(a), Some(b)) = (lhs_len, rhs_len) {
                        if a != b {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Dot product length mismatch: left length {a}, right length {b} (lengths must match)"
                                ),
                                span: merge_span(args[0].span, args[1].span),
                                code: "lint.shape.dot",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                }

                if name.eq_ignore_ascii_case("reshape") && args.len() >= 2 {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_shape = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape),
                        _ => None,
                    };
                    let mut dims: Vec<Option<usize>> = Vec::new();
                    let mut negative_count = 0usize;
                    let mut non_integer = false;
                    for arg in args.iter().skip(1) {
                        match parse_dim(arg) {
                            DimSpec::Known(value) => dims.push(Some(value)),
                            DimSpec::Negative => {
                                negative_count += 1;
                                dims.push(None);
                            }
                            DimSpec::NonInteger => {
                                non_integer = true;
                                dims.push(None);
                            }
                            DimSpec::Unknown => dims.push(None),
                        }
                    }
                    if negative_count > 1 {
                        diags.push(HirDiagnostic {
                            message: "Reshape dimension mismatch: more than one negative dimension (only one allowed)"
                                .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    } else if negative_count == 1 && non_integer {
                        diags.push(HirDiagnostic {
                            message: "Reshape dimension mismatch: negative dimensions require integer sizes"
                                .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                    if non_integer {
                        diags.push(HirDiagnostic {
                            message: "Reshape dimension mismatch: non-integer dimensions"
                                .to_string(),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.reshape",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                    if let Some(shape) =
                        runmat_builtins::shape_rules::constructor_shape_from_dims(&dims)
                    {
                        if let Some(input_shape) = input_shape {
                            if let (Some(in_count), Some(out_count)) =
                                (element_count(&input_shape), element_count(&shape))
                            {
                                if in_count != out_count {
                                    diags.push(HirDiagnostic {
                                        message: format!(
                                            "Reshape element count mismatch: input has {in_count}, output has {out_count} (must match)"
                                        ),
                                        span: merge_span(args[0].span, args[1].span),
                                        code: "lint.shape.reshape",
                                        severity: HirDiagnosticSeverity::Warning,
                                    });
                                }
                            }
                        }
                    }
                }

                if (name.eq_ignore_ascii_case("permute") || name.eq_ignore_ascii_case("ipermute"))
                    && args.len() >= 2
                {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_rank = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape.len()),
                        _ => None,
                    };
                    let order_rank = vector_literal_length(&args[1]);
                    if let (Some(in_rank), Some(ord_rank)) = (input_rank, order_rank) {
                        if in_rank != ord_rank {
                            diags.push(HirDiagnostic {
                                message: format!(
                                    "Permute rank mismatch: input rank {in_rank}, order length {ord_rank} (must match)"
                                ),
                                span: merge_span(args[0].span, args[1].span),
                                code: "lint.shape.permute",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                    }
                    if let HirExprKind::Tensor(rows) = &args[1].kind {
                        let mut seen: std::collections::BTreeSet<usize> =
                            std::collections::BTreeSet::new();
                        let mut duplicate = false;
                        let mut max_index = 0usize;
                        for row in rows {
                            for entry in row {
                                if let Some(value) = eval_const_num(entry) {
                                    let rounded = value.round();
                                    if (value - rounded).abs() <= 1e-9 && rounded >= 1.0 {
                                        let idx = rounded as usize;
                                        max_index = max_index.max(idx);
                                        if !seen.insert(idx) {
                                            duplicate = true;
                                        }
                                    }
                                }
                            }
                        }
                        if duplicate {
                            diags.push(HirDiagnostic {
                                message:
                                    "Permute order mismatch: duplicate dimensions in order vector"
                                        .to_string(),
                                span: args[1].span,
                                code: "lint.shape.permute",
                                severity: HirDiagnosticSeverity::Warning,
                            });
                        }
                        if let Some(in_rank) = input_rank {
                            if max_index > in_rank {
                                diags.push(HirDiagnostic {
                                    message: "Permute order mismatch: order references a dimension larger than the input rank"
                                        .to_string(),
                                    span: args[1].span,
                                    code: "lint.shape.permute",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }

                if name.eq_ignore_ascii_case("repmat") && args.len() >= 2 {
                    let mut reps: Vec<Option<usize>> = Vec::new();
                    let mut non_integer = false;
                    let mut negative = false;
                    for arg in args.iter().skip(1) {
                        match parse_dim(arg) {
                            DimSpec::Known(value) => reps.push(Some(value)),
                            DimSpec::NonInteger => non_integer = true,
                            DimSpec::Negative => negative = true,
                            _ => reps.push(None),
                        }
                    }
                    if non_integer || negative {
                        let reason = if non_integer {
                            "non-integer"
                        } else {
                            "negative"
                        };
                        diags.push(HirDiagnostic {
                            message: format!(
                                "Repmat dimension mismatch: {reason} replication factors"
                            ),
                            span: merge_span(args[0].span, args[1].span),
                            code: "lint.shape.repmat",
                            severity: HirDiagnosticSeverity::Warning,
                        });
                    }
                }

                if (name.eq_ignore_ascii_case("sum")
                    || name.eq_ignore_ascii_case("mean")
                    || name.eq_ignore_ascii_case("prod")
                    || name.eq_ignore_ascii_case("min")
                    || name.eq_ignore_ascii_case("max"))
                    && args.len() >= 2
                {
                    let input_ty = infer_expr_type_with_env(&args[0], env, returns);
                    let input_rank = match input_ty {
                        Type::Tensor { shape: Some(shape) }
                        | Type::Logical { shape: Some(shape) } => Some(shape.len()),
                        _ => None,
                    };
                    if let Some(rank) = input_rank {
                        if let DimSpec::Known(dim) = parse_dim(&args[1]) {
                            if dim == 0 || dim > rank {
                                diags.push(HirDiagnostic {
                                    message: format!(
                                        "Reduction dimension mismatch: dimension {dim} is out of range for rank {rank}"
                                    ),
                                    span: args[1].span,
                                    code: "lint.shape.reduction",
                                    severity: HirDiagnosticSeverity::Warning,
                                });
                            }
                        }
                    }
                }

                for arg in args {
                    walk_expr(arg, env, returns, diags);
                }
            }
            HirExprKind::MethodCall(_, _, args) => {
                for arg in args {
                    walk_expr(arg, env, returns, diags);
                }
            }
            HirExprKind::Member(base, _) | HirExprKind::MemberDynamic(base, _) => {
                walk_expr(base, env, returns, diags);
            }
            HirExprKind::AnonFunc { body, .. } => {
                walk_expr(body, env, returns, diags);
            }
            _ => {}
        }
    }

    fn walk_stmt(
        stmt: &HirStmt,
        env: &std::collections::HashMap<VarId, Type>,
        returns: &std::collections::HashMap<String, Vec<Type>>,
        func_envs: &std::collections::HashMap<String, std::collections::HashMap<VarId, Type>>,
        diags: &mut Vec<HirDiagnostic>,
    ) {
        match stmt {
            HirStmt::Assign(_, expr, _, _)
            | HirStmt::ExprStmt(expr, _, _)
            | HirStmt::MultiAssign(_, expr, _, _) => walk_expr(expr, env, returns, diags),
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                walk_expr(cond, env, returns, diags);
                for s in then_body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
                for (cond, body) in elseif_blocks {
                    walk_expr(cond, env, returns, diags);
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
                if let Some(body) = else_body {
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
            }
            HirStmt::While { cond, body, .. } => {
                walk_expr(cond, env, returns, diags);
                for s in body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
            }
            HirStmt::For { expr, body, .. } => {
                walk_expr(expr, env, returns, diags);
                for s in body {
                    walk_stmt(s, env, returns, func_envs, diags);
                }
            }
            HirStmt::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                walk_expr(expr, env, returns, diags);
                for (case_expr, case_body) in cases {
                    walk_expr(case_expr, env, returns, diags);
                    for s in case_body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
                if let Some(body) = otherwise {
                    for s in body {
                        walk_stmt(s, env, returns, func_envs, diags);
                    }
                }
            }
            HirStmt::Function { name, body, .. } => {
                let func_env = func_envs.get(name).cloned().unwrap_or_default();
                for s in body {
                    walk_stmt(s, &func_env, returns, func_envs, diags);
                }
            }
            HirStmt::ClassDef { members, .. } => {
                for member in members {
                    if let HirClassMember::Methods { body, .. } = member {
                        for s in body {
                            walk_stmt(s, env, returns, func_envs, diags);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut diags = Vec::new();
    let global_env = result.inferred_globals.clone();
    for stmt in &result.hir.body {
        walk_stmt(
            stmt,
            &global_env,
            &result.inferred_function_returns,
            &result.inferred_function_envs,
            &mut diags,
        );
    }
    diags
}

/// Collect all import statements in a program for downstream name resolution
pub fn collect_imports(prog: &HirProgram) -> Vec<(Vec<String>, bool)> {
    let mut imports = Vec::new();
    for stmt in &prog.body {
        if let HirStmt::Import { path, wildcard, .. } = stmt {
            imports.push((path.clone(), *wildcard));
        }
    }
    imports
}

/// Normalized import for easier downstream resolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedImport {
    /// Dot-joined package path (e.g., "pkg.sub") or class path (e.g., "Point")
    pub path: String,
    /// True if the import was a wildcard (e.g., pkg.*)
    pub wildcard: bool,
    /// For specific imports, the unqualified last segment (e.g., "Class"). None for wildcard
    pub unqualified: Option<String>,
}

/// Convert parsed imports into normalized string forms
pub fn normalize_imports(prog: &HirProgram) -> Vec<NormalizedImport> {
    let mut out = Vec::new();
    for stmt in &prog.body {
        if let HirStmt::Import { path, wildcard, .. } = stmt {
            // Support hierarchical aliases, including class paths (e.g., pkg.sub.Class)
            let path_str = path.join(".");
            let last = if *wildcard {
                None
            } else {
                path.last().cloned()
            };
            out.push(NormalizedImport {
                path: path_str,
                wildcard: *wildcard,
                unqualified: last,
            });
        }
    }
    out
}

/// Validate for obvious import ambiguities:
/// - Duplicate specific imports (same fully-qualified path)
/// - Specific imports that introduce the same unqualified name from different packages
pub fn validate_imports(prog: &HirProgram) -> Result<(), SemanticError> {
    use std::collections::{HashMap, HashSet};
    let norms = normalize_imports(prog);
    let mut seen_exact: HashSet<(String, bool)> = HashSet::new();
    for n in &norms {
        if !seen_exact.insert((n.path.clone(), n.wildcard)) {
            return Err(SemanticError::new(format!(
                "duplicate import '{}{}'",
                n.path,
                if n.wildcard { ".*" } else { "" }
            )));
        }
    }
    // Ambiguity among specifics with same unqualified name
    let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
    for n in &norms {
        if !n.wildcard {
            if let Some(uq) = &n.unqualified {
                by_name.entry(uq.clone()).or_default().push(n.path.clone());
            }
        }
    }
    for (uq, sources) in by_name {
        if sources.len() > 1 {
            return Err(SemanticError::new(format!(
                "ambiguous import for '{}': {}",
                uq,
                sources.join(", ")
            )));
        }
    }
    Ok(())
}

/// Validate classdef declarations for basic semantic correctness
/// Checks: duplicate property/method names within the same class,
/// conflicts between property and method names, and trivial superclass cycles (self parent)
pub fn validate_classdefs(prog: &HirProgram) -> Result<(), SemanticError> {
    use std::collections::HashSet;
    fn norm_attr_value(v: &str) -> String {
        let t = v.trim();
        let t = t.trim_matches('\'');
        t.to_ascii_lowercase()
    }
    fn validate_access_value(ctx: &str, v: &str) -> Result<(), SemanticError> {
        match v {
            "public" | "private" => Ok(()),
            other => Err(SemanticError::new(format!(
                "invalid access value '{other}' in {ctx} (allowed: public, private)",
            ))),
        }
    }
    for stmt in &prog.body {
        if let HirStmt::ClassDef {
            name,
            super_class,
            members,
            ..
        } = stmt
        {
            if let Some(sup) = super_class {
                if sup == name {
                    return Err(SemanticError::new(format!(
                        "Class '{name}' cannot inherit from itself"
                    )));
                }
            }
            let mut prop_names: HashSet<String> = HashSet::new();
            let mut method_names: HashSet<String> = HashSet::new();
            for m in members {
                match m {
                    HirClassMember::Properties {
                        names: props,
                        attributes,
                    } => {
                        // Enforce attributes: Access/GetAccess/SetAccess must be public/private; Static+Dependent invalid
                        let mut has_static = false;
                        let mut has_constant = false;
                        let mut _has_transient = false;
                        let mut _has_hidden = false;
                        let mut has_dependent = false;
                        let mut access_default: Option<String> = None;
                        let mut get_access: Option<String> = None;
                        let mut set_access: Option<String> = None;
                        for a in attributes {
                            if a.name.eq_ignore_ascii_case("Static") {
                                has_static = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Constant") {
                                has_constant = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Transient") {
                                _has_transient = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Hidden") {
                                _has_hidden = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Dependent") {
                                has_dependent = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Access") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "Access requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                access_default = Some(v);
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("GetAccess") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "GetAccess requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                get_access = Some(v);
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("SetAccess") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "SetAccess requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                set_access = Some(v);
                                continue;
                            }
                        }
                        if has_static && has_dependent {
                            return Err(SemanticError::new(format!(
                                "class '{name}' properties: attributes 'Static' and 'Dependent' cannot be combined"
                            )));
                        }
                        if has_constant && has_dependent {
                            return Err(SemanticError::new(format!(
                                "class '{name}' properties: attributes 'Constant' and 'Dependent' cannot be combined"
                            )));
                        }
                        // If Access provided without Get/Set overrides, it's fine; if overrides provided, also fine.
                        let _ = (access_default, get_access, set_access);
                        // Enforce property attribute semantics minimal subset: ensure no duplicate Static flags in conflict (placeholder)
                        for p in props {
                            if !prop_names.insert(p.clone()) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate property '{p}' in class {name}"
                                )));
                            }
                            if method_names.contains(p) {
                                return Err(SemanticError::new(format!(
                                    "Name '{p}' used for both property and method in class {name}"
                                )));
                            }
                        }
                    }
                    HirClassMember::Methods { body, attributes } => {
                        // Validate method attributes: Access must be public/private if present
                        let mut _has_static = false;
                        let mut has_abstract = false;
                        let mut has_sealed = false;
                        let mut _has_hidden = false;
                        for a in attributes {
                            if a.name.eq_ignore_ascii_case("Static") {
                                _has_static = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Abstract") {
                                has_abstract = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Sealed") {
                                has_sealed = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Hidden") {
                                _has_hidden = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Access") {
                                let v =
                                    a.value.as_ref().ok_or_else(|| {
                                        format!(
                                            "Access requires value in class '{name}' methods block",
                                        )
                                    })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' methods"), &v)?;
                            }
                        }
                        if has_abstract && has_sealed {
                            return Err(SemanticError::new(format!(
                                "class '{name}' methods: attributes 'Abstract' and 'Sealed' cannot be combined"
                            )));
                        }
                        // Extract method function names at top-level of methods block
                        for s in body {
                            if let HirStmt::Function { name: fname, .. } = s {
                                if !method_names.insert(fname.clone()) {
                                    return Err(SemanticError::new(format!(
                                        "Duplicate method '{fname}' in class {name}"
                                    )));
                                }
                                if prop_names.contains(fname) {
                                    return Err(SemanticError::new(format!(
                                        "Name '{fname}' used for both property and method in class {name}"
                                    )));
                                }
                            }
                        }
                    }
                    HirClassMember::Events { attributes, names } => {
                        // Events: currently no attributes enforced; names must be unique within class
                        for ev in names {
                            if method_names.contains(ev) || prop_names.contains(ev) {
                                return Err(SemanticError::new(format!(
                                    "Name '{ev}' used for event conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let mut seen = std::collections::HashSet::new();
                        for ev in names {
                            if !seen.insert(ev) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate event '{ev}' in class {name}"
                                )));
                            }
                        }
                        let _ = attributes; // placeholder for future attribute validation
                    }
                    HirClassMember::Enumeration { attributes, names } => {
                        // Enumeration: unique names; no conflicts with props/methods
                        for en in names {
                            if method_names.contains(en) || prop_names.contains(en) {
                                return Err(SemanticError::new(format!(
                                    "Name '{en}' used for enumeration conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let mut seen = std::collections::HashSet::new();
                        for en in names {
                            if !seen.insert(en) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate enumeration '{en}' in class {name}"
                                )));
                            }
                        }
                        let _ = attributes;
                    }
                    HirClassMember::Arguments { attributes, names } => {
                        // Arguments: ensure no conflicts with props/methods
                        for ar in names {
                            if method_names.contains(ar) || prop_names.contains(ar) {
                                return Err(SemanticError::new(format!(
                                    "Name '{ar}' used for arguments conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let _ = attributes;
                    }
                }
            }
        }
    }
    Ok(())
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
                    super::HirLValue::Var(v) => {
                        super::HirLValue::Var(var_map.get(v).copied().unwrap_or(*v))
                    }
                    super::HirLValue::Member(b, n) => {
                        super::HirLValue::Member(Box::new(remap_expr(b, var_map)), n.clone())
                    }
                    super::HirLValue::MemberDynamic(b, n) => super::HirLValue::MemberDynamic(
                        Box::new(remap_expr(b, var_map)),
                        Box::new(remap_expr(n, var_map)),
                    ),
                    super::HirLValue::Index(b, idxs) => super::HirLValue::Index(
                        Box::new(remap_expr(b, var_map)),
                        idxs.iter().map(|e| remap_expr(e, var_map)).collect(),
                    ),
                    super::HirLValue::IndexCell(b, idxs) => super::HirLValue::IndexCell(
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
                        HirClassMember::Arguments { attributes, names } => {
                            HirClassMember::Arguments {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
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
            HirExprKind::Member(base, name) => {
                HirExprKind::Member(Box::new(remap_expr(base, var_map)), name.clone())
            }
            HirExprKind::MemberDynamic(base, name) => HirExprKind::MemberDynamic(
                Box::new(remap_expr(base, var_map)),
                Box::new(remap_expr(name, var_map)),
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
                span: _,
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
            HirStmt::While {
                cond,
                body,
                span: _,
            } => {
                collect_expr_variables(cond, vars);
                for stmt in body {
                    collect_stmt_variables(stmt, vars);
                }
            }
            HirStmt::For {
                var,
                expr,
                body,
                span: _,
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
                span: _,
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
                span: _,
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
                    for e in row {
                        collect_expr_variables(e, vars);
                    }
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
            HirExprKind::MethodCall(base, _, args) => {
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
    var_names: Vec<Option<String>>,
    allow_unqualified_statics: bool,
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
            var_names: Vec::new(),
            allow_unqualified_statics: false,
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
        self.scopes[current].bindings.insert(name.clone(), id);
        self.var_types.push(Type::Unknown);
        self.var_names.push(Some(name));
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

    fn lower_stmts(&mut self, stmts: &[AstStmt]) -> Result<Vec<HirStmt>, SemanticError> {
        stmts.iter().map(|s| self.lower_stmt(s)).collect()
    }

    fn lower_stmt(&mut self, stmt: &AstStmt) -> Result<HirStmt, SemanticError> {
        let span = stmt.span();
        match stmt {
            AstStmt::ExprStmt(e, semicolon_terminated, _) => Ok(HirStmt::ExprStmt(
                self.lower_expr(e)?,
                *semicolon_terminated,
                span,
            )),
            AstStmt::Assign(name, expr, semicolon_terminated, _) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                let value = self.lower_expr(expr)?;
                if id.0 < self.var_types.len() {
                    self.var_types[id.0] = value.ty.clone();
                }
                Ok(HirStmt::Assign(id, value, *semicolon_terminated, span))
            }
            AstStmt::MultiAssign(names, expr, semicolon_terminated, _) => {
                let ids: Vec<Option<VarId>> = names
                    .iter()
                    .map(|n| {
                        if n == "~" {
                            None
                        } else {
                            Some(match self.lookup(n) {
                                Some(id) => id,
                                None => self.define(n.to_string()),
                            })
                        }
                    })
                    .collect();
                let value = self.lower_expr(expr)?;
                Ok(HirStmt::MultiAssign(
                    ids,
                    value,
                    *semicolon_terminated,
                    span,
                ))
            }
            AstStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                span: _,
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
                    span,
                })
            }
            AstStmt::While {
                cond,
                body,
                span: _,
            } => Ok(HirStmt::While {
                cond: self.lower_expr(cond)?,
                body: self.lower_stmts(body)?,
                span,
            }),
            AstStmt::For {
                var,
                expr,
                body,
                span: _,
            } => {
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
                    span,
                })
            }
            AstStmt::Switch {
                expr,
                cases,
                otherwise,
                span: _,
            } => {
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
                Ok(HirStmt::Switch {
                    expr: control,
                    cases: cases_hir,
                    otherwise: otherwise_hir,
                    span,
                })
            }
            AstStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
                span: _,
            } => {
                let try_hir = self.lower_stmts(try_body)?;
                let catch_var_id = catch_var.as_ref().map(|name| match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                });
                let catch_hir = self.lower_stmts(catch_body)?;
                Ok(HirStmt::TryCatch {
                    try_body: try_hir,
                    catch_var: catch_var_id,
                    catch_body: catch_hir,
                    span,
                })
            }
            AstStmt::Global(names, _) => {
                let pairs: Vec<(VarId, String)> = names
                    .iter()
                    .map(|n| {
                        let id = match self.lookup(n) {
                            Some(id) => id,
                            None => self.define(n.to_string()),
                        };
                        (id, n.clone())
                    })
                    .collect();
                Ok(HirStmt::Global(pairs, span))
            }
            AstStmt::Persistent(names, _) => {
                let pairs: Vec<(VarId, String)> = names
                    .iter()
                    .map(|n| {
                        let id = match self.lookup(n) {
                            Some(id) => id,
                            None => self.define(n.to_string()),
                        };
                        (id, n.clone())
                    })
                    .collect();
                Ok(HirStmt::Persistent(pairs, span))
            }
            AstStmt::Break(_) => Ok(HirStmt::Break(span)),
            AstStmt::Continue(_) => Ok(HirStmt::Continue(span)),
            AstStmt::Return(_) => Ok(HirStmt::Return(span)),
            AstStmt::Function {
                name,
                params,
                outputs,
                body,
                span: _,
            } => {
                self.push_scope();
                let param_ids: Vec<VarId> = params.iter().map(|p| self.define(p.clone())).collect();
                let output_ids: Vec<VarId> =
                    outputs.iter().map(|o| self.define(o.clone())).collect();
                let body_hir = self.lower_stmts(body)?;
                self.pop_scope();

                let has_varargin = params
                    .last()
                    .map(|s| s.as_str() == "varargin")
                    .unwrap_or(false);
                let has_varargout = outputs
                    .last()
                    .map(|s| s.as_str() == "varargout")
                    .unwrap_or(false);

                let func_stmt = HirStmt::Function {
                    name: name.clone(),
                    params: param_ids,
                    outputs: output_ids,
                    body: body_hir,
                    has_varargin,
                    has_varargout,
                    span,
                };

                // Register the function in the context for future calls
                self.functions.insert(name.clone(), func_stmt.clone());

                Ok(func_stmt)
            }
            AstStmt::ClassDef {
                name,
                super_class,
                members,
                span: _,
            } => {
                // Lightweight lowering of class blocks into HIR without deep semantic checks
                let members_hir = members
                    .iter()
                    .map(|m| match m {
                        parser::ClassMember::Properties { attributes, names } => {
                            HirClassMember::Properties {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Events { attributes, names } => {
                            HirClassMember::Events {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Enumeration { attributes, names } => {
                            HirClassMember::Enumeration {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Arguments { attributes, names } => {
                            HirClassMember::Arguments {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Methods { attributes, body } => {
                            match self.lower_stmts(body) {
                                Ok(s) => HirClassMember::Methods {
                                    attributes: attributes.clone(),
                                    body: s,
                                },
                                Err(_) => HirClassMember::Methods {
                                    attributes: attributes.clone(),
                                    body: Vec::new(),
                                },
                            }
                        }
                    })
                    .collect();
                Ok(HirStmt::ClassDef {
                    name: name.clone(),
                    super_class: super_class.clone(),
                    members: members_hir,
                    span,
                })
            }
            AstStmt::AssignLValue(lv, rhs, suppressed, _) => {
                // Lower true lvalue assignment into HirStmt::AssignLValue
                let hir_lv = self.lower_lvalue(lv)?;
                let value = self.lower_expr(rhs)?;
                // If target is a plain variable, update its type from RHS
                if let HirLValue::Var(var_id) = hir_lv {
                    if var_id.0 < self.var_types.len() {
                        self.var_types[var_id.0] = value.ty.clone();
                    }
                    return Ok(HirStmt::Assign(var_id, value, *suppressed, span));
                }
                Ok(HirStmt::AssignLValue(hir_lv, value, *suppressed, span))
            }
            AstStmt::Import { .. } => {
                // Import statements have no runtime effect in HIR
                if let AstStmt::Import { path, wildcard, .. } = stmt {
                    Ok(HirStmt::Import {
                        path: path.clone(),
                        wildcard: *wildcard,
                        span,
                    })
                } else {
                    unreachable!()
                }
            }
        }
    }

    fn lower_expr(&mut self, expr: &AstExpr) -> Result<HirExpr, SemanticError> {
        use parser::Expr::*;
        let span = expr.span();
        let (kind, ty) = match expr {
            Number(n, _) => (HirExprKind::Number(n.clone()), Type::Num),
            String(s, _) => (HirExprKind::String(s.clone()), Type::String),
            Ident(name, _) => {
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
                } else if self.allow_unqualified_statics {
                    (HirExprKind::Constant(name.clone()), Type::Unknown)
                } else {
                    let ident = format!("{}:UndefinedVariable", error_namespace());
                    return Err(SemanticError::new(format!("Undefined variable: {name}"))
                        .with_identifier(ident)
                        .with_span(span));
                }
            }
            Unary(op, e, _) => {
                let inner = self.lower_expr(e)?;
                let ty = inner.ty.clone();
                (HirExprKind::Unary(*op, Box::new(inner)), ty)
            }
            Binary(a, op, b, _) => {
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
                        if matches!(left_ty, Type::Tensor { .. })
                            || matches!(right_ty, Type::Tensor { .. })
                        {
                            Type::tensor()
                        } else {
                            Type::Num
                        }
                    }
                    // Element-wise operations preserve the matrix type if either operand is a matrix
                    BinOp::ElemMul | BinOp::ElemDiv | BinOp::ElemPow | BinOp::ElemLeftDiv => {
                        if matches!(left_ty, Type::Tensor { .. })
                            || matches!(right_ty, Type::Tensor { .. })
                        {
                            Type::tensor()
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
                    BinOp::Colon => Type::tensor(),
                };
                (
                    HirExprKind::Binary(Box::new(left), *op, Box::new(right)),
                    ty,
                )
            }
            AnonFunc {
                params,
                body,
                span: _,
            } => {
                // Lower body in a fresh scope with parameters bound to local VarIds
                let saved_len = self.scopes.len();
                self.push_scope();
                let mut param_ids: Vec<VarId> = Vec::with_capacity(params.len());
                for p in params {
                    param_ids.push(self.define(p.clone()));
                }
                let lowered_body = self.lower_expr(body)?;
                // restore scope
                while self.scopes.len() > saved_len {
                    self.pop_scope();
                }
                (
                    HirExprKind::AnonFunc {
                        params: param_ids,
                        body: Box::new(lowered_body),
                    },
                    Type::Unknown,
                )
            }
            FuncHandle(name, _) => (HirExprKind::FuncHandle(name.clone()), Type::Unknown),
            FuncCall(name, args, _) => {
                if name == "__register_test_classes" {
                    self.allow_unqualified_statics = true;
                }
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
                        span,
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
            Tensor(rows, _) => {
                let mut hir_rows = Vec::new();
                for row in rows {
                    let mut hir_row = Vec::new();
                    for expr in row {
                        hir_row.push(self.lower_expr(expr)?);
                    }
                    hir_rows.push(hir_row);
                }
                (HirExprKind::Tensor(hir_rows), Type::tensor())
            }
            Cell(rows, _) => {
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
            Index(expr, indices, _) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                let ty = base.ty.clone(); // Indexing preserves base type for now
                (HirExprKind::Index(Box::new(base), idx_exprs), ty)
            }
            IndexCell(expr, indices, _) => {
                let base = self.lower_expr(expr)?;
                let idx_exprs: Result<Vec<_>, _> =
                    indices.iter().map(|i| self.lower_expr(i)).collect();
                let idx_exprs = idx_exprs?;
                (
                    HirExprKind::IndexCell(Box::new(base), idx_exprs),
                    Type::Unknown,
                )
            }
            Range(start, step, end, _) => {
                let start_hir = self.lower_expr(start)?;
                let end_hir = self.lower_expr(end)?;
                let step_hir = step.as_ref().map(|s| self.lower_expr(s)).transpose()?;
                (
                    HirExprKind::Range(
                        Box::new(start_hir),
                        step_hir.map(Box::new),
                        Box::new(end_hir),
                    ),
                    Type::tensor(),
                )
            }
            Colon(_) => (HirExprKind::Colon, Type::tensor()),
            EndKeyword(_) => (HirExprKind::End, Type::Unknown),
            Member(base, name, _) => {
                let b = self.lower_expr(base)?;
                (
                    HirExprKind::Member(Box::new(b), name.clone()),
                    Type::Unknown,
                )
            }
            MemberDynamic(base, name_expr, _) => {
                let b = self.lower_expr(base)?;
                let n = self.lower_expr(name_expr)?;
                (
                    HirExprKind::MemberDynamic(Box::new(b), Box::new(n)),
                    Type::Unknown,
                )
            }
            MethodCall(base, name, args, _) => {
                let b = self.lower_expr(base)?;
                let lowered_args: Result<Vec<_>, _> =
                    args.iter().map(|a| self.lower_expr(a)).collect();
                (
                    HirExprKind::MethodCall(Box::new(b), name.clone(), lowered_args?),
                    Type::Unknown,
                )
            }
            MetaClass(name, _) => (HirExprKind::MetaClass(name.clone()), Type::String),
        };
        Ok(HirExpr { kind, ty, span })
    }

    fn lower_lvalue(&mut self, lv: &parser::LValue) -> Result<HirLValue, SemanticError> {
        use parser::LValue as ALV;
        Ok(match lv {
            ALV::Var(name) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                HirLValue::Var(id)
            }
            ALV::Member(base, name) => {
                // Special-case unknown identifier base to allow struct-like creation semantics (e.g., s.f = 4)
                if let parser::Expr::Ident(var_name, _) = &**base {
                    let id = match self.lookup(var_name) {
                        Some(id) => id,
                        None => self.define(var_name.clone()),
                    };
                    let ty = if id.0 < self.var_types.len() {
                        self.var_types[id.0].clone()
                    } else {
                        Type::Unknown
                    };
                    let b = HirExpr {
                        kind: HirExprKind::Var(id),
                        ty,
                        span: base.span(),
                    };
                    HirLValue::Member(Box::new(b), name.clone())
                } else {
                    let b = self.lower_expr(base)?;
                    HirLValue::Member(Box::new(b), name.clone())
                }
            }
            ALV::MemberDynamic(base, name_expr) => {
                let b = self.lower_expr(base)?;
                let n = self.lower_expr(name_expr)?;
                HirLValue::MemberDynamic(Box::new(b), Box::new(n))
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
                let arg_types: Vec<Type> = args.iter().map(|arg| arg.ty.clone()).collect();
                return builtin.infer_return_type(&arg_types);
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
        result_types.first().cloned().unwrap_or(Type::Unknown)
    }

    fn infer_outputs_types(&self, outputs: &[VarId], body: &[HirStmt]) -> Vec<Type> {
        use std::collections::HashMap;

        #[derive(Clone)]
        struct Analysis {
            exits: Vec<HashMap<VarId, Type>>,          // envs at return points
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

        #[allow(clippy::type_complexity)]
        #[allow(clippy::only_used_in_recursion)]
        fn analyze_stmts(
            #[allow(clippy::only_used_in_recursion)] _outputs: &[VarId],
            stmts: &[HirStmt],
            mut env: HashMap<VarId, Type>,
        ) -> Analysis {
            let mut exits = Vec::new();
            let mut i = 0usize;
            while i < stmts.len() {
                match &stmts[i] {
                    HirStmt::Assign(var, expr, _, _) => {
                        env.insert(*var, expr.ty.clone());
                    }
                    HirStmt::MultiAssign(vars, expr, _, _) => {
                        for v in vars.iter().flatten() {
                            env.insert(*v, expr.ty.clone());
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
                        cond: _,
                        then_body,
                        elseif_blocks,
                        else_body,
                        span: _,
                    } => {
                        let then_a = analyze_stmts(_outputs, then_body, env.clone());
                        let mut out_env = then_a.fallthrough.unwrap_or_else(|| env.clone());
                        let mut all_exits = then_a.exits;
                        for (c, b) in elseif_blocks {
                            let _ = c; // cond type unused in analysis
                            let a = analyze_stmts(_outputs, b, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = join_env(&out_env, &f);
                            }
                            all_exits.extend(a.exits);
                        }
                        if let Some(else_body) = else_body {
                            let a = analyze_stmts(_outputs, else_body, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = join_env(&out_env, &f);
                            }
                            all_exits.extend(a.exits);
                        } else {
                            // no else: join with incoming env
                            out_env = join_env(&out_env, &env);
                        }
                        env = out_env;
                        exits.extend(all_exits);
                    }
                    HirStmt::While {
                        cond: _,
                        body,
                        span: _,
                    } => {
                        // Approximate: analyze once and join with incoming env
                        let a = analyze_stmts(_outputs, body, env.clone());
                        if let Some(f) = a.fallthrough {
                            env = join_env(&env, &f);
                        }
                        exits.extend(a.exits);
                    }
                    HirStmt::For {
                        var,
                        expr,
                        body,
                        span: _,
                    } => {
                        // Assign loop var type from expr type
                        env.insert(*var, expr.ty.clone());
                        let a = analyze_stmts(_outputs, body, env.clone());
                        if let Some(f) = a.fallthrough {
                            env = join_env(&env, &f);
                        }
                        exits.extend(a.exits);
                    }
                    HirStmt::Switch {
                        expr: _,
                        cases,
                        otherwise,
                        span: _,
                    } => {
                        let mut out_env: Option<HashMap<VarId, Type>> = None;
                        for (_v, b) in cases {
                            let a = analyze_stmts(_outputs, b, env.clone());
                            if let Some(f) = a.fallthrough {
                                out_env = Some(match out_env {
                                    Some(curr) => join_env(&curr, &f),
                                    None => f,
                                });
                            }
                            exits.extend(a.exits);
                        }
                        if let Some(otherwise) = otherwise {
                            let a = analyze_stmts(_outputs, otherwise, env.clone());
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
                        catch_var: _,
                        catch_body,
                        span: _,
                    } => {
                        let a_try = analyze_stmts(_outputs, try_body, env.clone());
                        let a_catch = analyze_stmts(_outputs, catch_body, env.clone());
                        let mut out_env = a_try.fallthrough.unwrap_or_else(|| env.clone());
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
                        // Update env conservatively based on RHS type (specific lvalue target unknown)
                        // No binding updated unless it's a plain variable (handled elsewhere)
                        let _ = &expr.ty;
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
        for e in &analysis.exits {
            accumulate(e);
        }
        if let Some(f) = &analysis.fallthrough {
            accumulate(f);
        }
        per_output
    }
}

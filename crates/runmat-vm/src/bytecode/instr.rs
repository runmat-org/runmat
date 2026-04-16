use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pops: usize,
    pub pushes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmitLabel {
    Ans,
    Var(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndExpr {
    End,
    Const(f64),
    Var(usize),
    Call(String, Vec<EndExpr>),
    Add(Box<EndExpr>, Box<EndExpr>),
    Sub(Box<EndExpr>, Box<EndExpr>),
    Mul(Box<EndExpr>, Box<EndExpr>),
    Div(Box<EndExpr>, Box<EndExpr>),
    LeftDiv(Box<EndExpr>, Box<EndExpr>),
    Pow(Box<EndExpr>, Box<EndExpr>),
    Neg(Box<EndExpr>),
    Pos(Box<EndExpr>),
    Floor(Box<EndExpr>),
    Ceil(Box<EndExpr>),
    Round(Box<EndExpr>),
    Fix(Box<EndExpr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instr {
    // Constant and variable loads.
    LoadConst(f64),
    LoadComplex(f64, f64),
    LoadBool(bool),
    LoadString(String),
    LoadCharRow(String),
    LoadVar(usize),
    StoreVar(usize),

    // Scalar and matrix arithmetic.
    Add,
    Sub,
    Mul,
    RightDiv,
    LeftDiv,
    Pow,
    Neg,
    UPlus,
    Transpose,
    ConjugateTranspose,
    ElemMul,
    ElemDiv,
    ElemPow,
    ElemLeftDiv,
    LessEqual,
    Less,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,

    // Short-circuit logical control flow.
    AndAnd(usize),
    OrOr(usize),
    JumpIfFalse(usize),
    Jump(usize),
    Pop,

    // Expands a single value into N outputs, padding with zero values when needed.
    Unpack(usize),

    // Direct builtin invocation.
    CallBuiltin(String, usize),

    // Specialized lowering target for the stochastic evolution fast path.
    StochasticEvolution,

    // Array construction and direct indexing.
    CreateMatrix(usize, usize),
    CreateMatrixDynamic(usize),
    CreateRange(bool),
    Index(usize),

    // Slice indexing with compiler-encoded colon and plain `end` masks.
    IndexSlice(usize, usize, u32, u32),

    // General slice/index path carrying dynamic ranges and `end` arithmetic.
    IndexSliceExpr {
        dims: usize,
        numeric_count: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: Vec<usize>,
        range_has_step: Vec<bool>,
        range_start_exprs: Vec<Option<EndExpr>>,
        range_step_exprs: Vec<Option<EndExpr>>,
        range_end_exprs: Vec<EndExpr>,
        end_numeric_exprs: Vec<(usize, EndExpr)>,
    },

    // Assignment counterpart to `IndexSliceExpr`.
    StoreSliceExpr {
        dims: usize,
        numeric_count: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: Vec<usize>,
        range_has_step: Vec<bool>,
        range_start_exprs: Vec<Option<EndExpr>>,
        range_step_exprs: Vec<Option<EndExpr>>,
        range_end_exprs: Vec<EndExpr>,
        end_numeric_exprs: Vec<(usize, EndExpr)>,
    },

    // Cell array construction and indexing.
    CreateCell2D(usize, usize),
    IndexCell(usize),

    // Expands cell contents into a comma-separated list with fixed output arity.
    IndexCellExpand(usize, usize),

    // Indexed assignment updates the base value and pushes the updated base.
    StoreIndex(usize),
    StoreIndexCell(usize),

    // Slice assignment with compiler-encoded colon and plain `end` masks.
    StoreSlice(usize, usize, u32, u32),

    // Struct, object, and class member access.
    LoadMember(String),
    LoadMemberOrInit(String),
    LoadMemberDynamic,
    LoadMemberDynamicOrInit,
    StoreMember(String),
    StoreMemberOrInit(String),
    StoreMemberDynamic,
    StoreMemberDynamicOrInit,
    LoadMethod(String),
    CallMethod(String, usize),

    // Ambiguous `obj.name(...)` shape resolved at runtime as method call or member indexing.
    CallMethodOrMemberIndex(String, usize),

    // Closure and static class dispatch.
    CreateClosure(String, usize),
    LoadStaticProperty(String, String),
    CallStaticMethod(String, String, usize),

    // Registers a runtime class definition produced by `classdef` lowering.
    RegisterClass {
        name: String,
        super_class: Option<String>,
        properties: Vec<(String, bool, String, String)>,
        methods: Vec<(String, String, bool, String)>,
    },

    // `feval` keeps the callable value on the stack instead of naming the target statically.
    CallFeval(usize),
    CallFevalExpandMulti(Vec<ArgSpec>),

    // Stack and exception-control operations.
    Swap,
    EnterTry(usize, Option<usize>),
    PopTry,
    Return,
    ReturnValue,

    // User-function invocation variants.
    CallFunction(String, usize),

    // Calls a user function and shapes the result list to `out_count`.
    CallFunctionMulti(String, usize, usize),

    // Expands one argument position from cell indexing at runtime.
    CallFunctionExpandAt(String, usize, usize, usize),
    CallBuiltinExpandLast(String, usize, usize),
    CallBuiltinExpandAt(String, usize, usize, usize),
    CallFunctionExpandMulti(String, Vec<ArgSpec>),
    CallBuiltinExpandMulti(String, Vec<ArgSpec>),

    // Packs the top N values into row or column tensor form.
    PackToRow(usize),
    PackToCol(usize),

    // Local scope and local variable access.
    EnterScope(usize),
    ExitScope(usize),
    LoadLocal(usize),
    StoreLocal(usize),

    // Import registration for later unqualified resolution.
    RegisterImport {
        path: Vec<String>,
        wildcard: bool,
    },

    // Global and persistent declarations, including name-stable forms across units.
    DeclareGlobal(Vec<usize>),
    DeclarePersistent(Vec<usize>),
    DeclareGlobalNamed(Vec<usize>, Vec<String>),
    DeclarePersistentNamed(Vec<usize>, Vec<String>),

    // Emission instructions used to produce visible workspace outputs.
    EmitStackTop {
        label: EmitLabel,
    },
    EmitVar {
        var_index: usize,
        label: EmitLabel,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgSpec {
    pub is_expand: bool,
    pub num_indices: usize,
    pub expand_all: bool,
}

impl Instr {
    pub fn stack_effect(&self) -> Option<StackEffect> {
        fn effect(pops: usize, pushes: usize) -> Option<StackEffect> {
            Some(StackEffect { pops, pushes })
        }

        match self {
            Instr::LoadConst(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadVar(_)
            | Instr::LoadLocal(_) => effect(0, 1),
            Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Pop
            | Instr::JumpIfFalse(_)
            | Instr::AndAnd(_)
            | Instr::OrOr(_) => effect(1, 0),
            Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::Pow
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::LessEqual
            | Instr::Less
            | Instr::Greater
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual => effect(2, 1),
            Instr::Swap => effect(2, 2),
            Instr::Neg
            | Instr::UPlus
            | Instr::Transpose
            | Instr::ConjugateTranspose
            | Instr::LoadMember(_)
            | Instr::LoadMemberOrInit(_)
            | Instr::LoadMethod(_) => effect(1, 1),
            Instr::CallBuiltin(_, argc) | Instr::CallFunction(_, argc) => effect(*argc, 1),
            Instr::CallFunctionMulti(_, argc, out_count) => effect(*argc, *out_count),
            Instr::CallMethod(_, argc) | Instr::CallMethodOrMemberIndex(_, argc) => {
                effect(argc + 1, 1)
            }
            Instr::CallStaticMethod(_, _, argc) => effect(*argc, 1),
            Instr::CallFeval(argc) => effect(argc + 1, 1),
            Instr::CreateMatrix(rows, cols) | Instr::CreateCell2D(rows, cols) => {
                effect(rows * cols, 1)
            }
            Instr::CreateMatrixDynamic(rows) => effect(*rows, 1),
            Instr::CreateRange(has_step) => effect(if *has_step { 3 } else { 2 }, 1),
            Instr::Unpack(n) => effect(1, *n),
            Instr::Index(n) | Instr::IndexCell(n) => effect(n + 1, 1),
            Instr::IndexCellExpand(n, out_count) => effect(n + 1, *out_count),
            Instr::StoreIndex(n) | Instr::StoreIndexCell(n) => effect(n + 2, 1),
            Instr::IndexSlice(dims, numeric_count, _, _)
            | Instr::StoreSlice(dims, numeric_count, _, _) => {
                let pops = 1 + numeric_count;
                if matches!(self, Instr::StoreSlice(_, _, _, _)) {
                    effect(pops + 1, 1)
                } else {
                    let _ = dims;
                    effect(pops, 1)
                }
            }
            Instr::IndexSliceExpr {
                numeric_count,
                range_dims,
                ..
            } => effect(1 + numeric_count + range_dims.len(), 1),
            Instr::StoreSliceExpr {
                numeric_count,
                range_dims,
                ..
            } => effect(2 + numeric_count + range_dims.len(), 1),
            Instr::StoreMember(_)
            | Instr::StoreMemberOrInit(_)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit => effect(2, 1),
            Instr::LoadMemberDynamic | Instr::LoadMemberDynamicOrInit => effect(2, 1),
            Instr::CreateClosure(_, capture_count) => effect(*capture_count, 1),
            Instr::LoadStaticProperty(_, _) => effect(0, 1),
            Instr::RegisterClass { .. } => effect(0, 0),
            Instr::CallFevalExpandMulti(specs)
            | Instr::CallFunctionExpandMulti(_, specs)
            | Instr::CallBuiltinExpandMulti(_, specs) => {
                let fixed = specs.iter().filter(|s| !s.is_expand).count();
                let expanded: usize = specs
                    .iter()
                    .filter(|s| s.is_expand)
                    .map(|s| 1 + s.num_indices)
                    .sum();
                let handle = usize::from(matches!(self, Instr::CallFevalExpandMulti(_)));
                effect(handle + fixed + expanded, 1)
            }
            Instr::CallFunctionExpandAt(_, before, num_indices, after)
            | Instr::CallBuiltinExpandAt(_, before, num_indices, after) => {
                effect(before + after + 1 + num_indices, 1)
            }
            Instr::CallBuiltinExpandLast(_, fixed_argc, num_indices) => {
                effect(fixed_argc + 1 + num_indices, 1)
            }
            Instr::PackToRow(n) | Instr::PackToCol(n) => effect(*n, 1),
            Instr::EnterScope(_) | Instr::ExitScope(_) | Instr::Jump(_) | Instr::PopTry => {
                effect(0, 0)
            }
            Instr::EnterTry(_, _) => effect(0, 0),
            Instr::Return => effect(0, 0),
            Instr::ReturnValue => effect(1, 0),
            Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclarePersistent(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistentNamed(_, _) => effect(0, 0),
            Instr::EmitStackTop { .. } => effect(1, 1),
            Instr::EmitVar { .. } => effect(0, 0),
            Instr::StochasticEvolution => None,
        }
    }
}

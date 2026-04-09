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
    LoadConst(f64),
    LoadComplex(f64, f64),
    LoadBool(bool),
    LoadString(String),
    LoadCharRow(String),
    LoadVar(usize),
    StoreVar(usize),
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
    // Element-wise operations
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
    // Short-circuit logicals
    AndAnd(usize), // jump target if lhs is false (0)
    OrOr(usize),   // jump target if lhs is true (non-zero)
    JumpIfFalse(usize),
    Jump(usize),
    Pop,
    // Unpack a value into N outputs (uses OutputList when present, pads with 0.0)
    Unpack(usize),
    CallBuiltin(String, usize),
    StochasticEvolution,
    // User function call
    CreateMatrix(usize, usize),
    CreateMatrixDynamic(usize), // Number of rows, each row can have variable elements
    CreateRange(bool),          // true if step is provided, false if start:end
    Index(usize),               // Number of indices (all numeric)
    // Matrix slicing with colon and end support:
    // dims = total dims, numeric = how many numeric indices were pushed,
    // colon_mask bit i set => dim i is colon (0-based),
    // end_mask bit i set => dim i is plain 'end' (no arithmetic) and should resolve to that dim's length
    IndexSlice(usize, usize, u32, u32),
    // Unified expression-based slicing/indexing path with dynamic ranges and end arithmetic
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
    // Unified expression-based slice assignment path with dynamic ranges and end arithmetic
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
    // Cell arrays
    CreateCell2D(usize, usize), // rows, cols; pops rows*cols elements
    IndexCell(usize),           // Number of indices (1D or 2D supported)
    // Expand cell array contents into a comma-separated list (column-major),
    // pushing exactly out_count values onto the stack
    IndexCellExpand(usize, usize), // (num_indices, out_count)
    // L-value element assignments
    StoreIndex(usize), // like Index, but pops RHS and updates base, then pushes updated base
    StoreIndexCell(usize), // like IndexCell, but for cell arrays; pops RHS and updates base
    // Store slice with colon/end semantics (mirrors IndexSlice)
    StoreSlice(usize, usize, u32, u32),
    // Object/Class member/method operations
    LoadMember(String),        // base on stack -> member value
    LoadMemberOrInit(String), // base on stack -> member value (missing struct field => empty struct)
    LoadMemberDynamic,        // base, name on stack -> member value (structs and objects)
    LoadMemberDynamicOrInit, // base, name on stack -> member value (missing struct field => empty struct)
    StoreMember(String),     // base, rhs on stack -> updated base
    StoreMemberOrInit(String), // base, rhs on stack -> updated base (numeric zero base => struct)
    StoreMemberDynamic,      // base, name, rhs on stack -> updated base
    StoreMemberDynamicOrInit, // base, name, rhs on stack -> updated base (numeric zero base => struct)
    LoadMethod(String),       // base on stack -> method handle
    CallMethod(String, usize), // base on stack along with args
    CallMethodOrMemberIndex(String, usize),
    // Closures and handle invocation
    CreateClosure(String, usize), // function name and capture count; captures expected on stack
    // Static class access
    LoadStaticProperty(String, String),      // class, property
    CallStaticMethod(String, String, usize), // class, method, argc
    // Class definition at runtime
    RegisterClass {
        name: String,
        super_class: Option<String>,
        properties: Vec<(String, bool, String, String)>, // (name, is_static, get_access, set_access)
        methods: Vec<(String, String, bool, String)>, // (method name, function name, is_static, access)
    },
    // feval special path to support closures/user functions/method handles
    CallFeval(usize), // argc
    // feval with expansion specs for arguments
    CallFevalExpandMulti(Vec<ArgSpec>),
    // Stack manipulation
    Swap, // swap top two stack values
    // Try/Catch control flow
    EnterTry(usize, Option<usize>), // catch_pc, catch_var (global var index)
    PopTry,                         // pop current try frame
    Return,
    ReturnValue,                 // Return with a value from the stack
    CallFunction(String, usize), // Function name and argument count
    // Call a user function and push `out_count` return values (left-to-right), if fewer available push 0
    CallFunctionMulti(String, usize, usize),
    // Call a user function with one argument (at position) expanded from a cell indexing expression
    // (name, before_count, num_indices, after_count)
    CallFunctionExpandAt(String, usize, usize, usize),
    // Call a builtin with the last argument expanded from a cell indexing expression at runtime
    // (name, fixed_argc (count of preceding fixed args), num_indices for the cell indexing)
    CallBuiltinExpandLast(String, usize, usize),
    // Call a builtin with one argument (at position) expanded from a cell indexing expression
    // (name, before_count, num_indices, after_count)
    CallBuiltinExpandAt(String, usize, usize, usize),
    // Multi-arg expansion for user functions (parallel to builtin)
    CallFunctionExpandMulti(String, Vec<ArgSpec>),
    // Multi-arg expansion: specs vector corresponds to arguments in order.
    // For each ArgSpec { is_expand, num_indices }:
    // - if is_expand: pop indices (num_indices), then base; expand to value(s)
    // - else: pop one fixed argument value
    CallBuiltinExpandMulti(String, Vec<ArgSpec>),
    // Pack N top-of-stack values into a 1xN row tensor (left-to-right)
    PackToRow(usize),
    // Pack N top-of-stack values into an Nx1 column tensor (top is last)
    PackToCol(usize),
    // Scoping and call stack instructions
    EnterScope(usize), // Number of local variables to allocate
    ExitScope(usize),  // Number of local variables to deallocate
    LoadLocal(usize),  // Load from local variable (relative to current frame)
    StoreLocal(usize), // Store to local variable (relative to current frame)
    // Imports
    RegisterImport {
        path: Vec<String>,
        wildcard: bool,
    },
    // Global and Persistent declarations
    DeclareGlobal(Vec<usize>),
    DeclarePersistent(Vec<usize>),
    // New named variants to bind by source names across units
    DeclareGlobalNamed(Vec<usize>, Vec<String>),
    DeclarePersistentNamed(Vec<usize>, Vec<String>),
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
            Instr::StochasticEvolution => effect(4, 1),
            Instr::CreateMatrix(rows, cols) | Instr::CreateCell2D(rows, cols) => {
                effect(rows * cols, 1)
            }
            Instr::CreateRange(has_step) => effect(if *has_step { 3 } else { 2 }, 1),
            Instr::Index(num_indices) | Instr::IndexCell(num_indices) => effect(num_indices + 1, 1),
            Instr::IndexCellExpand(num_indices, out_count) => effect(num_indices + 1, *out_count),
            Instr::StoreIndex(num_indices) | Instr::StoreIndexCell(num_indices) => {
                effect(num_indices + 2, 1)
            }
            Instr::IndexSlice(_, numeric_count, _, _) => effect(numeric_count + 1, 1),
            Instr::StoreSlice(_, numeric_count, _, _) => effect(numeric_count + 2, 1),
            Instr::IndexSliceExpr {
                numeric_count,
                range_has_step,
                ..
            } => {
                let range_pops = range_has_step
                    .iter()
                    .map(|has_step| if *has_step { 2 } else { 1 })
                    .sum::<usize>();
                effect(1 + numeric_count + range_pops, 1)
            }
            Instr::StoreSliceExpr {
                numeric_count,
                range_has_step,
                ..
            } => {
                let range_pops = range_has_step
                    .iter()
                    .map(|has_step| if *has_step { 2 } else { 1 })
                    .sum::<usize>();
                effect(2 + numeric_count + range_pops, 1)
            }
            Instr::LoadMemberDynamic | Instr::LoadMemberDynamicOrInit => effect(2, 1),
            Instr::StoreMember(_) => effect(2, 1),
            Instr::StoreMemberOrInit(_) => effect(2, 1),
            Instr::StoreMemberDynamic => effect(3, 1),
            Instr::StoreMemberDynamicOrInit => effect(3, 1),
            Instr::CreateClosure(_, capture_count) => effect(*capture_count, 1),
            Instr::CallBuiltinExpandLast(_, fixed_argc, num_indices) => {
                effect(fixed_argc + num_indices + 1, 1)
            }
            Instr::CallBuiltinExpandAt(_, before_count, num_indices, after_count)
            | Instr::CallFunctionExpandAt(_, before_count, num_indices, after_count) => {
                effect(before_count + num_indices + after_count + 1, 1)
            }
            Instr::CallBuiltinExpandMulti(_, specs) | Instr::CallFunctionExpandMulti(_, specs) => {
                let pops = specs
                    .iter()
                    .map(|spec| {
                        if spec.is_expand {
                            1 + spec.num_indices
                        } else {
                            1
                        }
                    })
                    .sum::<usize>();
                effect(pops, 1)
            }
            Instr::CallFevalExpandMulti(specs) => {
                let pops = specs
                    .iter()
                    .map(|spec| {
                        if spec.is_expand {
                            1 + spec.num_indices
                        } else {
                            1
                        }
                    })
                    .sum::<usize>();
                effect(pops + 1, 1)
            }
            Instr::PackToRow(count) | Instr::PackToCol(count) => effect(*count, 1),
            Instr::Unpack(count) => effect(1, *count),
            Instr::EmitStackTop { .. }
            | Instr::EmitVar { .. }
            | Instr::Jump(_)
            | Instr::EnterTry(_, _)
            | Instr::PopTry
            | Instr::Return
            | Instr::EnterScope(_)
            | Instr::ExitScope(_)
            | Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclarePersistent(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::RegisterClass { .. } => effect(0, 0),
            Instr::LoadStaticProperty(_, _) => effect(0, 1),
            Instr::ReturnValue => effect(1, 0),
            Instr::CreateMatrixDynamic(_) => None,
        }
    }
}

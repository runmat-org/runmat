use serde::{Deserialize, Serialize};

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
    Div,
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
    // N-D range/selectors with per-dimension modes and end arithmetic offsets for ranges
    // dims: total dims; numeric_count: count of extra numeric scalar indices following; colon_mask/end_mask like IndexSlice;
    // range_dims: list of dimension indices that are ranges; for each, we expect start [, step] pushed in order; end_offsets align with range_dims
    IndexRangeEnd {
        dims: usize,
        numeric_count: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: Vec<usize>,
        range_has_step: Vec<bool>,
        end_offsets: Vec<i64>,
    },
    // 1-D range with end arithmetic: base on stack, then start [, step]
    Index1DRangeEnd {
        has_step: bool,
        offset: i64,
    },
    // Store with range+end arithmetic across dims; stack layout mirrors IndexRangeEnd plus RHS (on top)
    StoreRangeEnd {
        dims: usize,
        numeric_count: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: Vec<usize>,
        range_has_step: Vec<bool>,
        end_offsets: Vec<i64>,
    },
    // Extended slice: supports end arithmetic per-numeric index via offsets list.
    // Tuple items are (numeric_position_in_order, offset) representing 'end - offset'.
    IndexSliceEx(usize, usize, u32, u32, Vec<(usize, i64)>),
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
    // Store slice with end arithmetic offsets applied to numeric indices
    StoreSliceEx(usize, usize, u32, u32, Vec<(usize, i64)>),
    // Store with 1-D range having end arithmetic: base, start, [step], rhs
    StoreSlice1DRangeEnd {
        has_step: bool,
        offset: i64,
    },
    // Object/Class member/method operations
    LoadMember(String),        // base on stack -> member value
    LoadMemberDynamic,         // base, name on stack -> member value (structs and objects)
    StoreMember(String),       // base, rhs on stack -> updated base
    StoreMemberDynamic,        // base, name, rhs on stack -> updated base
    LoadMethod(String),        // base on stack -> method handle
    CallMethod(String, usize), // base on stack along with args
    // Closures and handle invocation
    CreateClosure(String, usize), // function name and capture count; captures expected on stack
    // Static class access
    LoadStaticProperty(String, String),      // class, property
    CallStaticMethod(String, String, usize), // class, method, argc
    // Package-qualified function/constructor call (e.g., Electrical.Resistor(...))
    CallQualified(String, usize), // qualified_name (e.g. "Electrical.Resistor"), argc
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
    CallBuiltinMulti(String, usize, usize),
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgSpec {
    pub is_expand: bool,
    pub num_indices: usize,
    pub expand_all: bool,
}

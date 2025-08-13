use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instr {
    LoadConst(f64),
    LoadString(String),
    LoadVar(usize),
    StoreVar(usize),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Transpose,
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
    // Cell arrays
    CreateCell2D(usize, usize), // rows, cols; pops rows*cols elements
    IndexCell(usize),           // Number of indices (1D or 2D supported)
    // Expand cell array contents into a comma-separated list (column-major),
    // pushing exactly out_count values onto the stack
    IndexCellExpand(usize, usize), // (num_indices, out_count)
    // L-value element assignments
    StoreIndex(usize),          // like Index, but pops RHS and updates base, then pushes updated base
    StoreIndexCell(usize),      // like IndexCell, but for cell arrays; pops RHS and updates base
    // Store slice with colon/end semantics (mirrors IndexSlice)
    StoreSlice(usize, usize, u32, u32),
    // Object/Class member/method operations
    LoadMember(String),          // base on stack -> member value
    StoreMember(String),         // base, rhs on stack -> updated base
    LoadMethod(String),          // base on stack -> method handle
    CallMethod(String, usize),   // base on stack along with args
    // Closures and handle invocation
    CreateClosure(String, usize), // function name and capture count; captures expected on stack
    // Static class access
    LoadStaticProperty(String, String), // class, property
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
    // Scoping and call stack instructions
    EnterScope(usize), // Number of local variables to allocate
    ExitScope(usize),  // Number of local variables to deallocate
    LoadLocal(usize),  // Load from local variable (relative to current frame)
    StoreLocal(usize), // Store to local variable (relative to current frame)
    // Imports
    RegisterImport { path: Vec<String>, wildcard: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgSpec {
    pub is_expand: bool,
    pub num_indices: usize,
    pub expand_all: bool,
}



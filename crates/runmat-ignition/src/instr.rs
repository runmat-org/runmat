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
    // Scoping and call stack instructions
    EnterScope(usize), // Number of local variables to allocate
    ExitScope(usize),  // Number of local variables to deallocate
    LoadLocal(usize),  // Load from local variable (relative to current frame)
    StoreLocal(usize), // Store to local variable (relative to current frame)
}



use runmat_builtins::Value;
use runmat_gc::{gc_register_root, gc_unregister_root, RootId, StackRoot, VariableArrayRoot};
use runmat_hir::{HirExpr, HirExprKind, HirProgram, HirStmt, VarId};
use runmat_runtime::call_builtin;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;

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
    CreateMatrix(usize, usize),
    CreateMatrixDynamic(usize), // Number of rows, each row can have variable elements
    CreateRange(bool),          // true if step is provided, false if start:end
    Index(usize),               // Number of indices
    Return,
    ReturnValue,                 // Return with a value from the stack
    CallFunction(String, usize), // Function name and argument count
    // Scoping and call stack instructions
    EnterScope(usize), // Number of local variables to allocate
    ExitScope(usize),  // Number of local variables to deallocate
    LoadLocal(usize),  // Load from local variable (relative to current frame)
    StoreLocal(usize), // Store to local variable (relative to current frame)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFunction {
    pub name: String,
    pub params: Vec<VarId>,
    pub outputs: Vec<VarId>,
    pub body: Vec<HirStmt>,
    pub local_var_count: usize, // Number of local variables this function needs
}

/// Represents a call frame in the call stack
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub return_address: usize,   // Instruction pointer to return to
    pub locals_start: usize,     // Start index in the locals array for this frame
    pub locals_count: usize,     // Number of local variables for this frame
    pub expected_outputs: usize, // Number of return values expected
}

/// Runtime execution context with call stack
#[derive(Debug)]
pub struct ExecutionContext {
    pub call_stack: Vec<CallFrame>,
    pub locals: Vec<Value>, // Local variables for all frames
    pub instruction_pointer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    pub var_count: usize,
    pub functions: HashMap<String, UserFunction>,
}

pub fn compile(prog: &HirProgram) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.compile_program(prog)?;
    Ok(Bytecode {
        instructions: c.instructions,
        var_count: c.var_count,
        functions: c.functions,
    })
}

/// Compile a program with pre-existing function definitions
pub fn compile_with_functions(
    prog: &HirProgram,
    existing_functions: &HashMap<String, UserFunction>,
) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    // Pre-populate with existing function definitions
    c.functions = existing_functions.clone();
    c.compile_program(prog)?;
    Ok(Bytecode {
        instructions: c.instructions,
        var_count: c.var_count,
        functions: c.functions,
    })
}

struct Compiler {
    instructions: Vec<Instr>,
    var_count: usize,
    loop_stack: Vec<LoopLabels>,
    functions: HashMap<String, UserFunction>,
}

/// RAII wrapper for GC root management during interpretation
struct InterpretContext {
    stack_root_id: Option<RootId>,
    vars_root_id: Option<RootId>,
}

impl InterpretContext {
    fn new(stack: &Vec<Value>, vars: &Vec<Value>) -> Result<Self, String> {
        let stack_root = Box::new(unsafe {
            StackRoot::new(stack as *const Vec<Value>, "interpreter_stack".to_string())
        });
        let vars_root = Box::new(unsafe {
            VariableArrayRoot::new(vars as *const Vec<Value>, "interpreter_vars".to_string())
        });

        let stack_root_id = gc_register_root(stack_root)
            .map_err(|e| format!("Failed to register stack root: {e:?}"))?;
        let vars_root_id = gc_register_root(vars_root)
            .map_err(|e| format!("Failed to register vars root: {e:?}"))?;

        Ok(InterpretContext {
            stack_root_id: Some(stack_root_id),
            vars_root_id: Some(vars_root_id),
        })
    }
}

impl Drop for InterpretContext {
    fn drop(&mut self) {
        if let Some(id) = self.stack_root_id.take() {
            let _ = gc_unregister_root(id);
        }
        if let Some(id) = self.vars_root_id.take() {
            let _ = gc_unregister_root(id);
        }
    }
}

struct LoopLabels {
    break_jumps: Vec<usize>,
    continue_jumps: Vec<usize>,
}

impl Compiler {
    fn new(prog: &HirProgram) -> Self {
        let mut max_var = 0;
        fn visit_expr(expr: &runmat_hir::HirExpr, max: &mut usize) {
            use runmat_hir::HirExprKind;
            match &expr.kind {
                HirExprKind::Var(id) => {
                    if id.0 + 1 > *max {
                        *max = id.0 + 1;
                    }
                }
                HirExprKind::Unary(_, e) => visit_expr(e, max),
                HirExprKind::Binary(left, _, right) => {
                    visit_expr(left, max);
                    visit_expr(right, max);
                }
                HirExprKind::Matrix(rows) => {
                    for row in rows {
                        for expr in row {
                            visit_expr(expr, max);
                        }
                    }
                }
                HirExprKind::Index(expr, indices) => {
                    visit_expr(expr, max);
                    for idx in indices {
                        visit_expr(idx, max);
                    }
                }
                HirExprKind::Range(start, step, end) => {
                    visit_expr(start, max);
                    if let Some(step) = step {
                        visit_expr(step, max);
                    }
                    visit_expr(end, max);
                }
                HirExprKind::FuncCall(_, args) => {
                    for arg in args {
                        visit_expr(arg, max);
                    }
                }
                HirExprKind::Number(_)
                | HirExprKind::String(_)
                | HirExprKind::Constant(_)
                | HirExprKind::Colon => {
                    // No variables here
                }
            }
        }

        fn visit_stmts(stmts: &[HirStmt], max: &mut usize) {
            for s in stmts {
                match s {
                    HirStmt::Assign(id, expr, _) => {
                        if id.0 + 1 > *max {
                            *max = id.0 + 1;
                        }
                        visit_expr(expr, max);
                    }
                    HirStmt::ExprStmt(expr, _) => {
                        visit_expr(expr, max);
                    }
                    HirStmt::If {
                        cond,
                        then_body,
                        elseif_blocks,
                        else_body,
                    } => {
                        visit_expr(cond, max);
                        visit_stmts(then_body, max);
                        for (cond, body) in elseif_blocks {
                            visit_expr(cond, max);
                            visit_stmts(body, max);
                        }
                        if let Some(body) = else_body {
                            visit_stmts(body, max);
                        }
                    }
                    HirStmt::While { cond, body } => {
                        visit_expr(cond, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::For { var, expr, body } => {
                        if var.0 + 1 > *max {
                            *max = var.0 + 1;
                        }
                        visit_expr(expr, max);
                        visit_stmts(body, max);
                    }
                    HirStmt::Function { .. } => {}
                    HirStmt::Break | HirStmt::Continue | HirStmt::Return => {}
                }
            }
        }
        visit_stmts(&prog.body, &mut max_var);
        Self {
            instructions: Vec::new(),
            var_count: max_var,
            loop_stack: Vec::new(),
            functions: HashMap::new(),
        }
    }

    fn emit(&mut self, instr: Instr) -> usize {
        let pc = self.instructions.len();
        self.instructions.push(instr);
        pc
    }

    fn compile_program(&mut self, prog: &HirProgram) -> Result<(), String> {
        for stmt in &prog.body {
            self.compile_stmt(stmt)?;
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &HirStmt) -> Result<(), String> {
        match stmt {
            HirStmt::ExprStmt(expr, _) => {
                self.compile_expr(expr)?;
                self.emit(Instr::Pop);
            }
            HirStmt::Assign(id, expr, _) => {
                self.compile_expr(expr)?;
                self.emit(Instr::StoreVar(id.0));
            }
            HirStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                // compile initial condition
                self.compile_expr(cond)?;
                let mut last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                // then-body
                for s in then_body {
                    self.compile_stmt(s)?;
                }
                let mut end_jumps = Vec::new();
                end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                // process elseif blocks
                for (c, b) in elseif_blocks {
                    self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                    self.compile_expr(c)?;
                    last_jump = self.emit(Instr::JumpIfFalse(usize::MAX));
                    for s in b {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.emit(Instr::Jump(usize::MAX)));
                }
                // else block
                self.patch(last_jump, Instr::JumpIfFalse(self.instructions.len()));
                if let Some(body) = else_body {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }
                let end = self.instructions.len();
                for j in end_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::While { cond, body } => {
                let start = self.instructions.len();
                self.compile_expr(cond)?;
                let jump_end = self.emit(Instr::JumpIfFalse(usize::MAX));
                let labels = LoopLabels {
                    break_jumps: Vec::new(),
                    continue_jumps: Vec::new(),
                };
                self.loop_stack.push(labels);
                for s in body {
                    self.compile_stmt(s)?;
                }
                let labels = self.loop_stack.pop().unwrap();
                for j in labels.continue_jumps {
                    self.patch(j, Instr::Jump(start));
                }
                self.emit(Instr::Jump(start));
                let end = self.instructions.len();
                self.patch(jump_end, Instr::JumpIfFalse(end));
                for j in labels.break_jumps {
                    self.patch(j, Instr::Jump(end));
                }
            }
            HirStmt::For { var, expr, body } => {
                if let HirExprKind::Range(start, step, end) = &expr.kind {
                    if step.is_some() {
                        return Err("step in range not supported".into());
                    }
                    self.compile_expr(start)?;
                    self.emit(Instr::StoreVar(var.0));
                    self.compile_expr(end)?;
                    let end_var = self.var_count;
                    self.var_count += 1;
                    self.emit(Instr::StoreVar(end_var));
                    let loop_start = self.instructions.len();
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadVar(end_var));
                    self.emit(Instr::LessEqual);
                    let jump_end = self.emit(Instr::JumpIfFalse(usize::MAX));
                    self.loop_stack.push(LoopLabels {
                        break_jumps: Vec::new(),
                        continue_jumps: Vec::new(),
                    });
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    let labels = self.loop_stack.pop().unwrap();
                    for j in labels.continue_jumps {
                        self.patch(j, Instr::Jump(self.instructions.len()));
                    }
                    self.emit(Instr::LoadVar(var.0));
                    self.emit(Instr::LoadConst(1.0));
                    self.emit(Instr::Add);
                    self.emit(Instr::StoreVar(var.0));
                    self.emit(Instr::Jump(loop_start));
                    let end = self.instructions.len();
                    self.patch(jump_end, Instr::JumpIfFalse(end));
                    for j in labels.break_jumps {
                        self.patch(j, Instr::Jump(end));
                    }
                } else {
                    return Err("for loop expects range".into());
                }
            }
            HirStmt::Break => {
                if let Some(labels) = self.loop_stack.last_mut() {
                    let idx = self.instructions.len();
                    self.instructions.push(Instr::Jump(usize::MAX));
                    labels.break_jumps.push(idx);
                } else {
                    return Err("break outside loop".into());
                }
            }
            HirStmt::Continue => {
                if let Some(labels) = self.loop_stack.last_mut() {
                    let idx = self.instructions.len();
                    self.instructions.push(Instr::Jump(usize::MAX));
                    labels.continue_jumps.push(idx);
                } else {
                    return Err("continue outside loop".into());
                }
            }
            HirStmt::Return => {
                self.emit(Instr::Return);
            }
            HirStmt::Function {
                name,
                params,
                outputs,
                body,
            } => {
                // Calculate the maximum variable ID used in this function's body
                // to determine how many local variables are needed
                let mut max_local_var = 0;
                fn visit_expr_for_vars(expr: &HirExpr, max: &mut usize) {
                    match &expr.kind {
                        HirExprKind::Var(id) => {
                            if id.0 + 1 > *max {
                                *max = id.0 + 1;
                            }
                        }
                        HirExprKind::Unary(_, e) => visit_expr_for_vars(e, max),
                        HirExprKind::Binary(a, _, b) => {
                            visit_expr_for_vars(a, max);
                            visit_expr_for_vars(b, max);
                        }
                        HirExprKind::Matrix(rows) => {
                            for row in rows {
                                for elem in row {
                                    visit_expr_for_vars(elem, max);
                                }
                            }
                        }
                        HirExprKind::Index(base, indices) => {
                            visit_expr_for_vars(base, max);
                            for idx in indices {
                                visit_expr_for_vars(idx, max);
                            }
                        }
                        HirExprKind::Range(start, step, end) => {
                            visit_expr_for_vars(start, max);
                            if let Some(step) = step {
                                visit_expr_for_vars(step, max);
                            }
                            visit_expr_for_vars(end, max);
                        }
                        HirExprKind::FuncCall(_, args) => {
                            for arg in args {
                                visit_expr_for_vars(arg, max);
                            }
                        }
                        _ => {}
                    }
                }

                fn visit_stmt_for_vars(stmt: &HirStmt, max: &mut usize) {
                    match stmt {
                        HirStmt::ExprStmt(expr, _) => visit_expr_for_vars(expr, max),
                        HirStmt::Assign(id, expr, _) => {
                            if id.0 + 1 > *max {
                                *max = id.0 + 1;
                            }
                            visit_expr_for_vars(expr, max);
                        }
                        HirStmt::If {
                            cond,
                            then_body,
                            elseif_blocks,
                            else_body,
                        } => {
                            visit_expr_for_vars(cond, max);
                            for stmt in then_body {
                                visit_stmt_for_vars(stmt, max);
                            }
                            for (cond, body) in elseif_blocks {
                                visit_expr_for_vars(cond, max);
                                for stmt in body {
                                    visit_stmt_for_vars(stmt, max);
                                }
                            }
                            if let Some(body) = else_body {
                                for stmt in body {
                                    visit_stmt_for_vars(stmt, max);
                                }
                            }
                        }
                        HirStmt::While { cond, body } => {
                            visit_expr_for_vars(cond, max);
                            for stmt in body {
                                visit_stmt_for_vars(stmt, max);
                            }
                        }
                        HirStmt::For { var, expr, body } => {
                            if var.0 + 1 > *max {
                                *max = var.0 + 1;
                            }
                            visit_expr_for_vars(expr, max);
                            for stmt in body {
                                visit_stmt_for_vars(stmt, max);
                            }
                        }
                        HirStmt::Function { .. } => {
                            // Nested functions - we don't count their variables
                        }
                        HirStmt::Break | HirStmt::Continue | HirStmt::Return => {}
                    }
                }

                for stmt in body {
                    visit_stmt_for_vars(stmt, &mut max_local_var);
                }

                // Store the function definition for later execution
                let user_func = UserFunction {
                    name: name.clone(),
                    params: params.clone(),
                    outputs: outputs.clone(),
                    body: body.clone(),
                    local_var_count: max_local_var,
                };
                self.functions.insert(name.clone(), user_func);
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &HirExpr) -> Result<(), String> {
        match &expr.kind {
            HirExprKind::Number(n) => {
                let val: f64 = n.parse().map_err(|_| "invalid number")?;
                self.emit(Instr::LoadConst(val));
            }
            HirExprKind::String(s) => {
                // Strip quotes from string literal for storage
                let clean_string = if s.starts_with('\'') && s.ends_with('\'') {
                    s[1..s.len() - 1].to_string()
                } else {
                    s.clone()
                };
                self.emit(Instr::LoadString(clean_string));
            }
            HirExprKind::Var(id) => {
                self.emit(Instr::LoadVar(id.0));
            }
            HirExprKind::Constant(name) => {
                // Look up the constant value and load it
                let constants = runmat_builtins::constants();
                if let Some(constant) = constants.iter().find(|c| c.name == name) {
                    if let runmat_builtins::Value::Num(val) = &constant.value {
                        self.emit(Instr::LoadConst(*val));
                    } else {
                        return Err(format!("Constant {name} is not a number"));
                    }
                } else {
                    return Err(format!("Unknown constant: {name}"));
                }
            }
            HirExprKind::Unary(op, e) => {
                self.compile_expr(e)?;
                match op {
                    runmat_parser::UnOp::Plus => {}
                    runmat_parser::UnOp::Minus => {
                        self.emit(Instr::Neg);
                    }
                    runmat_parser::UnOp::Transpose => {
                        self.emit(Instr::Transpose);
                    }
                }
            }
            HirExprKind::Binary(a, op, b) => {
                self.compile_expr(a)?;
                self.compile_expr(b)?;
                match op {
                    runmat_parser::BinOp::Add => self.emit(Instr::Add),
                    runmat_parser::BinOp::Sub => self.emit(Instr::Sub),
                    runmat_parser::BinOp::Mul => self.emit(Instr::Mul),
                    runmat_parser::BinOp::Div | runmat_parser::BinOp::LeftDiv => {
                        self.emit(Instr::Div)
                    }
                    runmat_parser::BinOp::Pow => self.emit(Instr::Pow),
                    runmat_parser::BinOp::ElemMul => self.emit(Instr::ElemMul),
                    runmat_parser::BinOp::ElemDiv | runmat_parser::BinOp::ElemLeftDiv => {
                        self.emit(Instr::ElemDiv)
                    }
                    runmat_parser::BinOp::ElemPow => self.emit(Instr::ElemPow),
                    // Comparison operations
                    runmat_parser::BinOp::Equal => self.emit(Instr::Equal),
                    runmat_parser::BinOp::NotEqual => self.emit(Instr::NotEqual),
                    runmat_parser::BinOp::Less => self.emit(Instr::Less),
                    runmat_parser::BinOp::LessEqual => self.emit(Instr::LessEqual),
                    runmat_parser::BinOp::Greater => self.emit(Instr::Greater),
                    runmat_parser::BinOp::GreaterEqual => self.emit(Instr::GreaterEqual),
                    runmat_parser::BinOp::Colon => {
                        return Err("colon operator not supported".into())
                    }
                };
            }
            HirExprKind::Range(start, step, end) => {
                // Compile range components
                self.compile_expr(start)?;
                if let Some(step) = step {
                    self.compile_expr(step)?;
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(true)); // Has step
                } else {
                    self.compile_expr(end)?;
                    self.emit(Instr::CreateRange(false)); // No step
                }
            }
            HirExprKind::FuncCall(name, args) => {
                // Compile arguments in order
                for arg in args {
                    self.compile_expr(arg)?;
                }

                // Check if this is a user-defined function first
                if self.functions.contains_key(name) {
                    self.emit(Instr::CallFunction(name.clone(), args.len()));
                } else {
                    self.emit(Instr::CallBuiltin(name.clone(), args.len()));
                }
            }
            HirExprKind::Matrix(matrix_data) => {
                let rows = matrix_data.len();

                // Check if any element is non-literal (variable, function call, etc.)
                let has_non_literals = matrix_data.iter().any(|row| {
                    row.iter().any(|expr| {
                        !matches!(
                            expr.kind,
                            HirExprKind::Number(_)
                                | HirExprKind::String(_)
                                | HirExprKind::Constant(_)
                        )
                    })
                });

                if has_non_literals {
                    // Use dynamic matrix creation for concatenation
                    // Compile each row as a separate unit
                    for row in matrix_data {
                        // Compile all elements in the row
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }
                    // Emit the dynamic matrix creation with row structure
                    let row_lengths: Vec<usize> = matrix_data.iter().map(|row| row.len()).collect();
                    for &row_len in &row_lengths {
                        self.emit(Instr::LoadConst(row_len as f64));
                    }
                    self.emit(Instr::CreateMatrixDynamic(rows));
                } else {
                    // Use traditional matrix creation for literal values
                    let cols = if rows > 0 { matrix_data[0].len() } else { 0 };

                    // Compile all matrix elements onto the stack in row-major order
                    for row in matrix_data {
                        for element in row {
                            self.compile_expr(element)?;
                        }
                    }

                    self.emit(Instr::CreateMatrix(rows, cols));
                }
            }
            HirExprKind::Index(base, indices) => {
                // Compile the base expression (the array/matrix)
                self.compile_expr(base)?;

                // Compile all index expressions
                for index in indices {
                    self.compile_expr(index)?;
                }

                // Emit index instruction with number of indices
                self.emit(Instr::Index(indices.len()));
            }
            HirExprKind::Colon => {
                return Err("colon expression not supported".into());
            }
        }
        Ok(())
    }

    fn patch(&mut self, idx: usize, instr: Instr) {
        self.instructions[idx] = instr;
    }
}

pub fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
) -> Result<Vec<Value>, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut vars = initial_vars.to_vec();

    // Ensure vars is large enough for the bytecode requirements
    if vars.len() < bytecode.var_count {
        vars.resize(bytecode.var_count, Value::Num(0.0));
    }
    let mut pc: usize = 0;

    // Initialize execution context for proper function call handling
    let mut context = ExecutionContext {
        call_stack: Vec::new(),
        locals: Vec::new(),
        instruction_pointer: 0,
    };

    // Register GC roots for stack and variables (RAII cleanup)
    let _gc_context = InterpretContext::new(&stack, &vars)?;
    while pc < bytecode.instructions.len() {
        match bytecode.instructions[pc].clone() {
            Instr::LoadConst(c) => stack.push(Value::Num(c)),
            Instr::LoadString(s) => stack.push(Value::String(s)),
            Instr::LoadVar(i) => stack.push(vars[i].clone()),
            Instr::StoreVar(i) => {
                let val = stack.pop().ok_or("stack underflow")?;

                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                }
                vars[i] = val;
            }
            Instr::LoadLocal(offset) => {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;
                    if local_index >= context.locals.len() {
                        return Err("Local variable index out of bounds".to_string());
                    }
                    stack.push(context.locals[local_index].clone());
                } else {
                    // LoadLocal outside function call - treat as loading from global vars
                    // This provides graceful fallback behavior for malformed bytecode
                    if offset < vars.len() {
                        stack.push(vars[offset].clone());
                    } else {
                        stack.push(Value::Num(0.0)); // Default value
                    }
                }
            }
            Instr::StoreLocal(offset) => {
                let val = stack.pop().ok_or("stack underflow")?;
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + offset;

                    // Ensure locals array is large enough
                    while context.locals.len() <= local_index {
                        context.locals.push(Value::Num(0.0));
                    }
                    context.locals[local_index] = val;
                } else {
                    // StoreLocal outside function call - treat as storing to global vars
                    // This provides graceful fallback behavior for malformed bytecode
                    if offset >= vars.len() {
                        vars.resize(offset + 1, Value::Num(0.0));
                    }
                    vars[offset] = val;
                }
            }
            Instr::EnterScope(local_count) => {
                // Allocate space for local variables
                for _ in 0..local_count {
                    context.locals.push(Value::Num(0.0));
                }
            }
            Instr::ExitScope(local_count) => {
                // Deallocate local variables
                for _ in 0..local_count {
                    context.locals.pop();
                }
            }
            Instr::Add => element_binary(&mut stack, runmat_runtime::elementwise_add)?,
            Instr::Sub => element_binary(&mut stack, runmat_runtime::elementwise_sub)?,
            Instr::Mul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::Div => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::Pow => element_binary(&mut stack, runmat_runtime::power)?,
            Instr::Neg => {
                let value = stack.pop().ok_or("stack underflow")?;
                let result = runmat_runtime::elementwise_neg(&value)?;
                stack.push(result);
            }
            Instr::Transpose => {
                let value = stack.pop().ok_or("stack underflow")?;
                let result = runmat_runtime::transpose(value)?;
                stack.push(result);
            }
            Instr::ElemMul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::ElemDiv => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::ElemPow => element_binary(&mut stack, runmat_runtime::elementwise_pow)?,
            Instr::ElemLeftDiv => {
                element_binary(&mut stack, |a, b| runmat_runtime::elementwise_div(b, a))?
            }
            Instr::LessEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 }));
            }
            Instr::Less => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a < b { 1.0 } else { 0.0 }));
            }
            Instr::Greater => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a > b { 1.0 } else { 0.0 }));
            }
            Instr::GreaterEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 }));
            }
            Instr::Equal => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a == b { 1.0 } else { 0.0 }));
            }
            Instr::NotEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a != b { 1.0 } else { 0.0 }));
            }
            Instr::JumpIfFalse(target) => {
                let cond: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                if cond == 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::Jump(target) => {
                pc = target;
                continue;
            }
            Instr::CallBuiltin(name, arg_count) => {
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(stack.pop().ok_or("stack underflow")?);
                }
                args.reverse(); // Arguments were pushed in reverse order
                let result = call_builtin(&name, &args)?;
                stack.push(result);
            }
            Instr::CallFunction(name, arg_count) => {
                // Get the function definition
                let func = bytecode
                    .functions
                    .get(&name)
                    .ok_or_else(|| format!("undefined function: {name}"))?
                    .clone();

                // Validate argument count - MATLAB requires exact match
                if arg_count != func.params.len() {
                    return Err(format!(
                        "Function '{}' expects {} arguments, got {} - Not enough input arguments",
                        name,
                        func.params.len(),
                        arg_count
                    ));
                }

                // Pop arguments from stack
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(stack.pop().ok_or("stack underflow")?);
                }
                args.reverse(); // Arguments were pushed in reverse order

                // Create complete variable remapping that includes all variables referenced in the function body
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();

                // Remap the function body to use local variable indices
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);

                // Execute function with proper parameter binding and isolated scope
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];

                // Bind parameters to function's local variables
                for (i, _param_id) in func.params.iter().enumerate() {
                    if i < args.len() && i < func_vars.len() {
                        func_vars[i] = args[i].clone();
                    }
                }

                // Populate function's local variables with global variable values for variables it references
                // This allows functions to access global workspace variables (MATLAB behavior)
                for (original_var_id, local_var_id) in &var_map {
                    let local_index = local_var_id.0;
                    let global_index = original_var_id.0;

                    // Only populate if it's not already set by parameter binding and if global variable exists
                    if local_index < func_vars.len() && global_index < vars.len() {
                        // Don't overwrite parameter values, but populate other referenced variables
                        let is_parameter = func
                            .params
                            .iter()
                            .any(|param_id| param_id == original_var_id);
                        if !is_parameter {
                            func_vars[local_index] = vars[global_index].clone();
                        }
                    }
                }

                // Execute the function in its own isolated execution context
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                };
                let func_bytecode = compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = interpret_function(&func_bytecode, func_vars)?;

                // Return the output variable value (first output variable)
                if let Some(output_var_id) = func.outputs.first() {
                    // Use the remapped local index instead of the original VarId
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);

                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0)); // Default if output variable not found
                    }
                } else {
                    stack.push(Value::Num(0.0)); // Default if no output variable
                }
            }
            Instr::CreateMatrix(rows, cols) => {
                let total_elements = rows * cols;
                let mut data = Vec::with_capacity(total_elements);

                // Pop elements from stack in reverse order (since stack is LIFO)
                for _ in 0..total_elements {
                    let val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    data.push(val);
                }

                // Reverse to get row-major order
                data.reverse();

                let matrix = runmat_builtins::Matrix::new(data, rows, cols)
                    .map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Matrix(matrix));
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                // Dynamic matrix creation with concatenation support
                let mut row_lengths = Vec::new();

                // Pop row lengths (in reverse order since they're pushed last)
                for _ in 0..num_rows {
                    let row_len: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    row_lengths.push(row_len as usize);
                }
                row_lengths.reverse(); // Correct the order

                // Now pop elements according to row structure (in reverse order)
                let mut rows_data = Vec::new();
                for &row_len in row_lengths.iter().rev() {
                    let mut row_values = Vec::new();
                    for _ in 0..row_len {
                        row_values.push(stack.pop().ok_or("stack underflow")?);
                    }
                    row_values.reverse(); // Correct the order within row
                    rows_data.push(row_values);
                }

                // Reverse rows to get correct order
                rows_data.reverse();

                // Use the concatenation logic to create the matrix
                let result = runmat_runtime::create_matrix_from_values(&rows_data)?;
                stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    // Stack: start, step, end
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let step: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;

                    let range_result = runmat_runtime::create_range(start, Some(step), end)?;
                    stack.push(range_result);
                } else {
                    // Stack: start, end
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;

                    let range_result = runmat_runtime::create_range(start, None, end)?;
                    stack.push(range_result);
                }
            }
            Instr::Index(num_indices) => {
                // Pop indices from stack (in reverse order)
                let mut indices = Vec::new();
                let count = num_indices;
                for _ in 0..count {
                    let index_val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    indices.push(index_val);
                }
                indices.reverse(); // Correct the order

                // Pop the base array/matrix
                let base = stack.pop().ok_or("stack underflow")?;

                // Perform the indexing operation using centralized function
                let result = runmat_runtime::perform_indexing(&base, &indices)?;
                stack.push(result);
            }
            Instr::Pop => {
                stack.pop();
            }
            Instr::Return => {
                if context.call_stack.is_empty() {
                    // Return from main program
                    break;
                } else {
                    // Return from function - restore previous call frame
                    let frame = context.call_stack.pop().unwrap();

                    // Clean up local variables
                    for _ in 0..frame.locals_count {
                        context.locals.pop();
                    }

                    // Return to caller
                    pc = frame.return_address;
                    continue;
                }
            }
            Instr::ReturnValue => {
                let return_value = stack.pop().ok_or("stack underflow")?;

                if context.call_stack.is_empty() {
                    // Return from main program with value
                    stack.push(return_value);
                    break;
                } else {
                    // Return from function with value
                    let frame = context.call_stack.pop().unwrap();

                    // Clean up local variables
                    for _ in 0..frame.locals_count {
                        context.locals.pop();
                    }

                    // Push return value for caller
                    stack.push(return_value);

                    // Return to caller
                    pc = frame.return_address;
                    continue;
                }
            }
        }
        pc += 1;
    }

    // Copy updated variables back to the initial_vars array
    for (i, var) in vars.iter().enumerate() {
        if i < initial_vars.len() {
            initial_vars[i] = var.clone();
        }
    }

    Ok(vars)
}

/// Interpret bytecode for a function call with proper variable scoping
fn interpret_function(bytecode: &Bytecode, mut vars: Vec<Value>) -> Result<Vec<Value>, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut pc: usize = 0;

    // Register GC roots for stack and variables (RAII cleanup)
    let _gc_context = InterpretContext::new(&stack, &vars)?;

    while pc < bytecode.instructions.len() {
        match bytecode.instructions[pc].clone() {
            Instr::LoadConst(c) => stack.push(Value::Num(c)),
            Instr::LoadString(s) => stack.push(Value::String(s)),
            Instr::LoadVar(i) => stack.push(vars[i].clone()),
            Instr::StoreVar(i) => {
                let val = stack.pop().ok_or("stack underflow")?;
                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                }
                vars[i] = val;
            }
            // Function-local variables use the same LoadVar/StoreVar for simplicity
            Instr::LoadLocal(_) => {
                return Err("LoadLocal not supported in function context".to_string())
            }
            Instr::StoreLocal(_) => {
                return Err("StoreLocal not supported in function context".to_string())
            }
            Instr::EnterScope(_) => {} // No-op in function context
            Instr::ExitScope(_) => {}  // No-op in function context
            Instr::Add => element_binary(&mut stack, runmat_runtime::elementwise_add)?,
            Instr::Sub => element_binary(&mut stack, runmat_runtime::elementwise_sub)?,
            Instr::Mul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::Div => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::Pow => element_binary(&mut stack, runmat_runtime::power)?,
            Instr::Neg => {
                let value = stack.pop().ok_or("stack underflow")?;
                let result = runmat_runtime::elementwise_neg(&value)?;
                stack.push(result);
            }
            Instr::Transpose => {
                let value = stack.pop().ok_or("stack underflow")?;
                let result = runmat_runtime::transpose(value)?;
                stack.push(result);
            }
            Instr::ElemMul => element_binary(&mut stack, runmat_runtime::elementwise_mul)?,
            Instr::ElemDiv => element_binary(&mut stack, runmat_runtime::elementwise_div)?,
            Instr::ElemPow => element_binary(&mut stack, runmat_runtime::elementwise_pow)?,
            Instr::ElemLeftDiv => {
                element_binary(&mut stack, |a, b| runmat_runtime::elementwise_div(b, a))?
            }
            Instr::LessEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 }));
            }
            Instr::Less => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a < b { 1.0 } else { 0.0 }));
            }
            Instr::Greater => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a > b { 1.0 } else { 0.0 }));
            }
            Instr::GreaterEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 }));
            }
            Instr::Equal => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a == b { 1.0 } else { 0.0 }));
            }
            Instr::NotEqual => {
                let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                stack.push(Value::Num(if a != b { 1.0 } else { 0.0 }));
            }
            Instr::JumpIfFalse(target) => {
                let cond: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                if cond == 0.0 {
                    pc = target;
                    continue;
                }
            }
            Instr::Jump(target) => {
                pc = target;
                continue;
            }
            Instr::CallBuiltin(name, arg_count) => {
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(stack.pop().ok_or("stack underflow")?);
                }
                args.reverse(); // Arguments were pushed in reverse order
                let result = call_builtin(&name, &args)?;
                stack.push(result);
            }
            Instr::CallFunction(name, arg_count) => {
                // Recursive function calls within functions
                // Get the function definition from the global function registry
                let func = bytecode
                    .functions
                    .get(&name)
                    .ok_or_else(|| format!("undefined function: {name}"))?
                    .clone();

                // Validate argument count - MATLAB requires exact match
                if arg_count != func.params.len() {
                    return Err(format!(
                        "Function '{}' expects {} arguments, got {} - Not enough input arguments",
                        name,
                        func.params.len(),
                        arg_count
                    ));
                }

                // Pop arguments from stack
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(stack.pop().ok_or("stack underflow")?);
                }
                args.reverse(); // Arguments were pushed in reverse order

                // Create complete variable remapping that includes all variables referenced in the function body
                let var_map = runmat_hir::remapping::create_complete_function_var_map(
                    &func.params,
                    &func.outputs,
                    &func.body,
                );
                let local_var_count = var_map.len();

                // Remap the function body to use local variable indices
                let remapped_body =
                    runmat_hir::remapping::remap_function_body(&func.body, &var_map);

                // Create function variable space and bind parameters
                let func_vars_count = local_var_count.max(func.params.len());
                let mut func_vars = vec![Value::Num(0.0); func_vars_count];
                for (i, _param_id) in func.params.iter().enumerate() {
                    if i < args.len() {
                        func_vars[i] = args[i].clone();
                    }
                }

                // Recursively call the function
                let func_program = runmat_hir::HirProgram {
                    body: remapped_body,
                };
                let func_bytecode = compile_with_functions(&func_program, &bytecode.functions)?;
                let func_result_vars = interpret_function(&func_bytecode, func_vars)?;

                // Return the output variable value (first output variable)
                if let Some(output_var_id) = func.outputs.first() {
                    // Use the remapped local index instead of the original VarId
                    let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);

                    if local_output_index < func_result_vars.len() {
                        stack.push(func_result_vars[local_output_index].clone());
                    } else {
                        stack.push(Value::Num(0.0)); // Default if output variable not found
                    }
                } else {
                    stack.push(Value::Num(0.0)); // Default if no output variable
                }
            }
            Instr::CreateMatrix(rows, cols) => {
                let total_elements = rows * cols;
                let mut data = Vec::with_capacity(total_elements);

                // Pop elements from stack in reverse order (since stack is LIFO)
                for _ in 0..total_elements {
                    let val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    data.push(val);
                }

                // Reverse to get row-major order
                data.reverse();

                let matrix = runmat_builtins::Matrix::new(data, rows, cols)
                    .map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Matrix(matrix));
            }
            Instr::CreateMatrixDynamic(num_rows) => {
                // Dynamic matrix creation with concatenation support
                let mut row_lengths = Vec::new();

                // Pop row lengths (in reverse order since they're pushed last)
                for _ in 0..num_rows {
                    let row_len: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    row_lengths.push(row_len as usize);
                }
                row_lengths.reverse(); // Correct the order

                // Now pop elements according to row structure (in reverse order)
                let mut rows_data = Vec::new();
                for &row_len in row_lengths.iter().rev() {
                    let mut row_values = Vec::new();
                    for _ in 0..row_len {
                        row_values.push(stack.pop().ok_or("stack underflow")?);
                    }
                    row_values.reverse(); // Correct the order within row
                    rows_data.push(row_values);
                }

                // Reverse rows to get correct order
                rows_data.reverse();

                // Use the concatenation logic to create the matrix
                let result = runmat_runtime::create_matrix_from_values(&rows_data)?;
                stack.push(result);
            }
            Instr::CreateRange(has_step) => {
                if has_step {
                    // Stack: start, step, end
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let step: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;

                    let range_result = runmat_runtime::create_range(start, Some(step), end)?;
                    stack.push(range_result);
                } else {
                    // Stack: start, end
                    let end: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    let start: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;

                    let range_result = runmat_runtime::create_range(start, None, end)?;
                    stack.push(range_result);
                }
            }
            Instr::Index(num_indices) => {
                // Pop indices from stack (in reverse order)
                let mut indices = Vec::new();
                let count = num_indices;
                for _ in 0..count {
                    let index_val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
                    indices.push(index_val);
                }
                indices.reverse(); // Correct the order

                // Pop the base array/matrix
                let base = stack.pop().ok_or("stack underflow")?;

                // Perform the indexing operation using centralized function
                let result = runmat_runtime::perform_indexing(&base, &indices)?;
                stack.push(result);
            }
            Instr::Pop => {
                stack.pop();
            }
            Instr::Return | Instr::ReturnValue => {
                // Function return - exit the function execution
                break;
            }
        }
        pc += 1;
    }

    Ok(vars)
}

fn element_binary<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
where
    F: Fn(&Value, &Value) -> Result<Value, String>,
{
    let b = stack.pop().ok_or("stack underflow")?;
    let a = stack.pop().ok_or("stack underflow")?;
    let result = f(&a, &b)?;
    stack.push(result);
    Ok(())
}

/// Interpret bytecode with default variable initialization
pub fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, String> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    interpret_with_vars(bytecode, &mut vars)
}

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, String> {
    let bc = compile(program)?;
    interpret(&bc)
}

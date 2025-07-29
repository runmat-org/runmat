use rustmat_builtins::Value;
use rustmat_runtime::call_builtin;
use rustmat_hir::{HirProgram, HirStmt, HirExpr, HirExprKind};
use std::convert::TryInto;

#[derive(Debug, Clone)]
pub enum Instr {
    LoadConst(f64),
    LoadVar(usize),
    StoreVar(usize),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
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
    Return,
}

#[derive(Debug)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    pub var_count: usize,
}

pub fn compile(prog: &HirProgram) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.compile_program(prog)?;
    Ok(Bytecode {
        instructions: c.instructions,
        var_count: c.var_count,
    })
}

struct Compiler {
    instructions: Vec<Instr>,
    var_count: usize,
    loop_stack: Vec<LoopLabels>,
}

struct LoopLabels {
    break_jumps: Vec<usize>,
    continue_jumps: Vec<usize>,
}

impl Compiler {
    fn new(prog: &HirProgram) -> Self {
        let mut max_var = 0;
        fn visit_stmts(stmts: &[HirStmt], max: &mut usize) {
            for s in stmts {
                match s {
                    HirStmt::Assign(id, _) => {
                        if id.0 + 1 > *max {
                            *max = id.0 + 1;
                        }
                    }
                    HirStmt::If {
                        then_body,
                        elseif_blocks,
                        else_body,
                        ..
                    } => {
                        visit_stmts(then_body, max);
                        for (_, b) in elseif_blocks {
                            visit_stmts(b, max);
                        }
                        if let Some(b) = else_body {
                            visit_stmts(b, max);
                        }
                    }
                    HirStmt::While { body, .. } => visit_stmts(body, max),
                    HirStmt::For { var, body, .. } => {
                        if var.0 + 1 > *max {
                            *max = var.0 + 1;
                        }
                        visit_stmts(body, max);
                    }
                    HirStmt::Function { .. } => {}
                    HirStmt::ExprStmt(_) | HirStmt::Break | HirStmt::Continue | HirStmt::Return => {
                    }
                }
            }
        }
        visit_stmts(&prog.body, &mut max_var);
        Self {
            instructions: Vec::new(),
            var_count: max_var,
            loop_stack: Vec::new(),
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
            HirStmt::ExprStmt(expr) => {
                self.compile_expr(expr)?;
                self.emit(Instr::Pop);
            }
            HirStmt::Assign(id, expr) => {
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
            HirStmt::Function { .. } => {
                return Err("function definitions not supported".into());
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
            HirExprKind::Var(id) => {
                self.emit(Instr::LoadVar(id.0));
            }
            HirExprKind::Unary(op, e) => {
                self.compile_expr(e)?;
                match op {
                    rustmat_parser::UnOp::Plus => {}
                    rustmat_parser::UnOp::Minus => {
                        self.emit(Instr::Neg);
                    }
                }
            }
            HirExprKind::Binary(a, op, b) => {
                self.compile_expr(a)?;
                self.compile_expr(b)?;
                match op {
                    rustmat_parser::BinOp::Add => self.emit(Instr::Add),
                    rustmat_parser::BinOp::Sub => self.emit(Instr::Sub),
                    rustmat_parser::BinOp::Mul => self.emit(Instr::Mul),
                    rustmat_parser::BinOp::Div | rustmat_parser::BinOp::LeftDiv => {
                        self.emit(Instr::Div)
                    }
                    rustmat_parser::BinOp::Pow => self.emit(Instr::Pow),
                    rustmat_parser::BinOp::Colon => {
                        return Err("colon operator not supported".into())
                    }
                };
            }
            HirExprKind::Range(..) => {
                return Err("standalone range not supported".into());
            }
            HirExprKind::FuncCall(name, args) => {
                // Compile arguments in order
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emit(Instr::CallBuiltin(name.clone(), args.len()));
            }
            HirExprKind::Matrix(matrix_data) => {
                let rows = matrix_data.len();
                let cols = if rows > 0 { matrix_data[0].len() } else { 0 };
                
                // Compile all matrix elements onto the stack in row-major order
                for row in matrix_data {
                    for element in row {
                        self.compile_expr(element)?;
                    }
                }
                
                self.emit(Instr::CreateMatrix(rows, cols));
            }
            HirExprKind::Index(..) | HirExprKind::Colon => {
                return Err("expression not supported".into());
            }
        }
        Ok(())
    }

    fn patch(&mut self, idx: usize, instr: Instr) {
        self.instructions[idx] = instr;
    }
}

pub fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    let mut pc: usize = 0;
    while pc < bytecode.instructions.len() {
        match bytecode.instructions[pc].clone() {
            Instr::LoadConst(c) => stack.push(Value::Num(c)),
            Instr::LoadVar(i) => stack.push(vars[i].clone()),
            Instr::StoreVar(i) => {
                let val = stack.pop().ok_or("stack underflow")?;
                if i >= vars.len() {
                    vars.resize(i + 1, Value::Num(0.0));
                }
                vars[i] = val;
            }
            Instr::Add => binary(&mut stack, |a, b| a + b)?,
            Instr::Sub => binary(&mut stack, |a, b| a - b)?,
            Instr::Mul => binary(&mut stack, |a, b| a * b)?,
            Instr::Div => binary(&mut stack, |a, b| a / b)?,
            Instr::Pow => binary(&mut stack, |a, b| a.powf(b))?,
            Instr::Neg => unary(&mut stack, |a| -a)?,
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
                
                let matrix = rustmat_builtins::Matrix::new(data, rows, cols)
                    .map_err(|e| format!("Matrix creation error: {e}"))?;
                stack.push(Value::Matrix(matrix));
            }
            Instr::Pop => {
                stack.pop();
            }
            Instr::Return => {
                break; // Halt execution immediately
            }

        }
        pc += 1;
    }
    Ok(vars)
}

fn binary<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
where
    F: Fn(f64, f64) -> f64,
{
    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
    stack.push(Value::Num(f(a, b)));
    Ok(())
}

fn unary<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
where
    F: Fn(f64) -> f64,
{
    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()?;
    stack.push(Value::Num(f(a)));
    Ok(())
}

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, String> {
    let bc = compile(program)?;
    interpret(&bc)
}

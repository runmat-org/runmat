use crate::instr::Instr;
use runmat_hir::{HirStmt, VarId};
use runmat_builtins::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFunction {
    pub name: String,
    pub params: Vec<VarId>,
    pub outputs: Vec<VarId>,
    pub body: Vec<HirStmt>,
    pub local_var_count: usize,
}

/// Represents a call frame in the call stack
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub return_address: usize,
    pub locals_start: usize,
    pub locals_count: usize,
    pub expected_outputs: usize,
}

/// Runtime execution context with call stack
#[derive(Debug)]
pub struct ExecutionContext {
    pub call_stack: Vec<CallFrame>,
    pub locals: Vec<Value>,
    pub instruction_pointer: usize,
    pub functions: std::collections::HashMap<String, UserFunction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    pub var_count: usize,
    pub functions: HashMap<String, UserFunction>,
}

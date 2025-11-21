use crate::instr::Instr;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionGroup;
use runmat_builtins::{Type, Value};
use runmat_hir::{HirStmt, VarId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFunction {
    pub name: String,
    pub params: Vec<VarId>,
    pub outputs: Vec<VarId>,
    pub body: Vec<HirStmt>,
    pub local_var_count: usize,
    pub has_varargin: bool,
    pub has_varargout: bool,
    #[serde(default)]
    pub var_types: Vec<Type>,
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
    #[serde(default)]
    pub var_types: Vec<Type>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub accel_graph: Option<AccelGraph>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_groups: Vec<FusionGroup>,
}

impl Bytecode {
    pub fn empty() -> Self {
        Self {
            instructions: Vec::new(),
            var_count: 0,
            functions: HashMap::new(),
            var_types: Vec::new(),
            #[cfg(feature = "native-accel")]
            accel_graph: None,
            #[cfg(feature = "native-accel")]
            fusion_groups: Vec::new(),
        }
    }

    pub fn with_instructions(instructions: Vec<Instr>, var_count: usize) -> Self {
        Self {
            instructions,
            var_count,
            ..Self::empty()
        }
    }
}

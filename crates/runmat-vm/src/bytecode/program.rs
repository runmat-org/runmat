use crate::bytecode::instr::Instr;
use crate::layout::VmAssemblyLayout;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionGroup;
use runmat_builtins::{Type, Value};
use runmat_hir::{FunctionId, LegacyHirStmt as HirStmt, VarId};
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
    #[serde(default)]
    pub source_id: Option<runmat_hir::SourceId>,
}

#[derive(Debug, Clone)]
pub struct CallFrame {
    pub function_name: String,
    pub return_address: usize,
    pub locals_start: usize,
    pub locals_count: usize,
    pub expected_outputs: usize,
}

#[derive(Debug)]
pub struct ExecutionContext {
    pub call_stack: Vec<CallFrame>,
    pub locals: Vec<Value>,
    pub instruction_pointer: usize,
    pub functions: std::collections::HashMap<String, UserFunction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFunctionBytecode {
    pub function: FunctionId,
    pub display_name: String,
    pub instructions: Vec<Instr>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    pub var_count: usize,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub capture_slots: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bytecode {
    pub instructions: Vec<Instr>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    #[serde(default)]
    pub source_id: Option<runmat_hir::SourceId>,
    pub var_count: usize,
    pub functions: HashMap<String, UserFunction>,
    #[serde(default)]
    pub semantic_functions: HashMap<FunctionId, SemanticFunctionBytecode>,
    #[serde(default)]
    pub var_types: Vec<Type>,
    #[serde(default)]
    pub var_names: HashMap<usize, String>,
    #[serde(default)]
    pub layout: Option<VmAssemblyLayout>,
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
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            source_id: None,
            var_count: 0,
            functions: HashMap::new(),
            semantic_functions: HashMap::new(),
            var_types: Vec::new(),
            var_names: HashMap::new(),
            layout: None,
            #[cfg(feature = "native-accel")]
            accel_graph: None,
            #[cfg(feature = "native-accel")]
            fusion_groups: Vec::new(),
        }
    }

    pub fn with_instructions(instructions: Vec<Instr>, var_count: usize) -> Self {
        let instr_spans = vec![runmat_hir::Span::default(); instructions.len()];
        let call_arg_spans = vec![None; instructions.len()];
        Self {
            instructions,
            instr_spans,
            call_arg_spans,
            var_count,
            ..Self::empty()
        }
    }
}

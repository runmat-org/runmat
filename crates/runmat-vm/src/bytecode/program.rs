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
    #[serde(default)]
    pub source_id: Option<runmat_hir::SourceId>,
    pub instructions: Vec<Instr>,
    #[serde(default)]
    pub instr_spans: Vec<runmat_hir::Span>,
    #[serde(default)]
    pub call_arg_spans: Vec<Option<Vec<runmat_hir::Span>>>,
    pub var_count: usize,
    pub input_slots: Vec<usize>,
    #[serde(default)]
    pub varargin_slot: Option<usize>,
    pub output_slots: Vec<usize>,
    pub capture_slots: Vec<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticFunctionRegistry {
    pub functions: HashMap<FunctionId, SemanticFunctionBytecode>,
    #[serde(default)]
    pub names: HashMap<String, FunctionId>,
    #[serde(default)]
    pub source_functions: HashMap<runmat_hir::SourceId, Vec<FunctionId>>,
}

impl SemanticFunctionRegistry {
    pub fn new(functions: HashMap<FunctionId, SemanticFunctionBytecode>) -> Self {
        let mut names = HashMap::new();
        let mut source_functions: HashMap<runmat_hir::SourceId, Vec<FunctionId>> = HashMap::new();
        let mut ids: Vec<_> = functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            if let Some(function) = functions.get(&id) {
                names.entry(function.display_name.clone()).or_insert(id);
                if let Some(source_id) = function.source_id {
                    source_functions.entry(source_id).or_default().push(id);
                }
            }
        }
        Self {
            functions,
            names,
            source_functions,
        }
    }

    pub fn get(&self, function: FunctionId) -> Option<&SemanticFunctionBytecode> {
        self.functions.get(&function)
    }

    pub fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        self.names.get(name).copied()
    }

    pub fn insert_replacing_name(&mut self, function: SemanticFunctionBytecode) {
        if let Some(previous) = self
            .names
            .insert(function.display_name.clone(), function.function)
        {
            self.remove(previous);
        }
        let function_id = function.function;
        if let Some(source_id) = function.source_id {
            let functions = self.source_functions.entry(source_id).or_default();
            if !functions.contains(&function_id) {
                functions.push(function_id);
            }
        }
        self.functions.insert(function_id, function);
    }

    pub fn remove(&mut self, function: FunctionId) -> Option<SemanticFunctionBytecode> {
        let removed = self.functions.remove(&function)?;
        if self.names.get(&removed.display_name) == Some(&function) {
            self.names.remove(&removed.display_name);
        }
        if let Some(source_id) = removed.source_id {
            if let Some(functions) = self.source_functions.get_mut(&source_id) {
                functions.retain(|id| *id != function);
                if functions.is_empty() {
                    self.source_functions.remove(&source_id);
                }
            }
        }
        Some(removed)
    }

    pub fn remove_source(&mut self, source: runmat_hir::SourceId) -> Vec<SemanticFunctionBytecode> {
        let ids = self.source_functions.remove(&source).unwrap_or_default();
        let mut removed = Vec::new();
        for id in ids {
            if let Some(function) = self.functions.remove(&id) {
                if self.names.get(&function.display_name) == Some(&id) {
                    self.names.remove(&function.display_name);
                }
                removed.push(function);
            }
        }
        removed
    }

    pub fn functions_for_source(&self, source: runmat_hir::SourceId) -> &[FunctionId] {
        self.source_functions
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
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
    pub semantic_function_registry: SemanticFunctionRegistry,
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
            semantic_function_registry: SemanticFunctionRegistry::default(),
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

    pub fn semantic_registry(&self) -> SemanticFunctionRegistry {
        if self.semantic_function_registry.functions.is_empty()
            && !self.semantic_functions.is_empty()
        {
            return SemanticFunctionRegistry::new(self.semantic_functions.clone());
        }
        self.semantic_function_registry.clone()
    }
}

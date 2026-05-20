#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
use crate::bytecode::instr::Instr;
use crate::layout::VmAssemblyLayout;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
#[cfg(feature = "native-accel")]
use runmat_accelerate::FusionGroup;
use runmat_builtins::{Type, Value};
use runmat_hir::FunctionId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    pub spawned_task_ids: HashSet<u64>,
    pub next_spawn_task_id: u64,
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
    #[serde(default)]
    pub varargout_slot: Option<usize>,
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
    #[serde(default)]
    pub semantic_async_metadata: SemanticAsyncMetadata,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub accel_graph: Option<AccelGraph>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub fusion_groups: Vec<FusionGroup>,
    #[cfg(feature = "native-accel")]
    #[serde(default)]
    pub semantic_fusion_metadata: SemanticFusionMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticAsyncMetadata {
    pub mir_spawn_site_count: usize,
    pub mir_spawn_sites: Vec<SemanticSpawnSite>,
    pub mir_await_site_count: usize,
    pub mir_await_sites: Vec<SemanticAwaitSite>,
    #[serde(default)]
    pub runtime_model: SemanticAsyncRuntimeModel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticAsyncRuntimeModel {
    LazyFutureDescriptorLane,
}

impl Default for SemanticAsyncRuntimeModel {
    fn default() -> Self {
        Self::LazyFutureDescriptorLane
    }
}

impl SemanticAsyncRuntimeModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LazyFutureDescriptorLane => "lazy_future_descriptor_lane",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSpawnSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAwaitSite {
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub resume: runmat_mir::BasicBlockId,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticFusionMetadata {
    pub mir_fusion_signal_count: usize,
    pub mir_fusion_candidate_group_count: usize,
    pub mir_fusion_candidate_groups: Vec<SemanticFusionCandidateGroup>,
    pub semantic_instruction_window_count: usize,
    #[serde(default)]
    pub semantic_instruction_windows: Vec<SemanticFusionInstructionWindow>,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFusionCandidateGroup {
    pub id: usize,
    pub signal_count: usize,
    pub function: runmat_hir::FunctionId,
    pub block: runmat_mir::BasicBlockId,
    pub stmt_start: usize,
    pub stmt_end: usize,
    #[serde(default)]
    pub source_span: runmat_hir::Span,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticFusionInstructionKind {
    Elementwise,
    Reduction,
    Matmul,
}

#[cfg(feature = "native-accel")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticFusionInstructionWindow {
    pub span: runmat_accelerate::graph::InstrSpan,
    pub kind: SemanticFusionInstructionKind,
}

impl Bytecode {
    pub fn empty() -> Self {
        Self {
            instructions: Vec::new(),
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            source_id: None,
            var_count: 0,
            semantic_functions: HashMap::new(),
            semantic_function_registry: SemanticFunctionRegistry::default(),
            var_types: Vec::new(),
            var_names: HashMap::new(),
            layout: None,
            semantic_async_metadata: SemanticAsyncMetadata::default(),
            #[cfg(feature = "native-accel")]
            accel_graph: None,
            #[cfg(feature = "native-accel")]
            fusion_groups: Vec::new(),
            #[cfg(feature = "native-accel")]
            semantic_fusion_metadata: SemanticFusionMetadata::default(),
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

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups(&self) -> Vec<FusionGroup> {
        if !self.fusion_groups.is_empty() {
            return self.fusion_groups.clone();
        }
        if self
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count
            == 0
            || self
                .semantic_fusion_metadata
                .semantic_instruction_windows
                .is_empty()
        {
            return Vec::new();
        }
        self.semantic_fusion_metadata
            .semantic_instruction_windows
            .iter()
            .enumerate()
            .map(|(id, window)| FusionGroup {
                id,
                kind: match window.kind {
                    SemanticFusionInstructionKind::Elementwise => {
                        runmat_accelerate::fusion::FusionKind::ElementwiseChain
                    }
                    SemanticFusionInstructionKind::Reduction => {
                        runmat_accelerate::fusion::FusionKind::Reduction
                    }
                    SemanticFusionInstructionKind::Matmul => {
                        runmat_accelerate::fusion::FusionKind::MatmulEpilogue
                    }
                },
                nodes: Vec::new(),
                shape: runmat_accelerate::graph::ShapeInfo::Unknown,
                span: window.span.clone(),
                pattern: None,
                stack_layout: None,
            })
            .collect()
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_fusion_groups_for_graph(&self, graph: &AccelGraph) -> Vec<FusionGroup> {
        let mut groups = self.runtime_fusion_groups();
        if groups.is_empty() {
            return groups;
        }
        if groups.iter().any(|group| group.stack_layout.is_none()) {
            annotate_fusion_groups_with_stack_layout(&self.instructions, graph, &mut groups);
        }
        groups
    }

    #[cfg(feature = "native-accel")]
    pub fn runtime_accel_graph_for_fusion(
        &self,
        runtime_groups: &[FusionGroup],
    ) -> Option<AccelGraph> {
        if runtime_groups.is_empty()
            || self
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count
                == 0
        {
            return None;
        }
        Some(build_accel_graph(&self.instructions, &self.var_types))
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::{Bytecode, SemanticFusionInstructionKind, SemanticFusionInstructionWindow};
    use runmat_accelerate::graph::InstrSpan;
    use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};

    #[test]
    fn runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty() {
        let mut bytecode = Bytecode::empty();
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 2, end: 4 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].span.start, 2);
        assert_eq!(groups[0].span.end, 4);
        assert!(groups[0].nodes.is_empty());
        assert_eq!(
            groups[0].kind,
            runmat_accelerate::fusion::FusionKind::ElementwiseChain
        );
    }

    #[test]
    fn runtime_fusion_groups_prefer_existing_bytecode_groups() {
        let mut bytecode = Bytecode::empty();
        bytecode.fusion_groups = vec![runmat_accelerate::fusion::FusionGroup {
            id: 7,
            kind: runmat_accelerate::fusion::FusionKind::ElementwiseChain,
            nodes: vec![1],
            shape: runmat_accelerate::graph::ShapeInfo::Unknown,
            span: InstrSpan { start: 5, end: 5 },
            pattern: None,
            stack_layout: None,
        }];
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 10, end: 20 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let groups = bytecode.runtime_fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 7);
        assert_eq!(groups[0].nodes, vec![1]);
        assert_eq!(groups[0].span.start, 5);
        assert_eq!(groups[0].span.end, 5);
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![crate::Instr::Add];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 0, end: 0 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let graph = bytecode.runtime_accel_graph_for_fusion(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should be materialized when semantic runtime groups exist and compile graph is missing"
        );
    }

    #[test]
    fn runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_present() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];
        bytecode.accel_graph = Some(crate::accel::graph::build_accel_graph(
            &bytecode.instructions,
            &bytecode.var_types,
        ));
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let graph = bytecode.runtime_accel_graph_for_fusion(&runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should still be materialized when compile graph metadata is present"
        );
    }

    #[test]
    fn runtime_accel_graph_ignores_stale_compile_graph_metadata() {
        let mut bytecode = Bytecode::empty();
        bytecode.instructions = vec![
            crate::Instr::LoadVar(0),
            crate::Instr::LoadVar(1),
            crate::Instr::Add,
        ];
        bytecode.var_types = vec![
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
            runmat_builtins::Type::Num,
        ];

        let stale_graph = crate::accel::graph::build_accel_graph(
            &[
                crate::Instr::LoadVar(0),
                crate::Instr::LoadVar(1),
                crate::Instr::Mul,
            ],
            &bytecode.var_types,
        );
        bytecode.accel_graph = Some(stale_graph);
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let runtime_groups = bytecode.runtime_fusion_groups();
        let graph = bytecode
            .runtime_accel_graph_for_fusion(&runtime_groups)
            .expect("runtime graph should be materialized from active bytecode instructions");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))),
            "runtime graph should reflect active bytecode instructions"
        );
        assert!(
            !graph
                .nodes
                .iter()
                .any(|node| matches!(node.label, AccelNodeLabel::Primitive(PrimitiveOp::Mul))),
            "stale compile graph metadata should not be reused at runtime"
        );
    }

    #[test]
    fn runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty() {
        let bytecode = Bytecode::empty();
        let graph = bytecode.runtime_accel_graph_for_fusion(&[]);
        assert!(
            graph.is_none(),
            "runtime graph materialization should remain gated when semantic runtime groups are absent"
        );
    }
}

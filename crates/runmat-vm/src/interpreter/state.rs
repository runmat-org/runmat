use crate::bytecode::program::ExecutionContext;
use crate::bytecode::Bytecode;
use log::debug;
#[cfg(feature = "native-accel")]
use runmat_accelerate::{prepare_fusion_plan, FusionPlan};
use runmat_builtins::Value;
use std::collections::HashMap;
#[cfg(feature = "native-accel")]
use std::sync::Arc;

#[derive(Debug)]
pub enum InterpreterOutcome {
    Completed(Vec<Value>),
}

#[derive(Debug)]
pub struct InterpreterState {
    pub bytecode: Bytecode,
    pub stack: Vec<Value>,
    pub vars: Vec<Value>,
    pub pc: usize,
    pub context: ExecutionContext,
    pub try_stack: Vec<(usize, Option<usize>)>,
    pub last_exception: Option<runmat_builtins::MException>,
    pub imports: Vec<(Vec<String>, bool)>,
    pub global_aliases: HashMap<usize, String>,
    pub persistent_aliases: HashMap<usize, String>,
    pub current_function_name: String,
    pub call_counts: Vec<(usize, usize)>,
    #[cfg(feature = "native-accel")]
    pub fusion_plan: Option<Arc<FusionPlan>>,
}

#[cfg(feature = "native-accel")]
fn runtime_fusion_groups(bytecode: &Bytecode) -> Vec<runmat_accelerate::fusion::FusionGroup> {
    if !bytecode.fusion_groups.is_empty() {
        return bytecode.fusion_groups.clone();
    }
    if bytecode
        .semantic_fusion_metadata
        .mir_fusion_candidate_group_count
        == 0
        || bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows
            .is_empty()
    {
        return Vec::new();
    }
    bytecode
        .semantic_fusion_metadata
        .semantic_instruction_windows
        .iter()
        .enumerate()
        .map(|(id, window)| runmat_accelerate::fusion::FusionGroup {
            id,
            kind: match window.kind {
                crate::bytecode::SemanticFusionInstructionKind::Elementwise => {
                    runmat_accelerate::fusion::FusionKind::ElementwiseChain
                }
                crate::bytecode::SemanticFusionInstructionKind::Reduction => {
                    runmat_accelerate::fusion::FusionKind::Reduction
                }
                crate::bytecode::SemanticFusionInstructionKind::Matmul => {
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

impl InterpreterState {
    pub fn new(
        bytecode: Bytecode,
        initial_vars: &mut [Value],
        current_function_name: Option<&str>,
        call_counts: Vec<(usize, usize)>,
    ) -> Self {
        let mut vars = initial_vars.to_vec();
        if vars.len() < bytecode.var_count {
            vars.resize(bytecode.var_count, Value::Num(0.0));
        }
        if bytecode.semantic_async_metadata.mir_spawn_site_count > 0
            || bytecode.semantic_async_metadata.mir_await_site_count > 0
        {
            debug!(
                "async semantics: compiled bytecode carries {} MIR spawn site(s) and {} MIR await site(s); runtime model={} with explicit spawn/await bytecode boundaries",
                bytecode.semantic_async_metadata.mir_spawn_site_count,
                bytecode.semantic_async_metadata.mir_await_site_count,
                bytecode.semantic_async_metadata.runtime_model.as_str()
            );
        }
        Self {
            stack: Vec::new(),
            context: ExecutionContext {
                call_stack: Vec::new(),
                locals: Vec::new(),
                instruction_pointer: 0,
                spawned_task_ids: std::collections::HashSet::new(),
                next_spawn_task_id: 0,
            },
            try_stack: Vec::new(),
            last_exception: None,
            imports: Vec::new(),
            global_aliases: HashMap::new(),
            persistent_aliases: HashMap::new(),
            vars,
            pc: 0,
            call_counts,
            current_function_name: current_function_name
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<main>".to_string()),
            #[cfg(feature = "native-accel")]
            // Runtime planning prefers compile-populated groups but falls back to
            // semantic instruction-window scaffolds when compile groups are absent.
            fusion_plan: prepare_fusion_plan(
                bytecode.accel_graph.as_ref(),
                &runtime_fusion_groups(&bytecode),
                bytecode
                    .semantic_fusion_metadata
                    .mir_fusion_candidate_group_count,
            ),
            bytecode,
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::runtime_fusion_groups;
    use crate::bytecode::{
        Bytecode, SemanticFusionInstructionKind, SemanticFusionInstructionWindow,
    };
    use runmat_accelerate::graph::InstrSpan;

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

        let groups = runtime_fusion_groups(&bytecode);
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

        let groups = runtime_fusion_groups(&bytecode);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, 7);
        assert_eq!(groups[0].nodes, vec![1]);
        assert_eq!(groups[0].span.start, 5);
        assert_eq!(groups[0].span.end, 5);
    }
}

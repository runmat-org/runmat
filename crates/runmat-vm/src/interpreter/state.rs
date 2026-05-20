#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
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
fn runtime_accel_graph_for_fusion(
    bytecode: &Bytecode,
    runtime_groups: &[runmat_accelerate::fusion::FusionGroup],
) -> Option<runmat_accelerate::graph::AccelGraph> {
    if bytecode.accel_graph.is_some()
        || runtime_groups.is_empty()
        || bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count
            == 0
    {
        return None;
    }
    Some(build_accel_graph(
        &bytecode.instructions,
        &bytecode.var_types,
    ))
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
            fusion_plan: {
                let runtime_groups = bytecode.runtime_fusion_groups();
                let runtime_graph = runtime_accel_graph_for_fusion(&bytecode, &runtime_groups);
                let accel_graph_ref = runtime_graph.as_ref().or(bytecode.accel_graph.as_ref());
                prepare_fusion_plan(
                    accel_graph_ref,
                    &runtime_groups,
                    bytecode
                        .semantic_fusion_metadata
                        .mir_fusion_candidate_group_count,
                )
            },
            bytecode,
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::runtime_accel_graph_for_fusion;
    use crate::bytecode::{
        Bytecode, SemanticFusionInstructionKind, SemanticFusionInstructionWindow,
    };
    use runmat_accelerate::graph::InstrSpan;

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
        let graph = runtime_accel_graph_for_fusion(&bytecode, &runtime_groups);
        assert!(
            graph.is_some(),
            "runtime graph should be materialized when semantic runtime groups exist and compile graph is missing"
        );
    }

    #[test]
    fn runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty() {
        let bytecode = Bytecode::empty();
        let graph = runtime_accel_graph_for_fusion(&bytecode, &[]);
        assert!(
            graph.is_none(),
            "runtime graph materialization should remain gated when semantic runtime groups are absent"
        );
    }
}

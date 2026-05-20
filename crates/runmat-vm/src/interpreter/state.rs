use crate::bytecode::program::ExecutionContext;
use crate::bytecode::Bytecode;
use log::debug;
#[cfg(feature = "native-accel")]
use runmat_accelerate::graph::AccelGraph;
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
    #[cfg(feature = "native-accel")]
    pub fusion_accel_graph: Option<AccelGraph>,
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
        #[cfg(feature = "native-accel")]
        let (fusion_plan, fusion_accel_graph) = {
            // Runtime planning prefers compile-populated groups but falls back to
            // semantic instruction-window scaffolds when compile groups are absent.
            let runtime_groups = bytecode.runtime_fusion_groups();
            let runtime_graph = bytecode.runtime_accel_graph_for_fusion(&runtime_groups);
            let accel_graph_ref = runtime_graph.as_ref().or(bytecode.accel_graph.as_ref());
            let runtime_groups = if let Some(graph) = accel_graph_ref {
                bytecode.runtime_fusion_groups_for_graph(graph)
            } else {
                runtime_groups
            };
            let fusion_plan = prepare_fusion_plan(
                accel_graph_ref,
                &runtime_groups,
                bytecode
                    .semantic_fusion_metadata
                    .mir_fusion_candidate_group_count,
            );
            let fusion_accel_graph = runtime_graph.or_else(|| bytecode.accel_graph.clone());
            (fusion_plan, fusion_accel_graph)
        };
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
            fusion_plan,
            #[cfg(feature = "native-accel")]
            fusion_accel_graph,
            bytecode,
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::InterpreterState;
    use crate::bytecode::{
        Bytecode, SemanticFusionInstructionKind, SemanticFusionInstructionWindow,
    };
    use runmat_accelerate::graph::InstrSpan;

    #[test]
    fn runtime_materialized_graph_is_retained_for_fusion_execution() {
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
        bytecode
            .semantic_fusion_metadata
            .mir_fusion_candidate_group_count = 1;
        bytecode
            .semantic_fusion_metadata
            .semantic_instruction_windows = vec![SemanticFusionInstructionWindow {
            span: InstrSpan { start: 2, end: 2 },
            kind: SemanticFusionInstructionKind::Elementwise,
        }];

        let mut initial_vars = Vec::new();
        let state = InterpreterState::new(bytecode, &mut initial_vars, Some("<main>"), Vec::new());
        assert!(
            state.fusion_plan.is_some(),
            "expected runtime fusion plan when semantic windows exist"
        );
        assert!(
            state.fusion_accel_graph.is_some(),
            "expected runtime accel graph to be retained for fusion execution"
        );
    }
}

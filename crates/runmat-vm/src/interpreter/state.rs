use crate::bytecode::{Bytecode, ExecutionContext};
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
        Self {
            stack: Vec::new(),
            context: ExecutionContext {
                call_stack: Vec::new(),
                locals: Vec::new(),
                instruction_pointer: 0,
                functions: bytecode.functions.clone(),
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
            fusion_plan: prepare_fusion_plan(
                bytecode.accel_graph.as_ref(),
                &bytecode.fusion_groups,
            ),
            bytecode,
        }
    }
}

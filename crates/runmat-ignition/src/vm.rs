use crate::functions::Bytecode;
#[cfg(test)]
use crate::instr::Instr;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub use runmat_vm::interpreter::api::{
    push_pending_workspace, set_call_stack_limit, set_error_namespace,
    take_updated_workspace_state, InterpreterOutcome, InterpreterState, PendingWorkspaceGuard,
    DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};

#[cfg(test)]
use runmat_vm::accel::fusion as accel_fusion;

#[cfg(test)]
use crate::instr::EndExpr;
#[cfg(test)]
use runmat_vm::indexing::selectors as idx_selectors;

type VmResult<T> = Result<T, RuntimeError>;

pub async fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    runmat_vm::interpret_with_vars(bytecode, initial_vars, current_function_name).await
}

pub async fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, RuntimeError> {
    runmat_vm::interpret(bytecode).await
}

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    runmat_vm::interpret_function(bytecode, vars).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    runmat_vm::interpret_function_with_counts(bytecode, vars, name, out_count, in_count).await
}

#[cfg(test)]
#[derive(Clone)]
enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    LinearIndices {
        values: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

#[cfg(test)]
fn from_vm_slice_selector(selector: idx_selectors::SliceSelector) -> SliceSelector {
    match selector {
        idx_selectors::SliceSelector::Colon => SliceSelector::Colon,
        idx_selectors::SliceSelector::Scalar(i) => SliceSelector::Scalar(i),
        idx_selectors::SliceSelector::Indices(v) => SliceSelector::Indices(v),
        idx_selectors::SliceSelector::LinearIndices { values, output_shape } => {
            SliceSelector::LinearIndices { values, output_shape }
        }
    }
}

#[cfg(test)]
async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    idx_selectors::selector_from_value_dim(value, dim_len)
        .await
        .map(from_vm_slice_selector)
}

#[cfg(test)]
async fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    idx_selectors::indices_from_value_linear(value, total_len).await
}

#[cfg(test)]
mod scalar_index_tests {
    use super::*;

    #[test]
    fn linear_false_bool_index_is_empty() {
        let indices =
            futures::executor::block_on(indices_from_value_linear(&Value::Bool(false), 4))
                .expect("false logical index should be empty");
        assert!(indices.is_empty());
    }

    #[test]
    fn linear_true_bool_index_selects_first() {
        let indices = futures::executor::block_on(indices_from_value_linear(&Value::Bool(true), 4))
            .expect("true logical index should select first element");
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn dim_false_bool_selector_is_empty() {
        let selector = futures::executor::block_on(selector_from_value_dim(&Value::Bool(false), 4))
            .expect("false logical selector should be empty");
        match selector {
            SliceSelector::Indices(indices) => assert!(indices.is_empty()),
            SliceSelector::Scalar(_)
            | SliceSelector::Colon
            | SliceSelector::LinearIndices { .. } => {
                panic!("expected empty indices selector")
            }
        }
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod fusion_span_barrier_tests {
    use super::*;
    use runmat_accelerate::InstrSpan;

    #[test]
    fn store_reload_span_with_one_live_result_is_legal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert_eq!(accel_fusion::fusion_span_live_result_count(&instructions, &span), Some(1));
        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn span_leaving_multiple_live_results_is_illegal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 3 };

        assert_eq!(accel_fusion::fusion_span_live_result_count(&instructions, &span), Some(2));
        assert!(accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn stored_value_observed_after_span_is_legal_when_materialized() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }

    #[test]
    fn overwritten_store_before_later_load_is_legal() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::Add,
            Instr::StoreVar(1),
            Instr::LoadVar(1),
            Instr::LoadConst(1.0),
            Instr::StoreVar(1),
            Instr::LoadVar(1),
        ];
        let span = InstrSpan { start: 0, end: 4 };

        assert!(!accel_fusion::fusion_span_has_vm_barrier(&instructions, &span));
    }
}

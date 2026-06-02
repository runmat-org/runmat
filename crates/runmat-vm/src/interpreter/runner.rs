use crate::accel::fusion as accel_fusion;
use crate::accel::residency as accel_residency;
use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::{InterpreterOutcome, InterpreterState};
use crate::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::{attach_span_from_pc, mex, set_vm_pc};
use crate::interpreter::timing::InterpreterTiming;
use crate::runtime::call_stack::attach_call_frames;
use crate::runtime::globals as runtime_globals;
use crate::runtime::workspace::{
    refresh_workspace_state, workspace_assign, workspace_clear, workspace_lookup, workspace_remove,
    workspace_snapshot,
};
use runmat_builtins::{CellArray, Value};
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, active_group_plan_clone, deactivate_fusion_plan, set_current_pc,
};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

type VmResult<T> = Result<T, RuntimeError>;
runmat_thread_local::runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

fn sync_initial_vars(initial: &mut [Value], vars: &[Value]) {
    for (i, var) in vars.iter().enumerate() {
        if i < initial.len() {
            initial[i] = var.clone();
        }
    }
}

fn ensure_workspace_resolver_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runtime_workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: workspace_lookup,
            snapshot: workspace_snapshot,
            globals: runtime_globals::workspace_global_names,
            assign: Some(workspace_assign),
            clear: Some(workspace_clear),
            remove: Some(workspace_remove),
        });
    });
}

fn ensure_wasm_builtins_registered() {
    #[cfg(target_arch = "wasm32")]
    {
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            runmat_runtime::builtins::wasm_registry::register_all();
        });
    }
}

#[cfg(feature = "native-accel")]
fn clear_residency(value: &Value) {
    accel_residency::clear_value(value);
}

pub async fn invoke_semantic_function_value(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<Value, RuntimeError> {
    let (value, _) = invoke_semantic_function_value_with_capture_updates(
        function,
        args,
        requested_outputs,
        function_registry,
    )
    .await?;
    Ok(value)
}

pub(crate) async fn invoke_semantic_function_value_with_capture_updates(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<(Value, Vec<Value>), RuntimeError> {
    let function_id = runmat_hir::FunctionId(function);
    let func = function_registry.get(function_id).ok_or_else(|| {
        let message = format!("Undefined semantic function: {function}");
        mex("UndefinedSemanticFunction", &message)
    })?;
    if args.len() < func.capture_slots.len() {
        let message = format!(
            "semantic function {} received too few arguments",
            func.display_name
        );
        return Err(mex("SemanticFunctionArity", &message));
    }
    let runtime_arg_count = args.len() - func.capture_slots.len();
    if runtime_arg_count > func.input_slots.len() && func.varargin_slot.is_none() {
        let message = format!(
            "semantic function {} expected {} inputs, got {}",
            func.display_name,
            func.input_slots.len(),
            runtime_arg_count
        );
        return Err(mex("TooManyInputs", &message));
    }
    if requested_outputs > func.output_slots.len() && func.varargout_slot.is_none() {
        let message = format!(
            "semantic function {} expected {} outputs, got {}",
            func.display_name,
            func.output_slots.len(),
            requested_outputs
        );
        return Err(mex("TooManyOutputs", &message));
    }

    let mut vars = vec![Value::Num(0.0); func.var_count];
    let mut missing_input_slots = HashSet::new();
    for (slot, value) in func.capture_slots.iter().zip(args.iter()) {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
    }
    for (slot, value) in func
        .input_slots
        .iter()
        .take(runtime_arg_count)
        .zip(args.iter().skip(func.capture_slots.len()))
    {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
    }
    let default_values_by_slot: HashMap<usize, Value> = func
        .argument_validations
        .iter()
        .filter_map(|validation| {
            validation.default_value.as_ref().map(|value| {
                let lowered = match value {
                    crate::bytecode::program::FunctionArgDefaultValue::Number(value) => {
                        Value::Num(*value)
                    }
                    crate::bytecode::program::FunctionArgDefaultValue::Bool(value) => {
                        Value::Bool(*value)
                    }
                    crate::bytecode::program::FunctionArgDefaultValue::String(value) => {
                        Value::String(value.clone())
                    }
                    crate::bytecode::program::FunctionArgDefaultValue::EmptyArray => Value::Tensor(
                        runmat_builtins::Tensor::new(Vec::new(), vec![0, 0])
                            .expect("empty default tensor"),
                    ),
                };
                (validation.input_slot, lowered)
            })
        })
        .collect();
    if runtime_arg_count < func.input_slots.len() {
        for slot in func.input_slots.iter().skip(runtime_arg_count) {
            if let Some(default_value) = default_values_by_slot.get(slot) {
                if *slot < vars.len() {
                    vars[*slot] = default_value.clone();
                }
            } else {
                missing_input_slots.insert(*slot);
            }
        }
    }
    validate_function_arguments(func, &vars, &missing_input_slots)?;
    if let Some(slot) = func.varargin_slot {
        let fixed_count = func.input_slots.len();
        let rest = if runtime_arg_count > fixed_count {
            args[func.capture_slots.len() + fixed_count..].to_vec()
        } else {
            Vec::new()
        };
        let cols = rest.len();
        let cell = CellArray::new(rest, 1, cols)
            .map_err(|err| mex("VararginPack", &format!("varargin: {err}")))?;
        if slot < vars.len() {
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.varargout_slot {
        if slot < vars.len() {
            let cell = CellArray::new(Vec::new(), 1, 0)
                .map_err(|err| mex("VarargoutPack", &format!("varargout: {err}")))?;
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.implicit_nargin_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(runtime_arg_count as f64);
        }
    }
    if let Some(slot) = func.implicit_nargout_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(requested_outputs as f64);
        }
    }

    let mut bytecode = Bytecode::with_instructions(func.instructions.clone(), func.var_count);
    bytecode.instr_spans = func.instr_spans.clone();
    bytecode.call_arg_spans = func.call_arg_spans.clone();
    bytecode.source_id = func.source_id;
    bytecode.bound_functions = function_registry.functions.clone();
    bytecode.function_registry = function_registry.clone();
    let result_vars = interpret_function_with_counts(
        &bytecode,
        vars,
        &func.display_name,
        requested_outputs,
        runtime_arg_count,
        missing_input_slots,
    )
    .await?;
    let output_values = collect_semantic_outputs(func, &result_vars, requested_outputs)?;
    let updated_captures = func
        .capture_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    #[cfg(feature = "native-accel")]
    clear_semantic_function_temp_residency(&result_vars, &output_values);
    Ok((
        output_value(output_values, requested_outputs),
        updated_captures,
    ))
}

fn validate_function_arguments(
    func: &crate::bytecode::program::FunctionBytecode,
    vars: &[Value],
    missing_input_slots: &HashSet<usize>,
) -> Result<(), RuntimeError> {
    for validation in &func.argument_validations {
        if missing_input_slots.contains(&validation.input_slot) {
            continue;
        }
        let Some(input_index) = func
            .input_slots
            .iter()
            .position(|slot| *slot == validation.input_slot)
        else {
            continue;
        };
        let value = vars
            .get(validation.input_slot)
            .ok_or_else(|| mex("InvalidInputSlot", "function argument slot out of bounds"))?;

        if let Some(size) = &validation.size {
            let (rows, cols) = value_shape_2d(value);
            if !dim_matches(&size.rows, rows) || !dim_matches(&size.cols, cols) {
                return Err(mex(
                    "ArgumentValidationSize",
                    &format!(
                        "Function '{}' argument #{} failed size validation",
                        func.display_name,
                        input_index + 1
                    ),
                ));
            }
        }

        if let Some(class_name) = &validation.class_name {
            if !value_matches_class(value, class_name) {
                return Err(mex(
                    "ArgumentValidationClass",
                    &format!(
                        "Function '{}' argument #{} failed class validation (expected {})",
                        func.display_name,
                        input_index + 1,
                        class_name
                    ),
                ));
            }
        }
        for validator in &validation.validators {
            match validator {
                crate::bytecode::program::FunctionArgValidator::Finite => {
                    if !value_is_finite(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeFinite validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::NumericOrLogical => {
                    if !value_is_numeric_or_logical(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNumericOrLogical validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Text => {
                    if !value_is_text(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeText validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Nonempty => {
                    if value_is_empty(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNonempty validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::ScalarOrEmpty => {
                    if !value_is_scalar_or_empty(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeScalarOrEmpty validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Real => {
                    if !value_is_real(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeReal validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Integer => {
                    if !value_is_integer(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeInteger validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Positive => {
                    if !value_is_positive(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBePositive validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Negative => {
                    if !value_is_negative(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNegative validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Nonnegative => {
                    if !value_is_nonnegative(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNonnegative validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Nonzero => {
                    if !value_is_nonzero(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNonzero validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::Nonpositive => {
                    if !value_is_nonpositive(value) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeNonpositive validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::GreaterThanOrEqual(threshold) => {
                    if !value_is_greater_than_or_equal(value, *threshold) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeGreaterThanOrEqual validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::LessThanOrEqual(threshold) => {
                    if !value_is_less_than_or_equal(value, *threshold) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeLessThanOrEqual validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::GreaterThan(threshold) => {
                    if !value_is_greater_than(value, *threshold) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeGreaterThan validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
                crate::bytecode::program::FunctionArgValidator::LessThan(threshold) => {
                    if !value_is_less_than(value, *threshold) {
                        return Err(mex(
                            "ArgumentValidationFunction",
                            &format!(
                                "Function '{}' argument #{} failed mustBeLessThan validation",
                                func.display_name,
                                input_index + 1
                            ),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn dim_matches(dim: &crate::bytecode::program::FunctionArgDim, actual: usize) -> bool {
    match dim {
        crate::bytecode::program::FunctionArgDim::Any => true,
        crate::bytecode::program::FunctionArgDim::Exact(expected) => *expected == actual,
    }
}

fn value_shape_2d(value: &Value) -> (usize, usize) {
    match value {
        Value::Tensor(t) => {
            let rows = t.shape.first().copied().unwrap_or(0);
            let cols = t.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::ComplexTensor(t) => {
            let rows = t.shape.first().copied().unwrap_or(0);
            let cols = t.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::LogicalArray(a) => {
            let rows = a.shape.first().copied().unwrap_or(0);
            let cols = a.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::Cell(c) => (c.rows, c.cols),
        Value::CharArray(c) => (c.rows, c.cols),
        Value::StringArray(s) => {
            let rows = s.shape.first().copied().unwrap_or(0);
            let cols = s.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        _ => (1, 1),
    }
}

fn value_matches_class(value: &Value, class_name: &str) -> bool {
    match class_name {
        "double" => match value {
            Value::Num(_) => true,
            Value::Tensor(t) => t.dtype.class_name() == "double",
            _ => false,
        },
        "single" => matches!(value, Value::Tensor(t) if t.dtype.class_name() == "single"),
        "logical" => matches!(value, Value::Bool(_) | Value::LogicalArray(_)),
        "char" => matches!(value, Value::CharArray(_) | Value::String(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_)),
        other => match value {
            Value::Object(obj) => obj.class_name == other,
            Value::HandleObject(handle) => handle.class_name == other,
            Value::ClassRef(name) => name == other,
            _ => false,
        },
    }
}

fn value_is_finite(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite()),
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite()),
        Value::LogicalArray(_) | Value::CharArray(_) => true,
        _ => false,
    }
}

fn value_is_numeric_or_logical(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
    )
}

fn value_is_text(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(chars) => chars.rows == 1,
        Value::Cell(cell) => cell.data.iter().all(|entry| match &**entry {
            Value::CharArray(chars) => chars.rows == 1,
            Value::String(_) => true,
            _ => false,
        }),
        _ => false,
    }
}

fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.shape.iter().product::<usize>() == 0,
        Value::ComplexTensor(t) => t.shape.iter().product::<usize>() == 0,
        Value::LogicalArray(a) => a.shape.iter().product::<usize>() == 0,
        Value::StringArray(s) => s.shape.iter().product::<usize>() == 0,
        Value::CharArray(c) => c.rows * c.cols == 0,
        Value::Cell(c) => c.shape.iter().product::<usize>() == 0,
        _ => false,
    }
}

fn value_is_scalar_or_empty(value: &Value) -> bool {
    let (rows, cols) = value_shape_2d(value);
    (rows == 1 && cols == 1) || (rows == 0 || cols == 0)
}

fn value_is_real(value: &Value) -> bool {
    match value {
        Value::Complex(_, im) => *im == 0.0,
        Value::ComplexTensor(t) => t.data.iter().all(|(_, im)| *im == 0.0),
        _ => true,
    }
}

fn value_is_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Num(v) => v.is_finite() && v.fract() == 0.0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && v.fract() == 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && re.fract() == 0.0,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && re.fract() == 0.0),
        _ => false,
    }
}

fn value_is_positive(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v > 0.0,
        Value::Int(v) => v.to_i64() > 0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v > 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re > 0.0,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re > 0.0),
        _ => false,
    }
}

fn value_is_negative(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v < 0.0,
        Value::Int(v) => v.to_i64() < 0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v < 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re < 0.0,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re < 0.0),
        _ => false,
    }
}

fn value_is_nonnegative(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v >= 0.0,
        Value::Int(v) => v.to_i64() >= 0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v >= 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re >= 0.0,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re >= 0.0),
        _ => false,
    }
}

fn value_is_nonzero(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v != 0.0,
        Value::Int(v) => v.to_i64() != 0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v != 0.0),
        Value::Complex(re, im) => re.is_finite() && im.is_finite() && (*re != 0.0 || *im != 0.0),
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite() && (*re != 0.0 || *im != 0.0)),
        _ => false,
    }
}

fn value_is_nonpositive(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v <= 0.0,
        Value::Int(v) => v.to_i64() <= 0,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v <= 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re <= 0.0,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re <= 0.0),
        _ => false,
    }
}

fn value_is_greater_than_or_equal(value: &Value, threshold: f64) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v >= threshold,
        Value::Int(v) => (v.to_i64() as f64) >= threshold,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v >= threshold),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re >= threshold,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re >= threshold),
        _ => false,
    }
}

fn value_is_less_than_or_equal(value: &Value, threshold: f64) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v <= threshold,
        Value::Int(v) => (v.to_i64() as f64) <= threshold,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v <= threshold),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re <= threshold,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re <= threshold),
        _ => false,
    }
}

fn value_is_greater_than(value: &Value, threshold: f64) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v > threshold,
        Value::Int(v) => (v.to_i64() as f64) > threshold,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v > threshold),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re > threshold,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re > threshold),
        _ => false,
    }
}

fn value_is_less_than(value: &Value, threshold: f64) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v < threshold,
        Value::Int(v) => (v.to_i64() as f64) < threshold,
        Value::Tensor(t) => t.data.iter().all(|v| v.is_finite() && *v < threshold),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && *re < threshold,
        Value::ComplexTensor(t) => t
            .data
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && *re < threshold),
        _ => false,
    }
}

fn collect_semantic_outputs(
    func: &crate::bytecode::program::FunctionBytecode,
    result_vars: &[Value],
    requested_outputs: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(requested_outputs.max(1));
    for slot in func.output_slots.iter().take(requested_outputs) {
        values.push(result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
    }
    if values.len() < requested_outputs {
        if let Some(slot) = func.varargout_slot {
            let available = match result_vars.get(slot) {
                Some(Value::Cell(cell)) => {
                    let expanded = crate::call::shared::expand_all_cell(cell)?;
                    let available = expanded.len();
                    for value in expanded {
                        if values.len() >= requested_outputs {
                            break;
                        }
                        values.push(value);
                    }
                    available
                }
                _ => 0,
            };
            if values.len() < requested_outputs {
                let need = requested_outputs - func.output_slots.len();
                let message = format!(
                    "Function '{}' returned {available} varargout values, {need} requested",
                    func.display_name
                );
                return Err(mex("VarargoutMismatch", &message));
            }
        }
    }
    while values.len() < requested_outputs {
        values.push(Value::Num(0.0));
    }
    Ok(values)
}

fn output_value(output_values: Vec<Value>, requested_outputs: usize) -> Value {
    match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => output_values.into_iter().next().unwrap_or(Value::Num(0.0)),
        _ => Value::OutputList(output_values.into_iter().take(requested_outputs).collect()),
    }
}

#[cfg(feature = "native-accel")]
fn clear_semantic_function_temp_residency(result_vars: &[Value], output_values: &[Value]) {
    let mut keep_values = output_values.to_vec();
    keep_values.extend(runtime_globals::collect_thread_roots());
    let keep = Value::OutputList(keep_values);
    for value in result_vars {
        accel_residency::clear_value_excluding(value, &keep);
    }
}

pub async fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut [Value],
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let state = Box::new(InterpreterState::new(
        bytecode.clone(),
        initial_vars,
        current_function_name,
        call_counts,
    ));
    match Box::pin(run_interpreter(state, initial_vars)).await {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            let current_name = current_function_name.unwrap_or("<main>");
            Err(attach_call_frames(bytecode, current_name, err))
        }
    }
}

async fn run_interpreter(
    state: Box<InterpreterState>,
    initial_vars: &mut [Value],
) -> VmResult<InterpreterOutcome> {
    let state = *state;
    Box::pin(run_interpreter_inner(state, initial_vars)).await
}

async fn run_interpreter_inner(
    state: InterpreterState,
    initial_vars: &mut [Value],
) -> VmResult<InterpreterOutcome> {
    let run_span = info_span!(
        "interpreter.run",
        function = state.current_function_name.as_str()
    );
    let _run_guard = run_span.enter();
    ensure_wasm_builtins_registered();
    ensure_workspace_resolver_registered();
    #[cfg(feature = "native-accel")]
    activate_fusion_plan(state.fusion_plan.clone());
    #[cfg(feature = "native-accel")]
    let _fusion_guard = FusionPlanGuard;
    let InterpreterState {
        mut stack,
        mut vars,
        mut pc,
        mut context,
        mut try_stack,
        mut last_exception,
        mut imports,
        mut global_aliases,
        mut persistent_aliases,
        mut missing_input_slots,
        current_function_name,
        call_counts,
        #[cfg(feature = "native-accel")]
            fusion_plan: _,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph,
        bytecode,
    } = state;
    let _source_context_guard =
        runmat_runtime::source_context::replace_current_source_id(bytecode.source_id);
    let _arity_call_counts_guard =
        runmat_runtime::builtins::introspection::arity_check::replace_call_counts(
            call_counts.clone(),
        );
    let function_registry = Arc::new(bytecode.function_registry());
    let previous_semantic_invoker = user_functions::current_semantic_function_invoker();
    let registry_for_function_invoker = Arc::clone(&function_registry);
    let _semantic_function_guard =
        user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function: usize, args: &[Value], requested_outputs: usize| {
                let args = args.to_vec();
                let previous_invoker = previous_semantic_invoker.clone();
                let function_registry = Arc::clone(&registry_for_function_invoker);
                Box::pin(async move {
                    let local_function = function_registry
                        .get(runmat_hir::FunctionId(function))
                        .is_some();
                    if !local_function {
                        if let Some(invoker) = previous_invoker {
                            return invoker(function, &args, requested_outputs).await;
                        }
                    }
                    invoke_semantic_function_value(
                        function,
                        &args,
                        requested_outputs,
                        &function_registry,
                    )
                    .await
                })
            },
        )));
    let previous_semantic_resolver = user_functions::current_semantic_function_resolver();
    let registry_for_function_resolver = Arc::clone(&function_registry);
    let _semantic_resolver_guard =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name: &str| {
            if let Some(function) = registry_for_function_resolver.resolve_name(name) {
                return Some(function.0);
            }
            previous_semantic_resolver
                .as_ref()
                .and_then(|resolver| resolver(name))
        })));
    let mut source_function_catalog = function_registry
        .functions
        .values()
        .filter_map(|function| {
            function.source_id.map(
                |source_id| runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.display_name.clone(),
                    function: function.function.0,
                },
            )
        })
        .collect::<Vec<_>>();
    source_function_catalog.sort_by_key(|info| info.function);
    let _source_function_catalog_guard =
        user_functions::install_source_function_catalog(Some(Arc::new(source_function_catalog)));
    CALL_COUNTS.with(|cc| {
        *cc.borrow_mut() = call_counts.clone();
    });
    let _workspace_guard = interp_engine::prepare_workspace_guard(&mut vars);
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let mut _gc_context = interp_engine::create_gc_context(&stack, &vars, thread_roots)?;
    let debug_stack = interp_engine::debug_stack_enabled();
    let mut interpreter_timing = InterpreterTiming::new();
    while pc < bytecode.instructions.len() {
        set_vm_pc(pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(pc);
        if let Err(err) = interp_engine::check_cancelled() {
            #[cfg(feature = "native-accel")]
            {
                for value in &stack {
                    clear_residency(value);
                }
                for value in &vars {
                    clear_residency(value);
                }
            }
            return Err(err);
        }
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) = (active_group_plan_clone(), fusion_accel_graph.as_ref())
        {
            if plan.group.span.start == pc {
                #[cfg(feature = "native-accel")]
                {
                    interp_engine::note_fusion_gate(
                        &mut interpreter_timing,
                        &plan,
                        &bytecode,
                        pc,
                        accel_fusion::fusion_span_has_vm_barrier(
                            &bytecode.instructions,
                            &plan.group.span,
                        ),
                        accel_fusion::fusion_span_live_result_count(
                            &bytecode.instructions,
                            &plan.group.span,
                        ),
                    );
                }
                let span = plan.group.span.clone();
                let has_barrier =
                    accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span);
                let _fusion_span = info_span!(
                    "fusion.execute",
                    span_start = plan.group.span.start,
                    span_end = plan.group.span.end,
                    kind = ?plan.group.kind
                )
                .entered();
                if !has_barrier {
                    match accel_fusion::try_execute_fusion_group(
                        &plan,
                        graph,
                        &mut stack,
                        &mut vars,
                        &mut context,
                    )
                    .await
                    {
                        Ok(result) => {
                            stack.push(result);
                            pc = plan.group.span.end + 1;
                            continue;
                        }
                        Err(err) => {
                            log::debug!("fusion fallback at pc {}: {}", pc, err);
                        }
                    }
                } else {
                    interp_engine::note_fusion_skip(pc, &span);
                }
            }
        }
        interp_engine::note_pre_dispatch(
            &mut interpreter_timing,
            debug_stack,
            pc,
            &bytecode.instructions[pc],
            stack.len(),
        );
        let call_counts_snapshot = CALL_COUNTS.with(|cc| cc.borrow().clone());
        let store_var_global_aliases = match &bytecode.instructions[pc] {
            Instr::StoreVar(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let store_local_global_aliases = match &bytecode.instructions[pc] {
            Instr::StoreLocal(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let mut clear_value_residency = |value: &Value| {
            #[cfg(feature = "native-accel")]
            clear_residency(value);
        };
        let mut store_var_before_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_var_after_store = |stored_index: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_var_global_aliases {
                runtime_globals::update_global_store(stored_index, stored_value, aliases);
            }
        };
        let mut store_local_before_local_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_before_var_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_after_store = |stored_offset: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_local_global_aliases {
                runtime_globals::update_global_store(stored_offset, stored_value, aliases);
            }
        };
        let mut store_local_after_fallback_store =
            |func_name: &str, stored_offset: usize, stored_value: &Value| {
                if let Some(ref aliases) = store_local_global_aliases {
                    runtime_globals::update_global_store(stored_offset, stored_value, aliases);
                }
                runtime_globals::update_persistent_local_store(
                    func_name,
                    stored_offset,
                    stored_value,
                );
            };
        let dispatch_result = interp_dispatch::dispatch_instruction(
            interp_dispatch::DispatchMeta {
                instr: &bytecode.instructions[pc],
                var_names: &bytecode.var_names,
                function_registry: &function_registry,
                source_id: bytecode.source_id,
                call_arg_spans: bytecode.call_arg_spans.get(pc).cloned().flatten(),
                call_counts: &call_counts_snapshot,
                current_function_name: &current_function_name,
            },
            interp_dispatch::DispatchState {
                stack: &mut stack,
                vars: &mut vars,
                context: &mut context,
                try_stack: &mut try_stack,
                last_exception: &mut last_exception,
                imports: &mut imports,
                global_aliases: &mut global_aliases,
                persistent_aliases: &mut persistent_aliases,
                missing_input_slots: &mut missing_input_slots,
                pc: &mut pc,
            },
            interp_dispatch::DispatchHooks {
                clear_value_residency: &mut clear_value_residency,
                store_var_before_overwrite: &mut store_var_before_overwrite,
                store_var_after_store: &mut store_var_after_store,
                store_local_before_local_overwrite: &mut store_local_before_local_overwrite,
                store_local_before_var_overwrite: &mut store_local_before_var_overwrite,
                store_local_after_store: &mut store_local_after_store,
                store_local_after_fallback_store: &mut store_local_after_fallback_store,
            },
        )
        .await;
        let dispatch_result = match dispatch_result {
            Ok(result) => result,
            Err(err) => match interp_dispatch::redirect_exception_to_catch(
                err,
                &mut try_stack,
                &mut vars,
                &mut last_exception,
                &mut pc,
                refresh_workspace_state,
            ) {
                interp_dispatch::ExceptionHandling::Caught => {
                    continue;
                }
                interp_dispatch::ExceptionHandling::Uncaught(err) => return Err(*err),
            },
        };
        if let Some(decision) = dispatch_result {
            match decision {
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::ContinueLoop) => {
                    continue
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::FallThrough) => {
                    pc += 1;
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::ContinueLoop)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::ContinueLoop) => {
                    continue
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return_value", None);
                    break;
                }
                interp_dispatch::DispatchHandled::Return(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::FallThrough)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::FallThrough) => {
                    pc += 1;
                    continue;
                }
            }
        }
        match bytecode.instructions[pc].clone() {
            Instr::EmitStackTop { .. }
            | Instr::EmitVar { .. }
            | Instr::AndAnd(_)
            | Instr::OrOr(_)
            | Instr::JumpIfFalse(_)
            | Instr::Jump(_)
            | Instr::LoadConst(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadLocal(_)
            | Instr::LoadVar(_)
            | Instr::LoadVarForIndexAssignment(_)
            | Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Swap
            | Instr::Pop
            | Instr::EnterTry(_, _)
            | Instr::PopTry
            | Instr::ReturnValue
            | Instr::Return
            | Instr::EnterScope(_)
            | Instr::LoadMember(_)
            | Instr::LoadMemberOrInit(_)
            | Instr::LoadMemberDynamic
            | Instr::LoadMemberDynamicOrInit
            | Instr::StoreMember(_)
            | Instr::StoreMemberOrInit(_)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit
            | Instr::Index(_)
            | Instr::IndexSlice(_, _, _, _)
            | Instr::IndexSliceExpr { .. }
            | Instr::IndexCell { .. }
            | Instr::IndexCellExpand { .. }
            | Instr::IndexCellList { .. }
            | Instr::StoreIndex(_)
            | Instr::StoreIndexCell { .. }
            | Instr::StoreIndexDelete(_)
            | Instr::StoreIndexCellDelete { .. }
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceDelete(_, _, _, _)
            | Instr::StoreSliceExpr { .. }
            | Instr::StoreSliceExprDelete { .. }
            | Instr::CallMethodOrMemberIndexMulti { .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. }
            | Instr::LoadMethod(_)
            | Instr::CreateFunctionHandle(_)
            | Instr::CreateExternalFunctionHandle(_)
            | Instr::CreateMethodFunctionHandle(_)
            | Instr::CreateBoundFunctionHandle(_, _)
            | Instr::CreateClosure(_, _)
            | Instr::CreateSemanticClosure(_, _, _)
            | Instr::LoadStaticProperty(_, _)
            | Instr::RegisterClass { .. }
            | Instr::CallFevalMulti(_, _)
            | Instr::CallFevalMultiUsingOutputSlot(_, _)
            | Instr::CallFevalExpandMultiOutput(_, _)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _)
            | Instr::CreateSemanticFuture(_, _, _)
            | Instr::CreateSemanticFutureExpandMultiOutput(_, _, _)
            | Instr::Spawn
            | Instr::Await
            | Instr::CallBuiltinMulti(_, _, _)
            | Instr::CallBuiltinMultiUsingOutputSlot(_, _, _)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. }
            | Instr::CallSemanticFunctionMulti(_, _, _)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
            | Instr::CallSemanticNestedFunctionMulti { .. }
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionMulti { .. }
            | Instr::CallFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionExpandMultiOutput { .. }
            | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
            | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. }
            | Instr::CallBuiltinExpandMultiOutput(_, _, _)
            | Instr::CallSuperConstructorExpandMultiOutput { .. }
            | Instr::CallSuperMethodExpandMultiOutput { .. }
            | Instr::ExitScope(_)
            | Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistent(_)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::CreateCell2D(_, _)
            | Instr::CreateStructLiteral(_)
            | Instr::CreateObjectLiteral { .. }
            | Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::Neg
            | Instr::UPlus
            | Instr::Transpose
            | Instr::ConjugateTranspose
            | Instr::Pow
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::LessEqual
            | Instr::Less
            | Instr::Greater
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual
            | Instr::LogicalNot
            | Instr::LogicalAnd
            | Instr::LogicalOr
            | Instr::Unpack(_)
            | Instr::CreateMatrix(_, _)
            | Instr::CreateMatrixDynamic(_)
            | Instr::CreateRange(_)
            | Instr::PackToRow(_)
            | Instr::PackToCol(_) => unreachable!("handled by dispatch_instruction"),
            Instr::StochasticEvolution => {
                let steps_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let scale_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let drift_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let state_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let evolved =
                    crate::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
                        state_value,
                        drift_value,
                        scale_value,
                        steps_value,
                    )
                    .await?;
                stack.push(evolved);
            }
        }
        if debug_stack {
            debug!(pc, stack_len = stack.len(), "[vm] after exec");
        }
        pc += 1;
    }
    interpreter_timing.flush_host_span("loop_complete", None);
    #[cfg(feature = "native-accel")]
    {
        let mut live_values = Vec::with_capacity(vars.len() + context.locals.len());
        live_values.extend(vars.iter().cloned());
        live_values.extend(context.locals.iter().cloned());
        let live_values = Value::OutputList(live_values);
        for value in &stack {
            accel_residency::clear_value_excluding(value, &live_values);
        }
    }
    sync_initial_vars(initial_vars, &vars);
    Ok(InterpreterOutcome::Completed(vars))
}

pub async fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    match interpret_with_vars(bytecode, &mut vars, Some("<main>")).await {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Err(e) => Err(e),
    }
}

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0, HashSet::new()).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vars;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
    });
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let mut state = InterpreterState::new(bytecode.clone(), &mut vars, Some(name), call_counts);
    state.missing_input_slots = missing_input_slots;
    let res = Box::pin(run_interpreter(Box::new(state), &mut vars)).await;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().pop();
    });
    let res = match res {
        Ok(InterpreterOutcome::Completed(values)) => Ok(values),
        Err(e) => Err(e),
    }?;
    runtime_globals::persist_declared_for_bytecode(bytecode, name, &vars);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::{
        collect_semantic_outputs, interpret_with_vars, output_value, run_interpreter_inner,
        value_is_empty, value_is_greater_than, value_is_greater_than_or_equal, value_is_integer,
        value_is_less_than, value_is_less_than_or_equal, value_is_negative, value_is_nonnegative,
        value_is_nonpositive, value_is_nonzero, value_is_numeric_or_logical, value_is_positive,
        value_is_real, value_is_scalar_or_empty, value_is_text,
    };
    use crate::bytecode::program::{Bytecode, FunctionBytecode};
    use crate::bytecode::Instr;
    use crate::interpreter::api::InterpreterState;
    use futures::executor::block_on;
    use runmat_builtins::{
        CellArray, Closure, HandleRef, ObjectInstance, StructValue, Tensor, Value,
    };
    use runmat_hir::FunctionId;
    use std::sync::{atomic::AtomicBool, Arc};
    #[cfg(feature = "native-accel")]
    use {
        once_cell::sync::Lazy,
        runmat_accelerate::simple_provider::InProcessProvider,
        runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard},
    };

    #[cfg(feature = "native-accel")]
    static TEST_PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

    #[cfg(feature = "native-accel")]
    fn upload_provider_handle(
        data: Vec<f64>,
        shape: Vec<usize>,
    ) -> runmat_accelerate_api::GpuTensorHandle {
        TEST_PROVIDER
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload should succeed")
    }

    fn test_function(varargout_slot: Option<usize>) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(0),
            display_name: "f".into(),
            source_id: None,
            instructions: vec![Instr::Return],
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count: 1,
            input_slots: Vec::new(),
            varargin_slot: None,
            implicit_nargin_slot: None,
            output_slots: Vec::new(),
            varargout_slot,
            implicit_nargout_slot: None,
            capture_slots: Vec::new(),
            argument_validations: Vec::new(),
        }
    }

    #[test]
    fn collect_outputs_zero_requested_does_not_consume_varargout() {
        let func = test_function(Some(0));
        let varargout = CellArray::new(vec![Value::Num(7.0)], 1, 1).expect("cell");
        let result_vars = vec![Value::Cell(varargout)];
        let outputs = collect_semantic_outputs(&func, &result_vars, 0).expect("collect");
        assert!(outputs.is_empty());
    }

    #[test]
    fn collect_outputs_one_requested_reads_varargout() {
        let func = test_function(Some(0));
        let varargout = CellArray::new(vec![Value::Num(7.0)], 1, 1).expect("cell");
        let result_vars = vec![Value::Cell(varargout)];
        let outputs = collect_semantic_outputs(&func, &result_vars, 1).expect("collect");
        assert_eq!(outputs, vec![Value::Num(7.0)]);
    }

    #[test]
    fn output_value_zero_requested_is_empty_output_list() {
        let value = output_value(vec![Value::Num(1.0)], 0);
        assert_eq!(value, Value::OutputList(Vec::new()));
    }

    #[test]
    fn output_value_multi_requested_returns_output_list() {
        let value = output_value(vec![Value::Num(1.0), Value::Num(2.0)], 2);
        assert_eq!(
            value,
            Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)])
        );
    }

    #[test]
    fn numeric_or_logical_validator_accepts_expected_domains() {
        assert!(value_is_numeric_or_logical(&Value::Num(1.0)));
        assert!(value_is_numeric_or_logical(&Value::Bool(true)));
        assert!(value_is_numeric_or_logical(&Value::Complex(1.0, 2.0)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_numeric_or_logical(&Value::Tensor(tensor)));
        assert!(!value_is_numeric_or_logical(&Value::String(
            "x".to_string()
        )));
        assert!(!value_is_numeric_or_logical(&Value::CharArray(
            runmat_builtins::CharArray::new("x".chars().collect(), 1, 1).expect("char")
        )));
    }

    #[test]
    fn text_validator_accepts_string_char_vector_and_cellstr() {
        assert!(value_is_text(&Value::String("x".to_string())));
        assert!(value_is_text(&Value::CharArray(
            runmat_builtins::CharArray::new("abc".chars().collect(), 1, 3).expect("char")
        )));
        assert!(value_is_text(&Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(
                        runmat_builtins::CharArray::new("a".chars().collect(), 1, 1).expect("char"),
                    ),
                    Value::String("b".to_string()),
                ],
                1,
                2,
            )
            .expect("cell"),
        )));
        assert!(!value_is_text(&Value::Num(1.0)));
    }

    #[test]
    fn nonempty_validator_rejects_empty_arrays_and_cells() {
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_empty(&Value::Tensor(empty_num)));
        let empty_char =
            runmat_builtins::CharArray::new(Vec::new(), 1, 0).expect("empty char array");
        assert!(value_is_empty(&Value::CharArray(empty_char)));
        let empty_cell = CellArray::new(Vec::new(), 0, 0).expect("empty cell");
        assert!(value_is_empty(&Value::Cell(empty_cell)));
        assert!(!value_is_empty(&Value::String("".to_string())));
        assert!(!value_is_empty(&Value::Num(1.0)));
    }

    #[test]
    fn scalar_or_empty_validator_accepts_scalar_or_empty_shapes() {
        assert!(value_is_scalar_or_empty(&Value::Num(1.0)));
        assert!(value_is_scalar_or_empty(&Value::Bool(true)));
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_scalar_or_empty(&Value::Tensor(empty_num)));
        let matrix = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("matrix");
        assert!(!value_is_scalar_or_empty(&Value::Tensor(matrix)));
    }

    #[test]
    fn real_validator_rejects_imaginary_values() {
        assert!(value_is_real(&Value::Num(1.0)));
        assert!(value_is_real(&Value::Complex(1.0, 0.0)));
        assert!(!value_is_real(&Value::Complex(1.0, 2.0)));
        let complex_real = runmat_builtins::ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1])
            .expect("complex tensor");
        let complex_imag = runmat_builtins::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1])
            .expect("complex tensor");
        assert!(value_is_real(&Value::ComplexTensor(complex_real)));
        assert!(!value_is_real(&Value::ComplexTensor(complex_imag)));
    }

    #[test]
    fn integer_validator_accepts_integer_valued_numeric_inputs() {
        assert!(value_is_integer(&Value::Int(
            runmat_builtins::IntValue::I64(3)
        )));
        assert!(value_is_integer(&Value::Num(3.0)));
        assert!(!value_is_integer(&Value::Num(3.5)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_integer(&Value::Tensor(tensor)));
        let non_integer = Tensor::new(vec![1.0, 2.5], vec![1, 2]).expect("tensor");
        assert!(!value_is_integer(&Value::Tensor(non_integer)));
        assert!(!value_is_integer(&Value::Bool(true)));
    }

    #[test]
    fn positive_validator_rejects_zero_and_negative_values() {
        assert!(value_is_positive(&Value::Num(1.0)));
        assert!(!value_is_positive(&Value::Num(0.0)));
        assert!(!value_is_positive(&Value::Num(-1.0)));
        assert!(value_is_positive(&Value::Int(
            runmat_builtins::IntValue::I64(2)
        )));
        assert!(!value_is_positive(&Value::Int(
            runmat_builtins::IntValue::I64(0)
        )));
        let positive = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_positive(&Value::Tensor(positive)));
        let mixed = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_positive(&Value::Tensor(mixed)));
    }

    #[test]
    fn negative_validator_rejects_zero_and_positive_values() {
        assert!(value_is_negative(&Value::Num(-1.0)));
        assert!(!value_is_negative(&Value::Num(0.0)));
        assert!(!value_is_negative(&Value::Num(1.0)));
        assert!(value_is_negative(&Value::Int(
            runmat_builtins::IntValue::I64(-2)
        )));
        let ok = Tensor::new(vec![-1.0, -2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_negative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![-1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_negative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonnegative_validator_accepts_zero_and_positive_values() {
        assert!(value_is_nonnegative(&Value::Num(0.0)));
        assert!(value_is_nonnegative(&Value::Num(2.0)));
        assert!(!value_is_nonnegative(&Value::Num(-1.0)));
        assert!(value_is_nonnegative(&Value::Int(
            runmat_builtins::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonnegative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonnegative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonzero_validator_rejects_zero_values() {
        assert!(value_is_nonzero(&Value::Num(1.0)));
        assert!(!value_is_nonzero(&Value::Num(0.0)));
        assert!(value_is_nonzero(&Value::Int(
            runmat_builtins::IntValue::I64(2)
        )));
        assert!(!value_is_nonzero(&Value::Int(
            runmat_builtins::IntValue::I64(0)
        )));
        assert!(value_is_nonzero(&Value::Complex(0.0, 1.0)));
        assert!(!value_is_nonzero(&Value::Complex(0.0, 0.0)));
        let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonzero(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonzero(&Value::Tensor(bad)));
    }

    #[test]
    fn nonpositive_validator_accepts_zero_and_negative_values() {
        assert!(value_is_nonpositive(&Value::Num(0.0)));
        assert!(value_is_nonpositive(&Value::Num(-2.0)));
        assert!(!value_is_nonpositive(&Value::Num(1.0)));
        assert!(value_is_nonpositive(&Value::Int(
            runmat_builtins::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonpositive(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonpositive(&Value::Tensor(bad)));
    }

    #[test]
    fn greater_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_greater_than_or_equal(&Value::Num(2.0), 0.0));
        assert!(value_is_greater_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_greater_than_or_equal(&Value::Num(-1.0), 0.0));
    }

    #[test]
    fn less_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_less_than_or_equal(&Value::Num(-1.0), 0.0));
        assert!(value_is_less_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_less_than_or_equal(&Value::Num(1.0), 0.0));
    }

    #[test]
    fn greater_than_and_less_than_validators_use_numeric_threshold() {
        assert!(value_is_greater_than(&Value::Num(2.0), 1.0));
        assert!(!value_is_greater_than(&Value::Num(1.0), 1.0));
        assert!(value_is_less_than(&Value::Num(-2.0), -1.0));
        assert!(!value_is_less_than(&Value::Num(-1.0), -1.0));
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn cancellation_clears_gpu_residency_for_live_values() {
        use runmat_accelerate::fusion_residency;
        use runmat_accelerate_api::GpuTensorHandle;

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 777_001,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let mut vars = vec![Value::GpuTensor(handle.clone())];
        let bytecode = Bytecode::with_instructions(vec![Instr::Return], vars.len());
        let cancelled = Arc::new(AtomicBool::new(true));
        let _interrupt_guard = runmat_runtime::interrupt::replace_interrupt(Some(cancelled));

        let err = block_on(interpret_with_vars(&bytecode, &mut vars, Some("<main>")))
            .expect_err("cancelled execution should return error");
        assert_eq!(err.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            !fusion_residency::is_resident(&handle),
            "cancelled execution should clear residency marks for live GPU handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn completion_clears_stack_only_gpu_residency() {
        use runmat_accelerate::fusion_residency;
        use runmat_accelerate_api::GpuTensorHandle;

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 777_002,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let bytecode = Bytecode::with_instructions(Vec::new(), 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let outcome = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(matches!(
            outcome,
            crate::interpreter::api::InterpreterOutcome::Completed(_)
        ));
        assert!(
            !fusion_residency::is_resident(&handle),
            "completion should clear residency marks for stack-only GPU handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_releases_stack_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![9.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "pop should clear residency for stack-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "pop should release provider storage for stack-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_preserves_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![11.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("interpreter should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "pop should preserve residency for handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "pop should not release provider storage for handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_local_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![15.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "exit scope should clear residency for local-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "exit scope should release provider storage for local-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![17.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "exit scope should preserve residency for handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_nested_handle_object_local_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![18.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.context.locals.push(Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        }));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete for nested handle-object local");
        assert!(
            !fusion_residency::is_resident(&handle),
            "exit scope should clear residency for nested handle-object local-only handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "exit scope should release provider storage for nested handle-object local-only handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![20.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let local_value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.context.locals.push(local_value.clone());
        state.vars = vec![local_value.clone()];

        let mut result_vars = vec![local_value];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("exit scope should complete for aliased nested handle-object local");
        assert!(
            fusion_residency::is_resident(&handle),
            "exit scope should preserve residency for nested handle-object handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for nested handle-object handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_other_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![19.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
        let mut seed_vars = vec![
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle.clone()),
        ];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle.clone()),
        ];

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in other vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in other vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![20.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::GpuTensor(handle.clone())];
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var should complete when alias lives in locals");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![22.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.vars = vec![Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        })];
        state.stack.push(Value::Num(0.0));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete for nested handle-object value");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store var overwrite should clear residency for nested handle-object handles when unaliased"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store var overwrite should release provider storage for nested handle-object handles when unaliased"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var()
    {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![24.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
        let mut seed_vars = vec![Value::Num(0.0), Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.vars = vec![nested.clone(), nested.clone()];
        state.stack.push(Value::Num(0.0));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete for aliased nested handle-object values");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in other vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in other vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![27.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.vars = vec![nested.clone()];
        state.stack.push(Value::Num(0.0));
        state.context.locals.push(nested);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store var overwrite should complete when alias lives in locals");
        assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![23.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::GpuTensor(handle.clone())];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for handles still live in vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_other_local() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![24.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 2,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local should complete when alias lives in other local");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for handles still live in other locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in other locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![25.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store local overwrite should clear residency for unaliased local handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store local overwrite should release provider storage for unaliased local handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![26.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        }));

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete for nested handle-object value");
        assert!(
            !fusion_residency::is_resident(&handle),
            "store local overwrite should clear residency for nested handle-object handles when unaliased"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "store local overwrite should release provider storage for nested handle-object handles when unaliased"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![28.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let local_value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(Value::Num(0.0));
        state.vars = vec![local_value.clone()];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals.push(local_value);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete for aliased nested handle-object value");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_local(
    ) {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![30.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode = Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(Value::Num(0.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 2,
                expected_outputs: 0,
            });
        state.context.locals.push(nested.clone());
        state.context.locals.push(nested);

        let mut result_vars = state.vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("store local overwrite should complete when alias lives in other local");
        assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in other locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in other locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_stack_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![21.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for stack-only handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for stack-only handle"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_preserves_provider_handle_when_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![31.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/await completion should preserve residency for live-var handle"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/await completion should not release provider storage for live-var handle"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_releases_stack_only_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![41.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop should complete");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/pop should clear residency for dropped spawned task payload"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/pop should release provider storage for dropped spawned task payload"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_preserves_provider_handle_when_payload_still_live_in_vars() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![51.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::GpuTensor(handle.clone())];

        let mut result_vars = vec![Value::GpuTensor(handle.clone())];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/pop should preserve residency for spawned payload handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/pop should not release provider storage for spawned payload handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_preserves_provider_handle_when_payload_still_live_in_locals() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![56.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::GpuTensor(handle.clone()));
        state.vars = vec![Value::Num(0.0)];
        state.context.locals.push(Value::GpuTensor(handle.clone()));

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop should complete");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/pop should preserve residency for spawned payload handles still referenced by locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/pop should not release provider storage for spawned payload handles still referenced by locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_releases_nested_closure_captured_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![61.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Closure(Closure {
            function_name: "worker".to_string(),
            bound_function: None,
            captures: vec![Value::GpuTensor(handle.clone())],
        }));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop should complete for closure payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/pop should clear residency for nested closure-captured payload handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/pop should release provider storage for nested closure-captured payload handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_nested_output_list_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![71.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state
            .stack
            .push(Value::OutputList(vec![Value::GpuTensor(handle.clone())]));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for nested output payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for nested output-list payload handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for nested output-list payload handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_nested_struct_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![81.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        state.stack.push(Value::Struct(payload));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for nested struct payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for nested struct payload handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for nested struct payload handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_nested_object_property_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![91.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = ObjectInstance::new("Payload".to_string());
        payload
            .properties
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        state.stack.push(Value::Object(payload));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for nested object payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for nested object-property payload handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for nested object-property payload handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_preserves_nested_object_property_handle_when_alias_live() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![101.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = ObjectInstance::new("Payload".to_string());
        payload
            .properties
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        state.stack.push(Value::Object(payload.clone()));
        state.vars = vec![Value::Object(payload.clone())];

        let mut result_vars = vec![Value::Object(payload)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for aliased nested object payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/await completion should preserve residency for nested object handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/await completion should not release provider storage for nested object handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_nested_cell_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![111.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let payload =
            CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell payload");
        state.stack.push(Value::Cell(payload));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for nested cell payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for nested cell payload handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for nested cell payload handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_preserves_nested_cell_handle_when_alias_live() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![121.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let payload =
            CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell payload");
        state.stack.push(Value::Cell(payload.clone()));
        state.vars = vec![Value::Cell(payload.clone())];

        let mut result_vars = vec![Value::Cell(payload)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for aliased nested cell payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/await completion should preserve residency for nested cell handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/await completion should not release provider storage for nested cell handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_releases_nested_handle_object_target_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![131.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let task_payload = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(task_payload);
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for nested handle-object payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/await completion should clear residency for nested handle-object target handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/await completion should release provider storage for nested handle-object target handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![141.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let task_payload = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(task_payload.clone());
        state.vars = vec![task_payload.clone()];

        let mut result_vars = vec![task_payload];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for aliased nested handle-object payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/await completion should preserve residency for nested handle-object target handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/await completion should not release provider storage for nested handle-object target handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live_in_locals(
    ) {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![146.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Await, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let task_payload = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(task_payload.clone());
        state.vars = vec![Value::Num(0.0)];
        state.context.locals.push(task_payload.clone());

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/await flow should complete for aliased nested handle-object payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/await completion should preserve residency for nested handle-object target handles still referenced by locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/await completion should not release provider storage for nested handle-object target handles still referenced by locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_releases_nested_handle_object_target_provider_handle() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![151.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        state.stack.push(Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        }));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop flow should complete for nested handle-object payload");
        assert!(
            !fusion_residency::is_resident(&handle),
            "spawn/pop should clear residency for nested handle-object target handles"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "spawn/pop should release provider storage for nested handle-object target handles"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![161.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let task_payload = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(task_payload.clone());
        state.vars = vec![task_payload.clone()];

        let mut result_vars = vec![task_payload];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop flow should complete for aliased nested handle-object payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/pop should preserve residency for nested handle-object target handles still referenced by vars"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/pop should not release provider storage for nested handle-object target handles still referenced by vars"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live_in_locals() {
        use runmat_accelerate::fusion_residency;

        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_provider_handle(vec![166.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        let bytecode =
            Bytecode::with_instructions(vec![Instr::Spawn, Instr::Pop, Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let task_payload = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        state.stack.push(task_payload.clone());
        state.vars = vec![Value::Num(0.0)];
        state.context.locals.push(task_payload);

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("spawn/pop flow should complete for aliased nested handle-object payload");
        assert!(
            fusion_residency::is_resident(&handle),
            "spawn/pop should preserve residency for nested handle-object target handles still referenced by locals"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "spawn/pop should not release provider storage for nested handle-object target handles still referenced by locals"
        );
        fusion_residency::clear(&handle);
        let _ = TEST_PROVIDER.free(&handle);
    }

    #[test]
    fn await_passes_through_non_spawn_value_operand() {
        let bytecode =
            Bytecode::with_instructions(vec![Instr::Await, Instr::StoreVar(0), Instr::Return], 1);
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(7.0));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("await should pass through non-task operand");
        assert_eq!(result_vars[0], Value::Num(7.0));
    }

    #[test]
    fn await_succeeds_after_spawn_handle_self_reassignment() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::Spawn,
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::Await,
                Instr::StoreVar(0),
                Instr::Return,
            ],
            1,
        );
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(9.0));
        state.vars = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("await should still succeed after self-reassignment of spawn handle");
        assert_eq!(result_vars[0], Value::Num(9.0));
    }

    #[test]
    fn await_succeeds_after_overwriting_one_spawn_handle_alias() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::Spawn,
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::StoreVar(1),
                Instr::LoadConst(0.0),
                Instr::StoreVar(0),
                Instr::LoadVar(1),
                Instr::Await,
                Instr::StoreVar(0),
                Instr::Return,
            ],
            2,
        );
        let mut seed_vars = vec![Value::Num(0.0), Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(9.0));
        state.vars = vec![Value::Num(0.0), Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0), Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("await should succeed when another alias still carries the spawn task handle");
        assert_eq!(result_vars[0], Value::Num(9.0));
    }

    #[test]
    fn await_succeeds_after_overwriting_one_local_spawn_handle_alias() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::Spawn,
                Instr::StoreLocal(0),
                Instr::LoadLocal(0),
                Instr::StoreLocal(1),
                Instr::LoadConst(0.0),
                Instr::StoreLocal(0),
                Instr::LoadLocal(1),
                Instr::Await,
                Instr::StoreVar(0),
                Instr::Return,
            ],
            1,
        );
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(9.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 2,
                expected_outputs: 0,
            });
        state.context.locals = vec![Value::Num(0.0), Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars)).expect(
            "await should succeed when another local alias still carries the spawn task handle",
        );
        assert_eq!(result_vars[0], Value::Num(9.0));
    }

    #[test]
    fn await_succeeds_after_overwriting_var_alias_when_local_spawn_handle_alias_live() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::Spawn,
                Instr::StoreLocal(0),
                Instr::LoadLocal(0),
                Instr::StoreVar(0),
                Instr::LoadConst(0.0),
                Instr::StoreVar(0),
                Instr::LoadLocal(0),
                Instr::Await,
                Instr::StoreVar(0),
                Instr::Return,
            ],
            1,
        );
        let mut seed_vars = vec![Value::Num(0.0)];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.stack.push(Value::Num(9.0));
        state.vars = vec![Value::Num(0.0)];
        state
            .context
            .call_stack
            .push(crate::bytecode::program::CallFrame {
                function_name: "<local>".to_string(),
                return_address: 0,
                locals_start: 0,
                locals_count: 1,
                expected_outputs: 0,
            });
        state.context.locals = vec![Value::Num(0.0)];

        let mut result_vars = vec![Value::Num(0.0)];
        let _ = block_on(run_interpreter_inner(state, &mut result_vars)).expect(
            "await should succeed when var alias is overwritten but local alias still carries the spawn task handle",
        );
        assert_eq!(result_vars[0], Value::Num(9.0));
    }

    #[test]
    fn await_succeeds_after_scope_exit_when_var_alias_keeps_spawn_task_id_live() {
        let mut task = runmat_builtins::StructValue::new();
        task.fields.insert(
            "__runmat_spawn_kind".to_string(),
            Value::String("task".to_string()),
        );
        task.fields.insert(
            "__runmat_spawn_id".to_string(),
            Value::Int(runmat_builtins::IntValue::U64(23)),
        );
        task.fields
            .insert("__runmat_spawn_payload".to_string(), Value::Num(4.0));
        let task_value = Value::Struct(task);

        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::ExitScope(1),
                Instr::LoadVar(0),
                Instr::Await,
                Instr::Return,
            ],
            1,
        );
        let mut seed_vars = vec![task_value.clone()];
        let mut state = InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
        state.context.locals.push(task_value);
        state.context.spawned_task_ids.insert(23);
        state.vars = seed_vars.clone();

        let mut result_vars = seed_vars.clone();
        let _ = block_on(run_interpreter_inner(state, &mut result_vars))
            .expect("await should succeed when var alias keeps the spawn task id live");
        assert!(
            matches!(result_vars[0], Value::Struct(_)),
            "await in this sequence does not overwrite var0"
        );
    }
}

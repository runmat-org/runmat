#![allow(
    clippy::await_holding_lock,
    clippy::enum_variant_names,
    clippy::get_first,
    clippy::io_other_error,
    clippy::needless_range_loop,
    clippy::redundant_closure,
    clippy::result_large_err,
    clippy::too_many_arguments,
    clippy::useless_conversion
)]
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use crate::builtins::common::format::format_variadic;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_gc_api::GcPtr;

pub mod dispatcher;

pub mod callsite;
pub mod console;
pub mod data;
pub mod interaction;
pub mod interrupt;
pub mod output_context;
pub mod output_count;
pub mod source_context;

pub mod arrays;
pub mod builtins;
pub mod comparison;
pub mod concatenation;
pub mod elementwise;
pub mod indexing;
pub mod matrix;
pub mod plotting_hooks;
pub mod replay;
pub mod runtime_error;
pub mod user_functions;
pub mod warning_store;
pub mod workspace;

/// Standard result type for runtime builtins.
pub type BuiltinResult<T> = Result<T, RuntimeError>;

pub const OBJECT_INDEX_PAREN: &str = "()";
pub const OBJECT_INDEX_BRACE: &str = "{}";
pub const OBJECT_INDEX_MEMBER: &str = ".";
pub const CALL_METHOD_BUILTIN_NAME: &str = "call_method";
pub const OBJECT_SUBSREF_METHOD: &str = "subsref";
pub const OBJECT_SUBSASGN_METHOD: &str = "subsasgn";
pub(crate) const IDENT_UNDEFINED_FUNCTION: &str = "RunMat:UndefinedFunction";

pub fn object_property_getter_name(field: &str) -> String {
    format!("get.{field}")
}

pub fn object_property_setter_name(field: &str) -> String {
    format!("set.{field}")
}

fn current_requested_outputs() -> usize {
    crate::output_count::current_output_count().unwrap_or(1)
}

fn undefined_callable_error(identity: &runmat_hir::CallableIdentity) -> RuntimeError {
    let detail = format!("Undefined function for callable identity {identity:?}");
    build_runtime_error(detail)
        .with_identifier(IDENT_UNDEFINED_FUNCTION)
        .build()
}

fn is_undefined_function_error(err: &RuntimeError) -> bool {
    err.identifier() == Some(IDENT_UNDEFINED_FUNCTION)
}

fn build_shape_checked_cell(
    values: Vec<Value>,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<runmat_builtins::CellArray, RuntimeError> {
    runmat_builtins::CellArray::new(values, rows, cols).map_err(|err| {
        build_runtime_error(format!("{context}: {err}"))
            .with_identifier("RunMat:ShapeMismatch")
            .build()
    })
}

fn runtime_descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    runtime_descriptor_error_with_message(builtin, error.message, error)
}

fn runtime_descriptor_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    runtime_descriptor_error_with_message(
        builtin,
        format!("{}: {}", error.message, detail.as_ref()),
        error,
    )
}

fn runtime_descriptor_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn object_receiver_class_name(receiver: &Value) -> Option<String> {
    match receiver {
        Value::Object(obj) => Some(obj.class_name.clone()),
        Value::HandleObject(handle) => {
            let target = unsafe { &*handle.target.as_raw() };
            Some(match target {
                Value::Object(obj) => obj.class_name.clone(),
                _ => handle.class_name.clone(),
            })
        }
        _ => None,
    }
}

fn class_member_identity(class_name: &str, member: &str) -> runmat_hir::CallableIdentity {
    runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
        runmat_hir::SymbolName(class_name.to_string()),
        runmat_hir::SymbolName(member.to_string()),
    ]))
}

pub(crate) fn qualified_name_segments(name: &str) -> Vec<runmat_hir::SymbolName> {
    name.split('.')
        .map(|segment| runmat_hir::SymbolName(segment.to_string()))
        .collect()
}

pub(crate) fn is_well_formed_qualified_name(name: &str) -> bool {
    let segments = qualified_name_segments(name);
    segments.len() > 1 && segments.iter().all(|segment| !segment.0.is_empty())
}

pub(crate) fn callable_identity_for_handle_name(
    name: &str,
) -> (
    runmat_hir::CallableIdentity,
    runmat_hir::CallableFallbackPolicy,
) {
    if is_well_formed_qualified_name(name) {
        let segments = qualified_name_segments(name);
        (
            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments)),
            runmat_hir::CallableFallbackPolicy::ExternalBoundary,
        )
    } else {
        (
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name.to_string())),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
        )
    }
}

pub(crate) fn external_callable_identity_for_name(name: &str) -> runmat_hir::CallableIdentity {
    if !is_well_formed_qualified_name(name) {
        runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
            runmat_hir::SymbolName(name.to_string()),
        ]))
    } else {
        let segments = qualified_name_segments(name);
        runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
    }
}

async fn dispatch_object_external_member(
    class_name: String,
    member: &str,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    dispatch_callable_with_policy(
        class_member_identity(&class_name, member),
        runmat_hir::CallableFallbackPolicy::ExternalBoundary,
        args,
        requested_outputs,
    )
    .await
}

async fn dispatch_named_with_requested_outputs(
    name: &str,
    args: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    call_builtin_async_with_outputs(name, args, requested_outputs).await
}

async fn dispatch_callable_with_policy(
    identity: runmat_hir::CallableIdentity,
    fallback_policy: runmat_hir::CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    let request = crate::user_functions::CallableRequest::resolved(
        identity.clone(),
        fallback_policy,
        args.clone(),
        requested_outputs,
    );
    if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await {
        return result;
    }

    if let Some(name) = fallback_policy.vm_fallback_name_for(&identity) {
        return dispatch_named_with_requested_outputs(&name, &args, requested_outputs).await;
    }

    Err(undefined_callable_error(&identity))
}

pub async fn call_feval_async_with_outputs(
    func_value: Value,
    args: &[Value],
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let _guard = crate::output_count::push_output_count(Some(requested_outputs));
    feval_builtin(func_value, args.to_vec()).await
}

pub use runtime_error::{
    build_runtime_error, replay_error, replay_error_with_source, CallFrame, ErrorContext,
    ReplayErrorKind, RuntimeError, RuntimeErrorBuilder,
};

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

// Ensure OpenBLAS is linked on non-macOS platforms when BLAS/LAPACK is enabled
#[cfg(all(feature = "blas-lapack", not(target_os = "macos")))]
extern crate openblas_src;

pub use dispatcher::{
    call_builtin, call_builtin_async, call_builtin_async_with_outputs, gather_if_needed,
    gather_if_needed_async, is_gpu_value, value_contains_gpu,
};

#[cfg(feature = "plot-core")]
pub use builtins::plotting::{
    export_figure_scene as runtime_plot_export_figure_scene,
    import_figure_scene_async as runtime_plot_import_figure_scene_async,
    import_figure_scene_from_path_async as runtime_plot_import_figure_scene_from_path_async,
};
pub use replay::{
    runtime_export_workspace_state, runtime_import_workspace_state, WorkspaceReplayMode,
};

pub use runmat_macros::{register_fusion_spec, register_gpu_spec};

// Pruned legacy re-exports; prefer builtins::* and explicit shims only
// Transitional root-level shims for widely used helpers
pub use arrays::create_range;
pub use concatenation::create_matrix_from_values;
pub use elementwise::{elementwise_div, elementwise_mul, elementwise_neg, elementwise_pow, power};
pub use indexing::perform_indexing;
// Explicitly re-export for external users of the VM that build matrices from values
// (kept above)
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

pub fn make_cell_with_shape(values: Vec<Value>, shape: Vec<usize>) -> Result<Value, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let ca = runmat_builtins::CellArray::new_with_shape(values, shape)
            .map_err(|e| format!("Cell creation error: {e}"))?;
        Ok(Value::Cell(ca))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let handles: Vec<GcPtr<Value>> = values
            .into_iter()
            .map(|v| runmat_gc::gc_allocate(v).map_err(|e| format!("Cell creation error: {e}")))
            .collect::<Result<_, _>>()?;
        let ca = runmat_builtins::CellArray::new_handles_with_shape(handles, shape)
            .map_err(|e| format!("Cell creation error: {e}"))?;
        Ok(Value::Cell(ca))
    }
}

pub(crate) fn make_cell(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, String> {
    make_cell_with_shape(values, vec![rows, cols])
}

fn to_string_scalar(v: &Value) -> Result<String, String> {
    let s: String = v.try_into()?;
    Ok(s)
}

fn to_string_array(v: &Value) -> Result<runmat_builtins::StringArray, String> {
    match v {
        Value::String(s) => runmat_builtins::StringArray::new(vec![s.clone()], vec![1, 1])
            .map_err(|e| e.to_string()),
        Value::StringArray(sa) => Ok(sa.clone()),
        Value::CharArray(ca) => {
            // Convert each row to a string; treat as column vector
            let mut out: Vec<String> = Vec::with_capacity(ca.rows);
            for r in 0..ca.rows {
                let mut s = String::with_capacity(ca.cols);
                for c in 0..ca.cols {
                    s.push(ca.data[r * ca.cols + c]);
                }
                out.push(s);
            }
            runmat_builtins::StringArray::new(out, vec![ca.rows, 1]).map_err(|e| e.to_string())
        }
        other => Err(format!("cannot convert to string array: {other:?}")),
    }
}

const STRTRIM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trimmed string scalar.",
}];

const STRTRIM_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String-like input value.",
}];

const STRTRIM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strtrim(text)",
    inputs: &STRTRIM_INPUTS,
    outputs: &STRTRIM_OUTPUT,
}];

pub const STRTRIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRTRIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "strtrim",
    descriptor(crate::STRTRIM_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn strtrim_builtin(a: Value) -> crate::BuiltinResult<Value> {
    let s = to_string_scalar(&a)?;
    Ok(Value::String(s.trim().to_string()))
}

// Adjust strjoin semantics: join rows (row-wise)
const STRJOIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Joined string array.",
}];

const STRJOIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text array.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter inserted between row elements.",
    },
];

const STRJOIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strjoin(text, delimiter)",
    inputs: &STRJOIN_INPUTS,
    outputs: &STRJOIN_OUTPUT,
}];

pub const STRJOIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRJOIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "strjoin",
    descriptor(crate::STRJOIN_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn strjoin_rowwise(a: Value, delim: Value) -> crate::BuiltinResult<Value> {
    let d = to_string_scalar(&delim)?;
    let sa = to_string_array(&a)?;
    let rows = *sa.shape.first().unwrap_or(&sa.data.len());
    let cols = *sa.shape.get(1).unwrap_or(&1);
    if rows == 0 || cols == 0 {
        return Ok(Value::StringArray(
            runmat_builtins::StringArray::new(Vec::new(), vec![0, 0]).unwrap(),
        ));
    }
    let mut out: Vec<String> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut s = String::new();
        for c in 0..cols {
            if c > 0 {
                s.push_str(&d);
            }
            s.push_str(&sa.data[r + c * rows]);
        }
        out.push(s);
    }
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(out, vec![rows, 1])
            .map_err(|e| format!("strjoin: {e}"))?,
    ))
}

// deal: distribute inputs to multiple outputs (via cell for expansion)
const DEAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distributed output values.",
}];

const DEAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values to distribute.",
}];

const DEAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = deal(varargin)",
    inputs: &DEAL_INPUTS,
    outputs: &DEAL_OUTPUT,
}];

pub const DEAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "deal",
    descriptor(crate::DEAL_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn deal_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count > 1 {
            return Ok(crate::output_count::output_list_with_padding(
                out_count, rest,
            ));
        }
    }
    // Return cell row vector of inputs for expansion
    let cols = rest.len();
    make_cell(rest, 1, cols).map_err(Into::into)
}

// Object/handle utilities used by interpreter lowering for OOP/func handles

const RETHROW_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const RETHROW_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error value to rethrow.",
}];

const RETHROW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "rethrow(err)",
    inputs: &RETHROW_INPUTS,
    outputs: &RETHROW_OUTPUT,
}];

pub const RETHROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RETHROW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "rethrow",
    descriptor(crate::RETHROW_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn rethrow_builtin(e: Value) -> crate::BuiltinResult<Value> {
    match e {
        Value::MException(me) => Err(build_runtime_error(me.message)
            .with_identifier(me.identifier)
            .build()),
        Value::String(s) => Err(build_runtime_error(s).build()),
        other => Err(build_runtime_error(format!("RunMat:error: {other:?}")).build()),
    }
}

const CALL_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Method return value(s).",
}];

const CALL_METHOD_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "base",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method arguments.",
    },
];

const CALL_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out] = call_method(base, method, varargin)",
    inputs: &CALL_METHOD_INPUTS,
    outputs: &CALL_METHOD_OUTPUT,
}];

const CALL_METHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CALL_METHOD.NAME_INVALID",
    identifier: Some("RunMat:CallMethodNameInvalid"),
    when: "The method name is empty or missing.",
    message: "call_method: method name must not be empty",
};

const CALL_METHOD_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CALL_METHOD.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "call_method: requires object receiver",
};

const CALL_METHOD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    CALL_METHOD_ERROR_NAME_INVALID,
    CALL_METHOD_ERROR_RECEIVER_INVALID,
];

pub const CALL_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &CALL_METHOD_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "call_method",
    descriptor(crate::CALL_METHOD_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn call_method_builtin(
    base: Value,
    method: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let method = method.trim().to_string();
    if method.is_empty() {
        return Err(runtime_descriptor_error(
            "call_method",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    match base {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = object_receiver_class_name(&receiver).ok_or_else(|| {
                runtime_descriptor_error("call_method", &CALL_METHOD_ERROR_RECEIVER_INVALID)
            })?;
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(receiver.clone());
            args.extend(rest);
            let requested_outputs = current_requested_outputs();
            match dispatch_object_external_member(
                class_name,
                &method,
                args.clone(),
                requested_outputs,
            )
            .await
            {
                Ok(v) => return Ok(v),
                Err(err) if is_undefined_function_error(&err) => {}
                Err(err) => return Err(err),
            }
            let (identity, fallback_policy) = callable_identity_for_handle_name(&method);
            dispatch_callable_with_policy(identity, fallback_policy, args, requested_outputs).await
        }
        other => Err(runtime_descriptor_error_with_detail(
            "call_method",
            &CALL_METHOD_ERROR_RECEIVER_INVALID,
            format!("unsupported receiver {other:?} for method '{method}'"),
        )),
    }
}

// Global dispatch helpers for overloaded indexing (subsref/subsasgn) to support fallback resolution paths
const SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object value.",
}];

const SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token ('()', '{}', '.').",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = subsasgn(obj, kind, payload, rhs)",
    inputs: &SUBSASGN_INPUTS,
    outputs: &SUBSASGN_OUTPUT,
}];

const SUBSASGN_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "subsasgn: requires object receiver",
};

const SUBSASGN_ERROR_METHOD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.METHOD_MISSING",
    identifier: Some("RunMat:MissingSubsasgn"),
    when: "Target class does not implement subsasgn.",
    message: "subsasgn: class does not define subsasgn for indexed assignment",
};

const SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SUBSASGN_ERROR_RECEIVER_INVALID,
    SUBSASGN_ERROR_METHOD_MISSING,
];

pub const SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &SUBSASGN_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "subsasgn",
    descriptor(crate::SUBSASGN_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn subsasgn_dispatch(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    match obj {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = object_receiver_class_name(&receiver).ok_or_else(|| {
                runtime_descriptor_error("subsasgn", &SUBSASGN_ERROR_RECEIVER_INVALID)
            })?;
            dispatch_object_external_member(
                class_name,
                OBJECT_SUBSASGN_METHOD,
                vec![receiver, Value::String(kind), payload, rhs],
                current_requested_outputs(),
            )
            .await
            .map_err(|err| {
                if is_undefined_function_error(&err) {
                    runtime_descriptor_error("subsasgn", &SUBSASGN_ERROR_METHOD_MISSING)
                } else {
                    err
                }
            })
        }
        other => Err(runtime_descriptor_error_with_detail(
            "subsasgn",
            &SUBSASGN_ERROR_RECEIVER_INVALID,
            format!("receiver must be object, got {other:?}"),
        )),
    }
}

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token ('()', '{}', '.').",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = subsref(obj, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

const SUBSREF_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "subsref: requires object receiver",
};

const SUBSREF_ERROR_METHOD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.METHOD_MISSING",
    identifier: Some("RunMat:MissingSubsref"),
    when: "Target class does not implement subsref.",
    message: "subsref: class does not define subsref for indexing operation",
};

const SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBSREF_ERROR_RECEIVER_INVALID, SUBSREF_ERROR_METHOD_MISSING];

pub const SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &SUBSREF_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "subsref",
    descriptor(crate::SUBSREF_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn subsref_dispatch(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    match obj {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = object_receiver_class_name(&receiver).ok_or_else(|| {
                runtime_descriptor_error("subsref", &SUBSREF_ERROR_RECEIVER_INVALID)
            })?;
            dispatch_object_external_member(
                class_name,
                OBJECT_SUBSREF_METHOD,
                vec![receiver, Value::String(kind), payload],
                current_requested_outputs(),
            )
            .await
            .map_err(|err| {
                if is_undefined_function_error(&err) {
                    runtime_descriptor_error("subsref", &SUBSREF_ERROR_METHOD_MISSING)
                } else {
                    err
                }
            })
        }
        other => Err(runtime_descriptor_error_with_detail(
            "subsref",
            &SUBSREF_ERROR_RECEIVER_INVALID,
            format!("receiver must be object, got {other:?}"),
        )),
    }
}

// -------- Handle classes & events --------

const NEW_HANDLE_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New handle-object instance.",
}];

const NEW_HANDLE_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for created object.",
}];

const NEW_HANDLE_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "handle = new_handle_object(class_name)",
        inputs: &NEW_HANDLE_OBJECT_INPUTS,
        outputs: &NEW_HANDLE_OBJECT_OUTPUT,
    }];

pub const NEW_HANDLE_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_HANDLE_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "new_handle_object",
    descriptor(crate::NEW_HANDLE_OBJECT_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn new_handle_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    // Create an underlying object instance and wrap it in a handle
    let obj = new_object_builtin(class_name.clone()).await?;
    let gc = runmat_gc::gc_allocate(obj).map_err(|e| format!("gc: {e}"))?;
    Ok(Value::HandleObject(runmat_builtins::HandleRef {
        class_name,
        target: gc,
        valid: true,
    }))
}

const ISVALID_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when handle/listener is valid.",
}];

const ISVALID_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to inspect.",
}];

const ISVALID_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvalid(value)",
    inputs: &ISVALID_INPUTS,
    outputs: &ISVALID_OUTPUT,
}];

pub const ISVALID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISVALID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "isvalid",
    descriptor(crate::ISVALID_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    match v {
        Value::HandleObject(h) => Ok(Value::Bool(h.valid)),
        Value::Listener(l) => Ok(Value::Bool(l.valid && l.enabled)),
        _ => Ok(Value::Bool(false)),
    }
}

use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct EventRegistry {
    next_id: u64,
    listeners: std::collections::HashMap<(usize, String), Vec<runmat_builtins::Listener>>,
}

static EVENT_REGISTRY: OnceLock<Mutex<EventRegistry>> = OnceLock::new();

fn events() -> &'static Mutex<EventRegistry> {
    EVENT_REGISTRY.get_or_init(|| Mutex::new(EventRegistry::default()))
}

pub(crate) fn canonicalize_callback_handle_for_semantic_resolution(callback: Value) -> Value {
    fn resolve_text_handle(text: &str) -> Option<Value> {
        let trimmed = text.trim();
        let rest = trimmed.strip_prefix('@')?;
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        let function = crate::user_functions::resolve_semantic_function_by_name(name)?;
        Some(Value::BoundFunctionHandle {
            name: name.to_string(),
            function,
        })
    }

    match callback {
        Value::String(text) => resolve_text_handle(&text).unwrap_or(Value::String(text)),
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = &array.data[0];
            resolve_text_handle(text).unwrap_or(Value::StringArray(array))
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            resolve_text_handle(&text).unwrap_or(Value::CharArray(chars))
        }
        Value::FunctionHandle(name) => {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name)
            {
                Value::BoundFunctionHandle { name, function }
            } else {
                Value::FunctionHandle(name)
            }
        }
        Value::ExternalFunctionHandle(name) => {
            if is_well_formed_qualified_name(&name) {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(&name)
                {
                    return Value::BoundFunctionHandle { name, function };
                }
            }
            Value::ExternalFunctionHandle(name)
        }
        Value::MethodFunctionHandle(name) => {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name)
            {
                Value::BoundFunctionHandle { name, function }
            } else {
                Value::MethodFunctionHandle(name)
            }
        }
        Value::Closure(mut closure) => {
            if closure.bound_function.is_none() {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(&closure.function_name)
                {
                    closure.bound_function = Some(function);
                }
            }
            Value::Closure(closure)
        }
        other => other,
    }
}

fn canonicalize_listener_callback(callback: Value) -> Value {
    canonicalize_callback_handle_for_semantic_resolution(callback)
}

const ADDLISTENER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "listener",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Created listener handle.",
}];

const ADDLISTENER_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target object or handle.",
    },
    BuiltinParamDescriptor {
        name: "event_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Event name.",
    },
    BuiltinParamDescriptor {
        name: "callback",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Callback handle or text handle.",
    },
];

const ADDLISTENER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "listener = addlistener(target, event_name, callback)",
    inputs: &ADDLISTENER_INPUTS,
    outputs: &ADDLISTENER_OUTPUT,
}];

const ADDLISTENER_ERROR_TARGET_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.TARGET_INVALID",
    identifier: Some("RunMat:AddListenerTargetInvalid"),
    when: "Target is not an object or handle object.",
    message: "addlistener: target must be handle or object",
};

const ADDLISTENER_ERRORS: [BuiltinErrorDescriptor; 1] = [ADDLISTENER_ERROR_TARGET_INVALID];

pub const ADDLISTENER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADDLISTENER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDLISTENER_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "addlistener",
    descriptor(crate::ADDLISTENER_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => {
            return Err(runtime_descriptor_error(
                "addlistener",
                &ADDLISTENER_ERROR_TARGET_INVALID,
            ))
        }
    };
    let mut reg = events().lock().unwrap();
    let id = {
        reg.next_id += 1;
        reg.next_id
    };
    let tgt_gc = match target {
        Value::HandleObject(h) => h.target,
        Value::Object(o) => {
            runmat_gc::gc_allocate(Value::Object(o)).map_err(|e| format!("gc: {e}"))?
        }
        _ => unreachable!(),
    };
    let callback = canonicalize_listener_callback(callback);
    let cb_gc = runmat_gc::gc_allocate(callback).map_err(|e| format!("gc: {e}"))?;
    let listener = runmat_builtins::Listener {
        id,
        target: tgt_gc,
        event_name: event_name.clone(),
        callback: cb_gc,
        enabled: true,
        valid: true,
    };
    reg.listeners
        .entry((key_ptr, event_name))
        .or_default()
        .push(listener.clone());
    Ok(Value::Listener(listener))
}

const NOTIFY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Always zero on success.",
}];

const NOTIFY_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target object or handle.",
    },
    BuiltinParamDescriptor {
        name: "event_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Event name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Event callback arguments.",
    },
];

const NOTIFY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = notify(target, event_name, varargin)",
    inputs: &NOTIFY_INPUTS,
    outputs: &NOTIFY_OUTPUT,
}];

const NOTIFY_ERROR_TARGET_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.TARGET_INVALID",
    identifier: Some("RunMat:NotifyTargetInvalid"),
    when: "Target is not an object or handle object.",
    message: "notify: target must be handle or object",
};

const NOTIFY_ERRORS: [BuiltinErrorDescriptor; 1] = [NOTIFY_ERROR_TARGET_INVALID];

pub const NOTIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NOTIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NOTIFY_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "notify",
    descriptor(crate::NOTIFY_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn notify_builtin(
    target: Value,
    event_name: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => {
            return Err(runtime_descriptor_error(
                "notify",
                &NOTIFY_ERROR_TARGET_INVALID,
            ))
        }
    };
    let mut to_call: Vec<runmat_builtins::Listener> = Vec::new();
    {
        let reg = events().lock().unwrap();
        if let Some(list) = reg.listeners.get(&(key_ptr, event_name.clone())) {
            for l in list {
                if l.valid && l.enabled {
                    to_call.push(l.clone());
                }
            }
        }
    }
    for l in to_call {
        // Call callback via feval-like protocol.
        let mut args = Vec::new();
        args.push(target.clone());
        args.extend(rest.iter().cloned());
        let cbv: Value = (*l.callback).clone();
        let should_dispatch = match &cbv {
            Value::String(s) => s.starts_with('@'),
            Value::StringArray(sa) => sa.data.len() == 1 && sa.data[0].starts_with('@'),
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                text.starts_with('@')
            }
            Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_) => true,
            _ => false,
        };
        if should_dispatch {
            let _ = call_feval_async_with_outputs(cbv.clone(), &args, 0).await?;
        }
    }
    Ok(Value::Num(0.0))
}

// Test-oriented dependent property handlers (global). If a class defines a Dependent
// property named 'p', the VM will try to call get.p / set.p. We provide generic
// implementations that read/write a conventional backing field 'p_backing'.
const GET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dependent property backing value.",
}];

const GET_P_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object receiver.",
}];

const GET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = get.p(obj)",
    inputs: &GET_P_INPUTS,
    outputs: &GET_P_OUTPUT,
}];

const GET_P_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET_P.RECEIVER_INVALID",
    identifier: Some("RunMat:GetPReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "get.p: requires object receiver",
};

const GET_P_ERRORS: [BuiltinErrorDescriptor; 1] = [GET_P_ERROR_RECEIVER_INVALID];

pub const GET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &GET_P_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "get.p",
    descriptor(crate::GET_P_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn get_p_builtin(obj: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            if let Some(v) = o.properties.get("p_backing") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        other => Err(runtime_descriptor_error_with_detail(
            "get.p",
            &GET_P_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

const SET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object receiver.",
}];

const SET_P_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to assign.",
    },
];

const SET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = set.p(obj, value)",
    inputs: &SET_P_INPUTS,
    outputs: &SET_P_OUTPUT,
}];

const SET_P_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET_P.RECEIVER_INVALID",
    identifier: Some("RunMat:SetPReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "set.p: requires object receiver",
};

const SET_P_ERRORS: [BuiltinErrorDescriptor; 1] = [SET_P_ERROR_RECEIVER_INVALID];

pub const SET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &SET_P_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "set.p",
    descriptor(crate::SET_P_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn set_p_builtin(obj: Value, val: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(mut o) => {
            o.properties.insert("p_backing".to_string(), val);
            Ok(Value::Object(o))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "set.p",
            &SET_P_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

const MAKE_ANON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle_text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Anonymous function text handle.",
}];

const MAKE_ANON_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Comma-separated parameter list.",
    },
    BuiltinParamDescriptor {
        name: "body",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Anonymous body expression text.",
    },
];

const MAKE_ANON_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "handle_text = make_anon(params, body)",
    inputs: &MAKE_ANON_INPUTS,
    outputs: &MAKE_ANON_OUTPUT,
}];

pub const MAKE_ANON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAKE_ANON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "make_anon",
    descriptor(crate::MAKE_ANON_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn make_anon_builtin(params: String, body: String) -> crate::BuiltinResult<Value> {
    Ok(Value::String(format!("@anon({params}) {body}")))
}

const NEW_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New object instance.",
}];

const NEW_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name.",
}];

const NEW_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = new_object(class_name)",
    inputs: &NEW_OBJECT_INPUTS,
    outputs: &NEW_OBJECT_OUTPUT,
}];

pub const NEW_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "new_object",
    descriptor(crate::NEW_OBJECT_DESCRIPTOR),
    builtin_path = "crate"
)]
pub(crate) async fn new_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    if let Some(def) = runmat_builtins::get_class(&class_name) {
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        // Walk up to root
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(cd) = runmat_builtins::get_class(&name) {
                chain.push(cd.clone());
                cursor = cd.parent.clone();
            } else {
                break;
            }
        }
        // Reverse to root-first
        chain.reverse();
        let mut obj = runmat_builtins::ObjectInstance::new(def.name.clone());
        // Apply defaults from root to leaf (leaf overrides effectively by later assignment)
        for cd in chain {
            for (k, p) in cd.properties.iter() {
                if !p.is_static {
                    if let Some(v) = &p.default_value {
                        obj.properties.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(
            class_name,
        )))
    }
}

// handle-object builtins removed for now

const CLASSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ref",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class reference value.",
}];

const CLASSREF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name.",
}];

const CLASSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = classref(class_name)",
    inputs: &CLASSREF_INPUTS,
    outputs: &CLASSREF_OUTPUT,
}];

pub const CLASSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "classref",
    descriptor(crate::CLASSREF_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn classref_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    Ok(Value::ClassRef(class_name))
}

const REGISTER_TEST_CLASSES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Registration status value.",
}];

const REGISTER_TEST_CLASSES_INPUTS: [BuiltinParamDescriptor; 0] = [];

const REGISTER_TEST_CLASSES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "status = __register_test_classes()",
        inputs: &REGISTER_TEST_CLASSES_INPUTS,
        outputs: &REGISTER_TEST_CLASSES_OUTPUT,
    }];

pub const REGISTER_TEST_CLASSES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGISTER_TEST_CLASSES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "__register_test_classes",
    descriptor(crate::REGISTER_TEST_CLASSES_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn register_test_classes_builtin() -> crate::BuiltinResult<Value> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "staticValue".to_string(),
        PropertyDef {
            name: "staticValue".to_string(),
            is_static: true,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(42.0)),
        },
    );
    props.insert(
        "secret".to_string(),
        PropertyDef {
            name: "secret".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Private,
            set_access: Access::Private,
            default_value: Some(Value::Num(99.0)),
        },
    );
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        "move".to_string(),
        MethodDef {
            name: "move".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Point.move".to_string(),
            implicit_class_argument: None,
        },
    );
    methods.insert(
        "origin".to_string(),
        MethodDef {
            name: "origin".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Point.origin".to_string(),
            implicit_class_argument: None,
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Point".to_string(),
        parent: None,
        properties: props,
        methods,
    });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    ns_props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(2.0)),
        },
    );
    let ns_methods = std::collections::HashMap::new();
    runmat_builtins::register_class(ClassDef {
        name: "pkg.PointNS".to_string(),
        parent: None,
        properties: ns_props,
        methods: ns_methods,
    });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Shape.area".to_string(),
            implicit_class_argument: None,
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Shape".to_string(),
        parent: None,
        properties: shape_props,
        methods: shape_methods,
    });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert(
        "r".to_string(),
        PropertyDef {
            name: "r".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    let mut circle_methods = std::collections::HashMap::new();
    circle_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Circle.area".to_string(),
            implicit_class_argument: None,
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Circle".to_string(),
        parent: Some("Shape".to_string()),
        properties: circle_props,
        methods: circle_methods,
    });

    // Constructor demo class: Ctor with static constructor method Ctor
    let ctor_props = std::collections::HashMap::new();
    let mut ctor_methods = std::collections::HashMap::new();
    ctor_methods.insert(
        "Ctor".to_string(),
        MethodDef {
            name: "Ctor".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Ctor.Ctor".to_string(),
            implicit_class_argument: None,
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Ctor".to_string(),
        parent: None,
        properties: ctor_props,
        methods: ctor_methods,
    });

    // Overloaded indexing demo class: OverIdx with subsref/subsasgn
    let overidx_props = std::collections::HashMap::new();
    let mut overidx_methods = std::collections::HashMap::new();
    overidx_methods.insert(
        OBJECT_SUBSREF_METHOD.to_string(),
        MethodDef {
            name: OBJECT_SUBSREF_METHOD.to_string(),
            is_static: false,
            access: Access::Public,
            function_name: format!("OverIdx.{OBJECT_SUBSREF_METHOD}"),
            implicit_class_argument: None,
        },
    );
    overidx_methods.insert(
        OBJECT_SUBSASGN_METHOD.to_string(),
        MethodDef {
            name: OBJECT_SUBSASGN_METHOD.to_string(),
            is_static: false,
            access: Access::Public,
            function_name: format!("OverIdx.{OBJECT_SUBSASGN_METHOD}"),
            implicit_class_argument: None,
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "OverIdx".to_string(),
        parent: None,
        properties: overidx_props,
        methods: overidx_methods,
    });

    // Class without indexing protocol methods, used by negative subsref/subsasgn contracts.
    runmat_builtins::register_class(ClassDef {
        name: "NoIdx".to_string(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    Ok(Value::Num(1.0))
}

#[cfg(feature = "test-classes")]
pub async fn test_register_classes() {
    let _ = register_test_classes_builtin().await;
}

// Example method implementation: Point.move(obj, dx, dy) -> updated obj
const POINT_MOVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated point object.",
}];

const POINT_MOVE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Point object receiver.",
    },
    BuiltinParamDescriptor {
        name: "dx",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X displacement.",
    },
    BuiltinParamDescriptor {
        name: "dy",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y displacement.",
    },
];

const POINT_MOVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Point.move(obj, dx, dy)",
    inputs: &POINT_MOVE_INPUTS,
    outputs: &POINT_MOVE_OUTPUT,
}];

const POINT_MOVE_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POINT_MOVE.RECEIVER_INVALID",
    identifier: Some("RunMat:PointMoveReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Point.move: requires object receiver",
};

const POINT_MOVE_ERRORS: [BuiltinErrorDescriptor; 1] = [POINT_MOVE_ERROR_RECEIVER_INVALID];

pub const POINT_MOVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_MOVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &POINT_MOVE_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "Point.move",
    descriptor(crate::POINT_MOVE_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn point_move_method(obj: Value, dx: f64, dy: f64) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(mut o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.properties.get("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.properties.get("y") {
                y = *v;
            }
            o.properties.insert("x".to_string(), Value::Num(x + dx));
            o.properties.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Point.move",
            &POINT_MOVE_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

const POINT_ORIGIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Origin point object.",
}];

const POINT_ORIGIN_INPUTS: [BuiltinParamDescriptor; 0] = [];

const POINT_ORIGIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Point.origin()",
    inputs: &POINT_ORIGIN_INPUTS,
    outputs: &POINT_ORIGIN_OUTPUT,
}];

pub const POINT_ORIGIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINT_ORIGIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Point.origin",
    descriptor(crate::POINT_ORIGIN_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn point_origin_method() -> crate::BuiltinResult<Value> {
    let mut o = runmat_builtins::ObjectInstance::new("Point".to_string());
    o.properties.insert("x".to_string(), Value::Num(0.0));
    o.properties.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o))
}

const SHAPE_AREA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "area",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Area value.",
}];

const SHAPE_AREA_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shape receiver.",
}];

const SHAPE_AREA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "area = Shape.area(obj)",
    inputs: &SHAPE_AREA_INPUTS,
    outputs: &SHAPE_AREA_OUTPUT,
}];

pub const SHAPE_AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHAPE_AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Shape.area",
    descriptor(crate::SHAPE_AREA_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn shape_area_method(_obj: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

const CIRCLE_AREA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "area",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Area value.",
}];

const CIRCLE_AREA_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Circle receiver.",
}];

const CIRCLE_AREA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "area = Circle.area(obj)",
    inputs: &CIRCLE_AREA_INPUTS,
    outputs: &CIRCLE_AREA_OUTPUT,
}];

const CIRCLE_AREA_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCLE_AREA.RECEIVER_INVALID",
    identifier: Some("RunMat:CircleAreaReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "Circle.area: requires object receiver",
};

const CIRCLE_AREA_ERRORS: [BuiltinErrorDescriptor; 1] = [CIRCLE_AREA_ERROR_RECEIVER_INVALID];

pub const CIRCLE_AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CIRCLE_AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &CIRCLE_AREA_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "Circle.area",
    descriptor(crate::CIRCLE_AREA_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn circle_area_method(obj: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            let r = if let Some(Value::Num(v)) = o.properties.get("r") {
                *v
            } else {
                0.0
            };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(runtime_descriptor_error_with_detail(
            "Circle.area",
            &CIRCLE_AREA_ERROR_RECEIVER_INVALID,
            format!("got {other:?}"),
        )),
    }
}

// --- Test-only helpers to validate constructors and subsref/subsasgn ---
const CTOR_CTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constructed Ctor object.",
}];

const CTOR_CTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Initial x value.",
}];

const CTOR_CTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = Ctor.Ctor(x)",
    inputs: &CTOR_CTOR_INPUTS,
    outputs: &CTOR_CTOR_OUTPUT,
}];

pub const CTOR_CTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CTOR_CTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "Ctor.Ctor",
    descriptor(crate::CTOR_CTOR_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn ctor_ctor_method(x: f64) -> crate::BuiltinResult<Value> {
    // Construct object with property 'x' initialized
    let mut o = runmat_builtins::ObjectInstance::new("Ctor".to_string());
    o.properties.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o))
}

// --- Test-only package functions to exercise import precedence ---
const PKGF_FOO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant test value.",
}];

const PKGF_FOO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const PKGF_FOO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = PkgF.foo()",
    inputs: &PKGF_FOO_INPUTS,
    outputs: &PKGF_FOO_OUTPUT,
}];

pub const PKGF_FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PKGF_FOO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "PkgF.foo",
    descriptor(crate::PKGF_FOO_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn pkgf_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(10.0))
}

const PKGG_FOO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant test value.",
}];

const PKGG_FOO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const PKGG_FOO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = PkgG.foo()",
    inputs: &PKGG_FOO_INPUTS,
    outputs: &PKGG_FOO_OUTPUT,
}];

pub const PKGG_FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PKGG_FOO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "PkgG.foo",
    descriptor(crate::PKGG_FOO_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn pkgg_foo() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(20.0))
}

const OVERIDX_SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value.",
}];

const OVERIDX_SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const OVERIDX_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.subsref(obj, kind, payload)",
    inputs: &OVERIDX_SUBSREF_INPUTS,
    outputs: &OVERIDX_SUBSREF_OUTPUT,
}];

const OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.SUBSREF_PAYLOAD_UNSUPPORTED",
    identifier: Some("RunMat:OverIdxSubsrefPayloadUnsupported"),
    when: "Indexing kind/payload combination is unsupported.",
    message: "OverIdx.subsref: unsupported payload",
};

const OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.SUBSASGN_PAYLOAD_UNSUPPORTED",
    identifier: Some("RunMat:OverIdxSubsasgnPayloadUnsupported"),
    when: "Indexing assignment kind/payload combination is unsupported.",
    message: "OverIdx.subsasgn: unsupported payload",
};

const OVERIDX_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.RECEIVER_INVALID",
    identifier: Some("RunMat:OverIdxReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "OverIdx: receiver must be object",
};

const OVERIDX_SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];

pub const OVERIDX_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_SUBSREF_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsref",
    descriptor(crate::OVERIDX_SUBSREF_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    // Simple sentinel implementation: return different values for '.' vs '()'
    match (obj, kind.as_str(), payload) {
        (Value::Object(_), OBJECT_INDEX_PAREN, Value::Cell(_)) => Ok(Value::Num(99.0)),
        (Value::Object(o), OBJECT_INDEX_BRACE, Value::Cell(_)) => {
            if let Some(v) = o.properties.get("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::String(field)) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsref",
            &OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
        )),
    }
}

const OVERIDX_SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object.",
}];

const OVERIDX_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const OVERIDX_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = OverIdx.subsasgn(obj, kind, payload, rhs)",
    inputs: &OVERIDX_SUBSASGN_INPUTS,
    outputs: &OVERIDX_SUBSASGN_OUTPUT,
}];

const OVERIDX_SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];

pub const OVERIDX_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_SUBSASGN_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.subsasgn",
    descriptor(crate::OVERIDX_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_subsasgn(
    mut obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    match (&mut obj, kind.as_str(), payload) {
        (Value::Object(o), OBJECT_INDEX_PAREN, Value::Cell(_)) => {
            // Store into 'last' property
            o.properties.insert("last".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_BRACE, Value::Cell(_)) => {
            o.properties.insert("lastCell".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::String(field)) => {
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), OBJECT_INDEX_MEMBER, Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        _ => Err(runtime_descriptor_error(
            "OverIdx.subsasgn",
            &OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
        )),
    }
}

fn overidx_expect_object(
    obj: Value,
    method: &str,
) -> crate::BuiltinResult<runmat_builtins::ObjectInstance> {
    match obj {
        Value::Object(o) => Ok(o),
        other => Err(runtime_descriptor_error_with_detail(
            "OverIdx",
            &OVERIDX_ERROR_RECEIVER_INVALID,
            format!("{method}: got {other:?}"),
        )),
    }
}

// --- Operator overloading methods for OverIdx (test scaffolding) ---
const OVERIDX_BINARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];

const OVERIDX_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object receiver.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right-hand operand.",
    },
];

const OVERIDX_UNARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];

const OVERIDX_UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "OverIdx object receiver.",
}];

const OVERIDX_PLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.plus(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_TIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.times(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MTIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mtimes(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_LT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.lt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_GT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.gt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_EQ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.eq(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_UPLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.uplus(obj)",
    inputs: &OVERIDX_UNARY_INPUTS,
    outputs: &OVERIDX_UNARY_OUTPUT,
}];

const OVERIDX_RDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.rdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MRDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mrdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_LDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.ldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_MLDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_AND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.and(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_OR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.or(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_XOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.xor(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];

const OVERIDX_OPERATOR_ERRORS: [BuiltinErrorDescriptor; 1] = [OVERIDX_ERROR_RECEIVER_INVALID];

pub const OVERIDX_PLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_PLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_TIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_TIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MTIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_LT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_GT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_GT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_EQ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_EQ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_UPLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_UPLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_RDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_RDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MRDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MRDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_LDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_MLDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MLDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_AND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_AND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_OR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_OR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

pub const OVERIDX_XOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_XOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &OVERIDX_OPERATOR_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.plus",
    descriptor(crate::OVERIDX_PLUS_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_plus(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.plus")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k + r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.times",
    descriptor(crate::OVERIDX_TIMES_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_times(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.times")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mtimes",
    descriptor(crate::OVERIDX_MTIMES_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_mtimes(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.mtimes")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.lt",
    descriptor(crate::OVERIDX_LT_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_lt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.lt")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k < r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.gt",
    descriptor(crate::OVERIDX_GT_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_gt(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.gt")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k > r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.eq",
    descriptor(crate::OVERIDX_EQ_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_eq(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.eq")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k - r).abs() < 1e-12 { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.uplus",
    descriptor(crate::OVERIDX_UPLUS_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_uplus(obj: Value) -> crate::BuiltinResult<Value> {
    // Identity
    Ok(obj)
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.rdivide",
    descriptor(crate::OVERIDX_RDIVIDE_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_rdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.rdivide")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k / r))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mrdivide",
    descriptor(crate::OVERIDX_MRDIVIDE_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_mrdivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_rdivide(obj, rhs).await
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.ldivide",
    descriptor(crate::OVERIDX_LDIVIDE_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_ldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.ldivide")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(r / k))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mldivide",
    descriptor(crate::OVERIDX_MLDIVIDE_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_mldivide(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    overidx_ldivide(obj, rhs).await
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.and",
    descriptor(crate::OVERIDX_AND_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_and(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.and")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) && (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.or",
    descriptor(crate::OVERIDX_OR_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_or(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.or")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) || (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.xor",
    descriptor(crate::OVERIDX_XOR_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn overidx_xor(obj: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let o = overidx_expect_object(obj, "OverIdx.xor")?;
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    let a = k != 0.0;
    let b = r != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
}

const FEVAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Function return value(s).",
}];

const FEVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle, handle text, closure, or object receiver.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Function call arguments.",
    },
];

const FEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = feval(f, varargin)",
    inputs: &FEVAL_INPUTS,
    outputs: &FEVAL_OUTPUT,
}];

const FEVAL_ERROR_HANDLE_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.HANDLE_NAME_INVALID",
    identifier: Some("RunMat:FevalHandleNameInvalid"),
    when: "A function or method handle name is empty.",
    message: "feval: function handle name must not be empty",
};

const FEVAL_ERROR_HANDLE_STRING_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.HANDLE_STRING_INVALID",
    identifier: Some("RunMat:FevalHandleStringInvalid"),
    when: "Text handle input does not start with '@'.",
    message: "feval: expected function handle string starting with '@'",
};

const FEVAL_ERROR_HANDLE_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.HANDLE_SHAPE_INVALID",
    identifier: Some("RunMat:FevalHandleShapeInvalid"),
    when: "Text handle input has invalid char/string array shape.",
    message: "feval: function handle text input must be scalar row text",
};

const FEVAL_ERROR_SEMANTIC_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.SEMANTIC_UNAVAILABLE",
    identifier: Some("RunMat:SemanticFunctionUnavailable"),
    when: "Semantic function identity cannot be invoked in current runtime state.",
    message: "feval: semantic function handle is unavailable",
};

const FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEVAL.FUNCTION_VALUE_UNSUPPORTED",
    identifier: Some("RunMat:FevalFunctionValueUnsupported"),
    when: "The first argument is not a supported callable value.",
    message: "feval: unsupported function value",
};

const FEVAL_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FEVAL_ERROR_HANDLE_NAME_INVALID,
    FEVAL_ERROR_HANDLE_STRING_INVALID,
    FEVAL_ERROR_HANDLE_SHAPE_INVALID,
    FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
    FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
];

pub const FEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEVAL_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "feval",
    descriptor(crate::FEVAL_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn feval_builtin(f: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    fn normalize_feval_handle_name(name: &str) -> Option<String> {
        let trimmed = name.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    async fn call_by_identity(
        identity: runmat_hir::CallableIdentity,
        fallback_policy: runmat_hir::CallableFallbackPolicy,
        args: &[Value],
        requested_outputs: usize,
    ) -> crate::BuiltinResult<Value> {
        dispatch_callable_with_policy(identity, fallback_policy, args.to_vec(), requested_outputs)
            .await
    }

    async fn call_by_name(
        name: &str,
        args: &[Value],
        requested_outputs: usize,
    ) -> crate::BuiltinResult<Value> {
        let normalized = normalize_feval_handle_name(name)
            .ok_or_else(|| runtime_descriptor_error("feval", &FEVAL_ERROR_HANDLE_NAME_INVALID))?;
        let (identity, fallback_policy) = callable_identity_for_handle_name(&normalized);
        call_by_identity(identity, fallback_policy, args, requested_outputs).await
    }

    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);

    match f {
        // Function handle strings like "@sin"
        Value::String(s) => {
            if let Some(name) = s.strip_prefix('@') {
                call_by_name(name, &rest, requested_outputs).await
            } else {
                Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_HANDLE_STRING_INVALID,
                    format!("got {s}"),
                ))
            }
        }
        // Also accept character row vector handles like '@max'
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                let s: String = ca.data.iter().collect();
                if let Some(name) = s.strip_prefix('@') {
                    call_by_name(name, &rest, requested_outputs).await
                } else {
                    Err(runtime_descriptor_error_with_detail(
                        "feval",
                        &FEVAL_ERROR_HANDLE_STRING_INVALID,
                        format!("got {s}"),
                    ))
                }
            } else {
                Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                    "char array must be a row vector",
                ))
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                let s = &sa.data[0];
                if let Some(name) = s.strip_prefix('@') {
                    call_by_name(name, &rest, requested_outputs).await
                } else {
                    Err(runtime_descriptor_error_with_detail(
                        "feval",
                        &FEVAL_ERROR_HANDLE_STRING_INVALID,
                        format!("got {s}"),
                    ))
                }
            } else {
                Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_HANDLE_SHAPE_INVALID,
                    "string array must be scalar",
                ))
            }
        }
        Value::FunctionHandle(name) => call_by_name(&name, &rest, requested_outputs).await,
        Value::ExternalFunctionHandle(name) => call_by_name(&name, &rest, requested_outputs).await,
        Value::MethodFunctionHandle(name) => {
            let method_name = name.trim().to_string();
            if method_name.is_empty() {
                return Err(runtime_descriptor_error(
                    "feval",
                    &FEVAL_ERROR_HANDLE_NAME_INVALID,
                ));
            }
            dispatch_callable_with_policy(
                runmat_hir::CallableIdentity::Method(runmat_hir::MethodId(method_name)),
                runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
                rest,
                requested_outputs,
            )
            .await
        }
        Value::BoundFunctionHandle { name, function } => {
            let request = crate::user_functions::CallableRequest::semantic(
                function,
                rest.clone(),
                requested_outputs,
            );
            if let Some(result) = crate::user_functions::try_call_semantic_descriptor(request).await
            {
                return result;
            }
            Err(runtime_descriptor_error_with_detail(
                "feval",
                &FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
                format!("semantic function handle '{name}' ({function}) is unavailable"),
            ))
        }
        Value::Closure(c) => {
            if let Some(function) = c.bound_function {
                let mut args = c.captures.clone();
                args.extend(rest);
                let request = crate::user_functions::CallableRequest::semantic(
                    function,
                    args.clone(),
                    requested_outputs,
                );
                if let Some(result) =
                    crate::user_functions::try_call_semantic_descriptor(request).await
                {
                    return result;
                }
                return Err(runtime_descriptor_error_with_detail(
                    "feval",
                    &FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
                    format!(
                        "semantic closure '{}' ({function}) is unavailable",
                        c.function_name
                    ),
                ));
            }

            if c.function_name == CALL_METHOD_BUILTIN_NAME && c.captures.len() >= 2 {
                let base = c.captures[0].clone();
                let method = match &c.captures[1] {
                    Value::String(name) => name.clone(),
                    Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect(),
                    _ => {
                        return Err(runtime_descriptor_error_with_detail(
                            "call_method",
                            &CALL_METHOD_ERROR_NAME_INVALID,
                            "closure captures must include method name text",
                        ))
                    }
                };
                let mut method_args = c.captures.iter().skip(2).cloned().collect::<Vec<_>>();
                method_args.extend(rest);
                return call_method_builtin(base, method, method_args).await;
            }

            let mut args = c.captures.clone();
            args.extend(rest);
            if let Some(function) =
                crate::user_functions::resolve_semantic_function_by_name(&c.function_name)
            {
                let request = crate::user_functions::CallableRequest::semantic(
                    function,
                    args.clone(),
                    requested_outputs,
                );
                if let Some(result) =
                    crate::user_functions::try_call_semantic_descriptor(request).await
                {
                    return result;
                }
            }
            call_by_name(&c.function_name, &args, requested_outputs).await
        }
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let payload = Value::Cell(build_shape_checked_cell(
                rest.clone(),
                1,
                rest.len(),
                "feval object index payload",
            )?);
            subsref_dispatch(receiver, OBJECT_INDEX_PAREN.to_string(), payload).await
        }
        other => Err(runtime_descriptor_error_with_detail(
            "feval",
            &FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}

const STR2FUNC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

const STR2FUNC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle text.",
}];

const STR2FUNC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = str2func(name)",
    inputs: &STR2FUNC_INPUTS,
    outputs: &STR2FUNC_OUTPUT,
}];

const STR2FUNC_ERROR_NAME_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2FUNC.NAME_SHAPE_INVALID",
    identifier: Some("RunMat:Str2FuncNameShapeInvalid"),
    when: "Input text value is not scalar row text.",
    message: "str2func: function name text input must be scalar row text",
};

const STR2FUNC_ERROR_NAME_TYPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2FUNC.NAME_TYPE_INVALID",
    identifier: Some("RunMat:Str2FuncNameTypeInvalid"),
    when: "Input is not string or char text.",
    message: "str2func: expected string/char function name",
};

const STR2FUNC_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2FUNC.NAME_INVALID",
    identifier: Some("RunMat:Str2FuncNameInvalid"),
    when: "Parsed function name is empty.",
    message: "str2func: function name must not be empty",
};

const STR2FUNC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STR2FUNC_ERROR_NAME_SHAPE_INVALID,
    STR2FUNC_ERROR_NAME_TYPE_INVALID,
    STR2FUNC_ERROR_NAME_INVALID,
];

pub const STR2FUNC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STR2FUNC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STR2FUNC_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "str2func",
    descriptor(crate::STR2FUNC_DESCRIPTOR),
    builtin_path = "crate"
)]
fn str2func_builtin(value: Value) -> crate::BuiltinResult<Value> {
    fn normalize_handle_name(text: &str) -> Option<String> {
        let trimmed = text.trim();
        let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    let name = match value {
        Value::String(text) => normalize_handle_name(&text),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_handle_name(&sa.data[0]),
        Value::StringArray(_) => {
            return Err(runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "string array must be scalar",
            ))
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_handle_name(&text)
        }
        Value::CharArray(_) => {
            return Err(runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "char array must be a row vector",
            ))
        }
        other => {
            return Err(runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_TYPE_INVALID,
                format!("got {other:?}"),
            ))
        }
    }
    .ok_or_else(|| runtime_descriptor_error("str2func", &STR2FUNC_ERROR_NAME_INVALID))?;

    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name) {
        Ok(Value::BoundFunctionHandle { name, function })
    } else if is_well_formed_qualified_name(&name) {
        Ok(Value::ExternalFunctionHandle(name))
    } else {
        Ok(Value::FunctionHandle(name))
    }
}

const FUNC2STR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function name string.",
}];

const FUNC2STR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

const FUNC2STR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = func2str(fh)",
    inputs: &FUNC2STR_INPUTS,
    outputs: &FUNC2STR_OUTPUT,
}];

const FUNC2STR_ERROR_HANDLE_TYPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FUNC2STR.HANDLE_TYPE_INVALID",
    identifier: Some("RunMat:Func2StrHandleTypeInvalid"),
    when: "Input is not a supported function handle value.",
    message: "func2str: expected function handle",
};

const FUNC2STR_ERRORS: [BuiltinErrorDescriptor; 1] = [FUNC2STR_ERROR_HANDLE_TYPE_INVALID];

pub const FUNC2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FUNC2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FUNC2STR_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "func2str",
    descriptor(crate::FUNC2STR_DESCRIPTOR),
    builtin_path = "crate"
)]
fn func2str_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name)
        | Value::BoundFunctionHandle { name, .. } => Ok(Value::String(name)),
        Value::Closure(closure) => Ok(Value::String(closure.function_name)),
        other => Err(runtime_descriptor_error_with_detail(
            "func2str",
            &FUNC2STR_ERROR_HANDLE_TYPE_INVALID,
            format!("got {other:?}"),
        )),
    }
}

// -------- Reductions: sum/prod/mean/any/all --------

#[allow(dead_code)]
fn tensor_sum_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().sum()
}

fn tensor_prod_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().product()
}

fn prod_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![1.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut p = 1.0;
                    for r in 0..rows {
                        p *= t.data[r + c * rows];
                    }
                    *oc = p;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("prod: {e}"))?,
                ))
            } else {
                Ok(Value::Num(tensor_prod_all(&t)))
            }
        }
        _ => Err("prod: expected tensor".to_string()),
    }
}

fn prod_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("prod: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![1.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut p = 1.0;
            for r in 0..rows {
                p *= t.data[r + c * rows];
            }
            *oc = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![1.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut p = 1.0;
            for c in 0..cols {
                p *= t.data[r + c * rows];
            }
            *orow = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else {
        Err("prod: dim out of range".to_string())
    }
}

const PROD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Product reduction result.",
}];

const PROD_SIG_INPUTS_1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const PROD_SIG_INPUTS_2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Reduction dimension.",
    },
];

const PROD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = prod(A)",
        inputs: &PROD_SIG_INPUTS_1,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, dim)",
        inputs: &PROD_SIG_INPUTS_2,
        outputs: &PROD_OUTPUT,
    },
];

pub const PROD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PROD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "prod",
    descriptor(crate::PROD_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn prod_var_builtin(a: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return (prod_all_or_cols(a)).map_err(Into::into);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return (prod_dim(a, *d)).map_err(Into::into),
            Value::Int(i) => return (prod_dim(a, i.to_i64() as f64)).map_err(Into::into),
            _ => {}
        }
    }
    Err(("prod: unsupported arguments".to_string()).into())
}

fn any_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 0.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] != 0.0 {
                            v = 1.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("any: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().any(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("any: expected tensor".to_string()),
    }
}

fn any_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("any: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 0.0;
            for r in 0..rows {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("any: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 0.0;
            for c in 0..cols {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("any: {e}"))?,
        ))
    } else {
        Err("any: dim out of range".to_string())
    }
}

const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Any-true reduction result.",
}];

const ANY_SIG_INPUTS_1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const ANY_SIG_INPUTS_2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Reduction dimension.",
    },
];

const ANY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = any(A)",
        inputs: &ANY_SIG_INPUTS_1,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = any(A, dim)",
        inputs: &ANY_SIG_INPUTS_2,
        outputs: &ANY_OUTPUT,
    },
];

pub const ANY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "any",
    descriptor(crate::ANY_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn any_var_builtin(a: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return (any_all_or_cols(a)).map_err(Into::into);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return (any_dim(a, *d)).map_err(Into::into),
            Value::Int(i) => return (any_dim(a, i.to_i64() as f64)).map_err(Into::into),
            _ => {}
        }
    }
    Err(("any: unsupported arguments".to_string()).into())
}

fn all_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 1.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] == 0.0 {
                            v = 0.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("all: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().all(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("all: expected tensor".to_string()),
    }
}

fn all_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("all: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 1.0;
            for r in 0..rows {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("all: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 1.0;
            for c in 0..cols {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("all: {e}"))?,
        ))
    } else {
        Err("all: dim out of range".to_string())
    }
}

const ALL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "All-true reduction result.",
}];

const ALL_SIG_INPUTS_1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const ALL_SIG_INPUTS_2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Reduction dimension.",
    },
];

const ALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = all(A)",
        inputs: &ALL_SIG_INPUTS_1,
        outputs: &ALL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = all(A, dim)",
        inputs: &ALL_SIG_INPUTS_2,
        outputs: &ALL_OUTPUT,
    },
];

pub const ALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "all",
    descriptor(crate::ALL_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn all_var_builtin(a: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return (all_all_or_cols(a)).map_err(Into::into);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return (all_dim(a, *d)).map_err(Into::into),
            Value::Int(i) => return (all_dim(a, i.to_i64() as f64)).map_err(Into::into),
            _ => {}
        }
    }
    Err(("all: unsupported arguments".to_string()).into())
}

const WARNING_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Always zero on success.",
}];

const WARNING_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning format string.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Format arguments.",
    },
];

const WARNING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = warning(fmt, varargin)",
    inputs: &WARNING_INPUTS,
    outputs: &WARNING_OUTPUT,
}];

pub const WARNING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WARNING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runmat_macros::runtime_builtin(
    name = "warning",
    sink = true,
    descriptor(crate::WARNING_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn warning_builtin(fmt: String, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let s = format_variadic(&fmt, &rest)?;
    tracing::warn!("Warning: {s}");
    Ok(Value::Num(0.0))
}

const GETMETHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bound method closure/handle.",
}];

const GETMETHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj_or_class",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver or class reference.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
];

const GETMETHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = getmethod(obj_or_class, name)",
    inputs: &GETMETHOD_INPUTS,
    outputs: &GETMETHOD_OUTPUT,
}];

const GETMETHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.NAME_INVALID",
    identifier: Some("RunMat:GetMethodNameInvalid"),
    when: "Method name is empty.",
    message: "getmethod: method name must not be empty",
};

const GETMETHOD_ERROR_RECEIVER_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.RECEIVER_UNSUPPORTED",
    identifier: Some("RunMat:GetMethodReceiverUnsupported"),
    when: "Receiver is neither object nor class reference.",
    message: "getmethod: unsupported receiver",
};

const GETMETHOD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    GETMETHOD_ERROR_NAME_INVALID,
    GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
];

pub const GETMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETMETHOD_ERRORS,
};

#[runmat_macros::runtime_builtin(
    name = "getmethod",
    descriptor(crate::GETMETHOD_DESCRIPTOR),
    builtin_path = "crate"
)]
async fn getmethod_builtin(obj: Value, name: String) -> crate::BuiltinResult<Value> {
    let method_name = name.trim();
    if method_name.is_empty() {
        return Err(runtime_descriptor_error(
            "getmethod",
            &GETMETHOD_ERROR_NAME_INVALID,
        ));
    }
    match obj {
        Value::Object(o) => {
            // Return a closure capturing the receiver; feval will call runtime builtin call_method
            Ok(Value::Closure(runmat_builtins::Closure {
                function_name: CALL_METHOD_BUILTIN_NAME.to_string(),
                bound_function: None,
                captures: vec![Value::Object(o), Value::String(method_name.to_string())],
            }))
        }
        Value::ClassRef(cls) => str2func_builtin(Value::String(format!("@{cls}.{method_name}"))),
        other => Err(runtime_descriptor_error_with_detail(
            "getmethod",
            &GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{register_class, Access, ClassDef, PropertyDef};
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    };

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn descriptor_migration_covers_lib_runtime_builtins() {
        let cases = [
            ("deal", "[varargout] = deal(varargin)"),
            ("rethrow", "rethrow(err)"),
            ("call_method", "[out] = call_method(base, method, varargin)"),
            ("subsasgn", "obj = subsasgn(obj, kind, payload, rhs)"),
            ("subsref", "out = subsref(obj, kind, payload)"),
            (
                "new_handle_object",
                "handle = new_handle_object(class_name)",
            ),
            (
                "addlistener",
                "listener = addlistener(target, event_name, callback)",
            ),
            ("notify", "status = notify(target, event_name, varargin)"),
            ("get.p", "value = get.p(obj)"),
            ("set.p", "obj = set.p(obj, value)"),
            ("make_anon", "handle_text = make_anon(params, body)"),
            ("new_object", "obj = new_object(class_name)"),
            ("classref", "ref = classref(class_name)"),
            (
                "__register_test_classes",
                "status = __register_test_classes()",
            ),
            ("Point.move", "obj = Point.move(obj, dx, dy)"),
            ("Circle.area", "area = Circle.area(obj)"),
            ("Ctor.Ctor", "obj = Ctor.Ctor(x)"),
            ("PkgF.foo", "value = PkgF.foo()"),
            ("OverIdx.plus", "out = OverIdx.plus(obj, rhs)"),
            (
                "OverIdx.subsref",
                "out = OverIdx.subsref(obj, kind, payload)",
            ),
            ("feval", "[varargout] = feval(f, varargin)"),
            ("str2func", "fh = str2func(name)"),
            ("func2str", "name = func2str(fh)"),
            ("getmethod", "fh = getmethod(obj_or_class, name)"),
        ];

        for (name, label) in cases {
            let builtin = runmat_builtins::builtin_function_by_name(name)
                .unwrap_or_else(|| panic!("builtin {name} not registered"));
            let descriptor = builtin
                .descriptor
                .unwrap_or_else(|| panic!("descriptor missing for {name}"));
            assert!(
                descriptor.signatures.iter().any(|sig| sig.label == label),
                "missing signature {label} for {name}"
            );
        }
    }

    #[test]
    fn feval_closure_uses_semantic_function_identity() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 42);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(7.0)) })
            },
        )));
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: "function_target".to_string(),
            bound_function: Some(42),
            captures: Vec::new(),
        });

        let result = block_on(feval_builtin(closure, vec![Value::Num(2.0)]))
            .expect("semantic closure feval succeeds");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn feval_semantic_function_handle_uses_semantic_identity() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 43);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(3.0)]);
                Box::pin(async { Ok(Value::Num(9.0)) })
            },
        )));
        let handle = Value::BoundFunctionHandle {
            name: "function_target".to_string(),
            function: 43,
        };

        let result = block_on(feval_builtin(handle, vec![Value::Num(3.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(result, Value::Num(9.0));
    }

    #[test]
    fn feval_semantic_function_handle_errors_when_semantic_invoker_unavailable() {
        let _guard = crate::user_functions::install_semantic_function_invoker(None);
        let handle = Value::BoundFunctionHandle {
            name: "function_target".to_string(),
            function: 9043,
        };

        let err = block_on(feval_builtin(handle, vec![Value::Num(3.0)])).expect_err(
            "semantic function handle should not fall back to name-based dispatch when unavailable",
        );
        assert_eq!(err.identifier(), Some("RunMat:SemanticFunctionUnavailable"));
        assert!(
            err.message()
                .contains("semantic function handle 'function_target' (9043) is unavailable"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn feval_name_only_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let result = block_on(feval_builtin(
            Value::FunctionHandle("resolved_target".to_string()),
            vec![Value::Num(4.0)],
        ))
        .expect("resolved name-only handle feval succeeds");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn feval_method_function_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_method").then_some(5045)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 5045);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(15.0)) })
            }),
        ));

        let result = block_on(feval_builtin(
            Value::MethodFunctionHandle("resolved_method".to_string()),
            vec![Value::Num(4.0)],
        ))
        .expect("resolved method handle feval succeeds");
        assert_eq!(result, Value::Num(15.0));
    }

    #[test]
    fn feval_method_function_handle_does_not_fallback_to_builtin_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let err = block_on(feval_builtin(
            Value::MethodFunctionHandle("sqrt".to_string()),
            vec![Value::Num(9.0)],
        ))
        .expect_err("method function handle should not fallback to builtin name dispatch");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn feval_name_only_closure_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(145)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 145);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(9.0), Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(13.0)) })
            }),
        ));

        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: "resolved_target".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        });

        let result = block_on(feval_builtin(closure, vec![Value::Num(4.0)]))
            .expect("resolved name-only closure feval succeeds");
        assert_eq!(result, Value::Num(13.0));
    }

    #[test]
    fn feval_name_only_closure_falls_back_when_semantic_invoker_unavailable() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "sin").then_some(245)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(None);

        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: "sin".to_string(),
            bound_function: None,
            captures: Vec::new(),
        });

        let result =
            block_on(feval_builtin(closure, vec![Value::Num(0.0)])).expect("sin fallback works");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn feval_external_function_handle_errors_when_unresolved() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let err = block_on(feval_builtin(
            Value::ExternalFunctionHandle("missing.external".to_string()),
            vec![Value::Num(1.0)],
        ))
        .expect_err("external function handle should error when unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("missing.external"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn feval_single_segment_external_function_handle_uses_runtime_name_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(4501)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 4501);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(12.0)) })
            }),
        ));

        let result = block_on(feval_builtin(
            Value::ExternalFunctionHandle("resolved_target".to_string()),
            vec![Value::Num(4.0)],
        ))
        .expect("single-segment external function handle should use runtime-name resolution");
        assert_eq!(result, Value::Num(12.0));
    }

    #[test]
    fn feval_rejects_string_without_at_with_identifier() {
        let err = block_on(feval_builtin(
            Value::String("sin".to_string()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval string handle without @ should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleStringInvalid"));
    }

    #[test]
    fn feval_rejects_char_handle_without_at_with_identifier() {
        let err = block_on(feval_builtin(
            Value::CharArray(runmat_builtins::CharArray::new_row("sin")),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval char handle without @ should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleStringInvalid"));
    }

    #[test]
    fn feval_rejects_non_row_char_handle_with_identifier() {
        let chars = runmat_builtins::CharArray::new(vec!['@', 's'], 2, 1)
            .expect("char array construction should succeed");
        let err = block_on(feval_builtin(
            Value::CharArray(chars),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval non-row char handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleShapeInvalid"));
    }

    #[test]
    fn feval_rejects_empty_at_string_handle_with_identifier() {
        let err = block_on(feval_builtin(
            Value::String("@".to_string()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval empty @string handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleNameInvalid"));
    }

    #[test]
    fn feval_rejects_empty_function_handle_value_with_identifier() {
        let err = block_on(feval_builtin(
            Value::FunctionHandle(String::new()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval empty function-handle value should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleNameInvalid"));
    }

    #[test]
    fn feval_trims_text_handle_name_for_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(9876)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9876);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(12.0)) })
            }),
        ));

        let value = block_on(feval_builtin(
            Value::String("@ resolved_target ".to_string()),
            vec![Value::Num(4.0)],
        ))
        .expect("trimmed text handle should resolve");
        assert_eq!(value, Value::Num(12.0));
    }

    #[test]
    fn str2func_returns_semantic_handle_when_resolver_can_resolve() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(145)
            })));
        let value = str2func_builtin(Value::String("resolved_target".to_string()))
            .expect("str2func should succeed");
        assert_eq!(
            value,
            Value::BoundFunctionHandle {
                name: "resolved_target".to_string(),
                function: 145,
            }
        );
    }

    #[test]
    fn str2func_returns_dynamic_handle_when_resolver_cannot_resolve() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::String("@missing_target".to_string()))
            .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("missing_target".to_string()));
    }

    #[test]
    fn str2func_returns_external_handle_for_qualified_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::String("Point.origin".to_string()))
            .expect("str2func should succeed");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn str2func_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::String("Point..origin".to_string()))
            .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".to_string()));
    }

    #[test]
    fn func2str_rejects_non_handle_with_identifier() {
        let err =
            func2str_builtin(Value::Num(1.0)).expect_err("func2str non-handle input should fail");
        assert_eq!(err.identifier(), Some("RunMat:Func2StrHandleTypeInvalid"));
    }

    #[test]
    fn str2func_rejects_empty_name_with_identifier() {
        let err = str2func_builtin(Value::String("   ".to_string()))
            .expect_err("empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_rejects_non_row_char_name_with_identifier() {
        let chars = runmat_builtins::CharArray::new(vec!['a', 'b'], 2, 1)
            .expect("char array construction should succeed");
        let err = str2func_builtin(Value::CharArray(chars))
            .expect_err("non-row char-array function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_rejects_non_text_name_with_identifier() {
        let err =
            str2func_builtin(Value::Num(1.0)).expect_err("non-text function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
    }

    #[test]
    fn str2func_accepts_scalar_string_array_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["@missing_target".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect("scalar string-array function name should succeed");
        assert_eq!(value, Value::FunctionHandle("missing_target".to_string()));
    }

    #[test]
    fn str2func_rejects_nonscalar_string_array_name_with_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = Value::StringArray(
            runmat_builtins::StringArray::new(vec!["@a".to_string(), "@b".to_string()], vec![1, 2])
                .expect("string array construction should succeed"),
        );
        let err =
            str2func_builtin(value).expect_err("nonscalar string-array function name must fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(445)
            })));
        let value = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["@resolved_target".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect("scalar string-array function name should resolve semantically");
        assert_eq!(
            value,
            Value::BoundFunctionHandle {
                name: "resolved_target".to_string(),
                function: 445,
            }
        );
    }

    #[test]
    fn str2func_scalar_string_array_returns_external_handle_for_qualified_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["Point.origin".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect("scalar string-array qualified name should succeed");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn str2func_scalar_string_array_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["Point..origin".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect("scalar string-array malformed qualified name should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".to_string()));
    }

    #[test]
    fn str2func_scalar_string_array_rejects_empty_name_with_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let err = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["   ".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect_err("scalar string-array empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_qualified_name_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.resolved_target").then_some(446)
            })));
        let value = str2func_builtin(Value::StringArray(
            runmat_builtins::StringArray::new(vec!["@pkg.resolved_target".to_string()], vec![1, 1])
                .expect("string array construction should succeed"),
        ))
        .expect("scalar string-array qualified function name should resolve semantically");
        assert_eq!(
            value,
            Value::BoundFunctionHandle {
                name: "pkg.resolved_target".to_string(),
                function: 446,
            }
        );
    }

    #[test]
    fn getmethod_classref_returns_typed_external_function_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = block_on(getmethod_builtin(
            Value::ClassRef("Point".to_string()),
            "origin".to_string(),
        ))
        .expect("getmethod should resolve classref method handle");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn getmethod_rejects_empty_method_name() {
        let err = block_on(getmethod_builtin(
            Value::ClassRef("Point".to_string()),
            "   ".to_string(),
        ))
        .expect_err("empty method name should be rejected");
        assert_eq!(err.identifier(), Some("RunMat:GetMethodNameInvalid"));
    }

    #[test]
    fn getmethod_rejects_unsupported_receiver_with_identifier() {
        let err = block_on(getmethod_builtin(Value::Num(1.0), "origin".to_string()))
            .expect_err("unsupported receiver should be rejected");
        assert_eq!(
            err.identifier(),
            Some("RunMat:GetMethodReceiverUnsupported")
        );
    }

    #[test]
    fn new_object_builtin_handles_class_parent_cycles() {
        let class_a = unique_class_name("runtime_ctor_cycle_a");
        let class_b = unique_class_name("runtime_ctor_cycle_b");

        let mut props_a = HashMap::new();
        props_a.insert(
            "fromA".to_string(),
            PropertyDef {
                name: "fromA".to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        let mut props_b = HashMap::new();
        props_b.insert(
            "fromB".to_string(),
            PropertyDef {
                name: "fromB".to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::Num(2.0)),
            },
        );

        register_class(ClassDef {
            name: class_a.clone(),
            parent: Some(class_b.clone()),
            properties: props_a,
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: class_b,
            parent: Some(class_a.clone()),
            properties: props_b,
            methods: HashMap::new(),
        });

        let value = block_on(new_object_builtin(class_a.clone()))
            .expect("constructor should terminate under parent-cycle metadata");
        let Value::Object(obj) = value else {
            panic!("expected object result");
        };
        assert_eq!(obj.class_name, class_a);
        assert_eq!(obj.properties.get("fromA"), Some(&Value::Num(1.0)));
        assert_eq!(obj.properties.get("fromB"), Some(&Value::Num(2.0)));
    }

    #[test]
    fn callable_identity_for_malformed_handle_name_stays_dynamic() {
        let (identity, fallback_policy) = callable_identity_for_handle_name("pkg..remote_inc");
        assert!(matches!(
            identity,
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name))
                if name == "pkg..remote_inc"
        ));
        assert_eq!(
            fallback_policy,
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
        );
    }

    #[test]
    fn unresolved_callable_without_display_name_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_hir::CallableIdentity::AnonymousFunction(runmat_hir::FunctionId(77)),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![],
            1,
        ))
        .expect_err("anonymous callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("AnonymousFunction(FunctionId(77))"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn unresolved_malformed_external_callable_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ])),
            runmat_hir::CallableFallbackPolicy::ExternalBoundary,
            vec![],
            1,
        ))
        .expect_err("malformed external callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message()
                .contains("ExternalName(QualifiedName([SymbolName(\"pkg\"), SymbolName(\"\"), SymbolName(\"remote\")]))"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn unresolved_method_callable_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_hir::CallableIdentity::Method(runmat_hir::MethodId(
                "missing_method".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![],
            1,
        ))
        .expect_err("method callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message()
                .contains("Method(MethodId(\"missing_method\"))"),
            "unexpected error: {err:?}"
        );
        assert!(
            !err.message()
                .contains("Undefined function 'missing_method'"),
            "method identity should not use fallback display-name text: {err:?}"
        );
    }

    #[test]
    fn feval_qualified_at_handle_errors_as_unresolved_external() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let err = block_on(feval_builtin(
            Value::String("@missing.external".to_string()),
            vec![Value::Num(1.0)],
        ))
        .expect_err("qualified @handle should error when unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("missing.external"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn func2str_extracts_name_from_function_handles() {
        assert_eq!(
            func2str_builtin(Value::FunctionHandle("sin".to_string())).expect("func2str"),
            Value::String("sin".to_string())
        );
        assert_eq!(
            func2str_builtin(Value::ExternalFunctionHandle("Point.origin".to_string()))
                .expect("func2str"),
            Value::String("Point.origin".to_string())
        );
        assert_eq!(
            func2str_builtin(Value::BoundFunctionHandle {
                name: "local_fn".to_string(),
                function: 44,
            })
            .expect("func2str"),
            Value::String("local_fn".to_string())
        );
        assert_eq!(
            func2str_builtin(Value::Closure(runmat_builtins::Closure {
                function_name: "captured_fn".to_string(),
                bound_function: None,
                captures: Vec::new(),
            }))
            .expect("func2str"),
            Value::String("captured_fn".to_string())
        );
    }

    #[test]
    fn none_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::None,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("runtime resolution should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn object_dispatch_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::ObjectDispatch,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn external_name_runtime_name_resolution_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("resolved_target".to_string()),
            ])),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn external_boundary_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("resolved_target".to_string()),
            ])),
            runmat_hir::CallableFallbackPolicy::ExternalBoundary,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("external boundary policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn external_boundary_policy_malformed_external_identity_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg..resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(vec![
                runmat_hir::SymbolName("pkg..resolved_target".to_string()),
            ])),
            runmat_hir::CallableFallbackPolicy::ExternalBoundary,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn runtime_name_resolution_policy_uses_semantic_resolver_after_object_probe() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("post-object-probe runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn method_identity_runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::Method(runmat_hir::MethodId(
                "resolved_target".to_string(),
            )),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("method runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn imported_identity_runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "Point.origin").then_some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::Imported(runmat_hir::DefPath {
                package: runmat_hir::PackageName("Point".to_string()),
                module: runmat_hir::QualifiedName(vec![
                    runmat_hir::SymbolName("Point".to_string()),
                    runmat_hir::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_hir::DefPathSegment::Function(
                    runmat_hir::SymbolName("origin".to_string()),
                )],
            }),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("imported runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn imported_identity_runtime_name_resolution_policy_rejects_malformed_path_without_semantic_probe(
    ) {
        let resolver_calls = Arc::new(AtomicUsize::new(0));
        let resolver_calls_for_closure = Arc::clone(&resolver_calls);
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(move |_| {
                resolver_calls_for_closure.fetch_add(1, Ordering::Relaxed);
                Some(45)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(11.0)) })
            }),
        ));

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_hir::CallableIdentity::Imported(runmat_hir::DefPath {
                package: runmat_hir::PackageName("Point".to_string()),
                module: runmat_hir::QualifiedName(vec![
                    runmat_hir::SymbolName("Point".to_string()),
                    runmat_hir::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_hir::DefPathSegment::Function(
                    runmat_hir::SymbolName("other".to_string()),
                )],
            }),
            runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(
            result.is_none(),
            "mismatched imported identity should not attempt semantic resolver"
        );
        assert_eq!(
            resolver_calls.load(Ordering::Relaxed),
            0,
            "malformed imported identity should be rejected before resolver probe"
        );
    }

    #[test]
    fn call_method_fallback_preserves_requested_outputs() {
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let base = Value::Object(runmat_builtins::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let result = block_on(call_method_builtin(
            base.clone(),
            "deal".to_string(),
            vec![Value::Num(9.0), Value::Num(10.0)],
        ))
        .expect("call_method fallback should succeed");
        match result {
            Value::OutputList(values) => {
                assert!(values.len() >= 2);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
            }
            other => {
                panic!("expected output list from multi-output call_method fallback, got {other:?}")
            }
        }
    }

    #[test]
    fn call_method_trims_method_name_for_resolution() {
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let base = Value::Object(runmat_builtins::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let result = block_on(call_method_builtin(
            base.clone(),
            "  deal  ".to_string(),
            vec![Value::Num(9.0), Value::Num(10.0)],
        ))
        .expect("call_method fallback should succeed after method-name trimming");
        match result {
            Value::OutputList(values) => {
                assert!(values.len() >= 2);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
            }
            other => {
                panic!("expected output list from trimmed-name call_method fallback, got {other:?}")
            }
        }
    }

    #[test]
    fn feval_call_method_closure_fast_path_preserves_requested_outputs() {
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let base = Value::Object(runmat_builtins::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: CALL_METHOD_BUILTIN_NAME.to_string(),
            bound_function: None,
            captures: vec![
                base.clone(),
                Value::String("deal".to_string()),
                Value::Num(9.0),
            ],
        });
        let result = block_on(feval_builtin(closure, vec![Value::Num(10.0)]))
            .expect("feval call_method closure should succeed");
        match result {
            Value::OutputList(values) => {
                assert!(values.len() >= 2);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
            }
            other => {
                panic!(
                    "expected output list from feval call_method closure fast path, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn feval_call_method_closure_fast_path_trims_method_name_for_resolution() {
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let base = Value::Object(runmat_builtins::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: CALL_METHOD_BUILTIN_NAME.to_string(),
            bound_function: None,
            captures: vec![
                base.clone(),
                Value::String("  deal  ".to_string()),
                Value::Num(9.0),
            ],
        });
        let result = block_on(feval_builtin(closure, vec![Value::Num(10.0)]))
            .expect("feval call_method closure should succeed after method-name trimming");
        match result {
            Value::OutputList(values) => {
                assert!(values.len() >= 2);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
            }
            other => {
                panic!(
                    "expected output list from trimmed call_method closure fast path, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn feval_call_method_closure_rejects_nontext_method_capture_with_identifier() {
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: CALL_METHOD_BUILTIN_NAME.to_string(),
            bound_function: None,
            captures: vec![
                Value::Object(runmat_builtins::ObjectInstance::new("Point".to_string())),
                Value::Num(1.0),
            ],
        });
        let err = block_on(feval_builtin(closure, Vec::new()))
            .expect_err("feval call_method closure should reject nontext method capture");
        assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
    }

    #[test]
    fn call_method_rejects_non_object_receiver_with_identifier() {
        let err = block_on(call_method_builtin(
            Value::Num(1.0),
            "origin".to_string(),
            Vec::new(),
        ))
        .expect_err("non-object receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn call_method_rejects_empty_method_name_with_identifier() {
        let err = block_on(call_method_builtin(
            Value::Object(runmat_builtins::ObjectInstance::new("Point".to_string())),
            "  ".to_string(),
            Vec::new(),
        ))
        .expect_err("empty method name should fail");
        assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
    }

    #[test]
    fn subsref_rejects_non_object_receiver_with_identifier() {
        let err = block_on(subsref_dispatch(
            Value::Num(1.0),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Num(2.0),
        ))
        .expect_err("non-object subsref receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn subsasgn_rejects_non_object_receiver_with_identifier() {
        let err = block_on(subsasgn_dispatch(
            Value::Num(1.0),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Num(2.0),
            Value::Num(3.0),
        ))
        .expect_err("non-object subsasgn receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn subsref_missing_protocol_errors_with_identifier() {
        let err = block_on(subsref_dispatch(
            Value::Object(runmat_builtins::ObjectInstance::new(
                "NoSubsrefProtocolClass".to_string(),
            )),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        ))
        .expect_err("missing subsref protocol should fail");
        assert_eq!(err.identifier(), Some("RunMat:MissingSubsref"));
    }

    #[test]
    fn subsasgn_missing_protocol_errors_with_identifier() {
        let err = block_on(subsasgn_dispatch(
            Value::Object(runmat_builtins::ObjectInstance::new(
                "NoSubsasgnProtocolClass".to_string(),
            )),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            Value::Num(3.0),
        ))
        .expect_err("missing subsasgn protocol should fail");
        assert_eq!(err.identifier(), Some("RunMat:MissingSubsasgn"));
    }

    #[test]
    fn get_p_rejects_non_object_receiver_with_identifier() {
        let err = block_on(get_p_builtin(Value::Num(1.0)))
            .expect_err("get.p should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:GetPReceiverInvalid"));
    }

    #[test]
    fn set_p_rejects_non_object_receiver_with_identifier() {
        let err = block_on(set_p_builtin(Value::Num(1.0), Value::Num(2.0)))
            .expect_err("set.p should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:SetPReceiverInvalid"));
    }

    #[test]
    fn point_move_rejects_non_object_receiver_with_identifier() {
        let err = block_on(point_move_method(Value::Num(1.0), 2.0, 3.0))
            .expect_err("Point.move should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:PointMoveReceiverInvalid"));
    }

    #[test]
    fn circle_area_rejects_non_object_receiver_with_identifier() {
        let err = block_on(circle_area_method(Value::Num(1.0)))
            .expect_err("Circle.area should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:CircleAreaReceiverInvalid"));
    }

    #[test]
    fn overidx_plus_rejects_non_object_receiver_with_identifier() {
        let err = block_on(overidx_plus(Value::Num(1.0), Value::Num(2.0)))
            .expect_err("OverIdx.plus should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:OverIdxReceiverInvalid"));
    }

    #[test]
    fn overidx_subsref_unsupported_payload_errors_with_identifier() {
        let err = block_on(overidx_subsref(
            Value::Object(runmat_builtins::ObjectInstance::new("OverIdx".to_string())),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Num(1.0),
        ))
        .expect_err("OverIdx.subsref unsupported payload should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:OverIdxSubsrefPayloadUnsupported")
        );
    }

    #[test]
    fn overidx_subsasgn_unsupported_payload_errors_with_identifier() {
        let err = block_on(overidx_subsasgn(
            Value::Object(runmat_builtins::ObjectInstance::new("OverIdx".to_string())),
            OBJECT_INDEX_PAREN.to_string(),
            Value::Num(1.0),
            Value::Num(2.0),
        ))
        .expect_err("OverIdx.subsasgn unsupported payload should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:OverIdxSubsasgnPayloadUnsupported")
        );
    }

    #[test]
    fn feval_object_receiver_routes_to_subsref_identifier() {
        let err = block_on(feval_builtin(
            Value::Object(runmat_builtins::ObjectInstance::new(
                "NoSubsrefProtocolClass".to_string(),
            )),
            vec![Value::Num(1.0)],
        ))
        .expect_err("feval(object, ...) should route through subsref dispatch");
        assert_eq!(err.identifier(), Some("RunMat:MissingSubsref"));
    }

    #[test]
    fn feval_unsupported_callable_value_errors_with_identifier() {
        let err = block_on(feval_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("numeric callable value should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:FevalFunctionValueUnsupported")
        );
    }

    #[test]
    fn shape_checked_cell_builder_maps_shape_identifier() {
        let err = super::build_shape_checked_cell(vec![Value::Num(1.0)], 2, 2, "test")
            .expect_err("expected shape mismatch");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn feval_accepts_scalar_string_array_handle() {
        let handle =
            runmat_builtins::StringArray::new(vec!["@sin".to_string()], vec![1, 1]).expect("sa");
        let result = block_on(feval_builtin(
            Value::StringArray(handle),
            vec![Value::Num(0.0)],
        ))
        .expect("string-array handle feval should succeed");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn feval_rejects_nonscalar_string_array_handle_with_identifier() {
        let handle = runmat_builtins::StringArray::new(
            vec!["@sin".to_string(), "@cos".to_string()],
            vec![1, 2],
        )
        .expect("sa");
        let err = block_on(feval_builtin(
            Value::StringArray(handle),
            vec![Value::Num(0.0)],
        ))
        .expect_err("nonscalar string-array handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleShapeInvalid"));
    }

    #[test]
    fn call_feval_async_with_outputs_preserves_unresolved_identifier() {
        let err = block_on(super::call_feval_async_with_outputs(
            Value::ExternalFunctionHandle("missing.external".to_string()),
            &[Value::Num(3.0)],
            1,
        ))
        .expect_err("unresolved external handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn addlistener_rejects_non_object_target_with_identifier() {
        let err = block_on(addlistener_builtin(
            Value::Num(1.0),
            "Changed".to_string(),
            Value::FunctionHandle("sin".to_string()),
        ))
        .expect_err("addlistener should reject non-object target");
        assert_eq!(err.identifier(), Some("RunMat:AddListenerTargetInvalid"));
    }

    #[test]
    fn notify_rejects_non_object_target_with_identifier() {
        let err = block_on(notify_builtin(
            Value::Num(1.0),
            "Changed".to_string(),
            Vec::new(),
        ))
        .expect_err("notify should reject non-object target");
        assert_eq!(err.identifier(), Some("RunMat:NotifyTargetInvalid"));
    }

    #[test]
    fn addlistener_function_handle_prefers_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "event_callback").then_some(61)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let listener = block_on(addlistener_builtin(
            target,
            "Changed".to_string(),
            Value::FunctionHandle("event_callback".to_string()),
        ))
        .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::BoundFunctionHandle { name, function }
                if name == "event_callback" && *function == 61
        ));
    }

    #[test]
    fn addlistener_external_function_handle_prefers_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.event_callback").then_some(62)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let listener = block_on(addlistener_builtin(
            target,
            "Changed".to_string(),
            Value::ExternalFunctionHandle("pkg.event_callback".to_string()),
        ))
        .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::BoundFunctionHandle { name, function }
                if name == "pkg.event_callback" && *function == 62
        ));
    }

    #[test]
    fn addlistener_string_handle_prefers_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "event_callback").then_some(63)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let listener = block_on(addlistener_builtin(
            target,
            "Changed".to_string(),
            Value::String("@event_callback".to_string()),
        ))
        .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::BoundFunctionHandle { name, function }
                if name == "event_callback" && *function == 63
        ));
    }

    #[test]
    fn addlistener_char_handle_prefers_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "event_callback").then_some(64)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let listener = block_on(addlistener_builtin(
            target,
            "Changed".to_string(),
            Value::CharArray(runmat_builtins::CharArray::new_row("@event_callback")),
        ))
        .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::BoundFunctionHandle { name, function }
                if name == "event_callback" && *function == 64
        ));
    }

    #[test]
    fn addlistener_string_array_handle_prefers_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "event_callback").then_some(66)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let callback =
            runmat_builtins::StringArray::new(vec!["@event_callback".to_string()], vec![1, 1])
                .expect("string array");
        let listener = block_on(addlistener_builtin(
            target,
            "Changed".to_string(),
            Value::StringArray(callback),
        ))
        .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::BoundFunctionHandle { name, function }
                if name == "event_callback" && *function == 66
        ));
    }

    #[test]
    fn addlistener_closure_prefers_embedded_semantic_identity_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "event_callback").then_some(65)
            })));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let callback = Value::Closure(runmat_builtins::Closure {
            function_name: "event_callback".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let listener = block_on(addlistener_builtin(target, "Changed".to_string(), callback))
            .expect("listener registered");
        let Value::Listener(listener) = listener else {
            panic!("expected listener value");
        };
        assert!(matches!(
            &*listener.callback,
            Value::Closure(runmat_builtins::Closure {
                function_name,
                bound_function: Some(65),
                captures,
            }) if function_name == "event_callback" && captures == &vec![Value::Num(9.0)]
        ));
    }

    #[test]
    fn notify_semantic_function_handle_uses_semantic_identity() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen_calls = Arc::clone(&calls);
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, requested_outputs| {
                assert_eq!(function, 44);
                assert_eq!(requested_outputs, 0);
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Value::HandleObject(_)));
                seen_calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(Value::Num(0.0)) })
            },
        )));
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let callback = Value::BoundFunctionHandle {
            name: "event_callback".to_string(),
            function: 44,
        };

        block_on(addlistener_builtin(
            target.clone(),
            "Changed".to_string(),
            callback,
        ))
        .expect("listener registered");
        block_on(notify_builtin(target, "Changed".to_string(), Vec::new()))
            .expect("notify succeeds");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn notify_char_handle_callback_surfaces_unresolved_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        block_on(addlistener_builtin(
            target.clone(),
            "Changed".to_string(),
            Value::CharArray(runmat_builtins::CharArray::new_row(
                "@definitely_missing_callback",
            )),
        ))
        .expect("listener registered");
        let err = block_on(notify_builtin(target, "Changed".to_string(), Vec::new()))
            .expect_err("unresolved char callback should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn notify_string_array_handle_callback_surfaces_unresolved_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let target =
            block_on(new_handle_object_builtin("EventTarget".to_string())).expect("handle target");
        let callback = runmat_builtins::StringArray::new(
            vec!["@definitely_missing_callback".to_string()],
            vec![1, 1],
        )
        .expect("string array");
        block_on(addlistener_builtin(
            target.clone(),
            "Changed".to_string(),
            Value::StringArray(callback),
        ))
        .expect("listener registered");
        let err = block_on(notify_builtin(target, "Changed".to_string(), Vec::new()))
            .expect_err("unresolved string-array callback should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn feval_semantic_handle_honors_zero_requested_outputs() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 46);
                assert_eq!(requested_outputs, 0);
                assert_eq!(args, &[Value::Num(5.0)]);
                Box::pin(async { Ok(Value::OutputList(Vec::new())) })
            },
        )));
        let _output_guard = crate::output_count::push_output_count(Some(0));
        let handle = Value::BoundFunctionHandle {
            name: "function_target".to_string(),
            function: 46,
        };

        let result = block_on(feval_builtin(handle, vec![Value::Num(5.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(result, Value::OutputList(Vec::new()));
    }

    #[test]
    fn feval_semantic_handle_honors_multi_requested_outputs() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 47);
                assert_eq!(requested_outputs, 2);
                assert_eq!(args, &[Value::Num(6.0)]);
                Box::pin(async { Ok(Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)])) })
            },
        )));
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let handle = Value::BoundFunctionHandle {
            name: "function_target".to_string(),
            function: 47,
        };

        let result = block_on(feval_builtin(handle, vec![Value::Num(6.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(
            result,
            Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)])
        );
    }

    #[test]
    fn feval_semantic_closure_errors_when_semantic_invoker_unavailable() {
        let _guard = crate::user_functions::install_semantic_function_invoker(None);
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: "function_target".to_string(),
            bound_function: Some(9044),
            captures: vec![Value::Num(1.0)],
        });

        let err = block_on(feval_builtin(closure, vec![Value::Num(2.0)]))
            .expect_err("semantic closure should not fall back to name-based dispatch");
        assert_eq!(err.identifier(), Some("RunMat:SemanticFunctionUnavailable"));
        assert!(
            err.message()
                .contains("semantic closure 'function_target' (9044) is unavailable"),
            "unexpected error: {err:?}"
        );
    }
}

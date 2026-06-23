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

use runmat_builtins::{BuiltinErrorDescriptor, Value};

pub mod dispatcher;

pub mod callsite;
pub mod console;
pub mod data;
pub mod interaction;
pub mod interrupt;
pub mod output_context;
pub mod output_count;
pub mod source_context;

pub mod builtins;
pub mod comparison;
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
pub const CALL_BOUND_METHOD_BUILTIN_NAME: &str = "__runmat_call_bound_method__";
pub const OBJECT_SUBSREF_METHOD: &str = "subsref";
pub const OBJECT_SUBSASGN_METHOD: &str = "subsasgn";
pub(crate) const IDENT_UNDEFINED_FUNCTION: &str = "RunMat:UndefinedFunction";
pub(crate) const HANDLE_VALID_FLAG_PROPERTY: &str = "__runmat_handle_valid__";

pub(crate) fn is_handle_valid(handle: &runmat_builtins::HandleRef) -> bool {
    if !handle.valid {
        return false;
    }
    runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(obj) => !matches!(
            obj.properties.get(HANDLE_VALID_FLAG_PROPERTY),
            Some(Value::Bool(false))
        ),
        _ => true,
    })
    .unwrap_or(false)
}

pub(crate) fn set_handle_valid(handle: &runmat_builtins::HandleRef, valid: bool) -> bool {
    runmat_gc::gc_with_value_mut(&handle.target, |target| match target {
        Value::Object(obj) => {
            obj.properties
                .insert(HANDLE_VALID_FLAG_PROPERTY.to_string(), Value::Bool(valid));
            true
        }
        _ => false,
    })
    .unwrap_or(false)
}

pub fn object_property_getter_name(field: &str) -> String {
    format!("get.{field}")
}

pub fn object_property_setter_name(field: &str) -> String {
    format!("set.{field}")
}

pub(crate) fn current_requested_outputs() -> usize {
    crate::output_count::current_output_count().unwrap_or(1)
}

fn undefined_callable_error(identity: &runmat_hir::CallableIdentity) -> RuntimeError {
    let detail = format!("Undefined function for callable identity {identity:?}");
    build_runtime_error(detail)
        .with_identifier(IDENT_UNDEFINED_FUNCTION)
        .build()
}

pub(crate) fn is_undefined_function_error(err: &RuntimeError) -> bool {
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

pub(crate) fn runtime_descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    runtime_descriptor_error_with_message(builtin, error.message, error)
}

pub(crate) fn runtime_descriptor_error_with_detail(
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

pub(crate) fn object_receiver_class_name(receiver: &Value) -> Option<String> {
    match receiver {
        Value::Object(obj) => Some(obj.class_name.clone()),
        Value::HandleObject(handle) => {
            let class_name = runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(obj) => obj.class_name.clone(),
                _ => handle.class_name.clone(),
            })
            .unwrap_or_else(|_| handle.class_name.clone());
            Some(class_name)
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

pub(crate) async fn dispatch_object_external_member(
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

pub(crate) async fn dispatch_callable_with_policy(
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
    call_builtin, call_builtin_async, call_builtin_async_with_outputs, class_access_context,
    gather_if_needed, gather_if_needed_async, is_gpu_value, push_class_access_context,
    value_contains_gpu,
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
pub use builtins::common::concatenation::create_matrix_from_values;
pub use builtins::common::elementwise::{
    elementwise_div, elementwise_mul, elementwise_neg, elementwise_pow, power,
};
pub use builtins::common::indexing::perform_indexing;
pub use builtins::common::matrix::value_matmul;
// Explicitly re-export for external users of the VM that build matrices from values
// (kept above)
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

pub fn make_cell_with_shape(values: Vec<Value>, shape: Vec<usize>) -> Result<Value, String> {
    let ca = runmat_builtins::CellArray::new_with_shape(values, shape)
        .map_err(|e| format!("Cell creation error: {e}"))?;
    Ok(Value::Cell(ca))
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

pub(crate) async fn strjoin_rowwise(a: Value, delim: Value) -> crate::BuiltinResult<Value> {
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

pub(crate) async fn deal_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
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

pub(crate) async fn rethrow_builtin(e: Value) -> crate::BuiltinResult<Value> {
    match e {
        Value::MException(me) => Err(build_runtime_error(me.message)
            .with_identifier(me.identifier)
            .build()),
        Value::String(s) => Err(build_runtime_error(s).build()),
        other => Err(build_runtime_error(format!("RunMat:error: {other:?}")).build()),
    }
}

// -------- Handle classes & events --------

pub(crate) async fn new_handle_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    // Create an underlying object instance and wrap it in a handle
    let obj = create_class_object(class_name.clone()).await?;
    let gc = runmat_gc::gc_allocate(obj).map_err(|e| format!("gc: {e}"))?;
    Ok(Value::HandleObject(runmat_builtins::HandleRef {
        class_name,
        target: gc,
        valid: true,
    }))
}

pub(crate) async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    match v {
        Value::HandleObject(h) => Ok(Value::Bool(crate::is_handle_valid(&h))),
        Value::Listener(l) => Ok(Value::Bool(l.valid && l.enabled)),
        _ => Ok(Value::Bool(false)),
    }
}

use std::cell::RefCell;

#[derive(Default)]
struct EventRegistry {
    next_id: u64,
    listeners: std::collections::HashMap<(usize, String), Vec<runmat_builtins::Listener>>,
}

thread_local! {
    static EVENT_REGISTRY: RefCell<EventRegistry> = RefCell::new(EventRegistry::default());
}

pub(crate) fn invalidate_listener_registration(listener_id: u64) {
    EVENT_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        for listeners in registry.listeners.values_mut() {
            for listener in listeners.iter_mut() {
                if listener.id == listener_id {
                    listener.valid = false;
                    listener.enabled = false;
                }
            }
        }
    });
}

pub(crate) fn canonicalize_callback_handle_for_semantic_resolution(callback: Value) -> Value {
    fn normalize_handle_name(text: &str) -> Option<String> {
        let trimmed = text.trim();
        let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    fn resolve_text_handle(text: &str) -> Option<Value> {
        let name = normalize_handle_name(text)?;
        let function = crate::user_functions::resolve_semantic_function_by_name(&name)?;
        Some(Value::BoundFunctionHandle { name, function })
    }

    match callback {
        Value::String(text) => resolve_text_handle(&text).unwrap_or_else(|| {
            crate::builtins::introspection::function_handle_text::dispatch_str2func(Value::String(
                text.clone(),
            ))
            .unwrap_or(Value::String(text))
        }),
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = &array.data[0];
            resolve_text_handle(text).unwrap_or_else(|| {
                crate::builtins::introspection::function_handle_text::dispatch_str2func(
                    Value::StringArray(array.clone()),
                )
                .unwrap_or(Value::StringArray(array))
            })
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            resolve_text_handle(&text).unwrap_or_else(|| {
                crate::builtins::introspection::function_handle_text::dispatch_str2func(
                    Value::CharArray(chars.clone()),
                )
                .unwrap_or(Value::CharArray(chars))
            })
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

pub(crate) async fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => runmat_gc::gc_handle_addr(&h.target),
        Value::Object(o) => o as *const _ as usize,
        _ => {
            return Err(
                build_runtime_error("addlistener: target must be handle or object")
                    .with_builtin("addlistener")
                    .with_identifier("RunMat:AddListenerTargetInvalid")
                    .build(),
            )
        }
    };
    let id = EVENT_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.next_id += 1;
        registry.next_id
    });
    let target_root = match target {
        Value::HandleObject(h) => runmat_gc::gc_root(h.target).map_err(|e| format!("gc: {e}"))?,
        Value::Object(o) => {
            runmat_gc::gc_allocate_rooted(Value::Object(o)).map_err(|e| format!("gc: {e}"))?
        }
        _ => unreachable!(),
    };
    let callback = canonicalize_listener_callback(callback);
    let callback_root = runmat_gc::gc_allocate_rooted(callback).map_err(|e| format!("gc: {e}"))?;
    let listener = runmat_builtins::Listener {
        id,
        target: target_root.handle(),
        event_name: event_name.clone(),
        callback: callback_root.handle(),
        enabled: true,
        valid: true,
    };
    EVENT_REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .listeners
            .entry((key_ptr, event_name))
            .or_default()
            .push(listener.clone());
    });
    Ok(Value::Listener(listener))
}

pub(crate) async fn notify_builtin(
    target: Value,
    event_name: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => runmat_gc::gc_handle_addr(&h.target),
        Value::Object(o) => o as *const _ as usize,
        _ => {
            return Err(
                build_runtime_error("notify: target must be handle or object")
                    .with_builtin("notify")
                    .with_identifier("RunMat:NotifyTargetInvalid")
                    .build(),
            )
        }
    };
    let mut to_call: Vec<runmat_builtins::Listener> = Vec::new();
    EVENT_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        if let Some(list) = registry.listeners.get(&(key_ptr, event_name.clone())) {
            for l in list {
                if l.valid && l.enabled {
                    to_call.push(l.clone());
                }
            }
        }
    });
    for l in to_call {
        // Call callback via feval-like protocol.
        let mut args = Vec::new();
        args.push(target.clone());
        args.extend(rest.iter().cloned());
        let cbv: Value = runmat_gc::gc_clone_value(&l.callback).map_err(|e| {
            build_runtime_error(format!("notify: invalid listener callback handle: {e}"))
                .with_builtin("notify")
                .with_identifier("RunMat:NotifyInvalidCallback")
                .build()
        })?;
        let should_dispatch = match &cbv {
            Value::String(s) => !s.trim().is_empty(),
            Value::StringArray(sa) => sa.data.len() == 1 && !sa.data[0].trim().is_empty(),
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                !text.trim().is_empty()
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
pub(crate) async fn get_p_builtin(obj: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(o) => {
            if let Some(v) = o.properties.get("p_backing") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        other => Err(build_runtime_error(format!(
            "get.p: requires object receiver (got {other:?})"
        ))
        .with_builtin("get.p")
        .with_identifier("RunMat:GetPReceiverInvalid")
        .build()),
    }
}

pub(crate) async fn set_p_builtin(obj: Value, val: Value) -> crate::BuiltinResult<Value> {
    match obj {
        Value::Object(mut o) => {
            o.properties.insert("p_backing".to_string(), val);
            Ok(Value::Object(o))
        }
        other => Err(build_runtime_error(format!(
            "set.p: requires object receiver (got {other:?})"
        ))
        .with_builtin("set.p")
        .with_identifier("RunMat:SetPReceiverInvalid")
        .build()),
    }
}

pub(crate) async fn make_anon_builtin(params: String, body: String) -> crate::BuiltinResult<Value> {
    Ok(Value::String(format!("@anon({params}) {body}")))
}

pub async fn create_class_object(class_name: String) -> crate::BuiltinResult<Value> {
    if runmat_builtins::is_class_abstract(&class_name) {
        return Err(build_runtime_error(format!(
            "Cannot instantiate abstract class '{}'.",
            class_name
        ))
        .with_identifier("RunMat:AbstractMethodMissing")
        .build());
    }
    if let Some(def) = runmat_builtins::get_class(&class_name) {
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        let mut is_handle_class = false;
        let mut visited = std::collections::HashSet::new();
        // Walk up to root
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if name.eq_ignore_ascii_case("handle") {
                is_handle_class = true;
                break;
            }
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(cd) = runmat_builtins::get_class(&name) {
                if cd
                    .parent
                    .as_ref()
                    .is_some_and(|parent| parent.eq_ignore_ascii_case("handle"))
                {
                    is_handle_class = true;
                }
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
        let empty_default = || {
            Value::Tensor(runmat_builtins::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"))
        };
        for cd in chain {
            for (k, p) in cd.properties.iter() {
                if !p.is_static {
                    obj.properties.insert(
                        k.clone(),
                        p.default_value.clone().unwrap_or_else(empty_default),
                    );
                }
            }
        }
        if is_handle_class {
            let gc = runmat_gc::gc_allocate(Value::Object(obj)).map_err(|e| format!("gc: {e}"))?;
            Ok(Value::HandleObject(runmat_builtins::HandleRef {
                class_name: def.name.clone(),
                target: gc,
                valid: true,
            }))
        } else {
            Ok(Value::Object(obj))
        }
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(
            class_name,
        )))
    }
}

pub async fn call_super_constructor(
    class_name: String,
    super_class_name: String,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let receiver = create_class_object(class_name).await?;
    let ctor_name = super_class_name
        .rsplit('.')
        .next()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(super_class_name.as_str());
    let ctor_lookup = runmat_builtins::lookup_method(&super_class_name, ctor_name)
        .or_else(|| runmat_builtins::lookup_method(&super_class_name, &super_class_name));
    let Some((ctor, _owner)) = ctor_lookup else {
        return Ok(receiver);
    };
    let Some(result) =
        crate::user_functions::try_call_semantic_function_by_name(&ctor.function_name, &args, 1)
            .await
    else {
        return Ok(receiver);
    };
    let ctor_result = result?;
    fn merge_parent_props_into_object(
        receiver_obj: &mut runmat_builtins::ObjectInstance,
        ctor_result: Value,
    ) {
        match ctor_result {
            Value::Object(parent_obj) => {
                for (name, value) in parent_obj.properties {
                    receiver_obj.properties.insert(name, value);
                }
            }
            Value::HandleObject(parent_handle) => {
                if let Ok(Value::Object(parent_obj)) =
                    runmat_gc::gc_clone_value(&parent_handle.target)
                {
                    for (name, value) in parent_obj.properties {
                        receiver_obj.properties.insert(name, value);
                    }
                }
            }
            Value::Struct(parent_fields) => {
                for (name, value) in parent_fields.fields {
                    receiver_obj.properties.insert(name, value);
                }
            }
            _ => {}
        }
    }
    match receiver {
        Value::Object(mut receiver_obj) => {
            merge_parent_props_into_object(&mut receiver_obj, ctor_result);
            Ok(Value::Object(receiver_obj))
        }
        Value::HandleObject(handle) => {
            let _ = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                if let Value::Object(receiver_obj) = target {
                    merge_parent_props_into_object(receiver_obj, ctor_result);
                }
            });
            Ok(Value::HandleObject(handle))
        }
        _ => Ok(receiver),
    }
}

pub async fn call_super_method(
    class_name: String,
    super_class_name: String,
    method_name: String,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let Some((method, owner)) = runmat_builtins::lookup_method(&super_class_name, &method_name)
    else {
        return Err(build_runtime_error(format!(
            "Undefined superclass method '{}@{}'",
            method_name, super_class_name
        ))
        .with_identifier("RunMat:UndefinedFunction")
        .build());
    };
    if method.is_static {
        return Err(build_runtime_error(format!(
            "Superclass method '{}@{}' is static and cannot be called with super method syntax.",
            method_name, super_class_name
        ))
        .with_identifier("RunMat:MethodStaticAccess")
        .build());
    }
    let access_allowed = match method.access {
        runmat_builtins::Access::Public => true,
        runmat_builtins::Access::Protected => {
            runmat_builtins::is_class_or_subclass(&class_name, &owner)
        }
        runmat_builtins::Access::Private => class_name == owner,
    };
    if !access_allowed {
        return Err(build_runtime_error(format!(
            "Method '{}@{}' is not accessible from class '{}'.",
            method_name, super_class_name, class_name
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    let Some(result) =
        crate::user_functions::try_call_semantic_function_by_name(&method.function_name, &args, 1)
            .await
    else {
        return Err(
            build_runtime_error(format!("Undefined function: {}", method.function_name))
                .with_identifier("RunMat:UndefinedFunction")
                .build(),
        );
    };
    result
}

// handle-object builtins removed for now

pub(crate) async fn classref_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    Ok(Value::ClassRef(class_name))
}

pub(crate) async fn register_test_classes_builtin() -> crate::BuiltinResult<Value> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_constant: false,
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
            is_constant: false,
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
            is_constant: false,
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
            is_constant: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_constant: false,
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
            is_constant: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_constant: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_abstract: false,
            is_sealed: false,
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
            is_abstract: false,
            is_sealed: false,
            access: Access::Public,
            function_name: format!("OverIdx.{OBJECT_SUBSASGN_METHOD}"),
            implicit_class_argument: None,
        },
    );
    for name in [
        "plus", "times", "mtimes", "lt", "gt", "eq", "uplus", "rdivide", "mrdivide", "ldivide",
        "mldivide", "and", "or", "xor",
    ] {
        overidx_methods.insert(
            name.to_string(),
            MethodDef {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: format!("OverIdx.{name}"),
                implicit_class_argument: None,
            },
        );
    }
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

pub(crate) const FEVAL_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FEVAL_ERROR_HANDLE_NAME_INVALID,
    FEVAL_ERROR_HANDLE_STRING_INVALID,
    FEVAL_ERROR_HANDLE_SHAPE_INVALID,
    FEVAL_ERROR_SEMANTIC_UNAVAILABLE,
    FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
];

pub(crate) async fn feval_builtin(f: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
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
                        return Err(build_runtime_error(
                            "call_method: closure captures must include method name text",
                        )
                        .with_builtin("call_method")
                        .with_identifier("RunMat:CallMethodNameInvalid")
                        .build())
                    }
                };
                let mut method_args = c.captures.iter().skip(2).cloned().collect::<Vec<_>>();
                method_args.extend(rest);
                return crate::builtins::introspection::call_method::dispatch_call_method(
                    base,
                    method,
                    method_args,
                )
                .await;
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
            crate::builtins::introspection::object_indexing::dispatch_subsref(
                receiver,
                OBJECT_INDEX_PAREN.to_string(),
                payload,
            )
            .await
        }
        other => Err(runtime_descriptor_error_with_detail(
            "feval",
            &FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::introspection::test_methods::*;
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
            ("functions", "info = functions(fh)"),
            ("inputname", "name = inputname(argNumber)"),
            ("localfunctions", "handles = localfunctions()"),
            ("narginchk", "narginchk(minArgs, maxArgs)"),
            ("nargoutchk", "nargoutchk(minArgs, maxArgs)"),
            ("mfilename", "name = mfilename()"),
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
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("resolved_target".to_string()),
        )
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
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("@missing_target".to_string()),
        )
        .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("missing_target".to_string()));
    }

    #[test]
    fn str2func_returns_external_handle_for_qualified_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("Point.origin".to_string()),
        )
        .expect("str2func should succeed");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn str2func_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("Point..origin".to_string()),
        )
        .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".to_string()));
    }

    #[test]
    fn func2str_rejects_non_handle_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_func2str(
            Value::Num(1.0),
        )
        .expect_err("func2str non-handle input should fail");
        assert_eq!(err.identifier(), Some("RunMat:Func2StrHandleTypeInvalid"));
    }

    #[test]
    fn str2func_rejects_empty_name_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("   ".to_string()),
        )
        .expect_err("empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_rejects_non_row_char_name_with_identifier() {
        let chars = runmat_builtins::CharArray::new(vec!['a', 'b'], 2, 1)
            .expect("char array construction should succeed");
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::CharArray(chars),
        )
        .expect_err("non-row char-array function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_rejects_non_text_name_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::Num(1.0),
        )
        .expect_err("non-text function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
    }

    #[test]
    fn str2func_accepts_scalar_string_array_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(vec!["@missing_target".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
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
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(value)
            .expect_err("nonscalar string-array function name must fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(445)
            })));
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(vec!["@resolved_target".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
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
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(vec!["Point.origin".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array qualified name should succeed");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn str2func_scalar_string_array_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(vec!["Point..origin".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array malformed qualified name should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".to_string()));
    }

    #[test]
    fn str2func_scalar_string_array_rejects_empty_name_with_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None);
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(vec!["   ".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect_err("scalar string-array empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_qualified_name_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.resolved_target").then_some(446)
            })));
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_builtins::StringArray::new(
                    vec!["@pkg.resolved_target".to_string()],
                    vec![1, 1],
                )
                .expect("string array construction should succeed"),
            ),
        )
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
        let value = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::ClassRef("Point".to_string()),
            "origin".to_string(),
        )
        .expect("getmethod should resolve classref method handle");
        assert_eq!(
            value,
            Value::ExternalFunctionHandle("Point.origin".to_string())
        );
    }

    #[test]
    fn getmethod_rejects_empty_method_name() {
        let err = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::ClassRef("Point".to_string()),
            "   ".to_string(),
        )
        .expect_err("empty method name should be rejected");
        assert_eq!(err.identifier(), Some("RunMat:GetMethodNameInvalid"));
    }

    #[test]
    fn getmethod_rejects_unsupported_receiver_with_identifier() {
        let err = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::Num(1.0),
            "origin".to_string(),
        )
        .expect_err("unsupported receiver should be rejected");
        assert_eq!(
            err.identifier(),
            Some("RunMat:GetMethodReceiverUnsupported")
        );
    }

    #[test]
    fn create_class_object_handles_class_parent_cycles() {
        let class_a = unique_class_name("runtime_ctor_cycle_a");
        let class_b = unique_class_name("runtime_ctor_cycle_b");

        let mut props_a = HashMap::new();
        props_a.insert(
            "fromA".to_string(),
            PropertyDef {
                name: "fromA".to_string(),
                is_static: false,
                is_constant: false,
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
                is_constant: false,
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

        let value = block_on(create_class_object(class_a.clone()))
            .expect("constructor should terminate under parent-cycle metadata");
        let Value::Object(obj) = value else {
            panic!("expected object result");
        };
        assert_eq!(obj.class_name, class_a);
        assert_eq!(obj.properties.get("fromA"), Some(&Value::Num(1.0)));
        assert_eq!(obj.properties.get("fromB"), Some(&Value::Num(2.0)));
    }

    #[test]
    fn create_class_object_abstract_class_reports_stable_identifier() {
        let class_name = unique_class_name("runtime_ctor_abstract");
        runmat_builtins::register_class_with_modifiers(
            ClassDef {
                name: class_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
            false,
            true,
        );

        let err = block_on(create_class_object(class_name))
            .expect_err("abstract class instantiation should fail");
        assert_eq!(err.identifier(), Some("RunMat:AbstractMethodMissing"));
        assert!(err.message().contains("Cannot instantiate abstract class"));
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
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::FunctionHandle("sin".to_string())
            )
            .expect("func2str"),
            Value::String("sin".to_string())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::ExternalFunctionHandle("Point.origin".to_string())
            )
            .expect("func2str"),
            Value::String("Point.origin".to_string())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::BoundFunctionHandle {
                    name: "local_fn".to_string(),
                    function: 44,
                }
            )
            .expect("func2str"),
            Value::String("local_fn".to_string())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::Closure(runmat_builtins::Closure {
                    function_name: "captured_fn".to_string(),
                    bound_function: None,
                    captures: Vec::new(),
                })
            )
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
        let result = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                base.clone(),
                "deal".to_string(),
                vec![Value::Num(9.0), Value::Num(10.0)],
            ),
        )
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
        let result = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                base.clone(),
                "  deal  ".to_string(),
                vec![Value::Num(9.0), Value::Num(10.0)],
            ),
        )
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
        let err = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                Value::Num(1.0),
                "origin".to_string(),
                Vec::new(),
            ),
        )
        .expect_err("non-object receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn call_method_rejects_empty_method_name_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                Value::Object(runmat_builtins::ObjectInstance::new("Point".to_string())),
                "  ".to_string(),
                Vec::new(),
            ),
        )
        .expect_err("empty method name should fail");
        assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
    }

    #[test]
    fn subsref_rejects_non_object_receiver_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsref(
                Value::Num(1.0),
                OBJECT_INDEX_PAREN.to_string(),
                Value::Num(2.0),
            ),
        )
        .expect_err("non-object subsref receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn subsasgn_rejects_non_object_receiver_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsasgn(
                Value::Num(1.0),
                OBJECT_INDEX_PAREN.to_string(),
                Value::Num(2.0),
                Value::Num(3.0),
            ),
        )
        .expect_err("non-object subsasgn receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn subsref_missing_protocol_errors_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsref(
                Value::Object(runmat_builtins::ObjectInstance::new(
                    "NoSubsrefProtocolClass".to_string(),
                )),
                OBJECT_INDEX_PAREN.to_string(),
                Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            ),
        )
        .expect_err("missing subsref protocol should fail");
        assert_eq!(err.identifier(), Some("RunMat:MissingSubsref"));
    }

    #[test]
    fn subsasgn_missing_protocol_errors_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsasgn(
                Value::Object(runmat_builtins::ObjectInstance::new(
                    "NoSubsasgnProtocolClass".to_string(),
                )),
                OBJECT_INDEX_PAREN.to_string(),
                Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
                Value::Num(3.0),
            ),
        )
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
    fn addlistener_preserves_object_target_when_callback_allocation_collects() {
        runmat_gc::gc_test_context(|| {
            let mut config = runmat_gc::GcConfig::default();
            config.young_generation_size = 64 * 1024 * 1024;
            config.minor_gc_threshold = 0.35;
            config.major_gc_threshold = 0.9;
            runmat_gc::gc_configure(config).expect("configure aggressive periodic minor GC");

            for i in 0..30 {
                let _ = runmat_gc::gc_allocate(Value::Num(i as f64)).expect("seed allocation");
            }

            let target = Value::Object(runmat_builtins::ObjectInstance::new(
                "EventTarget".to_string(),
            ));
            let listener = block_on(addlistener_builtin(
                target,
                "ChangedSoundness".to_string(),
                Value::FunctionHandle("sin".to_string()),
            ))
            .expect("listener registered");

            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let target = runmat_gc::gc_clone_value(&listener.target)
                .expect("listener target should survive construction");
            assert!(matches!(
                target,
                Value::Object(ref object) if object.class_name == "EventTarget"
            ));
            assert_eq!(
                runmat_gc::gc_clone_value(&listener.callback)
                    .expect("listener callback should survive construction"),
                Value::FunctionHandle("sin".to_string())
            );
        });
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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
        let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
        assert!(matches!(
            &callback,
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

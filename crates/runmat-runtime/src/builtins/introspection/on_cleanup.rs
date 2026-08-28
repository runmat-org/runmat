//! MATLAB-compatible `onCleanup` handle object.
use runmat_types::MemberAccess;

use std::collections::HashMap;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, HandleRef, ObjectInstance, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) const ON_CLEANUP_CLASS: &str = "onCleanup";
const CALLBACK_PROPERTY: &str = "__oncleanup_callback";
const ACTIVE_PROPERTY: &str = "__oncleanup_active";
const ON_CLEANUP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "onCleanup accepts only a function handle or closure and returns a handle object; numeric values are rejected as callbacks.",
};
const CANCEL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "cancel accepts only an onCleanup handle and returns a fixed double status; it has no integer data, control, or class-preserving form.",
};

#[cfg(test)]
pub(crate) static ON_CLEANUP_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const ON_CLEANUP_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle object that invokes the cleanup function when deleted or cleared.",
}];
const ON_CLEANUP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupFun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-input function handle to invoke during cleanup.",
}];
const ON_CLEANUP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "cleanupObj = onCleanup(cleanupFun)",
    inputs: &ON_CLEANUP_INPUTS,
    outputs: &ON_CLEANUP_OUTPUTS,
}];
const ON_CLEANUP_ERROR_INVALID_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_CALLBACK",
    identifier: Some("RunMat:onCleanup:InvalidCallback"),
    when: "The cleanup function is not a function handle or closure.",
    message: "onCleanup: cleanupFun must be a function handle",
};
const ON_CLEANUP_ERROR_INVALID_OBJECT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_OBJECT",
    identifier: Some("RunMat:onCleanup:InvalidObject"),
    when: "A cleanup operation targets a value that is not an onCleanup object.",
    message: "onCleanup: invalid cleanup object",
};
const ON_CLEANUP_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.GC",
    identifier: Some("RunMat:onCleanup:GcFailure"),
    when: "The cleanup object target cannot be allocated, rooted, or mutated.",
    message: "onCleanup: internal object storage failed",
};
const ON_CLEANUP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ON_CLEANUP_ERROR_INVALID_CALLBACK,
    ON_CLEANUP_ERROR_INVALID_OBJECT,
    ON_CLEANUP_ERROR_GC,
];
pub const ON_CLEANUP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ON_CLEANUP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ON_CLEANUP_ERRORS,
};

const DELETE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero after the cleanup callback has been invoked or skipped.",
}];
const DELETE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "onCleanup object to execute and deactivate.",
}];
const CANCEL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "onCleanup object to deactivate without running its callback.",
}];
const DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = delete(cleanupObj)",
    inputs: &DELETE_INPUTS,
    outputs: &DELETE_OUTPUTS,
}];
pub const ON_CLEANUP_DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &ON_CLEANUP_ERRORS,
};

const CANCEL_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero after the cleanup callback has been deactivated.",
}];
const CANCEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = cancel(cleanupObj)",
    inputs: &CANCEL_INPUTS,
    outputs: &CANCEL_OUTPUTS,
}];
pub const ON_CLEANUP_CANCEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CANCEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ON_CLEANUP_ERRORS,
};

#[runtime_builtin(
    name = "onCleanup",
    category = "introspection",
    summary = "Create an object that runs a function when deleted or cleared.",
    keywords = "onCleanup,cleanup,delete,clear,resource,RAII",
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::on_cleanup::ON_CLEANUP_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_builtin(callback: Value) -> BuiltinResult<Value> {
    ensure_on_cleanup_class_registered();
    let callback = canonicalize_cleanup_callback(callback)?;
    let mut object = ObjectInstance::new(ON_CLEANUP_CLASS.to_string());
    object
        .properties
        .insert(CALLBACK_PROPERTY.to_string(), callback.clone());
    object
        .properties
        .insert(ACTIVE_PROPERTY.to_string(), Value::Bool(true));
    object.properties.insert(
        crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
        Value::Bool(true),
    );

    let target = runmat_gc::gc_allocate(Value::Object(object))
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;

    Ok(Value::HandleObject(HandleRef {
        class_name: ON_CLEANUP_CLASS.to_string(),
        target,
        valid: true,
    }))
}

#[runtime_builtin(
    name = "__runmat_oncleanup_delete",
    category = "introspection",
    summary = "Execute an onCleanup object callback once.",
    keywords = "onCleanup,delete,cleanup",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_DELETE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_delete_builtin(value: Value) -> BuiltinResult<Value> {
    let cleanup_result = run_on_cleanup_value(&value).await;
    let invalidation_result = match &value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            if crate::set_handle_valid(handle, false) {
                Ok(())
            } else {
                Err(on_cleanup_error(
                    &ON_CLEANUP_ERROR_INVALID_OBJECT,
                    "onCleanup: failed to invalidate cleanup object",
                ))
            }
        }
        _ => Ok(()),
    };
    cleanup_result?;
    invalidation_result?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "cancel",
    category = "introspection",
    summary = "Deactivate an onCleanup object without running its callback.",
    keywords = "onCleanup,cancel,cleanup",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_CANCEL_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::on_cleanup::CANCEL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
async fn on_cleanup_cancel_builtin(value: Value) -> BuiltinResult<Value> {
    cancel_on_cleanup_value(&value)?;
    Ok(Value::Num(0.0))
}

pub(crate) async fn run_cleanup_for_workspace_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            run_on_cleanup_handle(handle).await
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                Box::pin(run_cleanup_for_workspace_value(item)).await?;
            }
            Ok(())
        }
        Value::Struct(struct_value) => {
            for item in struct_value.fields.values() {
                Box::pin(run_cleanup_for_workspace_value(item)).await?;
            }
            Ok(())
        }
        Value::Object(object) => {
            for item in object.properties.values() {
                Box::pin(run_cleanup_for_workspace_value(item)).await?;
            }
            Ok(())
        }
        Value::OutputList(items) => {
            for item in items {
                Box::pin(run_cleanup_for_workspace_value(item)).await?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(crate) async fn run_cleanup_for_workspace_values(values: &[Value]) -> BuiltinResult<()> {
    for value in values {
        run_cleanup_for_workspace_value(value).await?;
    }
    Ok(())
}

fn ensure_on_cleanup_class_registered() {
    if crate::class_registry::get_class(ON_CLEANUP_CLASS).is_some() {
        return;
    }
    let mut methods = HashMap::new();
    methods.insert(
        "delete".to_string(),
        crate::class_registry::RuntimeMethod {
            name: "delete".to_string(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "__runmat_oncleanup_delete".to_string(),
            implicit_class_argument: None,
        },
    );
    methods.insert(
        "cancel".to_string(),
        crate::class_registry::RuntimeMethod {
            name: "cancel".to_string(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "cancel".to_string(),
            implicit_class_argument: None,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: ON_CLEANUP_CLASS.to_string(),
        parent: Some("handle".to_string()),
        properties: HashMap::<String, crate::class_registry::RuntimeProperty>::new(),
        methods,
    });
}

fn canonicalize_cleanup_callback(callback: Value) -> BuiltinResult<Value> {
    match callback {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Ok(crate::canonicalize_callback_handle_for_semantic_resolution(
            callback,
        )),
        _ => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_CALLBACK,
            ON_CLEANUP_ERROR_INVALID_CALLBACK.message,
        )),
    }
}

async fn run_on_cleanup_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            run_on_cleanup_handle(handle).await
        }
        other => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            format!("onCleanup: expected onCleanup object, got {other:?}"),
        )),
    }
}

async fn run_on_cleanup_handle(handle: &HandleRef) -> BuiltinResult<()> {
    let Some(callback) = deactivate_handle_and_take_callback(handle)? else {
        return Ok(());
    };
    crate::call_feval_async_with_outputs(callback, &[], 0).await?;
    Ok(())
}

fn cancel_on_cleanup_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            let _ = deactivate_handle_and_take_callback(handle)?;
            Ok(())
        }
        other => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            format!("onCleanup: expected onCleanup object, got {other:?}"),
        )),
    }
}

fn deactivate_handle_and_take_callback(handle: &HandleRef) -> BuiltinResult<Option<Value>> {
    if !is_on_cleanup_handle(handle) || !crate::is_handle_valid(handle) {
        return Ok(None);
    }

    let callback = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        let active = matches!(
            object.properties.get(ACTIVE_PROPERTY),
            Some(Value::Bool(true))
        );
        if !active {
            return None;
        }
        object
            .properties
            .insert(ACTIVE_PROPERTY.to_string(), Value::Bool(false));
        object.properties.get(CALLBACK_PROPERTY).cloned()
    })
    .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;
    Ok(callback)
}

fn is_on_cleanup_handle(handle: &HandleRef) -> bool {
    handle.class_name == ON_CLEANUP_CLASS
}

fn on_cleanup_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("onCleanup");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    fn callback_invoker(counter: Arc<Mutex<usize>>) -> Arc<crate::user_functions::FunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            assert!(args.is_empty(), "cleanup callbacks receive no arguments");
            assert_eq!(requested_outputs, 0);
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                *counter.lock().unwrap() += 1;
                Ok(Value::Tensor(runmat_value::Tensor::zeros(vec![0, 0])))
            })
        })
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn on_cleanup_accepts_function_handles_and_runs_once_on_delete() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanup".to_string(),
            function: 7,
        }))
        .expect("create cleanup");

        block_on(run_on_cleanup_value(&cleanup)).expect("first cleanup");
        block_on(run_on_cleanup_value(&cleanup)).expect("second cleanup is no-op");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_method_invalidates_cleanup_handle() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanup".to_string(),
            function: 7,
        }))
        .expect("create cleanup");
        let Value::HandleObject(handle) = cleanup.clone() else {
            panic!("expected cleanup handle");
        };

        block_on(on_cleanup_delete_builtin(cleanup)).expect("delete cleanup");

        assert_eq!(*counter.lock().unwrap(), 1);
        assert!(
            !crate::is_handle_valid(&handle),
            "delete method should invalidate shared handle target"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancel_prevents_later_cleanup() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanup".to_string(),
            function: 7,
        }))
        .expect("create cleanup");

        cancel_on_cleanup_value(&cleanup).expect("cancel cleanup");
        block_on(run_on_cleanup_value(&cleanup)).expect("cleanup remains no-op");

        assert_eq!(*counter.lock().unwrap(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_numeric_and_non_function_handle_callbacks() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        for callback in [Value::Num(1.0), Value::Int(runmat_value::IntValue::I64(1))] {
            let err = block_on(on_cleanup_builtin(callback)).expect_err("expected error");
            assert_eq!(err.identifier(), Some("RunMat:onCleanup:InvalidCallback"));
        }
        let err = block_on(on_cleanup_cancel_builtin(Value::Int(
            runmat_value::IntValue::U64(1),
        )))
        .expect_err("integer is not an onCleanup handle");
        assert_eq!(err.identifier(), Some("RunMat:onCleanup:InvalidObject"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn workspace_cleanup_walks_cells() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanup".to_string(),
            function: 7,
        }))
        .expect("create cleanup");
        let cell = Value::Cell(CellArray::new(vec![cleanup], 1, 1).expect("cell"));

        block_on(run_cleanup_for_workspace_value(&cell)).expect("cleanup");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn workspace_cleanup_walks_structs_objects_and_output_lists() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup_a = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanupA".to_string(),
            function: 7,
        }))
        .expect("create cleanup a");
        let cleanup_b = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanupB".to_string(),
            function: 8,
        }))
        .expect("create cleanup b");
        let cleanup_c = block_on(on_cleanup_builtin(Value::BoundFunctionHandle {
            name: "cleanupC".to_string(),
            function: 9,
        }))
        .expect("create cleanup c");

        let mut struct_value = runmat_value::StructValue::new();
        struct_value.insert("cleanup", cleanup_a);
        let mut object = ObjectInstance::new("CleanupHolder".to_string());
        object.properties.insert("cleanup".to_string(), cleanup_b);
        let composite = Value::OutputList(vec![
            Value::Struct(struct_value),
            Value::Object(object),
            cleanup_c,
        ]);

        block_on(run_cleanup_for_workspace_value(&composite)).expect("cleanup");

        assert_eq!(*counter.lock().unwrap(), 3);
    }
}

//! MATLAB-compatible `memoize` and `MemoizedFunction` support.

use std::cell::RefCell;
use std::collections::HashMap;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ClassDef, ComplexTensor, HandleRef, IntValue, LogicalArray, MethodDef,
    ObjectInstance, PropertyDef, SparseTensor, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{
    build_runtime_error, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN,
    OBJECT_SUBSREF_METHOD,
};

pub(crate) const MEMOIZED_FUNCTION_CLASS: &str = "MemoizedFunction";
const FUNCTION_PROPERTY: &str = "Function";
const ENABLED_PROPERTY: &str = "Enabled";
const CACHE_SIZE_PROPERTY: &str = "CacheSize";
const CACHE_PROPERTY: &str = "__runmat_memoize_cache";
const TOTAL_HITS_PROPERTY: &str = "__runmat_memoize_total_hits";
const TOTAL_MISSES_PROPERTY: &str = "__runmat_memoize_total_misses";
const DEFAULT_CACHE_SIZE: usize = 10;

#[cfg(test)]
pub(crate) static MEMOIZE_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Clone)]
struct MemoizedCacheEntry {
    inputs: Vec<Value>,
    nargout: usize,
    output: Value,
    hit_count: usize,
}

thread_local! {
    static MEMOIZE_REGISTRY: RefCell<HashMap<String, runmat_gc::GcHandle>> = RefCell::new(HashMap::new());
}

const MEMOIZE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle to memoize.",
}];
const MEMOIZE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "memoizedFcn",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MemoizedFunction handle object.",
}];
const MEMOIZE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "memoizedFcn = memoize(fh)",
    inputs: &MEMOIZE_INPUTS,
    outputs: &MEMOIZE_OUTPUTS,
}];

const MEMOIZED_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "memoizedFcn",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MemoizedFunction object.",
}];
const STATUS_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero when the operation completes.",
}];
const STATS_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure with cache entries and hit/miss counters.",
}];
const CLEAR_CACHE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "clearCache(memoizedFcn)",
    inputs: &MEMOIZED_INPUTS,
    outputs: &STATUS_OUTPUTS,
}];
const STATS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = stats(memoizedFcn)",
    inputs: &MEMOIZED_INPUTS,
    outputs: &STATS_OUTPUTS,
}];
const CLEAR_ALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "clearAllMemoizedCaches",
    inputs: &[],
    outputs: &STATUS_OUTPUTS,
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MemoizedFunction receiver.",
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
const SUBSREF_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Result of function invocation or property access.",
}];
const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = subsref(obj, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUTS,
}];

const MEMOIZE_ERROR_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_FUNCTION",
    identifier: Some("RunMat:memoize:InvalidFunction"),
    when: "The input is not a function handle or closure.",
    message: "memoize: input must be a function handle",
};
const MEMOIZE_ERROR_INVALID_OBJECT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_OBJECT",
    identifier: Some("RunMat:memoize:InvalidObject"),
    when: "A memoized-function method receives a non-MemoizedFunction object.",
    message: "memoize: expected MemoizedFunction object",
};
const MEMOIZE_ERROR_INVALID_CACHE_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_CACHE_SIZE",
    identifier: Some("RunMat:memoize:InvalidCacheSize"),
    when: "CacheSize is not a positive finite integer scalar.",
    message: "memoize: CacheSize must be a positive integer scalar",
};
const MEMOIZE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_INDEX",
    identifier: Some("RunMat:memoize:InvalidIndex"),
    when: "The MemoizedFunction object is indexed with an unsupported indexing form.",
    message: "memoize: unsupported MemoizedFunction indexing",
};
const MEMOIZE_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.GC",
    identifier: Some("RunMat:memoize:GcFailure"),
    when: "The memoized object target cannot be allocated or accessed.",
    message: "memoize: internal object storage failed",
};
const MEMOIZE_CREATE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [MEMOIZE_ERROR_INVALID_FUNCTION, MEMOIZE_ERROR_GC];
const MEMOIZED_SUBSREF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    MEMOIZE_ERROR_INVALID_OBJECT,
    MEMOIZE_ERROR_INVALID_CACHE_SIZE,
    MEMOIZE_ERROR_INVALID_INDEX,
    MEMOIZE_ERROR_GC,
];
const CLEAR_CACHE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [MEMOIZE_ERROR_INVALID_OBJECT, MEMOIZE_ERROR_GC];
const STATS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEMOIZE_ERROR_INVALID_OBJECT,
    MEMOIZE_ERROR_INVALID_CACHE_SIZE,
    MEMOIZE_ERROR_GC,
];
const NO_MEMOIZE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const MEMOIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MEMOIZE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MEMOIZE_CREATE_ERRORS,
};
pub const MEMOIZED_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &MEMOIZED_SUBSREF_ERRORS,
};
pub const CLEAR_CACHE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_CACHE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEAR_CACHE_ERRORS,
};
pub const STATS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STATS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STATS_ERRORS,
};
pub const CLEAR_ALL_MEMOIZED_CACHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_ALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NO_MEMOIZE_ERRORS,
};

#[runtime_builtin(
    name = "memoize",
    category = "introspection",
    summary = "Create a memoized wrapper around a function handle.",
    keywords = "memoize,MemoizedFunction,function handle,cache",
    descriptor(crate::builtins::introspection::memoize::MEMOIZE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn memoize_builtin(function: Value) -> BuiltinResult<Value> {
    ensure_memoized_function_class_registered();
    let function = canonicalize_memoized_function(function)?;
    let key = function_key(&function);

    MEMOIZE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if let Some(target) = registry.get(&key).copied() {
            let existing = runmat_gc::gc_with_value(&target, |target_value| {
                matches!(
                    target_value,
                    Value::Object(object)
                        if object.class_name == MEMOIZED_FUNCTION_CLASS
                            && object.properties.get(FUNCTION_PROPERTY) == Some(&function)
                            && matches!(
                                object.properties.get(crate::HANDLE_VALID_FLAG_PROPERTY),
                                Some(Value::Bool(true))
                            )
                )
            })
            .unwrap_or(false);
            if existing {
                return Ok(handle_from_target(target));
            }
            registry.remove(&key);
        }

        let mut object = ObjectInstance::new(MEMOIZED_FUNCTION_CLASS.to_string());
        object
            .properties
            .insert(FUNCTION_PROPERTY.to_string(), function.clone());
        object
            .properties
            .insert(ENABLED_PROPERTY.to_string(), Value::Bool(true));
        object.properties.insert(
            CACHE_SIZE_PROPERTY.to_string(),
            Value::Num(DEFAULT_CACHE_SIZE as f64),
        );
        object.properties.insert(
            crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
            Value::Bool(true),
        );
        object
            .properties
            .insert(CACHE_PROPERTY.to_string(), empty_cache_value()?);
        object
            .properties
            .insert(TOTAL_HITS_PROPERTY.to_string(), Value::Num(0.0));
        object
            .properties
            .insert(TOTAL_MISSES_PROPERTY.to_string(), Value::Num(0.0));

        let target = runmat_gc::gc_allocate(Value::Object(object))
            .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?;
        let handle = handle_from_target(target);
        registry.insert(key, target);
        Ok(handle)
    })
}

#[runtime_builtin(
    name = "MemoizedFunction.subsref",
    category = "introspection",
    summary = "Dispatch MemoizedFunction indexing and invocation.",
    keywords = "memoize,MemoizedFunction,subsref",
    descriptor(crate::builtins::introspection::memoize::MEMOIZED_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn memoized_subsref_builtin(
    receiver: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<Value> {
    let handle = memoized_handle(&receiver)?;
    match kind.as_str() {
        OBJECT_INDEX_PAREN => call_memoized(handle, payload).await,
        OBJECT_INDEX_MEMBER => memoized_member(handle, payload),
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            format!("memoize: unsupported MemoizedFunction indexing kind {kind}"),
        )),
    }
}

#[runtime_builtin(
    name = "clearCache",
    category = "introspection",
    summary = "Clear the cache for a MemoizedFunction object.",
    keywords = "memoize,MemoizedFunction,clearCache,cache",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::memoize::CLEAR_CACHE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn clear_cache_builtin(receiver: Value) -> BuiltinResult<Value> {
    let handle = memoized_handle(&receiver)?;
    clear_cache_for_handle(handle)?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "stats",
    category = "introspection",
    summary = "Return cache statistics for a MemoizedFunction object.",
    keywords = "memoize,MemoizedFunction,stats,cache",
    descriptor(crate::builtins::introspection::memoize::STATS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn stats_builtin(receiver: Value) -> BuiltinResult<Value> {
    let handle = memoized_handle(&receiver)?;
    stats_for_handle(handle)
}

#[runtime_builtin(
    name = "clearAllMemoizedCaches",
    category = "introspection",
    summary = "Clear every MemoizedFunction cache.",
    keywords = "memoize,MemoizedFunction,clearAllMemoizedCaches,cache",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::memoize::CLEAR_ALL_MEMOIZED_CACHES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn clear_all_memoized_caches_builtin() -> BuiltinResult<Value> {
    MEMOIZE_REGISTRY.with(|registry| {
        registry.borrow_mut().retain(|_, target| {
            runmat_gc::gc_with_value_mut(target, |target_value| {
                let Value::Object(object) = target_value else {
                    return false;
                };
                if object.class_name != MEMOIZED_FUNCTION_CLASS {
                    return false;
                }
                reset_object_cache(object).is_ok()
            })
            .unwrap_or(false)
        });
    });
    Ok(Value::Num(0.0))
}

fn ensure_memoized_function_class_registered() {
    if runmat_builtins::get_class(MEMOIZED_FUNCTION_CLASS).is_some() {
        return;
    }

    let mut properties = HashMap::new();
    for (name, set_access, default_value) in [
        (FUNCTION_PROPERTY, Access::Private, None),
        (ENABLED_PROPERTY, Access::Public, Some(Value::Bool(true))),
        (
            CACHE_SIZE_PROPERTY,
            Access::Public,
            Some(Value::Num(DEFAULT_CACHE_SIZE as f64)),
        ),
    ] {
        properties.insert(
            name.to_string(),
            PropertyDef {
                name: name.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access,
                default_value,
            },
        );
    }

    let mut methods = HashMap::new();
    for (method_name, function_name) in [
        (OBJECT_SUBSREF_METHOD, "MemoizedFunction.subsref"),
        ("clearCache", "clearCache"),
        ("stats", "stats"),
    ] {
        methods.insert(
            method_name.to_string(),
            MethodDef {
                name: method_name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        );
    }

    runmat_builtins::register_class(ClassDef {
        name: MEMOIZED_FUNCTION_CLASS.to_string(),
        parent: Some("handle".to_string()),
        properties,
        methods,
    });
}

fn canonicalize_memoized_function(function: Value) -> BuiltinResult<Value> {
    match function {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Ok(crate::canonicalize_callback_handle_for_semantic_resolution(
            function,
        )),
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_FUNCTION,
            MEMOIZE_ERROR_INVALID_FUNCTION.message,
        )),
    }
}

fn handle_from_target(target: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: MEMOIZED_FUNCTION_CLASS.to_string(),
        target,
        valid: true,
    })
}

fn memoized_handle(value: &Value) -> BuiltinResult<&HandleRef> {
    match value {
        Value::HandleObject(handle)
            if handle.class_name == MEMOIZED_FUNCTION_CLASS && crate::is_handle_valid(handle) =>
        {
            Ok(handle)
        }
        Value::HandleObject(handle) if handle.class_name == MEMOIZED_FUNCTION_CLASS => {
            Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_OBJECT,
                "memoize: MemoizedFunction handle is invalid",
            ))
        }
        other => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            format!("memoize: expected MemoizedFunction object, got {other:?}"),
        )),
    }
}

fn memoized_state(handle: &HandleRef) -> BuiltinResult<(Value, bool, usize)> {
    runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) if object.class_name == MEMOIZED_FUNCTION_CLASS => {
            let function = object
                .properties
                .get(FUNCTION_PROPERTY)
                .cloned()
                .ok_or_else(|| {
                    memoize_error(
                        &MEMOIZE_ERROR_INVALID_OBJECT,
                        "memoize: missing Function property",
                    )
                })?;
            let enabled = match object.properties.get(ENABLED_PROPERTY) {
                Some(Value::Bool(value)) => *value,
                Some(Value::Num(value)) => *value != 0.0,
                Some(Value::Int(value)) => !value.is_zero(),
                Some(Value::LogicalArray(array)) if array.data.len() == 1 => array.data[0] != 0,
                Some(Value::Tensor(tensor)) if tensor.data.len() == 1 => tensor.data[0] != 0.0,
                _ => true,
            };
            let cache_size = parse_cache_size(object.properties.get(CACHE_SIZE_PROPERTY))?;
            Ok((function, enabled, cache_size))
        }
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: invalid object target",
        )),
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn parse_cache_size(value: Option<&Value>) -> BuiltinResult<usize> {
    let numeric = match value {
        Some(Value::Num(v)) => *v,
        Some(Value::Int(v)) => v.to_f64(),
        Some(Value::Tensor(t)) if t.data.len() == 1 => t.data[0],
        Some(Value::LogicalArray(a)) if a.data.len() == 1 => a.data[0] as f64,
        Some(Value::Bool(v)) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        None => DEFAULT_CACHE_SIZE as f64,
        _ => {
            return Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
                MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
            ))
        }
    };
    if !numeric.is_finite() || numeric < 1.0 || numeric.fract() != 0.0 {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
            MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
        ));
    }
    if numeric > usize::MAX as f64 {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
            MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
        ));
    }
    Ok(numeric as usize)
}

async fn call_memoized(handle: &HandleRef, payload: Value) -> BuiltinResult<Value> {
    let Value::Cell(cell) = payload else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            "memoize: function invocation payload must be a cell array",
        ));
    };
    let args = cell.data;
    let requested_outputs = crate::current_requested_outputs();
    let (function, enabled, cache_size) = memoized_state(handle)?;

    if enabled {
        if let Some(output) = take_cache_hit(handle, &args, requested_outputs, cache_size)? {
            return Ok(output);
        }
    }

    let output = crate::call_feval_async_with_outputs(function, &args, requested_outputs).await?;
    if enabled {
        store_cache_miss(handle, args, requested_outputs, output.clone(), cache_size)?;
    }
    Ok(output)
}

fn take_cache_hit(
    handle: &HandleRef,
    args: &[Value],
    requested_outputs: usize,
    cache_size: usize,
) -> BuiltinResult<Option<Value>> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<Option<Value>> {
        let object = memoized_object_mut(target)?;
        let mut cache = object_cache(object)?;
        trim_cache(&mut cache, cache_size);
        let mut hit = None;
        for entry in &mut cache {
            if entry.nargout == requested_outputs && values_equal_for_cache(&entry.inputs, args) {
                entry.hit_count = entry.hit_count.saturating_add(1);
                hit = Some(entry.output.clone());
                break;
            }
        }
        if hit.is_some() {
            increment_counter(object, TOTAL_HITS_PROPERTY)?;
        }
        set_object_cache(object, cache)?;
        Ok(hit)
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn store_cache_miss(
    handle: &HandleRef,
    args: Vec<Value>,
    requested_outputs: usize,
    output: Value,
    cache_size: usize,
) -> BuiltinResult<()> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        let object = memoized_object_mut(target)?;
        let mut cache = object_cache(object)?;
        increment_counter(object, TOTAL_MISSES_PROPERTY)?;
        cache.push(MemoizedCacheEntry {
            inputs: args,
            nargout: requested_outputs,
            output,
            hit_count: 0,
        });
        trim_cache(&mut cache, cache_size);
        set_object_cache(object, cache)?;
        Ok(())
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn trim_cache(cache: &mut Vec<MemoizedCacheEntry>, cache_size: usize) {
    if cache.len() > cache_size {
        let remove = cache.len() - cache_size;
        cache.drain(0..remove);
    }
}

fn memoized_member(handle: &HandleRef, payload: Value) -> BuiltinResult<Value> {
    let member = member_name(payload)?;
    match member.as_str() {
        "stats" => stats_for_handle(handle),
        "clearCache" => {
            clear_cache_for_handle(handle)?;
            Ok(Value::Num(0.0))
        }
        FUNCTION_PROPERTY | ENABLED_PROPERTY | CACHE_SIZE_PROPERTY => {
            runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(object) => object.properties.get(member.as_str()).cloned(),
                _ => None,
            })
            .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
            .ok_or_else(|| {
                memoize_error(
                    &MEMOIZE_ERROR_INVALID_INDEX,
                    format!("memoize: unknown MemoizedFunction member '{member}'"),
                )
            })
        }
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            format!("memoize: unknown MemoizedFunction member '{member}'"),
        )),
    }
}

fn member_name(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(name) => Ok(name),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        other => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            format!("memoize: invalid member selector {other:?}"),
        )),
    }
}

fn clear_cache_for_handle(handle: &HandleRef) -> BuiltinResult<()> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        let object = memoized_object_mut(target)?;
        reset_object_cache(object)
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn stats_for_handle(handle: &HandleRef) -> BuiltinResult<Value> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<Value> {
        let object = memoized_object_mut(target)?;
        let cache_size = parse_cache_size(object.properties.get(CACHE_SIZE_PROPERTY))?;
        let mut entries = object_cache(object)?;
        trim_cache(&mut entries, cache_size);
        set_object_cache(object, entries.clone())?;
        let total_hits = object_counter(object, TOTAL_HITS_PROPERTY);
        let total_misses = object_counter(object, TOTAL_MISSES_PROPERTY);

        let mut input_cells = Vec::with_capacity(entries.len());
        let mut nargout = Vec::with_capacity(entries.len());
        let mut output_cells = Vec::with_capacity(entries.len());
        let mut hit_count = Vec::with_capacity(entries.len());
        for entry in &entries {
            input_cells.push(Value::Cell(
                CellArray::new(entry.inputs.clone(), 1, entry.inputs.len())
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
            ));
            nargout.push(entry.nargout as f64);
            output_cells.push(Value::Cell(output_value_to_cell(
                &entry.output,
                entry.nargout,
            )?));
            hit_count.push(entry.hit_count as f64);
        }

        let cache_len = entries.len();
        let mut cache = StructValue::new();
        cache.insert(
            "Inputs",
            Value::Cell(
                CellArray::new(input_cells, 1, cache_len)
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
            ),
        );
        cache.insert(
            "Nargout",
            Value::Tensor(
                Tensor::new(nargout, vec![1, cache_len])
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
            ),
        );
        cache.insert(
            "Outputs",
            Value::Cell(
                CellArray::new(output_cells, 1, cache_len)
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
            ),
        );
        cache.insert(
            "HitCount",
            Value::Tensor(
                Tensor::new(hit_count, vec![1, cache_len])
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
            ),
        );
        cache.insert("TotalHits", Value::Num(total_hits as f64));
        cache.insert("TotalMisses", Value::Num(total_misses as f64));

        let mut stats = StructValue::new();
        stats.insert("Cache", Value::Struct(cache));
        Ok(Value::Struct(stats))
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn memoized_object_mut(target: &mut Value) -> BuiltinResult<&mut ObjectInstance> {
    match target {
        Value::Object(object) if object.class_name == MEMOIZED_FUNCTION_CLASS => Ok(object),
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: invalid object target",
        )),
    }
}

fn empty_cache_value() -> BuiltinResult<Value> {
    Ok(Value::Cell(CellArray::new(Vec::new(), 1, 0).map_err(
        |err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")),
    )?))
}

fn reset_object_cache(object: &mut ObjectInstance) -> BuiltinResult<()> {
    object
        .properties
        .insert(CACHE_PROPERTY.to_string(), empty_cache_value()?);
    object
        .properties
        .insert(TOTAL_HITS_PROPERTY.to_string(), Value::Num(0.0));
    object
        .properties
        .insert(TOTAL_MISSES_PROPERTY.to_string(), Value::Num(0.0));
    Ok(())
}

fn object_cache(object: &ObjectInstance) -> BuiltinResult<Vec<MemoizedCacheEntry>> {
    let Some(value) = object.properties.get(CACHE_PROPERTY) else {
        return Ok(Vec::new());
    };
    let Value::Cell(cell) = value else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache property is invalid",
        ));
    };
    cell.data.iter().map(decode_cache_entry).collect()
}

fn set_object_cache(
    object: &mut ObjectInstance,
    entries: Vec<MemoizedCacheEntry>,
) -> BuiltinResult<()> {
    let values = entries
        .into_iter()
        .map(encode_cache_entry)
        .collect::<BuiltinResult<Vec<_>>>()?;
    object.properties.insert(
        CACHE_PROPERTY.to_string(),
        Value::Cell(
            CellArray::new(values.clone(), 1, values.len())
                .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
        ),
    );
    Ok(())
}

fn encode_cache_entry(entry: MemoizedCacheEntry) -> BuiltinResult<Value> {
    let mut fields = StructValue::new();
    let input_len = entry.inputs.len();
    fields.insert(
        "Inputs",
        Value::Cell(
            CellArray::new(entry.inputs, 1, input_len)
                .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
        ),
    );
    fields.insert("Nargout", Value::Num(entry.nargout as f64));
    fields.insert("Output", entry.output);
    fields.insert("HitCount", Value::Num(entry.hit_count as f64));
    Ok(Value::Struct(fields))
}

fn decode_cache_entry(value: &Value) -> BuiltinResult<MemoizedCacheEntry> {
    let Value::Struct(fields) = value else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry is invalid",
        ));
    };
    let inputs = match fields.fields.get("Inputs") {
        Some(Value::Cell(cell)) => cell.data.clone(),
        _ => {
            return Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_OBJECT,
                "memoize: internal cache entry inputs are invalid",
            ))
        }
    };
    let nargout = numeric_counter(fields.fields.get("Nargout")).ok_or_else(|| {
        memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry output count is invalid",
        )
    })?;
    let output = fields.fields.get("Output").cloned().ok_or_else(|| {
        memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry output is missing",
        )
    })?;
    let hit_count = numeric_counter(fields.fields.get("HitCount")).unwrap_or(0);
    Ok(MemoizedCacheEntry {
        inputs,
        nargout,
        output,
        hit_count,
    })
}

fn increment_counter(object: &mut ObjectInstance, name: &str) -> BuiltinResult<()> {
    let next = object_counter(object, name).saturating_add(1);
    object
        .properties
        .insert(name.to_string(), Value::Num(next as f64));
    Ok(())
}

fn object_counter(object: &ObjectInstance, name: &str) -> usize {
    numeric_counter(object.properties.get(name)).unwrap_or(0)
}

fn numeric_counter(value: Option<&Value>) -> Option<usize> {
    match value {
        Some(Value::Num(value)) if value.is_finite() && *value >= 0.0 => Some(*value as usize),
        Some(Value::Int(value)) if value.to_i64() >= 0 => Some(value.to_i64() as usize),
        _ => None,
    }
}

fn output_value_to_cell(value: &Value, requested_outputs: usize) -> BuiltinResult<CellArray> {
    let values = match value {
        Value::OutputList(values) => values.clone(),
        _ if requested_outputs == 0 => Vec::new(),
        value => vec![value.clone()],
    };
    let cols = values.len();
    CellArray::new(values, 1, cols)
        .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))
}

fn values_equal_for_cache(lhs: &[Value], rhs: &[Value]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| value_equal_for_cache(left, right))
}

fn value_equal_for_cache(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Num(a), Value::Num(b)) => floats_equal_nan(*a, *b),
        (Value::Int(a), Value::Int(b)) => ints_equal(a, b),
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            floats_equal_nan(*ar, *br) && floats_equal_nan(*ai, *bi)
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::CharArray(a), Value::CharArray(b)) => {
            a.rows == b.rows && a.cols == b.cols && a.data == b.data
        }
        (Value::StringArray(a), Value::StringArray(b)) => string_arrays_equal(a, b),
        (Value::Tensor(a), Value::Tensor(b)) => tensors_equal(a, b),
        (Value::SparseTensor(a), Value::SparseTensor(b)) => sparse_tensors_equal(a, b),
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => complex_tensors_equal(a, b),
        (Value::LogicalArray(a), Value::LogicalArray(b)) => logical_arrays_equal(a, b),
        (Value::Cell(a), Value::Cell(b)) => cells_equal(a, b),
        (Value::Struct(a), Value::Struct(b)) => structs_equal(a, b),
        (Value::Object(a), Value::Object(b)) => objects_equal(a, b),
        (Value::HandleObject(a), Value::HandleObject(b)) => a == b,
        (Value::Listener(a), Value::Listener(b)) => a == b,
        (Value::OutputList(a), Value::OutputList(b)) => values_equal_for_cache(a, b),
        (Value::FunctionHandle(a), Value::FunctionHandle(b))
        | (Value::ExternalFunctionHandle(a), Value::ExternalFunctionHandle(b))
        | (Value::MethodFunctionHandle(a), Value::MethodFunctionHandle(b))
        | (Value::ClassRef(a), Value::ClassRef(b)) => a == b,
        (Value::Symbolic(a), Value::Symbolic(b)) => a == b,
        (
            Value::BoundFunctionHandle {
                name: an,
                function: af,
            },
            Value::BoundFunctionHandle {
                name: bn,
                function: bf,
            },
        ) => an == bn && af == bf,
        (Value::Closure(a), Value::Closure(b)) => {
            a.function_name == b.function_name
                && a.bound_function == b.bound_function
                && values_equal_for_cache(&a.captures, &b.captures)
        }
        (Value::MException(a), Value::MException(b)) => a == b,
        (Value::GpuTensor(a), Value::GpuTensor(b)) => a == b,
        _ => false,
    }
}

fn floats_equal_nan(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

fn ints_equal(a: &IntValue, b: &IntValue) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b) && a == b
}

fn tensors_equal(a: &Tensor, b: &Tensor) -> bool {
    a.shape == b.shape
        && a.dtype == b.dtype
        && a.data
            .iter()
            .zip(b.data.iter())
            .all(|(x, y)| floats_equal_nan(*x, *y))
}

fn sparse_tensors_equal(a: &SparseTensor, b: &SparseTensor) -> bool {
    a.rows == b.rows
        && a.cols == b.cols
        && a.col_ptrs == b.col_ptrs
        && a.row_indices == b.row_indices
        && a.values
            .iter()
            .zip(b.values.iter())
            .all(|(x, y)| floats_equal_nan(*x, *y))
}

fn complex_tensors_equal(a: &ComplexTensor, b: &ComplexTensor) -> bool {
    a.shape == b.shape
        && a.data
            .iter()
            .zip(b.data.iter())
            .all(|(x, y)| floats_equal_nan(x.0, y.0) && floats_equal_nan(x.1, y.1))
}

fn string_arrays_equal(a: &StringArray, b: &StringArray) -> bool {
    a.shape == b.shape && a.data == b.data
}

fn logical_arrays_equal(a: &LogicalArray, b: &LogicalArray) -> bool {
    a.shape == b.shape && a.data == b.data
}

fn cells_equal(a: &CellArray, b: &CellArray) -> bool {
    a.shape == b.shape && values_equal_for_cache(&a.data, &b.data)
}

fn structs_equal(a: &StructValue, b: &StructValue) -> bool {
    a.fields.len() == b.fields.len()
        && a.fields.iter().all(|(name, value)| {
            b.fields
                .get(name)
                .map(|other| value_equal_for_cache(value, other))
                .unwrap_or(false)
        })
}

fn objects_equal(a: &ObjectInstance, b: &ObjectInstance) -> bool {
    a.class_name == b.class_name && structs_equal(&object_properties(a), &object_properties(b))
}

fn object_properties(object: &ObjectInstance) -> StructValue {
    let mut fields = StructValue::new();
    for (name, value) in &object.properties {
        fields.insert(name.clone(), value.clone());
    }
    fields
}

fn function_key(function: &Value) -> String {
    match function {
        Value::FunctionHandle(name) => format!("fh:{name}"),
        Value::ExternalFunctionHandle(name) => format!("external:{name}"),
        Value::MethodFunctionHandle(name) => format!("method:{name}"),
        Value::BoundFunctionHandle { name, function } => format!("bound:{function}:{name}"),
        Value::Closure(closure) => format!(
            "closure:{}:{:?}:{}",
            closure.function_name,
            closure.bound_function,
            values_fingerprint(&closure.captures)
        ),
        other => format!("other:{other:?}"),
    }
}

fn values_fingerprint(values: &[Value]) -> String {
    values
        .iter()
        .map(value_fingerprint)
        .collect::<Vec<_>>()
        .join("|")
}

fn value_fingerprint(value: &Value) -> String {
    match value {
        Value::Num(value) if value.is_nan() => "num:NaN".to_string(),
        Value::Tensor(tensor) => format!("tensor:{:?}:{:?}", tensor.shape, tensor.data),
        Value::ComplexTensor(tensor) => format!("complex:{:?}:{:?}", tensor.shape, tensor.data),
        Value::Cell(cell) => format!("cell:{:?}:{}", cell.shape, values_fingerprint(&cell.data)),
        Value::Struct(struct_value) => format!(
            "struct:{}",
            struct_value
                .fields
                .iter()
                .map(|(name, value)| format!("{name}={}", value_fingerprint(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Closure(closure) => format!(
            "closure:{}:{:?}:{}",
            closure.function_name,
            closure.bound_function,
            values_fingerprint(&closure.captures)
        ),
        other => format!("{other:?}"),
    }
}

fn memoize_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("memoize");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) fn reset_memoize_registry_for_test() {
    MEMOIZE_REGISTRY.with(|registry| {
        registry.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::{Arc, Mutex};

    fn counting_invoker(counter: Arc<Mutex<usize>>) -> Arc<crate::user_functions::FunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            let counter = Arc::clone(&counter);
            let args = args.to_vec();
            Box::pin(async move {
                *counter.lock().unwrap() += 1;
                let base = match args.first() {
                    Some(Value::Num(value)) => *value,
                    Some(Value::Tensor(tensor)) => tensor.data.first().copied().unwrap_or(0.0),
                    _ => 0.0,
                };
                if requested_outputs == 2 {
                    Ok(Value::OutputList(vec![
                        Value::Num(base + 1.0),
                        Value::Num(base + 2.0),
                    ]))
                } else {
                    Ok(Value::Num(base + 1.0))
                }
            })
        })
    }

    fn echo_first_invoker() -> Arc<crate::user_functions::FunctionInvoker> {
        Arc::new(move |_function, args, _requested_outputs| {
            let first = args.first().cloned().unwrap_or(Value::Num(0.0));
            Box::pin(async move { Ok(first) })
        })
    }

    fn memoized() -> Value {
        block_on(memoize_builtin(Value::BoundFunctionHandle {
            name: "step".to_string(),
            function: 42,
        }))
        .expect("memoized function")
    }

    fn set_handle_property(handle: &HandleRef, name: &str, value: Value) {
        runmat_gc::gc_with_value_mut(&handle.target, |target| {
            let Value::Object(object) = target else {
                panic!("expected object target")
            };
            object.properties.insert(name.to_string(), value);
        })
        .expect("mutate handle target");
    }

    fn cache_entries(stats: &Value) -> &StructValue {
        let Value::Struct(stats) = stats else {
            panic!("expected stats struct")
        };
        let Some(Value::Struct(cache)) = stats.fields.get("Cache") else {
            panic!("expected Cache struct")
        };
        cache
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn memoize_returns_shared_handle_for_same_function() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();

        let first = memoized();
        let second = memoized();

        let (Value::HandleObject(first), Value::HandleObject(second)) = (first, second) else {
            panic!("expected handle objects");
        };
        assert_eq!(first.target, second.target);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repeated_call_hits_cache_for_same_inputs_and_nargout() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();

        let first = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(4.0)],
            1,
        ))
        .expect("first call");
        let second = block_on(crate::call_feval_async_with_outputs(
            f,
            &[Value::Num(4.0)],
            1,
        ))
        .expect("second call");

        assert_eq!(first, Value::Num(5.0));
        assert_eq!(second, Value::Num(5.0));
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_inputs_match_existing_cache_entry() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();

        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(f64::NAN)],
            1,
        ))
        .expect("first call");
        let _ = block_on(crate::call_feval_async_with_outputs(
            f,
            &[Value::Num(f64::NAN)],
            1,
        ))
        .expect("second call");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn requested_output_count_is_part_of_cache_key() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();

        let one = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(2.0)],
            1,
        ))
        .expect("one output");
        let two = block_on(crate::call_feval_async_with_outputs(
            f,
            &[Value::Num(2.0)],
            2,
        ))
        .expect("two outputs");

        assert_eq!(one, Value::Num(3.0));
        assert_eq!(
            two,
            Value::OutputList(vec![Value::Num(3.0), Value::Num(4.0)])
        );
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn disabled_memoized_function_bypasses_cache_and_stats() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, ENABLED_PROPERTY, Value::Bool(false));

        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(4.0)],
            1,
        ))
        .expect("first call");
        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(4.0)],
            1,
        ))
        .expect("second call");
        let stats = block_on(stats_builtin(f)).expect("stats");
        let cache = cache_entries(&stats);

        assert_eq!(*counter.lock().unwrap(), 2);
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));
        assert_eq!(cache.fields.get("TotalHits"), Some(&Value::Num(0.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cache_size_evicts_oldest_entries() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, CACHE_SIZE_PROPERTY, Value::Num(2.0));

        for value in [1.0, 2.0, 3.0] {
            let _ = block_on(crate::call_feval_async_with_outputs(
                f.clone(),
                &[Value::Num(value)],
                1,
            ))
            .expect("call");
        }
        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(1.0)],
            1,
        ))
        .expect("evicted call");
        let stats = block_on(stats_builtin(f)).expect("stats");
        let cache = cache_entries(&stats);

        let Some(Value::Cell(inputs)) = cache.fields.get("Inputs") else {
            panic!("expected Inputs cell")
        };
        assert_eq!(inputs.data.len(), 2);
        assert_eq!(*counter.lock().unwrap(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_cache_and_clear_all_reset_counters() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ));
        let f = memoized();

        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(8.0)],
            1,
        ))
        .expect("first call");
        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(8.0)],
            1,
        ))
        .expect("second call");
        block_on(clear_cache_builtin(f.clone())).expect("clear cache");
        let stats = block_on(stats_builtin(f.clone())).expect("stats");
        let cache = cache_entries(&stats);
        assert_eq!(cache.fields.get("TotalHits"), Some(&Value::Num(0.0)));
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));

        let _ = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            &[Value::Num(8.0)],
            1,
        ))
        .expect("after clear");
        block_on(clear_all_memoized_caches_builtin()).expect("clear all");
        let stats = block_on(stats_builtin(f)).expect("stats after all");
        let cache = cache_entries(&stats);
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_function_handle() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let err = block_on(memoize_builtin(Value::Num(1.0))).expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidFunction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_cache_size_errors_before_dispatch() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let f = memoized();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, CACHE_SIZE_PROPERTY, Value::Num(0.0));

        let err = block_on(crate::call_feval_async_with_outputs(
            f,
            &[Value::Num(4.0)],
            1,
        ))
        .expect_err("expected cache size error");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidCacheSize"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn member_access_returns_properties_and_stats() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let f = memoized();

        let enabled = block_on(memoized_subsref_builtin(
            f.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(ENABLED_PROPERTY.to_string()),
        ))
        .expect("enabled");
        let stats = block_on(memoized_subsref_builtin(
            f,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String("stats".to_string()),
        ))
        .expect("stats");

        assert_eq!(enabled, Value::Bool(true));
        assert!(matches!(stats, Value::Struct(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn internal_cache_property_is_not_exposed_as_public_member() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let f = memoized();

        let err = block_on(memoized_subsref_builtin(
            f,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(CACHE_PROPERTY.to_string()),
        ))
        .expect_err("internal cache property should not be public");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidIndex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalidated_memoized_handle_is_rejected() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let f = memoized();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        assert!(crate::set_handle_valid(handle, false));

        let err = block_on(stats_builtin(f)).expect_err("invalid handle should fail");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidObject"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handle_valued_outputs_are_reported_from_object_backed_cache() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let _guard =
            crate::user_functions::install_semantic_function_invoker(Some(echo_first_invoker()));
        let f = memoized();
        let payload_target = runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(
            "PayloadHandle".to_string(),
        )))
        .expect("payload allocation");
        let payload = Value::HandleObject(HandleRef {
            class_name: "PayloadHandle".to_string(),
            target: payload_target,
            valid: true,
        });

        let result = block_on(crate::call_feval_async_with_outputs(
            f.clone(),
            std::slice::from_ref(&payload),
            1,
        ))
        .expect("memoized handle output");
        let stats = block_on(stats_builtin(f)).expect("stats");
        let cache = cache_entries(&stats);
        let Some(Value::Cell(outputs)) = cache.fields.get("Outputs") else {
            panic!("expected Outputs cell")
        };
        let Some(Value::Cell(first_entry_outputs)) = outputs.data.first() else {
            panic!("expected first cached output cell")
        };

        assert_eq!(result, payload);
        assert_eq!(first_entry_outputs.data.first(), Some(&payload));
    }
}

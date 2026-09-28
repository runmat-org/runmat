//! MATLAB-compatible `containers.Map` constructor and methods for RunMat.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, CONTAINERS_MAP_ERROR_INTERNAL,
    CONTAINERS_MAP_ERROR_INVALID_ARGUMENT, CONTAINERS_MAP_ERROR_MISSING_KEY,
    MAP_RESIDENT_CONSTRUCTOR_EXTENSION, MAP_RESIDENT_ISKEY_EXTENSION,
    MAP_RESIDENT_REMOVE_EXTENSION, MAP_RESIDENT_SUBSASGN_EXTENSION, MAP_RESIDENT_SUBSREF_EXTENSION,
    MAP_RESIDENT_VALUES_EXTENSION,
};
use runmat_gc::GcHandle;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, HandleRef, IntValue, LogicalArray, ObjectInstance, StructValue, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("containers.Map");
const BUILTIN_CONSTRUCTOR: &str = "containers.Map";
const BUILTIN_KEYS: &str = "containers.Map.keys";
const BUILTIN_VALUES: &str = "containers.Map.values";
const BUILTIN_IS_KEY: &str = "containers.Map.isKey";
const BUILTIN_REMOVE: &str = "containers.Map.remove";
const BUILTIN_SUBSREF: &str = "containers.Map.subsref";
const BUILTIN_SUBSASGN: &str = "containers.Map.subsasgn";

fn contains_resident_value(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data.iter().any(contains_resident_value),
        Value::Struct(value) => value.fields.values().any(contains_resident_value),
        Value::StructArray(array) => array.any_value(contains_resident_value),
        Value::Object(value) => value.properties.values().any(contains_resident_value),
        Value::ObjectArray(array) => array.data().iter().any(contains_resident_value),
        Value::Closure(value) => value.captures.iter().any(contains_resident_value),
        Value::OutputList(values) => values.iter().any(contains_resident_value),
        // HandleObject targets are GC-managed identities, not inline values.
        // Resolving them here could traverse cycles or a different session's heap.
        _ => false,
    }
}

fn ensure_resident_extension(
    value: &Value,
    extension: &BuiltinExtensionDescriptor,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if contains_resident_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "containers.Map",
    op_kind: GpuOpKind::Custom("map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Map storage and outputs are host-resident; resident inputs gather only through explicitly gated RunMat extensions.",
};

fn map_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    builtin: &'static str,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw
        .strip_prefix("containers.Map:")
        .map(str::trim)
        .unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> RuntimeError {
    map_error_with_detail(error, "", builtin)
}

fn map_invalid(detail: impl AsRef<str>, builtin: &'static str) -> RuntimeError {
    map_error_with_detail(&CONTAINERS_MAP_ERROR_INVALID_ARGUMENT, detail, builtin)
}

fn map_internal(detail: impl AsRef<str>, builtin: &'static str) -> RuntimeError {
    map_error_with_detail(&CONTAINERS_MAP_ERROR_INTERNAL, detail, builtin)
}

fn map_error(message: impl Into<String>, builtin: &'static str) -> RuntimeError {
    map_invalid(message.into(), builtin)
}

fn attach_builtin_context(mut error: RuntimeError, builtin: &'static str) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(builtin);
    }
    error
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "containers.Map",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Handles act as fusion sinks; map construction terminates GPU fusion plans.",
};

mod registry;
use registry::{
    deactivate_map_registry_root_if_empty, ensure_containers_map_class_registered,
    ensure_map_registry_root_registered, MAP_REGISTRY, NEXT_ID,
};

mod store;
use store::*;

mod constructor_args;
use constructor_args::{parse_constructor_args, ConstructorArgs, KeyCandidate};
#[runtime_builtin(
    name = "containers.Map",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.iter().any(contains_resident_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_CONSTRUCTOR_EXTENSION,
            BUILTIN_CONSTRUCTOR,
        )?;
    }
    let mut host_args = Vec::with_capacity(args.len());
    for value in args {
        host_args.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| attach_builtin_context(err, BUILTIN_CONSTRUCTOR))?,
        );
    }
    let parsed = parse_constructor_args(host_args, BUILTIN_CONSTRUCTOR).await?;
    let store = build_store(parsed, BUILTIN_CONSTRUCTOR)?;
    allocate_handle(store, BUILTIN_CONSTRUCTOR)
}

#[runtime_builtin(
    name = "containers.Map.keys",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_keys(map: Value) -> crate::BuiltinResult<Value> {
    with_store(&map, BUILTIN_KEYS, |store| {
        let values = store.keys();
        make_row_cell(values, BUILTIN_KEYS)
    })
}

#[runtime_builtin(
    name = "containers.Map.values",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_values(map: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.iter().any(contains_resident_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_VALUES_EXTENSION,
            BUILTIN_VALUES,
        )?;
    }
    if rest.is_empty() {
        return with_store(&map, BUILTIN_VALUES, |store| {
            make_row_cell(store.values(), BUILTIN_VALUES)
        });
    }
    if rest.len() != 1 {
        return Err(map_error(
            "containers.Map: values expects M or M,keySet",
            BUILTIN_VALUES,
        ));
    }
    let Value::Cell(key_set) = &rest[0] else {
        return Err(map_error(
            "containers.Map: values keySet must be a cell array",
            BUILTIN_VALUES,
        ));
    };
    let mut keys = Vec::with_capacity(key_set.data.len());
    for key in &key_set.data {
        keys.push(
            gather_if_needed_async(key)
                .await
                .map_err(|err| attach_builtin_context(err, BUILTIN_VALUES))?,
        );
    }
    with_store(&map, BUILTIN_VALUES, |store| {
        let mut values = Vec::with_capacity(keys.len());
        for key in &keys {
            let normalized = normalize_key(key, store.key_type, BUILTIN_VALUES)?;
            values.push(store.get(&normalized).ok_or_else(|| {
                map_descriptor_error(&CONTAINERS_MAP_ERROR_MISSING_KEY, BUILTIN_VALUES)
            })?);
        }
        crate::make_cell_with_shape(values, vec![key_set.rows, key_set.cols])
            .map_err(|err| map_error(format!("containers.Map: {err}"), BUILTIN_VALUES))
    })
}

#[runtime_builtin(
    name = "containers.Map.isKey",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_is_key(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
    ensure_resident_extension(&key_spec, &MAP_RESIDENT_ISKEY_EXTENSION, BUILTIN_IS_KEY)?;
    let key_type = with_store(&map, BUILTIN_IS_KEY, |store| Ok(store.key_type))?;
    let collection = collect_key_spec(&key_spec, key_type, BUILTIN_IS_KEY).await?;
    with_store(&map, BUILTIN_IS_KEY, |store| {
        let mut flags = Vec::with_capacity(collection.values.len());
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type, BUILTIN_IS_KEY)?;
            flags.push(store.contains(&normalized));
        }
        if collection.values.len() == 1 {
            Ok(Value::Bool(flags[0]))
        } else {
            let data: Vec<u8> = flags.into_iter().map(|b| if b { 1 } else { 0 }).collect();
            let logical = LogicalArray::new(data, collection.shape)
                .map_err(|e| map_error(format!("containers.Map: {e}"), BUILTIN_IS_KEY))?;
            Ok(Value::LogicalArray(logical))
        }
    })
}

#[runtime_builtin(
    name = "containers.Map.remove",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_remove(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
    ensure_resident_extension(&key_spec, &MAP_RESIDENT_REMOVE_EXTENSION, BUILTIN_REMOVE)?;
    let key_type = with_store(&map, BUILTIN_REMOVE, |store| Ok(store.key_type))?;
    let collection = collect_key_spec(&key_spec, key_type, BUILTIN_REMOVE).await?;
    with_store_mut(&map, BUILTIN_REMOVE, |store| {
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type, BUILTIN_REMOVE)?;
            store.remove(&normalized, BUILTIN_REMOVE)?;
        }
        Ok(())
    })?;
    Ok(map)
}

#[runtime_builtin(
    name = "containers.Map.subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsref(map: Value, subscript: Value) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(map, path, containers_map_read_step, None).await
}

async fn containers_map_read_step(
    map: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    ensure_resident_extension(&payload, &MAP_RESIDENT_SUBSREF_EXTENSION, BUILTIN_SUBSREF)?;
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsref expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSREF,
        ));
    }
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => {
            let mut args = extract_key_arguments(&payload, BUILTIN_SUBSREF)?;
            if args.is_empty() {
                return Err(map_error(
                    "containers.Map: indexing requires at least one key",
                    BUILTIN_SUBSREF,
                ));
            }
            if args.len() != 1 {
                return Err(map_error(
                    "containers.Map: indexing expects a single key argument",
                    BUILTIN_SUBSREF,
                ));
            }
            let key_arg = args.remove(0);
            let key_type = with_store(&map, BUILTIN_SUBSREF, |store| Ok(store.key_type))?;
            let collection = collect_key_spec(&key_arg, key_type, BUILTIN_SUBSREF).await?;
            if collection.values.len() != 1 {
                return Err(map_error(
                    "containers.Map: indexing requires exactly one scalar key",
                    BUILTIN_SUBSREF,
                ));
            }
            with_store(&map, BUILTIN_SUBSREF, |store| {
                let normalized =
                    normalize_key(&collection.values[0], store.key_type, BUILTIN_SUBSREF)?;
                store.get(&normalized).ok_or_else(|| {
                    map_descriptor_error(&CONTAINERS_MAP_ERROR_MISSING_KEY, BUILTIN_SUBSREF)
                })
            })
        }
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = string_from_value(
                &payload,
                "containers.Map: property name must be text",
                BUILTIN_SUBSREF,
            )?;
            with_store(&map, BUILTIN_SUBSREF, |store| {
                match field.to_ascii_lowercase().as_str() {
                    "count" => Ok(Value::Int(IntValue::U64(store.len() as u64))),
                    "keytype" => char_array_value(store.key_type.matlab_name(), BUILTIN_SUBSREF),
                    "valuetype" => {
                        char_array_value(store.value_type.matlab_name(), BUILTIN_SUBSREF)
                    }
                    other => Err(map_error(
                        format!("containers.Map: no such property '{other}'"),
                        BUILTIN_SUBSREF,
                    )),
                }
            })
        }
        crate::object::indexing::ObjectIndexKind::Brace => Err(map_error(
            "containers.Map: curly-brace indexing is not supported.",
            BUILTIN_SUBSREF,
        )),
    }
}

#[runtime_builtin(
    name = "containers.Map.subsasgn",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsasgn(
    map: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        map,
        path,
        vec![rhs],
        containers_map_read_step,
        |map, step, mut values| async move {
            let rhs = values.pop().ok_or_else(|| {
                map_error(
                    "containers.Map: assignment value is missing",
                    BUILTIN_SUBSASGN,
                )
            })?;
            containers_map_write_step(map, step, rhs).await
        },
        None,
    )
    .await
}

async fn containers_map_write_step(
    map: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let payload = step.selector_value()?;
    if contains_resident_value(&payload) || contains_resident_value(&rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAP_RESIDENT_SUBSASGN_EXTENSION,
            BUILTIN_SUBSASGN,
        )?;
    }
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsasgn expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSASGN,
        ));
    }
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => {
            let mut args = extract_key_arguments(&payload, BUILTIN_SUBSASGN)?;
            if args.is_empty() {
                return Err(map_error(
                    "containers.Map: assignment requires at least one key",
                    BUILTIN_SUBSASGN,
                ));
            }
            if args.len() != 1 {
                return Err(map_error(
                    "containers.Map: assignment expects a single key argument",
                    BUILTIN_SUBSASGN,
                ));
            }
            let key_arg = args.remove(0);
            let key_type = with_store(&map, BUILTIN_SUBSASGN, |store| Ok(store.key_type))?;
            let KeyCollection {
                values: key_values, ..
            } = collect_key_spec(&key_arg, key_type, BUILTIN_SUBSASGN).await?;
            if key_values.len() != 1 {
                return Err(map_error(
                    "containers.Map: assignment requires exactly one scalar key",
                    BUILTIN_SUBSASGN,
                ));
            }
            let values =
                expand_assignment_values(rhs.clone(), key_values.len(), BUILTIN_SUBSASGN).await?;
            with_store_mut(&map, BUILTIN_SUBSASGN, move |store| {
                for (key_raw, value) in key_values.into_iter().zip(values.into_iter()) {
                    let (normalized, canonical) =
                        canonicalize_key(key_raw, store.key_type, BUILTIN_SUBSASGN)?;
                    let entry = MapEntry {
                        normalized,
                        key_value: canonical,
                        value,
                    };
                    store.set(entry, BUILTIN_SUBSASGN)?;
                }
                Ok(())
            })?;
            Ok(map)
        }
        crate::object::indexing::ObjectIndexKind::Member => Err(map_error(
            "containers.Map: property assignments are not supported.",
            BUILTIN_SUBSASGN,
        )),
        crate::object::indexing::ObjectIndexKind::Brace => Err(map_error(
            "containers.Map: curly-brace assignment is not supported.",
            BUILTIN_SUBSASGN,
        )),
    }
}

fn build_store(args: ConstructorArgs, builtin: &'static str) -> BuiltinResult<MapStore> {
    let mut store = MapStore::new(args.key_type, args.value_type, args.uniform_values);
    for (candidate, value) in args.keys.into_iter().zip(args.values.into_iter()) {
        store.insert_new(
            MapEntry {
                normalized: candidate.normalized,
                key_value: candidate.canonical,
                value,
            },
            builtin,
        )?;
    }
    Ok(store)
}

fn allocate_handle(store: MapStore, builtin: &'static str) -> BuiltinResult<Value> {
    ensure_containers_map_class_registered();

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    ensure_map_registry_root_registered(builtin)?;
    MAP_REGISTRY.with(|registry| {
        registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry is already borrowed", builtin))?
            .insert(id, store);
        Ok::<(), RuntimeError>(())
    })?;
    let mut storage = ObjectInstance::new(CLASS_IDENTITY);
    storage
        .properties
        .insert("id".to_string(), Value::Int(IntValue::U64(id)));
    let gc = match runmat_gc::gc_allocate(Value::Object(storage)) {
        Ok(gc) => gc,
        Err(e) => {
            MAP_REGISTRY.with(|registry| {
                if let Ok(mut registry) = registry.try_borrow_mut() {
                    registry.remove(&id);
                }
            });
            deactivate_map_registry_root_if_empty();
            return Err(map_error(format!("containers.Map: {e}"), builtin));
        }
    };
    MAP_REGISTRY.with(|registry| {
        let mut registry = registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry is already borrowed", builtin))?;
        let store = registry
            .get_mut(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        store.storage = Some(gc);
        Ok::<(), RuntimeError>(())
    })?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_IDENTITY.owned(),
        target: gc,
        valid: true,
    }))
}

fn with_store<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let id = map_id(handle, builtin)?;
    MAP_REGISTRY.with(|registry| {
        let registry = registry
            .try_borrow()
            .map_err(|_| map_internal("containers.Map: registry already borrowed", builtin))?;
        let store = registry
            .get(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        f(store)
    })
}

fn with_store_mut<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&mut MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let id = map_id(handle, builtin)?;
    MAP_REGISTRY.with(|registry| {
        let mut registry = registry
            .try_borrow_mut()
            .map_err(|_| map_internal("containers.Map: registry already borrowed", builtin))?;
        let store = registry
            .get_mut(&id)
            .ok_or_else(|| map_internal("containers.Map: internal storage not found", builtin))?;
        f(store)
    })
}

fn extract_handle<'a>(value: &'a Value, builtin: &'static str) -> BuiltinResult<&'a HandleRef> {
    match value {
        Value::HandleObject(handle) => Ok(handle),
        _ => Err(map_error(
            "containers.Map: expected a containers.Map handle",
            builtin,
        )),
    }
}

fn ensure_handle(handle: &HandleRef, builtin: &'static str) -> BuiltinResult<()> {
    if !crate::is_handle_valid(handle) {
        return Err(map_error("containers.Map: handle is invalid", builtin));
    }
    if !handle.class_name.is(CLASS_IDENTITY) {
        return Err(map_error(
            format!(
                "containers.Map: expected handle of class '{}', got '{}'",
                CLASS_IDENTITY, handle.class_name
            ),
            builtin,
        ));
    }
    Ok(())
}

fn map_id(handle: &HandleRef, builtin: &'static str) -> BuiltinResult<u64> {
    let storage = runmat_gc::gc_clone_value(&handle.target).map_err(|e| {
        map_internal(
            format!("containers.Map: invalid handle storage: {e}"),
            builtin,
        )
    })?;
    let id_value = match &storage {
        Value::Object(object) if object.class_name.is(CLASS_IDENTITY) => {
            object.properties.get("id")
        }
        Value::Struct(StructValue { fields }) => fields.get("id"),
        other => {
            return Err(map_internal(
                format!("containers.Map: internal storage has unexpected shape {other:?}"),
                builtin,
            ));
        }
    };
    match id_value {
        Some(Value::Int(IntValue::U64(id))) => Ok(*id),
        Some(Value::Int(other)) => {
            let id = other.to_i64();
            if id < 0 {
                Err(map_internal(
                    "containers.Map: negative map identifier",
                    builtin,
                ))
            } else {
                Ok(id as u64)
            }
        }
        Some(Value::Num(n)) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => {
            if *n >= u64::MAX as f64 {
                Err(map_internal(
                    "containers.Map: map identifier out of range",
                    builtin,
                ))
            } else {
                Ok(*n as u64)
            }
        }
        _ => Err(map_internal(
            "containers.Map: corrupted storage identifier",
            builtin,
        )),
    }
}

mod input_conversion;
use input_conversion::*;

mod value_conversion;
use value_conversion::*;

fn make_row_cell(values: Vec<Value>, builtin: &'static str) -> BuiltinResult<Value> {
    let cols = values.len();
    crate::make_cell_with_shape(values, vec![1, cols])
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
}

fn extract_key_arguments(payload: &Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(ptr.clone());
            }
            Ok(out)
        }
        other => Err(map_error(
            format!("containers.Map: expected key arguments in a cell array, got {other:?}"),
            builtin,
        )),
    }
}

async fn expand_assignment_values(
    value: Value,
    expected: usize,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    if expected == 1 {
        Ok(vec![host])
    } else {
        let values = flatten_values(&host, builtin).await?;
        if values.len() != expected {
            return Err(map_error(
                format!(
                    "containers.Map: assignment with {} keys requires {} values (got {})",
                    expected,
                    expected,
                    values.len()
                ),
                builtin,
            ));
        }
        Ok(values)
    }
}

struct KeyCollection {
    values: Vec<Value>,
    shape: Vec<usize>,
}

async fn collect_key_spec(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<KeyCollection> {
    let host = gather_if_needed_async(value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    match &host {
        Value::Cell(cell) => {
            let mut values = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                values.push(
                    gather_if_needed_async(ptr)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(KeyCollection {
                values,
                shape: vec![cell.rows, cell.cols],
            })
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(KeyCollection {
            values: vec![Value::String(sa.data[0].clone())],
            shape: vec![1, 1],
        }),
        Value::CharArray(ca) if ca.rows == 1 => Ok(KeyCollection {
            values: vec![Value::CharArray(ca.clone())],
            shape: vec![1, 1],
        }),
        Value::Tensor(t) if key_type != KeyType::Char && t.len() == 1 => Ok(KeyCollection {
            values: tensor_elements_to_values(t),
            shape: vec![1, 1],
        }),
        Value::StringArray(_) | Value::CharArray(_) | Value::Tensor(_) => Err(map_error(
            "containers.Map: multiple keys must be supplied in a cell array",
            builtin,
        )),
        _ => Ok(KeyCollection {
            values: vec![host.clone()],
            shape: vec![1, 1],
        }),
    }
}

pub fn map_length(value: &Value) -> Option<usize> {
    if let Value::HandleObject(handle) = value {
        if crate::is_handle_valid(handle) && handle.class_name.is(CLASS_IDENTITY) {
            if let Ok(id) = map_id(handle, BUILTIN_CONSTRUCTOR) {
                return MAP_REGISTRY.with(|registry| {
                    registry
                        .try_borrow()
                        .ok()
                        .and_then(|registry| registry.get(&id).map(|store| store.len()))
                });
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "tests.rs"]
pub(crate) mod tests;

//! MATLAB-compatible `containers.Map` constructor and methods for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, CONTAINERS_MAP_ERROR_INTERNAL,
    CONTAINERS_MAP_ERROR_INVALID_ARGUMENT, CONTAINERS_MAP_ERROR_MISSING_KEY,
    MAP_RESIDENT_CONSTRUCTOR_EXTENSION, MAP_RESIDENT_ISKEY_EXTENSION,
    MAP_RESIDENT_REMOVE_EXTENSION, MAP_RESIDENT_SUBSASGN_EXTENSION, MAP_RESIDENT_SUBSREF_EXTENSION,
    MAP_RESIDENT_VALUES_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, HandleRef, IntValue, LogicalArray, ObjectInstance, Tensor, Value};

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
        Value::Cell(cell) => cell.data().iter().any(contains_resident_value),
        Value::Struct(value) => value.fields.values().any(contains_resident_value),
        Value::StructArray(array) => array.any_value(contains_resident_value),
        Value::Object(value) => value.properties().values().any(contains_resident_value),
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

mod class_registration;
use class_registration::ensure_containers_map_class_registered;

mod store;
use store::*;

mod constructor_args;
use constructor_args::{parse_constructor_args, ConstructorArgs, KeyCandidate};
#[runtime_builtin(
    name = "containers.Map",
    binding_variant = "default",
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_builtin(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
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
    let _root = root_map_receiver(&map, BUILTIN_KEYS)?;
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
async fn containers_map_values(
    map: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let _root = root_map_receiver(&map, BUILTIN_VALUES)?;
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
    let mut keys = Vec::with_capacity(key_set.data().len());
    for key in key_set.data() {
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
            values.push(store.get_value(&normalized).ok_or_else(|| {
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
    let _root = root_map_receiver(&map, BUILTIN_IS_KEY)?;
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
    let _root = root_map_receiver(&map, BUILTIN_REMOVE)?;
    ensure_resident_extension(&key_spec, &MAP_RESIDENT_REMOVE_EXTENSION, BUILTIN_REMOVE)?;
    let key_type = with_store(&map, BUILTIN_REMOVE, |store| Ok(store.key_type))?;
    let collection = collect_key_spec(&key_spec, key_type, BUILTIN_REMOVE).await?;
    let normalized = collection
        .values
        .iter()
        .map(|value| normalize_key(value, key_type, BUILTIN_REMOVE))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut seen = std::collections::HashSet::new();
    if normalized.iter().any(|key| !seen.insert(key)) {
        return Err(map_descriptor_error(
            &CONTAINERS_MAP_ERROR_MISSING_KEY,
            BUILTIN_REMOVE,
        ));
    }
    with_store(&map, BUILTIN_REMOVE, |store| {
        for key in &normalized {
            if !store.contains(key) {
                return Err(map_descriptor_error(
                    &CONTAINERS_MAP_ERROR_MISSING_KEY,
                    BUILTIN_REMOVE,
                ));
            }
        }
        Ok(())
    })?;
    with_store_mut(&map, BUILTIN_REMOVE, |store, _, displaced| {
        for key in &normalized {
            displaced.push(store.remove(key, BUILTIN_REMOVE)?);
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
async fn containers_map_subsref(
    map: Value,
    subscript: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _root = root_map_receiver(&map, BUILTIN_SUBSREF)?;
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(map, path, containers_map_read_step, None).await
}

async fn containers_map_read_step(
    map: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let payload = step.selector_value()?;
    ensure_resident_extension(&payload, &MAP_RESIDENT_SUBSREF_EXTENSION, BUILTIN_SUBSREF)?;
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsref expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSREF,
        ));
    }
    let value = match step.kind() {
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
                store.get_value(&normalized).ok_or_else(|| {
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
    }?;
    crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
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
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _root = root_map_receiver(&map, BUILTIN_SUBSASGN)?;
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
) -> crate::BuiltinResult<crate::execution::RootedValue> {
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
    let value = match step.kind() {
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
            let (value_type, uniform_values, uniform_class) =
                with_store(&map, BUILTIN_SUBSASGN, |store| {
                    Ok((
                        store.value_type,
                        store.uniform_values,
                        store.uniform_class.clone(),
                    ))
                })?;
            let (normalized, canonical) = canonicalize_key(
                key_values.into_iter().next().unwrap(),
                key_type,
                BUILTIN_SUBSASGN,
            )?;
            let value = values.into_iter().next().unwrap();
            runmat_value::validate_no_transient_sequence(&value)
                .map_err(|error| map_error(format!("containers.Map: {error}"), BUILTIN_SUBSASGN))?;
            let value = value_type.normalize(value, BUILTIN_SUBSASGN)?;
            if uniform_values {
                let class = value_class_from_value(&value);
                if uniform_class
                    .as_ref()
                    .is_some_and(|existing| existing != &class)
                {
                    return Err(map_error(
                        "containers.Map: UniformValues=true requires all values to share the same MATLAB class.",
                        BUILTIN_SUBSASGN,
                    ));
                }
            }
            let entry = MapEntry {
                normalized,
                key_value: canonical,
                value,
            };
            with_store_mut(&map, BUILTIN_SUBSASGN, move |store, owner, displaced| {
                runmat_gc::gc_record_handle_write(&owner, &entry.key_value);
                runmat_gc::gc_record_handle_write(&owner, &entry.value);
                if let Some(previous) = store.set_prepared(entry) {
                    displaced.push(previous);
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
    }?;
    crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
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

    let storage = ObjectInstance::new(CLASS_IDENTITY).with_containers_map_payload(store);
    let target = runmat_gc::gc_allocate(Value::Object(storage))
        .map_err(|error| map_internal(format!("containers.Map: {error}"), builtin))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_IDENTITY.owned().into(),
        target,
        valid: true,
    }))
}

fn with_store<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return Err(map_internal(
                "containers.Map: invalid handle storage",
                builtin,
            ));
        };
        if !object.class_name().is(CLASS_IDENTITY) {
            return Err(map_internal(
                "containers.Map: invalid handle class",
                builtin,
            ));
        }
        let store = object
            .containers_map_payload()
            .ok_or_else(|| map_internal("containers.Map: missing map payload", builtin))?;
        f(store)
    })
    .map_err(|error| map_internal(format!("containers.Map: {error}"), builtin))?
}

fn with_store_mut<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&mut MapStore, runmat_gc::GcHandle, &mut Vec<MapEntry>) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let mut displaced = Vec::new();
    let result = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return Err(map_internal(
                "containers.Map: invalid handle storage",
                builtin,
            ));
        };
        if !object.class_name().is(CLASS_IDENTITY) {
            return Err(map_internal(
                "containers.Map: invalid handle class",
                builtin,
            ));
        }
        let store = object
            .containers_map_payload_mut()
            .ok_or_else(|| map_internal("containers.Map: missing map payload", builtin))?;
        f(store, handle.target, &mut displaced)
    })
    .map_err(|error| map_internal(format!("containers.Map: {error}"), builtin));
    // Value drops may release provider leases or reenter owners. The managed
    // object's exclusive borrow must end before any displaced entry is dropped.
    drop(displaced);
    result?
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

fn root_map_receiver(map: &Value, builtin: &'static str) -> BuiltinResult<runmat_gc::ExplicitRoot> {
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    runmat_gc::gc_root(handle.target)
        .map_err(|error| map_internal(format!("containers.Map: {error}"), builtin))
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
            let mut out = Vec::with_capacity(cell.data().len());
            for ptr in cell.data() {
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
            let mut values = Vec::with_capacity(cell.data().len());
            for ptr in cell.data() {
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
        Value::StringArray(sa) if sa.data().len() == 1 => Ok(KeyCollection {
            values: vec![Value::String(sa.data()[0].clone())],
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
            return runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(object) if object.class_name().is(CLASS_IDENTITY) => {
                    object.containers_map_payload().map(MapStore::len)
                }
                _ => None,
            })
            .ok()
            .flatten();
        }
    }
    None
}

#[cfg(test)]
#[path = "tests.rs"]
pub(crate) mod tests;

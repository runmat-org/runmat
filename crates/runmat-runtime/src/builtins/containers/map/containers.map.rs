//! MATLAB-compatible `containers.Map` constructor and methods for RunMat.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};

use once_cell::sync::Lazy;
use runmat_builtins::{CharArray, HandleRef, IntValue, LogicalArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::containers::type_resolvers::{
    map_cell_type, map_handle_type, map_is_key_type, map_unknown_type,
};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CLASS_NAME: &str = "containers.Map";
const MISSING_KEY_ERR: &str = "containers.Map: The specified key is not present in this container.";
const BUILTIN_CONSTRUCTOR: &str = "containers.Map";
const BUILTIN_KEYS: &str = "containers.Map.keys";
const BUILTIN_VALUES: &str = "containers.Map.values";
const BUILTIN_IS_KEY: &str = "containers.Map.isKey";
const BUILTIN_REMOVE: &str = "containers.Map.remove";
const BUILTIN_SUBSREF: &str = "containers.Map.subsref";
const BUILTIN_SUBSASGN: &str = "containers.Map.subsasgn";

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
    notes: "Map storage is host-resident; GPU inputs are gathered only when split into multiple entries.",
};

fn map_error(message: impl Into<String>, builtin: &'static str) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
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

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static MAP_REGISTRY: Lazy<RwLock<HashMap<u64, MapStore>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyType {
    Char,
    String,
    Double,
    Single,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Logical,
}

impl KeyType {
    fn matlab_name(self) -> &'static str {
        match self {
            KeyType::Char => "char",
            KeyType::String => "string",
            KeyType::Double => "double",
            KeyType::Single => "single",
            KeyType::Int32 => "int32",
            KeyType::UInt32 => "uint32",
            KeyType::Int64 => "int64",
            KeyType::UInt64 => "uint64",
            KeyType::Logical => "logical",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(value, "containers.Map: expected a KeyType string", builtin)?;
        match text.to_ascii_lowercase().as_str() {
            "char" | "character" => Ok(KeyType::Char),
            "string" => Ok(KeyType::String),
            "double" => Ok(KeyType::Double),
            "single" => Ok(KeyType::Single),
            "int32" => Ok(KeyType::Int32),
            "uint32" => Ok(KeyType::UInt32),
            "int64" => Ok(KeyType::Int64),
            "uint64" => Ok(KeyType::UInt64),
            "logical" => Ok(KeyType::Logical),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported KeyType '{other}'. Valid types: char, string, double, int32, uint32, int64, uint64, logical."
                ),
                builtin,
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    Any,
    Char,
    String,
    Double,
    Single,
    Logical,
}

impl ValueType {
    fn matlab_name(self) -> &'static str {
        match self {
            ValueType::Any => "any",
            ValueType::Char => "char",
            ValueType::String => "string",
            ValueType::Double => "double",
            ValueType::Single => "single",
            ValueType::Logical => "logical",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(
            value,
            "containers.Map: expected a ValueType string",
            builtin,
        )?;
        match text.to_ascii_lowercase().as_str() {
            "any" => Ok(ValueType::Any),
            "char" | "character" => Ok(ValueType::Char),
            "string" => Ok(ValueType::String),
            "double" => Ok(ValueType::Double),
            "single" => Ok(ValueType::Single),
            "logical" => Ok(ValueType::Logical),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported ValueType '{other}'. Valid types: any, char, string, double, single, logical."
                ),
                builtin,
            )),
        }
    }

    fn normalize(&self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        match self {
            ValueType::Any => Ok(value),
            ValueType::Char => {
                let chars = char_array_from_value(&value, builtin)?;
                Ok(Value::CharArray(chars))
            }
            ValueType::String => {
                let text = string_from_value(
                    &value,
                    "containers.Map: values must be string scalars",
                    builtin,
                )?;
                Ok(Value::String(text))
            }
            ValueType::Double | ValueType::Single => normalize_numeric_value(value, builtin),
            ValueType::Logical => normalize_logical_value(value, builtin),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum NormalizedKey {
    String(String),
    Float(u64),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

#[derive(Clone)]
struct MapEntry {
    normalized: NormalizedKey,
    key_value: Value,
    value: Value,
}

struct MapStore {
    key_type: KeyType,
    value_type: ValueType,
    uniform_values: bool,
    uniform_class: Option<ValueClass>,
    entries: Vec<MapEntry>,
    index: HashMap<NormalizedKey, usize>,
}

impl MapStore {
    fn new(key_type: KeyType, value_type: ValueType, uniform_values: bool) -> Self {
        Self {
            key_type,
            value_type,
            uniform_values,
            uniform_class: None,
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, key: &NormalizedKey) -> bool {
        self.index.contains_key(key)
    }

    fn get(&self, key: &NormalizedKey) -> Option<Value> {
        self.index
            .get(key)
            .map(|&idx| self.entries[idx].value.clone())
    }

    fn insert_new(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
        if self.index.contains_key(&entry.normalized) {
            return Err(map_error(
                "containers.Map: Duplicate key name was provided.",
                builtin,
            ));
        }
        entry.value = self.normalize_value(entry.value, builtin)?;
        self.track_uniform_class(&entry.value, builtin)?;
        let idx = self.entries.len();
        self.entries.push(entry.clone());
        self.index.insert(entry.normalized, idx);
        Ok(())
    }

    fn set(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
        entry.value = self.normalize_value(entry.value, builtin)?;
        self.track_uniform_class(&entry.value, builtin)?;
        if let Some(&idx) = self.index.get(&entry.normalized) {
            self.entries[idx].value = entry.value.clone();
            self.entries[idx].key_value = entry.key_value;
        } else {
            let idx = self.entries.len();
            self.entries.push(entry.clone());
            self.index.insert(entry.normalized, idx);
        }
        Ok(())
    }

    fn remove(&mut self, key: &NormalizedKey, builtin: &'static str) -> BuiltinResult<()> {
        let idx = match self.index.get(key) {
            Some(&idx) => idx,
            None => {
                return Err(map_error(MISSING_KEY_ERR, builtin));
            }
        };
        self.entries.remove(idx);
        self.index.clear();
        for (pos, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.normalized.clone(), pos);
        }
        if self.entries.is_empty() {
            self.uniform_class = None;
        }
        Ok(())
    }

    fn keys(&self) -> Vec<Value> {
        self.entries
            .iter()
            .map(|entry| entry.key_value.clone())
            .collect()
    }

    fn values(&self) -> Vec<Value> {
        self.entries
            .iter()
            .map(|entry| entry.value.clone())
            .collect()
    }

    fn normalize_value(&self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        self.value_type.normalize(value, builtin)
    }

    fn track_uniform_class(&mut self, value: &Value, builtin: &'static str) -> BuiltinResult<()> {
        if !self.uniform_values {
            return Ok(());
        }
        let class = ValueClass::from_value(value);
        if let Some(existing) = &self.uniform_class {
            if existing != &class {
                return Err(map_error(
                    "containers.Map: UniformValues=true requires all values to share the same MATLAB class.",
                    builtin,
                ));
            }
        } else {
            self.uniform_class = Some(class);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValueClass {
    Char,
    String,
    Double,
    Logical,
    Int,
    UInt,
    Cell,
    Struct,
    Object,
    Other(&'static str),
}

impl ValueClass {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::CharArray(_) => ValueClass::Char,
            Value::String(_) | Value::StringArray(_) => ValueClass::String,
            Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) => ValueClass::Double,
            Value::Bool(_) | Value::LogicalArray(_) => ValueClass::Logical,
            Value::Int(i) => match i {
                IntValue::I8(_) | IntValue::I16(_) | IntValue::I32(_) | IntValue::I64(_) => {
                    ValueClass::Int
                }
                IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => {
                    ValueClass::UInt
                }
            },
            Value::Cell(_) => ValueClass::Cell,
            Value::Struct(_) => ValueClass::Struct,
            Value::Object(_) | Value::HandleObject(_) | Value::Listener(_) => ValueClass::Object,
            _ => ValueClass::Other("other"),
        }
    }
}

struct ConstructorArgs {
    key_type: KeyType,
    value_type: ValueType,
    uniform_values: bool,
    keys: Vec<KeyCandidate>,
    values: Vec<Value>,
}

struct KeyCandidate {
    normalized: NormalizedKey,
    canonical: Value,
}

#[runtime_builtin(
    name = "containers.Map",
    category = "containers/map",
    summary = "Create MATLAB-style dictionary objects that map keys to values.",
    keywords = "map,containers.Map,dictionary,hash map,lookup",
    accel = "metadata",
    sink = true,
    type_resolver(map_handle_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = parse_constructor_args(args, BUILTIN_CONSTRUCTOR).await?;
    let store = build_store(parsed, BUILTIN_CONSTRUCTOR)?;
    allocate_handle(store, BUILTIN_CONSTRUCTOR)
}

#[runtime_builtin(
    name = "containers.Map.keys",
    type_resolver(map_cell_type),
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
    type_resolver(map_cell_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_values(map: Value) -> crate::BuiltinResult<Value> {
    with_store(&map, BUILTIN_VALUES, |store| {
        let values = store.values();
        make_row_cell(values, BUILTIN_VALUES)
    })
}

#[runtime_builtin(
    name = "containers.Map.isKey",
    type_resolver(map_is_key_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_is_key(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
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
    type_resolver(map_handle_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_remove(map: Value, key_spec: Value) -> crate::BuiltinResult<Value> {
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
    type_resolver(map_unknown_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsref(
    map: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<Value> {
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsref expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSREF,
        ));
    }
    match kind.as_str() {
        "()" => {
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
            with_store(&map, BUILTIN_SUBSREF, |store| {
                if collection.values.is_empty() {
                    return crate::make_cell_with_shape(Vec::new(), collection.shape.clone())
                        .map_err(|e| map_error(format!("containers.Map: {e}"), BUILTIN_SUBSREF));
                }
                if collection.values.len() == 1 {
                    let normalized =
                        normalize_key(&collection.values[0], store.key_type, BUILTIN_SUBSREF)?;
                    store
                        .get(&normalized)
                        .ok_or_else(|| map_error(MISSING_KEY_ERR, BUILTIN_SUBSREF))
                } else {
                    let mut results = Vec::with_capacity(collection.values.len());
                    for value in &collection.values {
                        let normalized = normalize_key(value, store.key_type, BUILTIN_SUBSREF)?;
                        let stored = store
                            .get(&normalized)
                            .ok_or_else(|| map_error(MISSING_KEY_ERR, BUILTIN_SUBSREF))?;
                        results.push(stored);
                    }
                    crate::make_cell_with_shape(results, collection.shape.clone())
                        .map_err(|e| map_error(format!("containers.Map: {e}"), BUILTIN_SUBSREF))
                }
            })
        }
        "." => {
            let field = string_from_value(
                &payload,
                "containers.Map: property name must be text",
                BUILTIN_SUBSREF,
            )?;
            with_store(&map, BUILTIN_SUBSREF, |store| {
                match field.to_ascii_lowercase().as_str() {
                    "count" => Ok(Value::Num(store.len() as f64)),
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
        "{}" => Err(map_error(
            "containers.Map: curly-brace indexing is not supported.",
            BUILTIN_SUBSREF,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported indexing kind '{other}'"),
            BUILTIN_SUBSREF,
        )),
    }
}

#[runtime_builtin(
    name = "containers.Map.subsasgn",
    type_resolver(map_handle_type),
    builtin_path = "crate::builtins::containers::map::containers_map"
)]
async fn containers_map_subsasgn(
    map: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    if !matches!(map, Value::HandleObject(_)) {
        return Err(map_error(
            format!("containers.Map: subsasgn expects a containers.Map handle, got {map:?}"),
            BUILTIN_SUBSASGN,
        ));
    }
    match kind.as_str() {
        "()" => {
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
        "." => Err(map_error(
            "containers.Map: property assignments are not supported.",
            BUILTIN_SUBSASGN,
        )),
        "{}" => Err(map_error(
            "containers.Map: curly-brace assignment is not supported.",
            BUILTIN_SUBSASGN,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported assignment kind '{other}'"),
            BUILTIN_SUBSASGN,
        )),
    }
}

async fn parse_constructor_args(
    args: Vec<Value>,
    builtin: &'static str,
) -> BuiltinResult<ConstructorArgs> {
    let mut index = 0usize;
    let mut keys_input: Option<Value> = None;
    let mut values_input: Option<Value> = None;

    if index < args.len() && keyword_of(&args[index]).is_none() {
        if args.len() < 2 {
            return Err(map_error(
                "containers.Map: constructor requires both keys and values when either is provided.",
                builtin,
            ));
        }
        keys_input = Some(args[index].clone());
        values_input = Some(args[index + 1].clone());
        index += 2;
    }

    let mut key_type = KeyType::Char;
    let mut value_type = ValueType::Any;
    let mut uniform_values = false;
    while index < args.len() {
        let keyword = keyword_of(&args[index]).ok_or_else(|| {
            map_error(
                "containers.Map: expected option name (e.g. 'KeyType')",
                builtin,
            )
        })?;
        index += 1;
        let Some(value) = args.get(index) else {
            return Err(map_error(
                format!("containers.Map: missing value for option '{keyword}'"),
                builtin,
            ));
        };
        index += 1;
        match keyword.as_str() {
            "keytype" => key_type = KeyType::parse(value, builtin)?,
            "valuetype" => value_type = ValueType::parse(value, builtin)?,
            "uniformvalues" => {
                uniform_values = bool_from_value(
                    value,
                    "containers.Map: UniformValues must be logical",
                    builtin,
                )?
            }
            "comparisonmethod" => {
                let text = string_from_value(
                    value,
                    "containers.Map: ComparisonMethod must be a string",
                    builtin,
                )?;
                let lowered = text.to_ascii_lowercase();
                if lowered != "strcmp" {
                    return Err(map_error(
                        "containers.Map: only ComparisonMethod='strcmp' is supported.",
                        builtin,
                    ));
                }
            }
            other => {
                return Err(map_error(
                    format!("containers.Map: unrecognised option '{other}'"),
                    builtin,
                ));
            }
        }
    }

    let keys = match keys_input {
        Some(value) => prepare_keys(value, key_type, builtin).await?,
        None => Vec::new(),
    };

    let values = match values_input {
        Some(value) => prepare_values(value, builtin).await?,
        None => Vec::new(),
    };

    if keys.len() != values.len() {
        return Err(map_error(
            format!(
                "containers.Map: number of keys ({}) must match number of values ({})",
                keys.len(),
                values.len()
            ),
            builtin,
        ));
    }

    Ok(ConstructorArgs {
        key_type,
        value_type,
        uniform_values,
        keys,
        values,
    })
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
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    MAP_REGISTRY
        .write()
        .map_err(|_| map_error("containers.Map: registry lock poisoned", builtin))?
        .insert(id, store);
    let mut struct_value = StructValue::new();
    struct_value
        .fields
        .insert("id".to_string(), Value::Int(IntValue::U64(id)));
    let storage = Value::Struct(struct_value);
    let gc = runmat_gc::gc_allocate(storage)
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_NAME.to_string(),
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
    let guard = MAP_REGISTRY
        .read()
        .map_err(|_| map_error("containers.Map: registry lock poisoned", builtin))?;
    let store = guard
        .get(&id)
        .ok_or_else(|| map_error("containers.Map: internal storage not found", builtin))?;
    f(store)
}

fn with_store_mut<F, R>(map: &Value, builtin: &'static str, f: F) -> BuiltinResult<R>
where
    F: FnOnce(&mut MapStore) -> BuiltinResult<R>,
{
    let handle = extract_handle(map, builtin)?;
    ensure_handle(handle, builtin)?;
    let id = map_id(handle, builtin)?;
    let mut guard = MAP_REGISTRY
        .write()
        .map_err(|_| map_error("containers.Map: registry lock poisoned", builtin))?;
    let store = guard
        .get_mut(&id)
        .ok_or_else(|| map_error("containers.Map: internal storage not found", builtin))?;
    f(store)
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
    if !handle.valid {
        return Err(map_error("containers.Map: handle is invalid", builtin));
    }
    if handle.class_name != CLASS_NAME {
        return Err(map_error(
            format!(
                "containers.Map: expected handle of class '{}', got '{}'",
                CLASS_NAME, handle.class_name
            ),
            builtin,
        ));
    }
    Ok(())
}

fn map_id(handle: &HandleRef, builtin: &'static str) -> BuiltinResult<u64> {
    let storage = unsafe { &*handle.target.as_raw() };
    match storage {
        Value::Struct(StructValue { fields }) => match fields.get("id") {
            Some(Value::Int(IntValue::U64(id))) => Ok(*id),
            Some(Value::Int(other)) => {
                let id = other.to_i64();
                if id < 0 {
                    Err(map_error(
                        "containers.Map: negative map identifier",
                        builtin,
                    ))
                } else {
                    Ok(id as u64)
                }
            }
            Some(Value::Num(n)) if *n >= 0.0 => Ok(*n as u64),
            _ => Err(map_error(
                "containers.Map: corrupted storage identifier",
                builtin,
            )),
        },
        other => Err(map_error(
            format!("containers.Map: internal storage has unexpected shape {other:?}"),
            builtin,
        )),
    }
}

async fn prepare_keys(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<KeyCandidate>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    let flattened = flatten_keys(&host, key_type, builtin).await?;
    let mut out = Vec::with_capacity(flattened.len());
    for raw_key in flattened {
        let (normalized, canonical) = canonicalize_key(raw_key, key_type, builtin)?;
        out.push(KeyCandidate {
            normalized,
            canonical,
        });
    }
    Ok(out)
}

async fn prepare_values(value: Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|err| attach_builtin_context(err, builtin))?;
    flatten_values(&host, builtin).await
}

async fn flatten_keys(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                let element = unsafe { &*ptr.as_raw() };
                if matches!(element, Value::Cell(_)) {
                    return Err(map_error(
                        "containers.Map: nested cell arrays are not supported for keys",
                        builtin,
                    ));
                }
                out.push(
                    gather_if_needed_async(element)
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(arr) => {
            if key_type != KeyType::Logical {
                return Err(map_error(
                    "containers.Map: logical arrays can only be used with KeyType='logical'",
                    builtin,
                ));
            }
            Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect())
        }
        Value::Tensor(t) => {
            if !t.shape.is_empty() && t.data.len() != 1 && !is_vector_shape(&t.shape) {
                return Err(map_error(
                    "containers.Map: numeric keys must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(t.data.iter().map(|&v| Value::Num(v)).collect())
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {
            Ok(vec![value.clone()])
        }
        Value::GpuTensor(_) => Err(map_error(
            "containers.Map: GPU keys must be gathered to the host before construction",
            builtin,
        )),
        other => Err(map_error(
            format!("containers.Map: unsupported key container {other:?}"),
            builtin,
        )),
    }
}

async fn flatten_values(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(
                    gather_if_needed_async(unsafe { &*ptr.as_raw() })
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca, builtin)?),
        Value::LogicalArray(arr) => Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect()),
        Value::Tensor(t) => {
            if !t.shape.is_empty() && !is_vector_shape(&t.shape) && t.data.len() != 1 {
                return Err(map_error(
                    "containers.Map: numeric values must be scalar or vector shaped",
                    builtin,
                ));
            }
            Ok(t.data.iter().map(|&v| Value::Num(v)).collect())
        }
        _ => Ok(vec![value.clone()]),
    }
}

fn char_array_rows(ca: &CharArray, builtin: &'static str) -> BuiltinResult<Vec<Value>> {
    if ca.rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let mut text = String::with_capacity(ca.cols);
        for col in 0..ca.cols {
            text.push(ca.data[row * ca.cols + col]);
        }
        let chars: Vec<char> = text.chars().collect();
        let array = CharArray::new(chars.clone(), 1, chars.len())
            .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
        out.push(Value::CharArray(array));
    }
    Ok(out)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

fn canonicalize_key(
    value: Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<(NormalizedKey, Value)> {
    let normalized = normalize_key(&value, key_type, builtin)?;
    let canonical = match key_type {
        KeyType::Char => Value::CharArray(char_array_from_value(&value, builtin)?),
        KeyType::String => Value::String(string_from_value(
            &value,
            "containers.Map: keys must be string scalars",
            builtin,
        )?),
        KeyType::Double => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
            builtin,
        )?),
        KeyType::Single => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
            builtin,
        )?),
        KeyType::Int32 => Value::Int(IntValue::I32(integer_from_value(
            &value,
            i32::MIN as i64,
            i32::MAX as i64,
            "containers.Map: int32 keys must be integers",
            builtin,
        )? as i32)),
        KeyType::UInt32 => Value::Int(IntValue::U32(unsigned_from_value(
            &value,
            u32::MAX as u64,
            "containers.Map: uint32 keys must be unsigned integers",
            builtin,
        )? as u32)),
        KeyType::Int64 => Value::Int(IntValue::I64(integer_from_value(
            &value,
            i64::MIN,
            i64::MAX,
            "containers.Map: int64 keys must be integers",
            builtin,
        )?)),
        KeyType::UInt64 => Value::Int(IntValue::U64(unsigned_from_value(
            &value,
            u64::MAX,
            "containers.Map: uint64 keys must be unsigned integers",
            builtin,
        )?)),
        KeyType::Logical => Value::Bool(bool_from_value(
            &value,
            "containers.Map: logical keys must be logical scalars",
            builtin,
        )?),
    };
    Ok((normalized, canonical))
}

fn normalize_key(
    value: &Value,
    key_type: KeyType,
    builtin: &'static str,
) -> BuiltinResult<NormalizedKey> {
    match key_type {
        KeyType::Char | KeyType::String => {
            let text =
                string_from_value(value, "containers.Map: keys must be text scalars", builtin)?;
            Ok(NormalizedKey::String(text))
        }
        KeyType::Double | KeyType::Single => {
            let numeric = numeric_from_value(
                value,
                "containers.Map: keys must be numeric scalars",
                builtin,
            )?;
            if !numeric.is_finite() {
                return Err(map_error(
                    "containers.Map: keys must be finite numeric scalars",
                    builtin,
                ));
            }
            let canonical = if numeric == 0.0 { 0.0 } else { numeric };
            Ok(NormalizedKey::Float(canonical.to_bits()))
        }
        KeyType::Int32 | KeyType::Int64 => {
            let bounds = if key_type == KeyType::Int32 {
                (i32::MIN as i64, i32::MAX as i64)
            } else {
                (i64::MIN, i64::MAX)
            };
            let value = integer_from_value(
                value,
                bounds.0,
                bounds.1,
                "containers.Map: integer keys must be whole numbers",
                builtin,
            )?;
            Ok(NormalizedKey::Int(value))
        }
        KeyType::UInt32 | KeyType::UInt64 => {
            let limit = if key_type == KeyType::UInt32 {
                u32::MAX as u64
            } else {
                u64::MAX
            };
            let value = unsigned_from_value(
                value,
                limit,
                "containers.Map: unsigned keys must be non-negative integers",
                builtin,
            )?;
            Ok(NormalizedKey::UInt(value))
        }
        KeyType::Logical => {
            let flag = bool_from_value(
                value,
                "containers.Map: logical keys must be logical scalars",
                builtin,
            )?;
            Ok(NormalizedKey::Bool(flag))
        }
    }
}

fn string_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        _ => Err(map_error(context, builtin)),
    }
}

fn char_array_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.clone()),
        Value::String(s) => {
            let chars: Vec<char> = s.chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let chars: Vec<char> = sa.data[0].chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
        }
        _ => Err(map_error(
            "containers.Map: keys must be character vectors",
            builtin,
        )),
    }
}

fn char_array_value(text: &str, builtin: &'static str) -> BuiltinResult<Value> {
    let chars: Vec<char> = text.chars().collect();
    CharArray::new(chars.clone(), 1, chars.len())
        .map(Value::CharArray)
        .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))
}

fn normalize_numeric_value(value: Value, builtin: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Num(_) | Value::Tensor(_) => Ok(value),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::Bool(b) => Ok(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(arr) => {
            let data: Vec<f64> = arr
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, arr.shape.clone())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::Tensor(tensor))
        }
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::GpuTensor(_) => Err(map_error(
            "containers.Map: values must be numeric when ValueType is 'double' or 'single'",
            builtin,
        )),
    }
}

fn normalize_logical_value(value: Value, builtin: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Int(i) => Ok(Value::Bool(i.to_i64() != 0)),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Tensor(t) => {
            let flags: Vec<u8> = t
                .data
                .iter()
                .map(|&v| if v != 0.0 { 1 } else { 0 })
                .collect();
            let logical = LogicalArray::new(flags, t.shape.clone())
                .map_err(|e| map_error(format!("containers.Map: {e}"), builtin))?;
            Ok(Value::LogicalArray(logical))
        }
        Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Struct(_)
        | Value::Cell(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::GpuTensor(_) => Err(map_error(
            "containers.Map: values must be logical when ValueType is 'logical'",
            builtin,
        )),
    }
}

fn numeric_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            Ok(if arr.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(map_error(context, builtin)),
    }
}

fn integer_from_value(
    value: &Value,
    min: i64,
    max: i64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<i64> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(map_error(context, builtin));
            }
            if (*n < min as f64) || (*n > max as f64) {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as i64)
        }
        Value::Bool(b) => {
            let v = if *b { 1 } else { 0 };
            if v < min || v > max {
                return Err(map_error(context, builtin));
            }
            Ok(v)
        }
        _ => Err(map_error(context, builtin)),
    }
}

fn unsigned_from_value(
    value: &Value,
    max: u64,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<u64> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < 0 || v as u64 > max {
                return Err(map_error(context, builtin));
            }
            Ok(v as u64)
        }
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || *n > max as f64 {
                return Err(map_error(context, builtin));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(map_error(context, builtin));
            }
            Ok(n.round() as u64)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(map_error(context, builtin)),
    }
}

fn bool_from_value(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(arr) if arr.data.len() == 1 => Ok(arr.data[0] != 0),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        _ => Err(map_error(context, builtin)),
    }
}

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
                out.push(unsafe { &*ptr.as_raw() }.clone());
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
    let values = flatten_values(&host, builtin).await?;
    if expected == 1 {
        if values.is_empty() {
            return Err(map_error(
                "containers.Map: assignment requires a value",
                builtin,
            ));
        }
        Ok(vec![values.into_iter().next().unwrap()])
    } else {
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
                    gather_if_needed_async(unsafe { &*ptr.as_raw() })
                        .await
                        .map_err(|err| attach_builtin_context(err, builtin))?,
                );
            }
            Ok(KeyCollection {
                values,
                shape: vec![cell.rows, cell.cols],
            })
        }
        Value::StringArray(sa) => Ok(KeyCollection {
            values: sa.data.iter().map(|s| Value::String(s.clone())).collect(),
            shape: vec![sa.rows(), sa.cols()],
        }),
        Value::CharArray(ca) => {
            let rows = if ca.rows == 0 { 0 } else { ca.rows };
            Ok(KeyCollection {
                values: char_array_rows(ca, builtin)?,
                shape: vec![rows, 1],
            })
        }
        Value::LogicalArray(arr) if key_type == KeyType::Logical => Ok(KeyCollection {
            values: arr.data.iter().map(|&b| Value::Bool(b != 0)).collect(),
            shape: arr.shape.clone(),
        }),
        Value::Tensor(t) if key_type != KeyType::Char && key_type != KeyType::String => {
            Ok(KeyCollection {
                values: t.data.iter().map(|&n| Value::Num(n)).collect(),
                shape: t.shape.clone(),
            })
        }
        _ => Ok(KeyCollection {
            values: vec![host.clone()],
            shape: vec![1, 1],
        }),
    }
}

pub fn map_length(value: &Value) -> Option<usize> {
    if let Value::HandleObject(handle) = value {
        if handle.valid && handle.class_name == CLASS_NAME {
            if let Ok(id) = map_id(handle, BUILTIN_CONSTRUCTOR) {
                if let Ok(registry) = MAP_REGISTRY.read() {
                    return registry.get(&id).map(|store| store.len());
                }
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::Type;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message.clone()
    }

    fn containers_map_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::containers_map_builtin(args))
    }

    fn containers_map_keys(map: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_keys(map))
    }

    fn containers_map_is_key(map: Value, key_spec: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_is_key(map, key_spec))
    }

    fn containers_map_remove(map: Value, key_spec: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_remove(map, key_spec))
    }

    fn containers_map_subsref(map: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
        block_on(super::containers_map_subsref(map, kind, payload))
    }

    fn containers_map_subsasgn(
        map: Value,
        kind: String,
        payload: Value,
        rhs: Value,
    ) -> BuiltinResult<Value> {
        block_on(super::containers_map_subsasgn(map, kind, payload, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn construct_empty_map_defaults() {
        let map = containers_map_builtin(Vec::new()).expect("map");
        let count = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("Count".to_string()),
        )
        .expect("Count");
        assert_eq!(count, Value::Num(0.0));

        let key_type = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("KeyType".to_string()),
        )
        .expect("KeyType");
        assert_eq!(
            key_type,
            Value::CharArray(CharArray::new("char".chars().collect(), 1, 4).unwrap())
        );

        let value_type = containers_map_subsref(
            map.clone(),
            ".".to_string(),
            Value::from("ValueType".to_string()),
        )
        .expect("ValueType");
        assert_eq!(
            value_type,
            Value::CharArray(CharArray::new("any".chars().collect(), 1, 3).unwrap())
        );
    }

    #[test]
    fn map_type_resolvers_basics() {
        assert_eq!(map_handle_type(&[Type::Unknown]), Type::Unknown);
        assert_eq!(map_cell_type(&[]), Type::cell());
        assert_eq!(map_is_key_type(&[Type::String]), Type::logical());
        assert_eq!(map_unknown_type(&[]), Type::Unknown);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_with_cells_lookup() {
        let keys = crate::make_cell(vec![Value::from("apple"), Value::from("pear")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(5.0), Value::Num(7.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let apple = containers_map_subsref(
            map.clone(),
            "()".to_string(),
            crate::make_cell(vec![Value::from("apple")], 1, 1).unwrap(),
        )
        .expect("lookup");
        assert_eq!(apple, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_rejects_duplicate_keys() {
        let keys = crate::make_cell(vec![Value::from("dup"), Value::from("dup")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("duplicate check");
        let message = error_message(err);
        assert!(message.contains("Duplicate key name"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constructor_errors_when_value_count_mismatch() {
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("count mismatch");
        let message = error_message(err);
        assert!(message.contains("number of keys"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn comparison_method_rejects_unknown_values() {
        let keys = crate::make_cell(vec![Value::from("a")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = containers_map_builtin(vec![
            keys,
            values,
            Value::from("ComparisonMethod"),
            Value::from("caseinsensitive"),
        ])
        .expect_err("comparison method");
        let message = error_message(err);
        assert!(message.contains("ComparisonMethod"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn key_type_single_roundtrip() {
        let map = containers_map_builtin(vec![Value::from("KeyType"), Value::from("single")])
            .expect("map");
        let key_type = containers_map_subsref(map.clone(), ".".to_string(), Value::from("KeyType"))
            .expect("keytype");
        assert_eq!(
            key_type,
            Value::CharArray(CharArray::new("single".chars().collect(), 1, 6).unwrap())
        );

        let payload = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let map = containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(7.0))
            .expect("assign");
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        assert!(matches!(value, Value::Num(n) if (n - 7.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_type_double_converts_integers() {
        let keys = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Int(IntValue::I32(7))], 1, 1).unwrap();
        let map = containers_map_builtin(vec![
            keys,
            values,
            Value::from("KeyType"),
            Value::from("double"),
            Value::from("ValueType"),
            Value::from("double"),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        assert!(matches!(value, Value::Num(n) if (n - 7.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_type_logical_converts_numeric_arrays() {
        let keys = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
        let tensor = Tensor::new(vec![0.0, 2.0, -3.0], vec![3, 1]).unwrap();
        let values = crate::make_cell(vec![Value::Tensor(tensor.clone())], 1, 1).unwrap();
        let map = containers_map_builtin(vec![
            keys,
            values,
            Value::from("ValueType"),
            Value::from("logical"),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        match value {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![3, 1]);
                assert_eq!(arr.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uniform_values_enforced_on_assignment() {
        let map = containers_map_builtin(vec![Value::from("UniformValues"), Value::from(true)])
            .expect("map");
        let payload = crate::make_cell(vec![Value::from("x")], 1, 1).unwrap();
        let map = containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(1.0))
            .expect("assign");
        let err = containers_map_subsasgn(map, "()".to_string(), payload, Value::from("text"))
            .expect_err("uniform enforcement");
        let message = error_message(err);
        assert!(message.contains("UniformValues"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assignment_updates_and_inserts() {
        let map = containers_map_builtin(Vec::new()).expect("map");
        let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
        let updated = containers_map_subsasgn(
            map.clone(),
            "()".to_string(),
            payload.clone(),
            Value::Num(1.0),
        )
        .expect("assign");
        let updated = containers_map_subsasgn(
            updated.clone(),
            "()".to_string(),
            payload.clone(),
            Value::Num(5.0),
        )
        .expect("update");
        let beta_payload = crate::make_cell(vec![Value::from("beta")], 1, 1).unwrap();
        let updated = containers_map_subsasgn(
            updated.clone(),
            "()".to_string(),
            beta_payload,
            Value::Num(9.0),
        )
        .expect("insert");
        let value = containers_map_subsref(updated, "()".to_string(), payload).expect("lookup");
        assert_eq!(value, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsref_multiple_keys_preserves_shape() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let request = crate::make_cell(vec![Value::from("a"), Value::from("c")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![request], 1, 1).unwrap();
        let result =
            containers_map_subsref(map.clone(), "()".to_string(), payload).expect("lookup");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);
                assert_eq!(cell.get(0, 0).expect("cell 0,0"), Value::Num(1.0));
                assert_eq!(cell.get(0, 1).expect("cell 0,1"), Value::Num(3.0));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsref_empty_key_collection_returns_empty_cell() {
        let keys = crate::make_cell(vec![Value::from("z")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(42.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
        let result = containers_map_subsref(map, "()".to_string(), payload).expect("lookup empty");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 0);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected empty cell, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsasgn_with_cell_keys_updates_all_targets() {
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_spec = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
        let new_values = crate::make_cell(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap();
        let updated = containers_map_subsasgn(map.clone(), "()".to_string(), payload, new_values)
            .expect("assign");
        let a_payload = crate::make_cell(vec![Value::from("a")], 1, 1).unwrap();
        let b_payload = crate::make_cell(vec![Value::from("b")], 1, 1).unwrap();
        let a_value =
            containers_map_subsref(updated.clone(), "()".to_string(), a_payload).expect("a lookup");
        let b_value =
            containers_map_subsref(updated, "()".to_string(), b_payload).expect("b lookup");
        assert_eq!(a_value, Value::Num(10.0));
        assert_eq!(b_value, Value::Num(20.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn assignment_value_count_mismatch_errors() {
        let keys = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_spec = crate::make_cell(vec![Value::from("x"), Value::from("y")], 1, 2).unwrap();
        let payload = crate::make_cell(vec![key_spec], 1, 1).unwrap();
        let rhs = crate::make_cell(vec![Value::Num(99.0)], 1, 1).unwrap();
        let err =
            containers_map_subsasgn(map, "()".to_string(), payload, rhs).expect_err("value count");
        let message = error_message(err);
        assert!(message.contains("requires 2 values"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn subsasgn_empty_key_collection_is_noop() {
        let keys = crate::make_cell(vec![Value::from("root")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(7.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let empty_keys = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let payload = crate::make_cell(vec![empty_keys], 1, 1).unwrap();
        let rhs = crate::make_cell(Vec::new(), 1, 0).unwrap();
        let updated =
            containers_map_subsasgn(map.clone(), "()".to_string(), payload, rhs).expect("assign");
        let lookup_payload = crate::make_cell(vec![Value::from("root")], 1, 1).unwrap();
        let value =
            containers_map_subsref(updated, "()".to_string(), lookup_payload).expect("lookup");
        assert_eq!(value, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn keys_values_iskey_remove() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let key_list = containers_map_keys(map.clone()).expect("keys");
        match key_list {
            Value::Cell(cell) => assert_eq!(cell.data.len(), 3),
            other => panic!("expected cell array, got {other:?}"),
        }
        let mask = containers_map_is_key(
            map.clone(),
            crate::make_cell(vec![Value::from("a"), Value::from("z")], 1, 2).unwrap(),
        )
        .expect("mask");
        match mask {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
        let removed = containers_map_remove(
            map.clone(),
            crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
        )
        .expect("remove");
        let mask = containers_map_is_key(
            removed,
            crate::make_cell(vec![Value::from("b")], 1, 1).unwrap(),
        )
        .expect("mask");
        assert_eq!(mask, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_missing_key_returns_error() {
        let keys = crate::make_cell(vec![Value::from("key")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        let err = containers_map_remove(
            map,
            crate::make_cell(vec![Value::from("missing")], 1, 1).unwrap(),
        )
        .expect_err("remove missing");
        let message = error_message(err);
        assert_eq!(message, MISSING_KEY_ERR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_delegates_to_map_count() {
        let keys = crate::make_cell(
            vec![Value::from("a"), Value::from("b"), Value::from("c")],
            1,
            3,
        )
        .unwrap();
        let values = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .unwrap();
        let map = containers_map_builtin(vec![keys, values]).expect("map");
        assert_eq!(map_length(&map), Some(3));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn map_constructor_gathers_gpu_values() {
        test_support::with_test_provider(|provider| {
            let keys = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
            let data = vec![1.0, 2.0, 3.0];
            let shape = vec![3, 1];
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let values = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).unwrap();
            let map = containers_map_builtin(vec![keys, values]).expect("map");
            let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
            let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
            match value {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, shape);
                    assert_eq!(t.data, data);
                }
                other => panic!("expected tensor, got {:?}", other),
            }
        });
    }
}

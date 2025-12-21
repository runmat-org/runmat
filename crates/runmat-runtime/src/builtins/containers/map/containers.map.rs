//! MATLAB-compatible `containers.Map` constructor and methods for RunMat.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};

use once_cell::sync::Lazy;
use runmat_builtins::{CharArray, HandleRef, IntValue, LogicalArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

const CLASS_NAME: &str = "containers.Map";
const MISSING_KEY_ERR: &str = "containers.Map: The specified key is not present in this container.";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "containers.Map"
category: "containers/map"
keywords: ["containers.Map", "map", "dictionary", "hash map", "lookup"]
summary: "Create MATLAB-compatible dictionary objects that map keys to values."
references:
  - https://www.mathworks.com/help/matlab/ref/containers.map-class.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Map storage lives on the host. GPU inputs are gathered when constructing maps or fetching values."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::containers::map::containers_map::tests"
  integration: "builtins::containers::map::containers_map::tests::doc_examples_present"
---

# What does the `containers.Map` function do in MATLAB / RunMat?
`containers.Map` builds dictionary objects that associate unique keys with values. Keys can be
character vectors, string scalars, or numeric scalars (double/int32/uint32/int64/uint64/logical).
Values default to MATLAB's `'any'` semantics, letting you store arbitrary scalars, arrays, structs,
or handle objects. Each map tracks insertion order, supports key-based indexing, and exposes methods
such as `keys`, `values`, `isKey`, and `remove`.

## How does the `containers.Map` function behave in MATLAB / RunMat?
- Keys must be unique. Constructing a map or assigning a key that already exists overwrites the
  stored value (matching MATLAB's behaviour).
- The `KeyType`, `ValueType`, and `Count` properties are readable with dot-indexing.
- `map(key)` returns the associated value; requesting a missing key raises the MATLAB-compatible
  error *"The specified key is not present in this container."*
- Assignments of the form `map(key) = value` update or insert entries.
- Methods `keys(map)`, `values(map)`, `isKey(map, keySpec)`, and `remove(map, keySpec)` are fully
  compatible. When `keySpec` is a cell or string array, the result matches MATLAB's shape.
- GPU tensors presented as values are gathered to host memory before insertion. When values or keys
  arrive on the GPU and need to be expanded element-wise (for example, vector-valued constructor
  arguments), RunMat downloads them to materialise individual scalars.
- The `'UniformValues'` flag is accepted; when `true`, RunMat validates that every inserted value has
  the same MATLAB class. Retrieval still returns a cell array, matching MATLAB behaviour when the
  value type is `'any'`.

## `containers.Map` Function GPU Execution Behaviour
The data structure itself resides on the CPU. When you construct a map with GPU arrays, RunMat first
downloads the inputs so it can perform MATLAB-compatible validation and coercion. Maps never retain
device buffers internally, so the GPU provider does not need to implement special hooks for this
builtin.

## Examples of using the `containers.Map` function in MATLAB / RunMat

### Create an empty map with default types
```matlab
m = containers.Map();
m.KeyType
m.ValueType
m.Count
```

Expected output:
```matlab
ans =
    'char'
ans =
    'any'
ans =
     0
```

### Build a map from paired cell arrays
```matlab
keys = {'apple', 'pear', 'banana'};
vals = {42, [1 2 3], true};
fruit = containers.Map(keys, vals);
energy = fruit('apple');
```

Expected output:
```matlab
energy =
    42
```

### Update an existing key and add a new one
```matlab
fruit('apple') = 99;
fruit('peach') = struct('ripe', true);
```

Expected output:
```matlab
fruit('apple')
ans =
    99
```

### Query keys, values, and membership
```matlab
allKeys = keys(fruit);
allVals = values(fruit);
mask = isKey(fruit, {'apple', 'durian'});
```

Expected output:
```matlab
allKeys =
  1×4 cell array
    {'apple'}    {'pear'}    {'banana'}    {'peach'}

allVals =
  1×4 cell array
    {[99]}    {[1 2 3]}    {[1]}    {1×1 struct}

mask =
  1×2 logical array
     1     0
```

### Remove keys and inspect the map length
```matlab
remove(fruit, {'pear', 'banana'});
n = length(fruit);
remaining = keys(fruit);
```

Expected output:
```matlab
n =
     2
remaining =
  1×2 cell array
    {'apple'}    {'peach'}
```

## FAQ

### Which key types are supported?
`containers.Map` accepts `'char'`, `'string'`, `'double'`, `'single'`, `'int32'`, `'uint32'`, `'int64'`,
`'uint64'`, and `'logical'`. Keys supplied during construction or assignment are coerced to the
declared type and must be scalar.

### What happens when I provide duplicate keys at construction time?
Duplicate keys raise the same error as MATLAB: *"Duplicate key name was provided."* During
assignment, duplicate keys overwrite the existing value.

### Does RunMat honour `'UniformValues', true`?
Yes. When this option is set, RunMat enforces that each inserted value matches the MATLAB class of
the first value. Retrieval still uses cell arrays, mirroring MATLAB when `'ValueType'` is `'any'`.

### Can I store GPU arrays as map values?
Yes. RunMat automatically gathers GPU tensors to host memory before inserting them so it can apply
the same validation and coercion rules as MATLAB. This ensures constructors that rely on vector
expansion continue to produce predictable host-side values.

### How does `length(map)` behave?
`length(map)` returns the number of stored keys (identical to the `Count` property). `size(map)`
remains `[1 1]`, matching MATLAB's handle semantics.

### What error is raised when a key is missing?
Indexing a missing key produces the MATLAB-compatible error message *"The specified key is not
present in this container."*

### Does the map preserve insertion order?
Yes. `keys(map)` and `values(map)` return entries in the order they were first inserted, matching the
behaviour of MATLAB's `containers.Map`.

### Is the implementation thread-safe?
Yes. A global read/write lock guards the backing storage so concurrent reads are allowed while write
operations remain exclusive.

### How do I remove every entry?
Call `remove(map, keys(map))` or reassign a new empty map. RunMat currently keeps the internal
storage until the handle is cleared, matching MATLAB's lifetime semantics.

### What happens if I pass a non-scalar key?
Keys must be scalar. Passing vectors, matrices, or nested cell arrays of keys raises a descriptive
error pointing to the offending argument.

## See Also
[keys](./containers.Map.keys), [values](./containers.Map.values), [isKey](./containers.Map.isKey),
[remove](./containers.Map.remove), [length](../../array/introspection/length)
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "containers.Map",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Handles act as fusion sinks; map construction terminates GPU fusion plans.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("containers.Map", DOC_MD);

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

    fn parse(value: &Value) -> Result<Self, String> {
        let text = string_from_value(value, "containers.Map: expected a KeyType string")?;
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
            other => Err(format!(
                "containers.Map: unsupported KeyType '{other}'. Valid types: char, string, double, int32, uint32, int64, uint64, logical."
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

    fn parse(value: &Value) -> Result<Self, String> {
        let text = string_from_value(value, "containers.Map: expected a ValueType string")?;
        match text.to_ascii_lowercase().as_str() {
            "any" => Ok(ValueType::Any),
            "char" | "character" => Ok(ValueType::Char),
            "string" => Ok(ValueType::String),
            "double" => Ok(ValueType::Double),
            "single" => Ok(ValueType::Single),
            "logical" => Ok(ValueType::Logical),
            other => Err(format!(
                "containers.Map: unsupported ValueType '{other}'. Valid types: any, char, string, double, single, logical."
            )),
        }
    }

    fn normalize(&self, value: Value) -> Result<Value, String> {
        match self {
            ValueType::Any => Ok(value),
            ValueType::Char => {
                let chars = char_array_from_value(&value)?;
                Ok(Value::CharArray(chars))
            }
            ValueType::String => {
                let text =
                    string_from_value(&value, "containers.Map: values must be string scalars")?;
                Ok(Value::String(text))
            }
            ValueType::Double | ValueType::Single => normalize_numeric_value(value),
            ValueType::Logical => normalize_logical_value(value),
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

    fn insert_new(&mut self, mut entry: MapEntry) -> Result<(), String> {
        if self.index.contains_key(&entry.normalized) {
            return Err("containers.Map: Duplicate key name was provided.".to_string());
        }
        entry.value = self.normalize_value(entry.value)?;
        self.track_uniform_class(&entry.value)?;
        let idx = self.entries.len();
        self.entries.push(entry.clone());
        self.index.insert(entry.normalized, idx);
        Ok(())
    }

    fn set(&mut self, mut entry: MapEntry) -> Result<(), String> {
        entry.value = self.normalize_value(entry.value)?;
        self.track_uniform_class(&entry.value)?;
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

    fn remove(&mut self, key: &NormalizedKey) -> Result<(), String> {
        let idx = match self.index.get(key) {
            Some(&idx) => idx,
            None => {
                return Err(MISSING_KEY_ERR.to_string());
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

    fn normalize_value(&self, value: Value) -> Result<Value, String> {
        self.value_type.normalize(value)
    }

    fn track_uniform_class(&mut self, value: &Value) -> Result<(), String> {
        if !self.uniform_values {
            return Ok(());
        }
        let class = ValueClass::from_value(value);
        if let Some(existing) = &self.uniform_class {
            if existing != &class {
                return Err("containers.Map: UniformValues=true requires all values to share the same MATLAB class."
                    .to_string());
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
    sink = true
)]
fn containers_map_builtin(args: Vec<Value>) -> Result<Value, String> {
    let parsed = parse_constructor_args(args)?;
    let store = build_store(parsed)?;
    allocate_handle(store)
}

#[runtime_builtin(name = "containers.Map.keys")]
fn containers_map_keys(map: Value) -> Result<Value, String> {
    with_store(&map, |store| {
        let values = store.keys();
        make_row_cell(values)
    })
}

#[runtime_builtin(name = "containers.Map.values")]
fn containers_map_values(map: Value) -> Result<Value, String> {
    with_store(&map, |store| {
        let values = store.values();
        make_row_cell(values)
    })
}

#[runtime_builtin(name = "containers.Map.isKey")]
fn containers_map_is_key(map: Value, key_spec: Value) -> Result<Value, String> {
    with_store(&map, |store| {
        let collection = collect_key_spec(&key_spec, store.key_type)?;
        let mut flags = Vec::with_capacity(collection.values.len());
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type)?;
            flags.push(store.contains(&normalized));
        }
        if collection.values.len() == 1 {
            Ok(Value::Bool(flags[0]))
        } else {
            let data: Vec<u8> = flags.into_iter().map(|b| if b { 1 } else { 0 }).collect();
            let logical = LogicalArray::new(data, collection.shape)
                .map_err(|e| format!("containers.Map: {e}"))?;
            Ok(Value::LogicalArray(logical))
        }
    })
}

#[runtime_builtin(name = "containers.Map.remove")]
fn containers_map_remove(map: Value, key_spec: Value) -> Result<Value, String> {
    with_store_mut(&map, |store| {
        let collection = collect_key_spec(&key_spec, store.key_type)?;
        for value in &collection.values {
            let normalized = normalize_key(value, store.key_type)?;
            store.remove(&normalized)?;
        }
        Ok(())
    })?;
    Ok(map)
}

#[runtime_builtin(name = "containers.Map.subsref")]
fn containers_map_subsref(map: Value, kind: String, payload: Value) -> Result<Value, String> {
    if !matches!(map, Value::HandleObject(_)) {
        return Err(format!(
            "containers.Map: subsref expects a containers.Map handle, got {map:?}"
        ));
    }
    match kind.as_str() {
        "()" => {
            let mut args = extract_key_arguments(&payload)?;
            if args.is_empty() {
                return Err("containers.Map: indexing requires at least one key".to_string());
            }
            if args.len() != 1 {
                return Err("containers.Map: indexing expects a single key argument".to_string());
            }
            let key_arg = args.remove(0);
            with_store(&map, |store| {
                let collection = collect_key_spec(&key_arg, store.key_type)?;
                if collection.values.is_empty() {
                    return crate::make_cell_with_shape(Vec::new(), collection.shape.clone())
                        .map_err(|e| format!("containers.Map: {e}"));
                }
                if collection.values.len() == 1 {
                    let normalized = normalize_key(&collection.values[0], store.key_type)?;
                    store
                        .get(&normalized)
                        .ok_or_else(|| MISSING_KEY_ERR.to_string())
                } else {
                    let mut results = Vec::with_capacity(collection.values.len());
                    for value in &collection.values {
                        let normalized = normalize_key(value, store.key_type)?;
                        let stored = store
                            .get(&normalized)
                            .ok_or_else(|| MISSING_KEY_ERR.to_string())?;
                        results.push(stored);
                    }
                    crate::make_cell_with_shape(results, collection.shape.clone())
                        .map_err(|e| format!("containers.Map: {e}"))
                }
            })
        }
        "." => {
            let field = string_from_value(&payload, "containers.Map: property name must be text")?;
            with_store(&map, |store| match field.to_ascii_lowercase().as_str() {
                "count" => Ok(Value::Num(store.len() as f64)),
                "keytype" => char_array_value(store.key_type.matlab_name()),
                "valuetype" => char_array_value(store.value_type.matlab_name()),
                other => Err(format!("containers.Map: no such property '{other}'")),
            })
        }
        "{}" => Err("containers.Map: curly-brace indexing is not supported.".to_string()),
        other => Err(format!(
            "containers.Map: unsupported indexing kind '{other}'"
        )),
    }
}

#[runtime_builtin(name = "containers.Map.subsasgn")]
fn containers_map_subsasgn(
    map: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> Result<Value, String> {
    if !matches!(map, Value::HandleObject(_)) {
        return Err(format!(
            "containers.Map: subsasgn expects a containers.Map handle, got {map:?}"
        ));
    }
    match kind.as_str() {
        "()" => {
            let mut args = extract_key_arguments(&payload)?;
            if args.is_empty() {
                return Err("containers.Map: assignment requires at least one key".to_string());
            }
            if args.len() != 1 {
                return Err("containers.Map: assignment expects a single key argument".to_string());
            }
            let key_arg = args.remove(0);
            with_store_mut(&map, move |store| {
                let KeyCollection {
                    values: key_values, ..
                } = collect_key_spec(&key_arg, store.key_type)?;
                let values = expand_assignment_values(rhs.clone(), key_values.len())?;
                for (key_raw, value) in key_values.into_iter().zip(values.into_iter()) {
                    let (normalized, canonical) = canonicalize_key(key_raw, store.key_type)?;
                    let entry = MapEntry {
                        normalized,
                        key_value: canonical,
                        value,
                    };
                    store.set(entry)?;
                }
                Ok(())
            })?;
            Ok(map)
        }
        "." => Err("containers.Map: property assignments are not supported.".to_string()),
        "{}" => Err("containers.Map: curly-brace assignment is not supported.".to_string()),
        other => Err(format!(
            "containers.Map: unsupported assignment kind '{other}'"
        )),
    }
}

fn parse_constructor_args(args: Vec<Value>) -> Result<ConstructorArgs, String> {
    let mut index = 0usize;
    let mut keys_input: Option<Value> = None;
    let mut values_input: Option<Value> = None;

    if index < args.len() && keyword_of(&args[index]).is_none() {
        if args.len() < 2 {
            return Err("containers.Map: constructor requires both keys and values when either is provided."
                .to_string());
        }
        keys_input = Some(args[index].clone());
        values_input = Some(args[index + 1].clone());
        index += 2;
    }

    let mut key_type = KeyType::Char;
    let mut value_type = ValueType::Any;
    let mut uniform_values = false;
    while index < args.len() {
        let keyword = keyword_of(&args[index])
            .ok_or_else(|| "containers.Map: expected option name (e.g. 'KeyType')".to_string())?;
        index += 1;
        let Some(value) = args.get(index) else {
            return Err(format!(
                "containers.Map: missing value for option '{keyword}'"
            ));
        };
        index += 1;
        match keyword.as_str() {
            "keytype" => key_type = KeyType::parse(value)?,
            "valuetype" => value_type = ValueType::parse(value)?,
            "uniformvalues" => {
                uniform_values =
                    bool_from_value(value, "containers.Map: UniformValues must be logical")?
            }
            "comparisonmethod" => {
                let text =
                    string_from_value(value, "containers.Map: ComparisonMethod must be a string")?;
                let lowered = text.to_ascii_lowercase();
                if lowered != "strcmp" {
                    return Err(
                        "containers.Map: only ComparisonMethod='strcmp' is supported.".to_string(),
                    );
                }
            }
            other => {
                return Err(format!("containers.Map: unrecognised option '{other}'"));
            }
        }
    }

    let keys = match keys_input {
        Some(value) => prepare_keys(value, key_type)?,
        None => Vec::new(),
    };

    let values = match values_input {
        Some(value) => prepare_values(value)?,
        None => Vec::new(),
    };

    if keys.len() != values.len() {
        return Err(format!(
            "containers.Map: number of keys ({}) must match number of values ({})",
            keys.len(),
            values.len()
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

fn build_store(args: ConstructorArgs) -> Result<MapStore, String> {
    let mut store = MapStore::new(args.key_type, args.value_type, args.uniform_values);
    for (candidate, value) in args.keys.into_iter().zip(args.values.into_iter()) {
        store.insert_new(MapEntry {
            normalized: candidate.normalized,
            key_value: candidate.canonical,
            value,
        })?;
    }
    Ok(store)
}

fn allocate_handle(store: MapStore) -> Result<Value, String> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    MAP_REGISTRY
        .write()
        .map_err(|_| "containers.Map: registry lock poisoned".to_string())?
        .insert(id, store);
    let mut struct_value = StructValue::new();
    struct_value
        .fields
        .insert("id".to_string(), Value::Int(IntValue::U64(id)));
    let storage = Value::Struct(struct_value);
    let gc = runmat_gc::gc_allocate(storage).map_err(|e| format!("containers.Map: {e}"))?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_NAME.to_string(),
        target: gc,
        valid: true,
    }))
}

fn with_store<F, R>(map: &Value, f: F) -> Result<R, String>
where
    F: FnOnce(&MapStore) -> Result<R, String>,
{
    let handle = extract_handle(map)?;
    ensure_handle(handle)?;
    let id = map_id(handle)?;
    let guard = MAP_REGISTRY
        .read()
        .map_err(|_| "containers.Map: registry lock poisoned".to_string())?;
    let store = guard
        .get(&id)
        .ok_or_else(|| "containers.Map: internal storage not found".to_string())?;
    f(store)
}

fn with_store_mut<F, R>(map: &Value, f: F) -> Result<R, String>
where
    F: FnOnce(&mut MapStore) -> Result<R, String>,
{
    let handle = extract_handle(map)?;
    ensure_handle(handle)?;
    let id = map_id(handle)?;
    let mut guard = MAP_REGISTRY
        .write()
        .map_err(|_| "containers.Map: registry lock poisoned".to_string())?;
    let store = guard
        .get_mut(&id)
        .ok_or_else(|| "containers.Map: internal storage not found".to_string())?;
    f(store)
}

fn extract_handle(value: &Value) -> Result<&HandleRef, String> {
    match value {
        Value::HandleObject(handle) => Ok(handle),
        _ => Err("containers.Map: expected a containers.Map handle".to_string()),
    }
}

fn ensure_handle(handle: &HandleRef) -> Result<(), String> {
    if !handle.valid {
        return Err("containers.Map: handle is invalid".to_string());
    }
    if handle.class_name != CLASS_NAME {
        return Err(format!(
            "containers.Map: expected handle of class '{}', got '{}'",
            CLASS_NAME, handle.class_name
        ));
    }
    Ok(())
}

fn map_id(handle: &HandleRef) -> Result<u64, String> {
    let storage = unsafe { &*handle.target.as_raw() };
    match storage {
        Value::Struct(StructValue { fields }) => match fields.get("id") {
            Some(Value::Int(IntValue::U64(id))) => Ok(*id),
            Some(Value::Int(other)) => {
                let id = other.to_i64();
                if id < 0 {
                    Err("containers.Map: negative map identifier".to_string())
                } else {
                    Ok(id as u64)
                }
            }
            Some(Value::Num(n)) if *n >= 0.0 => Ok(*n as u64),
            _ => Err("containers.Map: corrupted storage identifier".to_string()),
        },
        other => Err(format!(
            "containers.Map: internal storage has unexpected shape {other:?}"
        )),
    }
}

fn prepare_keys(value: Value, key_type: KeyType) -> Result<Vec<KeyCandidate>, String> {
    let host = gather_if_needed(&value).map_err(|e| format!("containers.Map: {e}"))?;
    let flattened = flatten_keys(&host, key_type)?;
    let mut out = Vec::with_capacity(flattened.len());
    for raw_key in flattened {
        let (normalized, canonical) = canonicalize_key(raw_key, key_type)?;
        out.push(KeyCandidate {
            normalized,
            canonical,
        });
    }
    Ok(out)
}

fn prepare_values(value: Value) -> Result<Vec<Value>, String> {
    let host = gather_if_needed(&value).map_err(|e| format!("containers.Map: {e}"))?;
    flatten_values(&host)
}

fn flatten_keys(value: &Value, key_type: KeyType) -> Result<Vec<Value>, String> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                let element = unsafe { &*ptr.as_raw() };
                if matches!(element, Value::Cell(_)) {
                    return Err(
                        "containers.Map: nested cell arrays are not supported for keys".to_string(),
                    );
                }
                out.push(gather_if_needed(element).map_err(|e| format!("containers.Map: {e}"))?);
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca)),
        Value::LogicalArray(arr) => {
            if key_type != KeyType::Logical {
                return Err(
                    "containers.Map: logical arrays can only be used with KeyType='logical'"
                        .to_string(),
                );
            }
            Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect())
        }
        Value::Tensor(t) => {
            if !t.shape.is_empty() && t.data.len() != 1 && !is_vector_shape(&t.shape) {
                return Err(
                    "containers.Map: numeric keys must be scalar or vector shaped".to_string(),
                );
            }
            Ok(t.data.iter().map(|&v| Value::Num(v)).collect())
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => {
            Ok(vec![value.clone()])
        }
        Value::GpuTensor(_) => Err(
            "containers.Map: GPU keys must be gathered to the host before construction".to_string(),
        ),
        other => Err(format!(
            "containers.Map: unsupported key container {other:?}"
        )),
    }
}

fn flatten_values(value: &Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(
                    gather_if_needed(unsafe { &*ptr.as_raw() })
                        .map_err(|e| format!("containers.Map: {e}"))?,
                );
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa
            .data
            .iter()
            .map(|text| Value::String(text.clone()))
            .collect()),
        Value::CharArray(ca) => Ok(char_array_rows(ca)),
        Value::LogicalArray(arr) => Ok(arr.data.iter().map(|&b| Value::Bool(b != 0)).collect()),
        Value::Tensor(t) => {
            if !t.shape.is_empty() && !is_vector_shape(&t.shape) && t.data.len() != 1 {
                return Err(
                    "containers.Map: numeric values must be scalar or vector shaped".to_string(),
                );
            }
            Ok(t.data.iter().map(|&v| Value::Num(v)).collect())
        }
        _ => Ok(vec![value.clone()]),
    }
}

fn char_array_rows(ca: &CharArray) -> Vec<Value> {
    if ca.rows == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let mut text = String::with_capacity(ca.cols);
        for col in 0..ca.cols {
            text.push(ca.data[row * ca.cols + col]);
        }
        out.push(Value::CharArray(
            CharArray::new(text.chars().collect(), 1, text.chars().count())
                .expect("char array new"),
        ));
    }
    out
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

fn canonicalize_key(value: Value, key_type: KeyType) -> Result<(NormalizedKey, Value), String> {
    let normalized = normalize_key(&value, key_type)?;
    let canonical = match key_type {
        KeyType::Char => Value::CharArray(char_array_from_value(&value)?),
        KeyType::String => Value::String(string_from_value(
            &value,
            "containers.Map: keys must be string scalars",
        )?),
        KeyType::Double => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
        )?),
        KeyType::Single => Value::Num(numeric_from_value(
            &value,
            "containers.Map: keys must be numeric scalars",
        )?),
        KeyType::Int32 => Value::Int(IntValue::I32(integer_from_value(
            &value,
            i32::MIN as i64,
            i32::MAX as i64,
            "containers.Map: int32 keys must be integers",
        )? as i32)),
        KeyType::UInt32 => Value::Int(IntValue::U32(unsigned_from_value(
            &value,
            u32::MAX as u64,
            "containers.Map: uint32 keys must be unsigned integers",
        )? as u32)),
        KeyType::Int64 => Value::Int(IntValue::I64(integer_from_value(
            &value,
            i64::MIN,
            i64::MAX,
            "containers.Map: int64 keys must be integers",
        )?)),
        KeyType::UInt64 => Value::Int(IntValue::U64(unsigned_from_value(
            &value,
            u64::MAX,
            "containers.Map: uint64 keys must be unsigned integers",
        )?)),
        KeyType::Logical => Value::Bool(bool_from_value(
            &value,
            "containers.Map: logical keys must be logical scalars",
        )?),
    };
    Ok((normalized, canonical))
}

fn normalize_key(value: &Value, key_type: KeyType) -> Result<NormalizedKey, String> {
    match key_type {
        KeyType::Char | KeyType::String => {
            let text = string_from_value(value, "containers.Map: keys must be text scalars")?;
            Ok(NormalizedKey::String(text))
        }
        KeyType::Double | KeyType::Single => {
            let numeric =
                numeric_from_value(value, "containers.Map: keys must be numeric scalars")?;
            if !numeric.is_finite() {
                return Err("containers.Map: keys must be finite numeric scalars".to_string());
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
            )?;
            Ok(NormalizedKey::UInt(value))
        }
        KeyType::Logical => {
            let flag = bool_from_value(
                value,
                "containers.Map: logical keys must be logical scalars",
            )?;
            Ok(NormalizedKey::Bool(flag))
        }
    }
}

fn string_from_value(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        _ => Err(context.to_string()),
    }
}

fn char_array_from_value(value: &Value) -> Result<CharArray, String> {
    match value {
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.clone()),
        Value::String(s) => {
            let chars: Vec<char> = s.chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| format!("containers.Map: {e}"))
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let chars: Vec<char> = sa.data[0].chars().collect();
            CharArray::new(chars.clone(), 1, chars.len())
                .map_err(|e| format!("containers.Map: {e}"))
        }
        _ => Err("containers.Map: keys must be character vectors".to_string()),
    }
}

fn char_array_value(text: &str) -> Result<Value, String> {
    let chars: Vec<char> = text.chars().collect();
    CharArray::new(chars.clone(), 1, chars.len())
        .map(Value::CharArray)
        .map_err(|e| format!("containers.Map: {e}"))
}

fn normalize_numeric_value(value: Value) -> Result<Value, String> {
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
            let tensor =
                Tensor::new(data, arr.shape.clone()).map_err(|e| format!("containers.Map: {e}"))?;
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
        | Value::GpuTensor(_)
        | Value::Symbolic(_) => Err(
            "containers.Map: values must be numeric when ValueType is 'double' or 'single'"
                .to_string(),
        ),
    }
}

fn normalize_logical_value(value: Value) -> Result<Value, String> {
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
                .map_err(|e| format!("containers.Map: {e}"))?;
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
        | Value::GpuTensor(_)
        | Value::Symbolic(_) => {
            Err("containers.Map: values must be logical when ValueType is 'logical'".to_string())
        }
    }
}

fn numeric_from_value(value: &Value, context: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(arr) if arr.data.len() == 1 => {
            Ok(if arr.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(context.to_string()),
    }
}

fn integer_from_value(value: &Value, min: i64, max: i64, context: &str) -> Result<i64, String> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < min || v > max {
                return Err(context.to_string());
            }
            Ok(v)
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(context.to_string());
            }
            if (*n < min as f64) || (*n > max as f64) {
                return Err(context.to_string());
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(context.to_string());
            }
            Ok(n.round() as i64)
        }
        Value::Bool(b) => {
            let v = if *b { 1 } else { 0 };
            if v < min || v > max {
                return Err(context.to_string());
            }
            Ok(v)
        }
        _ => Err(context.to_string()),
    }
}

fn unsigned_from_value(value: &Value, max: u64, context: &str) -> Result<u64, String> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < 0 || v as u64 > max {
                return Err(context.to_string());
            }
            Ok(v as u64)
        }
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || *n > max as f64 {
                return Err(context.to_string());
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(context.to_string());
            }
            Ok(n.round() as u64)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(context.to_string()),
    }
}

fn bool_from_value(value: &Value, context: &str) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(arr) if arr.data.len() == 1 => Ok(arr.data[0] != 0),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        _ => Err(context.to_string()),
    }
}

fn make_row_cell(values: Vec<Value>) -> Result<Value, String> {
    let cols = values.len();
    crate::make_cell_with_shape(values, vec![1, cols])
}

fn extract_key_arguments(payload: &Value) -> Result<Vec<Value>, String> {
    match payload {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                out.push(unsafe { &*ptr.as_raw() }.clone());
            }
            Ok(out)
        }
        other => Err(format!(
            "containers.Map: expected key arguments in a cell array, got {other:?}"
        )),
    }
}

fn expand_assignment_values(value: Value, expected: usize) -> Result<Vec<Value>, String> {
    let host = gather_if_needed(&value).map_err(|e| format!("containers.Map: {e}"))?;
    let values = flatten_values(&host)?;
    if expected == 1 {
        if values.is_empty() {
            return Err("containers.Map: assignment requires a value".to_string());
        }
        Ok(vec![values.into_iter().next().unwrap()])
    } else {
        if values.len() != expected {
            return Err(format!(
                "containers.Map: assignment with {} keys requires {} values (got {})",
                expected,
                expected,
                values.len()
            ));
        }
        Ok(values)
    }
}

struct KeyCollection {
    values: Vec<Value>,
    shape: Vec<usize>,
}

fn collect_key_spec(value: &Value, key_type: KeyType) -> Result<KeyCollection, String> {
    let host = gather_if_needed(value).map_err(|e| format!("containers.Map: {e}"))?;
    match &host {
        Value::Cell(cell) => {
            let mut values = Vec::with_capacity(cell.data.len());
            for ptr in &cell.data {
                values.push(
                    gather_if_needed(unsafe { &*ptr.as_raw() })
                        .map_err(|e| format!("containers.Map: {e}"))?,
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
                values: char_array_rows(ca),
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
            if let Ok(id) = map_id(handle) {
                if let Ok(registry) = MAP_REGISTRY.read() {
                    return registry.get(&id).map(|store| store.len());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

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

    #[test]
    fn constructor_rejects_duplicate_keys() {
        let keys = crate::make_cell(vec![Value::from("dup"), Value::from("dup")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("duplicate check");
        assert!(err.contains("Duplicate key name"));
    }

    #[test]
    fn constructor_errors_when_value_count_mismatch() {
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = containers_map_builtin(vec![keys, values]).expect_err("count mismatch");
        assert!(err.contains("number of keys"));
    }

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
        assert!(err.contains("ComparisonMethod"));
    }

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

    #[test]
    fn uniform_values_enforced_on_assignment() {
        let map = containers_map_builtin(vec![Value::from("UniformValues"), Value::from(true)])
            .expect("map");
        let payload = crate::make_cell(vec![Value::from("x")], 1, 1).unwrap();
        let map = containers_map_subsasgn(map, "()".to_string(), payload.clone(), Value::Num(1.0))
            .expect("assign");
        let err = containers_map_subsasgn(map, "()".to_string(), payload, Value::from("text"))
            .expect_err("uniform enforcement");
        assert!(err.contains("UniformValues"));
    }

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
        assert!(err.contains("requires 2 values"));
    }

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
        assert_eq!(err, MISSING_KEY_ERR);
    }

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

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let examples = test_support::doc_examples(DOC_MD);
        assert!(!examples.is_empty());
    }
}

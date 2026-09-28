use super::*;
use super::constructor_args::int_value_dtype;
use runmat_value::NumericDType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KeyType {
    Char,
    Double,
    Single,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl KeyType {
    pub(super) fn matlab_name(self) -> &'static str {
        match self {
            KeyType::Char => "char",
            KeyType::Double => "double",
            KeyType::Single => "single",
            KeyType::Int32 => "int32",
            KeyType::UInt32 => "uint32",
            KeyType::Int64 => "int64",
            KeyType::UInt64 => "uint64",
        }
    }

    pub(super) fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(value, "containers.Map: expected a KeyType string", builtin)?;
        match text.to_ascii_lowercase().as_str() {
            "char" | "character" => Ok(KeyType::Char),
            "double" => Ok(KeyType::Double),
            "single" => Ok(KeyType::Single),
            "int32" => Ok(KeyType::Int32),
            "uint32" => Ok(KeyType::UInt32),
            "int64" => Ok(KeyType::Int64),
            "uint64" => Ok(KeyType::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported KeyType '{other}'. Valid types: char, double, single, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ValueType {
    Any,
    Char,
    Double,
    Single,
    Logical,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl ValueType {
    pub(super) fn matlab_name(self) -> &'static str {
        match self {
            ValueType::Any => "any",
            ValueType::Char => "char",
            ValueType::Double => "double",
            ValueType::Single => "single",
            ValueType::Logical => "logical",
            ValueType::Int8 => "int8",
            ValueType::UInt8 => "uint8",
            ValueType::Int16 => "int16",
            ValueType::UInt16 => "uint16",
            ValueType::Int32 => "int32",
            ValueType::UInt32 => "uint32",
            ValueType::Int64 => "int64",
            ValueType::UInt64 => "uint64",
        }
    }

    pub(super) fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(
            value,
            "containers.Map: expected a ValueType string",
            builtin,
        )?;
        match text.to_ascii_lowercase().as_str() {
            "any" => Ok(ValueType::Any),
            "char" | "character" => Ok(ValueType::Char),
            "double" => Ok(ValueType::Double),
            "single" => Ok(ValueType::Single),
            "logical" => Ok(ValueType::Logical),
            "int8" => Ok(ValueType::Int8),
            "uint8" => Ok(ValueType::UInt8),
            "int16" => Ok(ValueType::Int16),
            "uint16" => Ok(ValueType::UInt16),
            "int32" => Ok(ValueType::Int32),
            "uint32" => Ok(ValueType::UInt32),
            "int64" => Ok(ValueType::Int64),
            "uint64" => Ok(ValueType::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported ValueType '{other}'. Valid types: any, char, logical, double, single, int8, uint8, int16, uint16, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }

    pub(super) fn normalize(&self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        match self {
            ValueType::Any => Ok(value),
            ValueType::Char => {
                let chars = char_array_from_value(&value, builtin)?;
                Ok(Value::CharArray(chars))
            }
            ValueType::Double => normalize_numeric_value(value, NumericDType::F64, builtin),
            ValueType::Single => normalize_numeric_value(value, NumericDType::F32, builtin),
            ValueType::Logical => normalize_logical_value(value, builtin),
            integer_type => normalize_integer_value(value, *integer_type, builtin),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum NormalizedKey {
    String(String),
    Float(u64),
    Int(i64),
    UInt(u64),
}

#[derive(Clone)]
pub(super) struct MapEntry {
    pub(super) normalized: NormalizedKey,
    pub(super) key_value: Value,
    pub(super) value: Value,
}

pub(super) struct MapStore {
    pub(super) storage: Option<GcHandle>,
    pub(super) key_type: KeyType,
    pub(super) value_type: ValueType,
    pub(super) uniform_values: bool,
    pub(super) uniform_class: Option<ValueClass>,
    pub(super) entries: Vec<MapEntry>,
    pub(super) index: HashMap<NormalizedKey, usize>,
}

impl MapStore {
    pub(super) fn new(key_type: KeyType, value_type: ValueType, uniform_values: bool) -> Self {
        Self {
            storage: None,
            key_type,
            value_type,
            uniform_values,
            uniform_class: None,
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn contains(&self, key: &NormalizedKey) -> bool {
        self.index.contains_key(key)
    }

    pub(super) fn get(&self, key: &NormalizedKey) -> Option<Value> {
        self.index
            .get(key)
            .map(|&idx| self.entries[idx].value.clone())
    }

    pub(super) fn insert_new(
        &mut self,
        mut entry: MapEntry,
        builtin: &'static str,
    ) -> BuiltinResult<()> {
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

    pub(super) fn set(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
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

    pub(super) fn remove(
        &mut self,
        key: &NormalizedKey,
        builtin: &'static str,
    ) -> BuiltinResult<()> {
        let idx = match self.index.get(key) {
            Some(&idx) => idx,
            None => {
                return Err(map_descriptor_error(
                    &CONTAINERS_MAP_ERROR_MISSING_KEY,
                    builtin,
                ));
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

    pub(super) fn keys(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.key_value.clone())
            .collect()
    }

    pub(super) fn values(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.value.clone())
            .collect()
    }

    pub(super) fn sorted_entries(&self) -> Vec<&MapEntry> {
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| match (&left.normalized, &right.normalized) {
            (NormalizedKey::String(left), NormalizedKey::String(right)) => left.cmp(right),
            (NormalizedKey::Int(left), NormalizedKey::Int(right)) => left.cmp(right),
            (NormalizedKey::UInt(left), NormalizedKey::UInt(right)) => left.cmp(right),
            (NormalizedKey::Float(left), NormalizedKey::Float(right)) => f64::from_bits(*left)
                .partial_cmp(&f64::from_bits(*right))
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        });
        entries
    }

    pub(super) fn normalize_value(
        &self,
        value: Value,
        builtin: &'static str,
    ) -> BuiltinResult<Value> {
        self.value_type.normalize(value, builtin)
    }

    pub(super) fn track_uniform_class(
        &mut self,
        value: &Value,
        builtin: &'static str,
    ) -> BuiltinResult<()> {
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
pub(super) enum ValueClass {
    Char,
    String,
    Numeric(NumericDType),
    Logical,
    Cell,
    Struct,
    Object,
    Other(&'static str),
}

impl ValueClass {
    pub(super) fn from_value(value: &Value) -> Self {
        match value {
            Value::CharArray(_) => ValueClass::Char,
            Value::String(_) | Value::StringArray(_) => ValueClass::String,
            Value::Num(_) => ValueClass::Numeric(NumericDType::F64),
            Value::Tensor(tensor) => ValueClass::Numeric(tensor.numeric_dtype()),
            Value::ComplexTensor(tensor) => ValueClass::Numeric(tensor.numeric_dtype()),
            Value::Bool(_) | Value::LogicalArray(_) => ValueClass::Logical,
            Value::Int(value) => ValueClass::Numeric(int_value_dtype(value)),
            Value::Cell(_) => ValueClass::Cell,
            Value::Struct(_) | Value::StructArray(_) => ValueClass::Struct,
            Value::ObjectArray(_)
            | Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_) => ValueClass::Object,
            _ => ValueClass::Other("other"),
        }
    }
}

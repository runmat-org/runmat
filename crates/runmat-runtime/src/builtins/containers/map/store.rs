//! Map language rules layered on GC-owned, data-only storage.

use super::constructor_args::int_value_dtype;
use super::*;
use runmat_value::NumericDType;

pub(super) use runmat_value::{
    ContainersMapEntry as MapEntry, ContainersMapKey as NormalizedKey,
    ContainersMapKeyType as KeyType, ContainersMapStorage as MapStore,
    ContainersMapValueClass as ValueClass, ContainersMapValueType as ValueType,
};

pub(super) trait KeyTypeRules {
    fn matlab_name(self) -> &'static str;
    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self>
    where
        Self: Sized;
}

impl KeyTypeRules for KeyType {
    fn matlab_name(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Double => "double",
            Self::Single => "single",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(value, "containers.Map: expected a KeyType string", builtin)?;
        match text.to_ascii_lowercase().as_str() {
            "char" | "character" => Ok(Self::Char),
            "double" => Ok(Self::Double),
            "single" => Ok(Self::Single),
            "int32" => Ok(Self::Int32),
            "uint32" => Ok(Self::UInt32),
            "int64" => Ok(Self::Int64),
            "uint64" => Ok(Self::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported KeyType '{other}'. Valid types: char, double, single, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }
}

pub(super) trait ValueTypeRules {
    fn matlab_name(self) -> &'static str;
    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self>
    where
        Self: Sized;
    fn normalize(self, value: Value, builtin: &'static str) -> BuiltinResult<Value>;
}

impl ValueTypeRules for ValueType {
    fn matlab_name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Char => "char",
            Self::Double => "double",
            Self::Single => "single",
            Self::Logical => "logical",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
        }
    }

    fn parse(value: &Value, builtin: &'static str) -> BuiltinResult<Self> {
        let text = string_from_value(
            value,
            "containers.Map: expected a ValueType string",
            builtin,
        )?;
        match text.to_ascii_lowercase().as_str() {
            "any" => Ok(Self::Any),
            "char" | "character" => Ok(Self::Char),
            "double" => Ok(Self::Double),
            "single" => Ok(Self::Single),
            "logical" => Ok(Self::Logical),
            "int8" => Ok(Self::Int8),
            "uint8" => Ok(Self::UInt8),
            "int16" => Ok(Self::Int16),
            "uint16" => Ok(Self::UInt16),
            "int32" => Ok(Self::Int32),
            "uint32" => Ok(Self::UInt32),
            "int64" => Ok(Self::Int64),
            "uint64" => Ok(Self::UInt64),
            other => Err(map_error(
                format!(
                    "containers.Map: unsupported ValueType '{other}'. Valid types: any, char, logical, double, single, int8, uint8, int16, uint16, int32, uint32, int64, uint64."
                ),
                builtin,
            )),
        }
    }

    fn normalize(self, value: Value, builtin: &'static str) -> BuiltinResult<Value> {
        match self {
            Self::Any => Ok(value),
            Self::Char => Ok(Value::CharArray(char_array_from_value(&value, builtin)?)),
            Self::Double => normalize_numeric_value(value, NumericDType::F64, builtin),
            Self::Single => normalize_numeric_value(value, NumericDType::F32, builtin),
            Self::Logical => normalize_logical_value(value, builtin),
            integer_type => normalize_integer_value(value, integer_type, builtin),
        }
    }
}

pub(super) trait MapStoreRules {
    fn get_value(&self, key: &NormalizedKey) -> Option<Value>;
    fn insert_new(&mut self, entry: MapEntry, builtin: &'static str) -> BuiltinResult<()>;
    fn set_prepared(&mut self, entry: MapEntry) -> Option<MapEntry>;
    fn remove(&mut self, key: &NormalizedKey, builtin: &'static str) -> BuiltinResult<MapEntry>;
    fn keys(&self) -> Vec<Value>;
    fn values(&self) -> Vec<Value>;
    fn track_uniform_class(&mut self, value: &Value, builtin: &'static str) -> BuiltinResult<()>;
}

impl MapStoreRules for MapStore {
    fn get_value(&self, key: &NormalizedKey) -> Option<Value> {
        self.get(key).map(|entry| entry.value.clone())
    }

    fn insert_new(&mut self, mut entry: MapEntry, builtin: &'static str) -> BuiltinResult<()> {
        runmat_value::validate_no_transient_sequence(&entry.value)
            .map_err(|error| map_error(format!("containers.Map: {error}"), builtin))?;
        if self.contains(&entry.normalized) {
            return Err(map_error(
                "containers.Map: Duplicate key name was provided.",
                builtin,
            ));
        }
        entry.value = self.value_type.normalize(entry.value, builtin)?;
        self.track_uniform_class(&entry.value, builtin)?;
        self.insert_unique(entry);
        Ok(())
    }

    fn set_prepared(&mut self, entry: MapEntry) -> Option<MapEntry> {
        if self.uniform_values && self.uniform_class.is_none() {
            self.uniform_class = Some(value_class_from_value(&entry.value));
        }
        self.replace_or_insert(entry)
    }

    fn remove(&mut self, key: &NormalizedKey, builtin: &'static str) -> BuiltinResult<MapEntry> {
        self.remove_entry(key)
            .ok_or_else(|| map_descriptor_error(&CONTAINERS_MAP_ERROR_MISSING_KEY, builtin))
    }

    fn keys(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.key_value.clone())
            .collect()
    }

    fn values(&self) -> Vec<Value> {
        self.sorted_entries()
            .into_iter()
            .map(|entry| entry.value.clone())
            .collect()
    }

    fn track_uniform_class(&mut self, value: &Value, builtin: &'static str) -> BuiltinResult<()> {
        if !self.uniform_values {
            return Ok(());
        }
        let class = value_class_from_value(value);
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

pub(super) fn value_class_from_value(value: &Value) -> ValueClass {
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
        Value::ObjectArray(_) | Value::Object(_) | Value::HandleObject(_) | Value::Listener(_) => {
            ValueClass::Object
        }
        _ => ValueClass::Other("other"),
    }
}

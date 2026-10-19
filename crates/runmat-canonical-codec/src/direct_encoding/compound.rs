use super::{encoding_error, invalid, serializer::Serializer, State};
use crate::CanonicalCodecError;
use serde::{ser, Serialize};
use std::collections::BTreeMap;

pub(super) struct Array<'a, 'control> {
    output: &'a mut Vec<u8>,
    state: &'a mut State<'control>,
    body: Vec<u8>,
    count: usize,
}
impl<'a, 'control> Array<'a, 'control> {
    pub(super) fn new(
        serializer: Serializer<'a, 'control>,
        _len: Option<usize>,
    ) -> Result<Self, CanonicalCodecError> {
        serializer.state.check()?;
        Ok(Self {
            output: serializer.output,
            state: serializer.state,
            body: Vec::new(),
            count: 0,
        })
    }
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("collection count overflow"))?;
        value.serialize(Serializer {
            output: &mut self.body,
            state: self.state,
        })
    }
    fn finish(self) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        let before = self.output.len();
        minicbor::Encoder::new(&mut *self.output)
            .array(self.count as u64)
            .map_err(encoding_error)?;
        self.state.charge(self.output.len() - before)?;
        // Payload bytes were charged at their scalar/header production sites.
        self.output.extend_from_slice(&self.body);
        Ok(())
    }
}
macro_rules! array {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Array<'_, '_> {
            type Ok = ();
            type Error = CanonicalCodecError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
                self.element(value)
            }
            fn end(self) -> Result<(), Self::Error> {
                self.finish()
            }
        }
    };
}
array!(SerializeSeq, serialize_element);
array!(SerializeTuple, serialize_element);
array!(SerializeTupleStruct, serialize_field);
array!(SerializeTupleVariant, serialize_field);

pub(super) struct Map<'a, 'control> {
    output: &'a mut Vec<u8>,
    state: &'a mut State<'control>,
    entries: BTreeMap<String, Vec<u8>>,
    pending: Option<String>,
}
impl<'a, 'control> Map<'a, 'control> {
    pub(super) fn new(
        serializer: Serializer<'a, 'control>,
        _len: Option<usize>,
    ) -> Result<Self, CanonicalCodecError> {
        serializer.state.check()?;
        Ok(Self {
            output: serializer.output,
            state: serializer.state,
            entries: BTreeMap::new(),
            pending: None,
        })
    }
    fn entry<T: Serialize + ?Sized>(
        &mut self,
        key: String,
        value: &T,
    ) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        let existing = self.entries.contains_key(&key);
        if !existing {
            let mut key_bytes = Vec::new();
            minicbor::Encoder::new(&mut key_bytes)
                .str(&key)
                .map_err(encoding_error)?;
            self.state.charge(key_bytes.len())?;
        }
        let mut encoded = Vec::new();
        value.serialize(Serializer {
            output: &mut encoded,
            state: self.state,
        })?;
        if let Some(previous) = self.entries.insert(key, encoded) {
            // serde_json's original map projection retains the last duplicate.
            self.state.wire_bytes -= previous.len();
        }
        Ok(())
    }
    fn finish(self) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        if self.pending.is_some() {
            return Err(invalid("map key lacks its value"));
        }
        let before = self.output.len();
        minicbor::Encoder::new(&mut *self.output)
            .map(self.entries.len() as u64)
            .map_err(encoding_error)?;
        self.state.charge(self.output.len() - before)?;
        for (key, value) in self.entries {
            self.state.check()?;
            minicbor::Encoder::new(&mut *self.output)
                .str(&key)
                .map_err(encoding_error)?;
            self.output.extend_from_slice(&value);
        }
        Ok(())
    }
}

struct KeyProjection<'a, T: ?Sized>(&'a T);
impl<T: Serialize + ?Sized> Serialize for KeyProjection<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(self.0, &())?;
        map.end()
    }
}
fn key<T: Serialize + ?Sized>(value: &T, state: &State<'_>) -> Result<String, CanonicalCodecError> {
    state.check()?;
    // Reuse the established JSON scalar-key conversion; this constructs only
    // a single key/null entry, never the map's values or a whole contract tree.
    let projected = crate::controlled_serde::to_value(&KeyProjection(value), state.control)
        .map_err(|error| CanonicalCodecError::invalid("canonical map key", error.to_string()))?;
    let serde_json::Value::Object(map) = projected else {
        return Err(invalid("invalid map key"));
    };
    let (key, _) = map
        .into_iter()
        .next()
        .ok_or_else(|| invalid("missing map key"))?;
    Ok(key)
}
impl ser::SerializeMap for Map<'_, '_> {
    type Ok = ();
    type Error = CanonicalCodecError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        if self.pending.is_some() {
            return Err(invalid("map key precedes an unresolved key"));
        }
        self.pending = Some(key(value, self.state)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        let key = self
            .pending
            .take()
            .ok_or_else(|| invalid("map value lacks its key"))?;
        self.entry(key, value)
    }
    fn serialize_entry<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        key_value: &K,
        value: &V,
    ) -> Result<(), Self::Error> {
        if self.pending.is_some() {
            return Err(invalid("map entry precedes an unresolved key"));
        }
        self.entry(key(key_value, self.state)?, value)
    }
    fn end(self) -> Result<(), Self::Error> {
        self.finish()
    }
}
macro_rules! map {
    ($trait:ident) => {
        impl ser::$trait for Map<'_, '_> {
            type Ok = ();
            type Error = CanonicalCodecError;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<(), Self::Error> {
                self.entry(key.to_owned(), value)
            }
            fn end(self) -> Result<(), Self::Error> {
                self.finish()
            }
        }
    };
}
map!(SerializeStruct);
map!(SerializeStructVariant);

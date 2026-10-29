use super::{encoding_error, invalid, serializer::Serializer, State};
use crate::CanonicalCodecError;
use serde::{ser, Serialize};
use std::collections::BTreeMap;

/// Sequences with a declared length write their header first and stream elements into the
/// parent buffer; only sequences of unknown length hold their body until the count is known.
pub(super) struct Array<'a, 'control> {
    output: &'a mut Vec<u8>,
    state: &'a mut State<'control>,
    body: Option<Vec<u8>>,
    expected: Option<usize>,
    count: usize,
}
impl<'a, 'control> Array<'a, 'control> {
    pub(super) fn new(
        serializer: Serializer<'a, 'control>,
        len: Option<usize>,
    ) -> Result<Self, CanonicalCodecError> {
        serializer.state.check()?;
        let Serializer { output, state, .. } = serializer;
        if let Some(len) = len {
            header(output, state, Header::Array(len as u64))?;
        }
        Ok(Self {
            output,
            state,
            body: len.is_none().then(Vec::new),
            expected: len,
            count: 0,
        })
    }
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("collection count overflow"))?;
        if self.expected.is_some_and(|len| self.count > len) {
            return Err(invalid("sequence is longer than its declared length"));
        }
        let Some(body) = self.body.as_mut() else {
            value.serialize(Serializer::child(self.output, self.state, value))?;
            self.state.flush(self.output, false);
            return Ok(());
        };
        value.serialize(Serializer::child(body, self.state, value))
    }
    fn finish(self) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        match self.body {
            None => {
                if self.expected != Some(self.count) {
                    return Err(invalid("sequence is shorter than its declared length"));
                }
                Ok(())
            }
            Some(body) => {
                header(self.output, self.state, Header::Array(self.count as u64))?;
                // Payload bytes were charged at their scalar/header production sites.
                self.output.extend_from_slice(&body);
                Ok(())
            }
        }
    }
}

enum Header {
    Array(u64),
    Map(u64),
}
fn header(
    output: &mut Vec<u8>,
    state: &mut State<'_>,
    header: Header,
) -> Result<(), CanonicalCodecError> {
    let before = output.len();
    let mut encoder = minicbor::Encoder::new(&mut *output);
    match header {
        Header::Array(count) => encoder.array(count),
        Header::Map(count) => encoder.map(count),
    }
    .map_err(encoding_error)?;
    state.charge(output.len() - before)
}
fn key_bytes(key: &str) -> Result<Vec<u8>, CanonicalCodecError> {
    let mut bytes = Vec::new();
    minicbor::Encoder::new(&mut bytes)
        .str(key)
        .map_err(encoding_error)?;
    Ok(bytes)
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

/// Canonical maps order keys by UTF-8 bytes.
///
/// * Structs with probed fields write their header first, stream each field that is next in
///   canonical order, and hold only fields that arrive before a smaller key.
/// * Maps with a declared length stream while keys strictly increase; the first out-of-order
///   or repeated key moves the streamed entries into the held inventory.
/// * Maps of unknown length hold every entry until the count is known.
pub(super) struct Map<'a, 'control> {
    output: &'a mut Vec<u8>,
    state: &'a mut State<'control>,
    mode: Mode,
    pending: Option<String>,
}
enum Mode {
    Ordered {
        order: Vec<&'static str>,
        next: usize,
        held: BTreeMap<&'static str, Vec<u8>>,
    },
    Streaming {
        /// Absolute position (see `State::position`) of the map header.
        header_start: usize,
        /// The header is pinned in a streamed root buffer and must be unpinned on exit.
        pinned: bool,
        expected: usize,
        /// Key and absolute position of each streamed entry, in strictly increasing key order.
        entries: Vec<(String, usize)>,
    },
    Held(BTreeMap<String, Vec<u8>>),
}
impl<'a, 'control> Map<'a, 'control> {
    pub(super) fn new(
        serializer: Serializer<'a, 'control>,
        len: Option<usize>,
    ) -> Result<Self, CanonicalCodecError> {
        serializer.state.check()?;
        let Serializer { output, state, .. } = serializer;
        let mode = match len {
            Some(len) => {
                let header_start = state.position(output);
                header(output, state, Header::Map(len as u64))?;
                let pinned = state.pin(output, header_start);
                Mode::Streaming {
                    header_start,
                    pinned,
                    expected: len,
                    entries: Vec::new(),
                }
            }
            None => Mode::Held(BTreeMap::new()),
        };
        Ok(Self {
            output,
            state,
            mode,
            pending: None,
        })
    }
    pub(super) fn structure(
        serializer: Serializer<'a, 'control>,
        len: usize,
    ) -> Result<Self, CanonicalCodecError> {
        let Some(mut order) = serializer.keys.clone().filter(|keys| keys.len() == len) else {
            return Self::new(serializer, Some(len));
        };
        order.sort_unstable();
        if order.windows(2).any(|pair| pair[0] == pair[1]) {
            return Self::new(serializer, Some(len));
        }
        serializer.state.check()?;
        let Serializer { output, state, .. } = serializer;
        header(output, state, Header::Map(len as u64))?;
        Ok(Self {
            output,
            state,
            mode: Mode::Ordered {
                order,
                next: 0,
                held: BTreeMap::new(),
            },
            pending: None,
        })
    }
    fn field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CanonicalCodecError> {
        let Mode::Ordered { order, next, held } = &mut self.mode else {
            return self.entry(key.to_owned(), value);
        };
        self.state.check()?;
        let key_bytes = key_bytes(key)?;
        self.state.charge(key_bytes.len())?;
        if order.get(*next) == Some(&key) {
            self.output.extend_from_slice(&key_bytes);
            value.serialize(Serializer::child(self.output, self.state, value))?;
            *next += 1;
            while let Some(entry) = order.get(*next).and_then(|key| held.remove(key)) {
                self.state.check()?;
                self.output.extend_from_slice(&entry);
                *next += 1;
            }
            self.state.flush(self.output, false);
            return Ok(());
        }
        let Ok(position) = order.binary_search(&key) else {
            return Err(invalid("struct field differs from its probed field set"));
        };
        if position < *next || held.contains_key(key) {
            return Err(invalid("struct field repeats within one value"));
        }
        let mut entry = key_bytes;
        value.serialize(Serializer::child(&mut entry, self.state, value))?;
        held.insert(order[position], entry);
        Ok(())
    }
    fn entry<T: Serialize + ?Sized>(
        &mut self,
        key: String,
        value: &T,
    ) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        if let Mode::Streaming { entries, .. } = &self.mode {
            if entries.last().is_none_or(|(last, _)| *last < key) {
                let start = self.state.position(self.output);
                let key_bytes = key_bytes(&key)?;
                self.state.charge(key_bytes.len())?;
                self.output.extend_from_slice(&key_bytes);
                value.serialize(Serializer::child(self.output, self.state, value))?;
                let Mode::Streaming { entries, .. } = &mut self.mode else {
                    unreachable!("streaming mode is unchanged by its value");
                };
                entries.push((key, start));
                // Bytes after this map's own pinned header stay until the map completes.
                self.state.flush(self.output, false);
                return Ok(());
            }
            self.hold_streamed()?;
        }
        let Mode::Held(entries) = &mut self.mode else {
            return Err(invalid("struct field differs from its probed field set"));
        };
        let existing = entries.contains_key(&key);
        if !existing {
            self.state.charge(key_bytes(&key)?.len())?;
        }
        let mut encoded = Vec::new();
        value.serialize(Serializer::child(&mut encoded, self.state, value))?;
        let Mode::Held(entries) = &mut self.mode else {
            unreachable!("held mode is unchanged by its value");
        };
        if let Some(previous) = entries.insert(key, encoded) {
            // serde_json's original map projection retains the last duplicate.
            self.state.wire_bytes -= previous.len();
        }
        Ok(())
    }
    /// Move streamed entries into the held inventory. Their key bytes stay charged as held
    /// keys; the header is rewritten with the final count.
    fn hold_streamed(&mut self) -> Result<(), CanonicalCodecError> {
        let Mode::Streaming {
            header_start,
            pinned,
            entries,
            ..
        } = std::mem::replace(&mut self.mode, Mode::Held(BTreeMap::new()))
        else {
            return Ok(());
        };
        if pinned {
            self.state.unpin(header_start);
        }
        let tail = self
            .output
            .split_off(self.state.index(self.output, header_start));
        let body_start = entries
            .first()
            .map_or(tail.len(), |(_, start)| start - header_start);
        self.state.wire_bytes -= body_start;
        let mut held = BTreeMap::new();
        let ends = entries
            .iter()
            .skip(1)
            .map(|(_, start)| start - header_start)
            .chain(std::iter::once(tail.len()))
            .collect::<Vec<_>>();
        for ((key, start), end) in entries.into_iter().zip(ends) {
            self.state.check()?;
            let value_start = start - header_start + key_bytes(&key)?.len();
            held.insert(key, tail[value_start..end].to_vec());
        }
        self.mode = Mode::Held(held);
        Ok(())
    }
    fn finish(mut self) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        if self.pending.is_some() {
            return Err(invalid("map key lacks its value"));
        }
        match &mut self.mode {
            Mode::Ordered { order, next, held } => {
                if *next != order.len() || !held.is_empty() {
                    return Err(invalid("struct fields differ from their probed field set"));
                }
                Ok(())
            }
            Mode::Streaming {
                header_start,
                pinned,
                expected,
                entries,
            } => {
                if *expected != entries.len() {
                    // A declared length that differs from the entries keeps the established
                    // count semantics of the held path.
                    self.hold_streamed()?;
                    return self.finish();
                }
                if *pinned {
                    let header_start = *header_start;
                    self.state.unpin(header_start);
                    self.state.flush(self.output, false);
                }
                Ok(())
            }
            Mode::Held(entries) => {
                let entries = std::mem::take(entries);
                header(self.output, self.state, Header::Map(entries.len() as u64))?;
                for (key, value) in entries {
                    self.state.check()?;
                    minicbor::Encoder::new(&mut *self.output)
                        .str(&key)
                        .map_err(encoding_error)?;
                    self.output.extend_from_slice(&value);
                }
                Ok(())
            }
        }
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
                self.field(key, value)
            }
            fn end(self) -> Result<(), Self::Error> {
                self.finish()
            }
        }
    };
}
map!(SerializeStruct);
map!(SerializeStructVariant);

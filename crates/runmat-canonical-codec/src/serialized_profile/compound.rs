use serde::{ser, Serialize};

use super::{invalid, CanonicalCodecError, Counter, Profiler};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    Sequence,
    Map,
    /// A struct or struct variant: fields with static names, ordered by the encoder.
    Struct,
}

/// Growth slack of a held `Vec` buffer: at most twice its length once complete, and three
/// times while reallocating.
const HELD_GROWTH: u64 = 3;
const HELD_RETAINED: u64 = 2;
const MAX_ORDERED_FIELDS: usize = 32;
const MAX_KEY_TEXT: usize = 64;

pub(super) struct Compound<'a> {
    counter: &'a mut Counter,
    depth: usize,
    expected: Option<usize>,
    count: usize,
    kind: Kind,
    pending_value: bool,
    ancestor_map_metadata: u64,
    /// Encoded bytes at the start of this compound's body.
    body_start: u64,
    /// Peak held-buffer bytes of any child, relative to this compound's sink.
    child_peak: u64,
    /// Struct fields in arrival order: name, entry bytes, and that field's own peak.
    /// Profiling stays allocation-free; wider structs use the conservative held charge.
    fields: [(&'static str, u64, u64); MAX_ORDERED_FIELDS],
    field_count: usize,
    /// Map key order: last key text, while keys strictly increase and are known.
    last_key: KeyText,
    increasing: bool,
    entry_start: u64,
}

impl<'a> Compound<'a> {
    pub(super) fn new(
        counter: &'a mut Counter,
        depth: usize,
        expected: Option<usize>,
        kind: Kind,
    ) -> Result<Self, CanonicalCodecError> {
        if expected.is_some_and(|len| len > counter.limits.maximum_collection_items) {
            return Err(invalid("collection length exceeds the contract limit"));
        }
        counter.node(depth)?;
        counter.encoded(expected.map_or(9, |length| super::wire::header(length as u64)))?;
        let ancestor_map_metadata = counter.live_map_metadata;
        let body_start = counter.encoded_bytes;
        Ok(Self {
            counter,
            depth,
            expected,
            count: 0,
            kind,
            pending_value: false,
            ancestor_map_metadata,
            body_start,
            child_peak: 0,
            fields: [("", 0, 0); MAX_ORDERED_FIELDS],
            field_count: 0,
            last_key: KeyText::default(),
            increasing: true,
            entry_start: 0,
        })
    }

    fn item(&mut self) -> Result<(), CanonicalCodecError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("collection length overflow"))?;
        if self.count > self.counter.limits.maximum_collection_items
            || self.expected.is_some_and(|len| self.count > len)
        {
            return Err(invalid(
                "collection length exceeds its declared or contract limit",
            ));
        }
        Ok(())
    }

    /// Profiles one child value and returns its own held-buffer peak.
    fn child<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<u64, CanonicalCodecError> {
        let depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| invalid("depth overflow"))?;
        self.counter.depth(depth)?;
        self.counter.child_peak = 0;
        value.serialize(Profiler::value(self.counter, depth))?;
        let peak = self.counter.child_peak;
        self.child_peak = self.child_peak.max(peak);
        Ok(peak)
    }

    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        if self.kind != Kind::Sequence {
            return Err(invalid("sequence element in map"));
        }
        self.item()?;
        self.child(value).map(|_| ())
    }

    fn key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        if self.kind == Kind::Sequence || self.pending_value {
            return Err(invalid("invalid map key order"));
        }
        self.entry_start = self.counter.encoded_bytes;
        if self.kind == Kind::Map && self.increasing {
            match KeyText::of(value) {
                Some(key) if self.count == 0 || self.last_key.text() < key.text() => {
                    self.last_key = key
                }
                _ => self.increasing = false,
            }
        }
        self.item()?;
        value.serialize(Profiler {
            counter: self.counter,
            depth: self.depth,
            key: true,
        })?;
        self.pending_value = true;
        Ok(())
    }

    fn value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<u64, CanonicalCodecError> {
        if self.kind == Kind::Sequence || !self.pending_value {
            return Err(invalid("map value has no key"));
        }
        let peak = self.child(value)?;
        self.pending_value = false;
        Ok(peak)
    }

    fn field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CanonicalCodecError> {
        self.key(key)?;
        let peak = self.value(value)?;
        if self.kind == Kind::Struct {
            let bytes = self.counter.encoded_bytes - self.entry_start;
            if let Some(slot) = self.fields.get_mut(self.field_count) {
                *slot = (key, bytes, peak);
            }
            self.field_count += 1;
        }
        Ok(())
    }

    /// Held-buffer peak of this compound relative to its parent's sink, mirroring the
    /// direct encoder's streaming decisions.
    fn held_peak(&self) -> Option<u64> {
        let body = self.counter.encoded_bytes - self.body_start;
        match self.kind {
            Kind::Sequence if self.expected.is_some() => Some(self.child_peak),
            Kind::Sequence => body.checked_mul(HELD_GROWTH)?.checked_add(self.child_peak),
            Kind::Map if self.increasing && self.expected == Some(self.count) => {
                Some(self.child_peak)
            }
            // Fallback moves streamed entries into held buffers (one split copy plus one
            // copy per entry) before re-emitting them.
            Kind::Map => body
                .checked_mul(HELD_GROWTH + 1)?
                .checked_add(self.child_peak),
            Kind::Struct => {
                let fields = match self.fields.get(..self.field_count) {
                    Some(fields) if self.expected == Some(self.field_count) => fields,
                    // Wider structs, or a count the encoder cannot stream, are charged as held.
                    _ => {
                        return body
                            .checked_mul(HELD_GROWTH + 1)?
                            .checked_add(self.child_peak)
                    }
                };
                // A field is held when a smaller key arrives after it. Held fields are
                // conservatively counted until the struct ends.
                let mut peak = 0u64;
                let mut held = 0u64;
                for (index, &(key, bytes, child)) in fields.iter().enumerate() {
                    if fields[..index].iter().any(|field| field.0 == key) {
                        // The encoder treats a repeated field as an unordered map.
                        return body
                            .checked_mul(HELD_GROWTH + 1)?
                            .checked_add(self.child_peak);
                    }
                    if fields[index + 1..].iter().any(|field| field.0 < key) {
                        let own = bytes.checked_mul(HELD_GROWTH)?;
                        peak = peak.max(held.checked_add(own)?.checked_add(child)?);
                        held = held.checked_add(bytes.checked_mul(HELD_RETAINED)?)?;
                    } else {
                        peak = peak.max(held.checked_add(child)?);
                    }
                }
                Some(peak)
            }
        }
    }

    fn finish(self) -> Result<(), CanonicalCodecError> {
        if self.pending_value || self.expected.is_some_and(|len| len != self.count) {
            return Err(invalid(
                "incomplete collection or mismatched declared length",
            ));
        }
        if self.kind != Kind::Sequence {
            self.counter.live_map_metadata = self.ancestor_map_metadata;
        }
        self.counter.child_peak = self
            .held_peak()
            .ok_or_else(|| invalid("held encoding workspace overflow"))?;
        Ok(())
    }
}

macro_rules! sequence {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Compound<'_> {
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
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);

impl ser::SerializeMap for Compound<'_> {
    type Ok = ();
    type Error = CanonicalCodecError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Self::Error> {
        self.key(key)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.value(value).map(|_| ())
    }
    fn end(self) -> Result<(), Self::Error> {
        self.finish()
    }
}

macro_rules! structure {
    ($trait:ident) => {
        impl ser::$trait for Compound<'_> {
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
structure!(SerializeStruct);
structure!(SerializeStructVariant);

/// Map key text as the encoder spells it, held inline, for keys whose spelling is known
/// without the shared JSON key projection. `None` means the profile assumes held entries.
#[derive(Clone, Copy)]
struct KeyText {
    bytes: [u8; MAX_KEY_TEXT],
    len: usize,
}
impl Default for KeyText {
    fn default() -> Self {
        Self {
            bytes: [0; MAX_KEY_TEXT],
            len: 0,
        }
    }
}
impl KeyText {
    fn text(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn from_str(value: &str) -> Result<Self, super::super::CanonicalCodecError> {
        let mut key = Self::default();
        key.bytes
            .get_mut(..value.len())
            .ok_or_else(|| invalid("unprofiled key"))?
            .copy_from_slice(value.as_bytes());
        key.len = value.len();
        Ok(key)
    }
    fn of<T: Serialize + ?Sized>(value: &T) -> Option<Self> {
        value.serialize(Text).ok()
    }
}
struct Text;
impl ser::Serializer for Text {
    type Ok = KeyText;
    type Error = super::super::CanonicalCodecError;
    type SerializeSeq = ser::Impossible<KeyText, Self::Error>;
    type SerializeTuple = ser::Impossible<KeyText, Self::Error>;
    type SerializeTupleStruct = ser::Impossible<KeyText, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<KeyText, Self::Error>;
    type SerializeMap = ser::Impossible<KeyText, Self::Error>;
    type SerializeStruct = ser::Impossible<KeyText, Self::Error>;
    type SerializeStructVariant = ser::Impossible<KeyText, Self::Error>;
    fn serialize_str(self, value: &str) -> Result<KeyText, Self::Error> {
        KeyText::from_str(value)
    }
    fn serialize_char(self, value: char) -> Result<KeyText, Self::Error> {
        KeyText::from_str(value.encode_utf8(&mut [0; 4]))
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<KeyText, Self::Error> {
        KeyText::from_str(variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<KeyText, Self::Error> {
        value.serialize(self)
    }
    fn serialize_bool(self, _: bool) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_i8(self, _: i8) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_i16(self, _: i16) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_i32(self, _: i32) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_i64(self, _: i64) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_u8(self, _: u8) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_u16(self, _: u16) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_u32(self, _: u32) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_u64(self, _: u64) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_f32(self, _: f32) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_f64(self, _: f64) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_none(self) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_unit(self) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<KeyText, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err(invalid("unprofiled key"))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(invalid("unprofiled key"))
    }
}

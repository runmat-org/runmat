//! Streaming Serde projection: no value tree, encoded buffer, or copied strings.

use serde::{ser, Serialize};

use crate::{
    materialization_resident_bytes_estimate, value::MaterializationProfile, CanonicalCodecError,
    CanonicalLimits,
};

mod compound;
#[cfg(test)]
mod tests;
use compound::{Compound, Kind};

// Maximum accepted prefix/domain lengths, plus conservative array/string CBOR headers.
const ENVELOPE_BYTES: u64 = 128 + 9 + 9 + 128;

/// Conservative encoding and materialization estimates from one borrowed traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SerializedCanonicalProfile {
    pub encoded_bytes_upper_bound: u64,
    pub materialization_resident_bytes_estimate: u64,
}

/// Projects canonical materialization residency directly from a borrowed Serde value.
///
/// The traversal itself allocates no value tree, output buffer, or owned strings. Custom
/// `Serialize`/`Display` implementations (including Serde-generated buffering for some advanced
/// representations) may allocate independently. Errors allocate their diagnostic message.
/// Counters, per-container lengths, string lengths, and nesting are checked during traversal.
/// Encoded bytes reserve the largest valid canonical envelope and nine-byte CBOR headers/scalars,
/// so this can conservatively reject values that a particular smaller envelope would accept.
/// Nesting also counts transparent `Some`/newtype wrappers, bounding traversal stack depth even
/// though those wrappers do not add nodes to the canonical value tree.
/// Nonfinite numbers, integers outside the JSON number range, and non-scalar map keys fail closed.
/// This is the shared materialization estimate, not an exact typed-object heap measurement.
pub fn serialized_materialization_resident_bytes_estimate<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<u64, CanonicalCodecError> {
    Ok(profile_serialized(value, limits)?.materialization_resident_bytes_estimate)
}

/// Profiles both canonical wire size and materialization without constructing either.
///
/// Uses the same traversal, conservative envelope, validation and custom-serializer caveats as
/// [`serialized_materialization_resident_bytes_estimate`]. The wire bound includes one contract
/// envelope; separately encoded partitions must each be profiled with their own envelope.
pub fn profile_serialized<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: 0,
        limits,
    };
    counter.encoded(ENVELOPE_BYTES)?;
    value.serialize(Profiler::value(&mut counter, 0))?;
    Ok(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: counter.encoded_bytes,
        materialization_resident_bytes_estimate: materialization_resident_bytes_estimate(
            counter.profile,
            counter.encoded_bytes,
        )?,
    })
}

fn invalid(reason: &'static str) -> CanonicalCodecError {
    CanonicalCodecError::invalid("serialized materialization profile", reason)
}

impl ser::Error for CanonicalCodecError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        Self::invalid("serialized materialization profile", message.to_string())
    }
}

struct Counter {
    profile: MaterializationProfile,
    encoded_bytes: u64,
    limits: CanonicalLimits,
}

impl Counter {
    fn encoded(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.encoded_bytes = self
            .encoded_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("encoded byte count overflow"))?;
        if self.encoded_bytes > self.limits.maximum_encoded_bytes as u64 {
            return Err(invalid("projected encoded bytes exceed the contract limit"));
        }
        Ok(())
    }

    fn depth(&self, depth: usize) -> Result<(), CanonicalCodecError> {
        if depth > self.limits.maximum_nesting_depth {
            Err(invalid("nesting depth exceeds the contract limit"))
        } else {
            Ok(())
        }
    }

    fn node(&mut self, depth: usize) -> Result<(), CanonicalCodecError> {
        self.depth(depth)?;
        self.profile.value_nodes = self
            .profile
            .value_nodes
            .checked_add(1)
            .ok_or_else(|| invalid("value count overflow"))?;
        self.encoded(9)
    }

    fn string(&mut self, len: usize, key: bool, depth: usize) -> Result<(), CanonicalCodecError> {
        self.depth(depth)?;
        if len > self.limits.maximum_string_bytes {
            return Err(invalid("string bytes exceed the contract limit"));
        }
        if key {
            self.profile.map_entries = self
                .profile
                .map_entries
                .checked_add(1)
                .ok_or_else(|| invalid("map entry count overflow"))?;
            self.encoded(9)?;
        } else {
            self.node(depth)?;
        }
        self.profile.string_bytes = self
            .profile
            .string_bytes
            .checked_add(len as u64)
            .ok_or_else(|| invalid("string byte count overflow"))?;
        self.encoded(len as u64)
    }
}

struct Profiler<'a> {
    counter: &'a mut Counter,
    depth: usize,
    key: bool,
}

impl<'a> Profiler<'a> {
    fn value(counter: &'a mut Counter, depth: usize) -> Self {
        Self {
            counter,
            depth,
            key: false,
        }
    }

    fn scalar(self) -> Result<(), CanonicalCodecError> {
        if self.key {
            return Err(invalid("unsupported map key"));
        }
        self.counter.node(self.depth)
    }

    fn display(self, value: impl std::fmt::Display) -> Result<(), CanonicalCodecError> {
        use std::fmt::Write;
        struct StringCount {
            len: usize,
            maximum: usize,
        }
        impl std::fmt::Write for StringCount {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                self.len = self.len.checked_add(value.len()).ok_or(std::fmt::Error)?;
                if self.len > self.maximum {
                    return Err(std::fmt::Error);
                }
                Ok(())
            }
        }
        let mut count = StringCount {
            len: 0,
            maximum: self.counter.limits.maximum_string_bytes.min(
                (self.counter.limits.maximum_encoded_bytes as u64)
                    .saturating_sub(self.counter.encoded_bytes)
                    .saturating_sub(9) as usize,
            ),
        };
        write!(&mut count, "{value}")
            .map_err(|_| invalid("formatted string failed or exceeded the contract limit"))?;
        self.counter.string(count.len, self.key, self.depth)
    }

    fn compound(self, len: Option<usize>, kind: Kind) -> Result<Compound<'a>, CanonicalCodecError> {
        if self.key {
            return Err(invalid("compound map keys are unsupported"));
        }
        Compound::new(self.counter, self.depth, len, kind)
    }

    fn variant(self, variant: &str) -> Result<Profiler<'a>, CanonicalCodecError> {
        if self.key {
            return Err(invalid("non-unit enum map keys are unsupported"));
        }
        if self.counter.limits.maximum_collection_items == 0 {
            return Err(invalid("enum wrapper exceeds the collection limit"));
        }
        self.counter.node(self.depth)?;
        self.counter.string(variant.len(), true, self.depth)?;
        let depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| invalid("depth overflow"))?;
        self.counter.depth(depth)?;
        Ok(Self::value(self.counter, depth))
    }

    fn transparent<T: Serialize + ?Sized>(self, value: &T) -> Result<(), CanonicalCodecError> {
        // Bound Rust traversal even for wrappers that vanish from the serialized value tree.
        let depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| invalid("depth overflow"))?;
        self.counter.depth(depth)?;
        value.serialize(Self { depth, ..self })
    }
}

macro_rules! integer {
    ($($name:ident($ty:ty)),* $(,)?) => { $(
        fn $name(self, value: $ty) -> Result<(), Self::Error> {
            if self.key { self.display(value) } else { self.scalar() }
        }
    )* };
}

impl<'a> ser::Serializer for Profiler<'a> {
    type Ok = ();
    type Error = CanonicalCodecError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = Compound<'a>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    integer!(
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64)
    );

    fn serialize_i128(self, value: i128) -> Result<(), Self::Error> {
        if !self.key && (value < i64::MIN as i128 || value > u64::MAX as i128) {
            return Err(invalid("integer is outside the JSON number range"));
        }
        if self.key {
            self.display(value)
        } else {
            self.scalar()
        }
    }
    fn serialize_u128(self, value: u128) -> Result<(), Self::Error> {
        if !self.key && value > u64::MAX as u128 {
            return Err(invalid("integer is outside the JSON number range"));
        }
        if self.key {
            self.display(value)
        } else {
            self.scalar()
        }
    }
    fn serialize_bool(self, value: bool) -> Result<(), Self::Error> {
        if self.key {
            self.display(value)
        } else {
            self.scalar()
        }
    }
    fn serialize_f32(self, value: f32) -> Result<(), Self::Error> {
        self.serialize_f64(f64::from(value))
    }
    fn serialize_f64(self, value: f64) -> Result<(), Self::Error> {
        if !value.is_finite() {
            return Err(invalid("nonfinite number"));
        }
        // Float map-key spellings are serializer-specific. Refuse to undercount them.
        if self.key {
            return Err(invalid("floating point map keys are unsupported"));
        }
        self.scalar()
    }
    fn serialize_char(self, value: char) -> Result<(), Self::Error> {
        self.counter.string(value.len_utf8(), self.key, self.depth)
    }
    fn serialize_str(self, value: &str) -> Result<(), Self::Error> {
        self.counter.string(value.len(), self.key, self.depth)
    }
    fn serialize_bytes(self, values: &[u8]) -> Result<(), Self::Error> {
        use ser::SerializeSeq;
        let mut sequence = self.compound(Some(values.len()), Kind::Sequence)?;
        for value in values {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
    fn serialize_none(self) -> Result<(), Self::Error> {
        self.scalar()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        if self.key {
            return Err(invalid("optional map keys are unsupported"));
        }
        self.transparent(value)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        self.scalar()
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        self.scalar()
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), Self::Error> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.transparent(value)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(self.variant(variant)?)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.compound(len, Kind::Sequence)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.compound(Some(len), Kind::Sequence)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.compound(Some(len), Kind::Sequence)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.variant(variant)?.compound(Some(len), Kind::Sequence)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.compound(len, Kind::Map)
    }
    fn serialize_struct(
        self,
        _: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.compound(Some(len), Kind::Map)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.variant(variant)?.compound(Some(len), Kind::Map)
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        self.display(value)
    }
}

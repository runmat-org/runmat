//! Streaming Serde projection: no value tree, encoded buffer, or copied strings.

use serde::{ser, Serialize};

use crate::{
    materialization_resident_bytes_estimate, value::MaterializationProfile, CanonicalCodecError,
    CanonicalLimits,
};

mod compound;
mod integer_values;
mod materialization_profile;
pub use materialization_profile::{
    profile_direct_materialization, profile_serialized, profile_serialized_shape,
    profile_serialized_with_control, profile_typed_resident_bytes,
    serialized_materialization_resident_bytes_estimate,
};
#[cfg(test)]
mod tests;
mod wire;
use compound::{Compound, Kind};

// Maximum accepted prefix/domain lengths, plus conservative array/string CBOR headers.
const ENVELOPE_BYTES: u64 = 128 + 9 + 9 + 128;

/// Conservative encoding and materialization estimates from one borrowed traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SerializedCanonicalProfile {
    pub encoded_bytes_upper_bound: u64,
    pub materialization_resident_bytes_estimate: u64,
}

impl SerializedCanonicalProfile {
    /// Requested workspace for Serde-to-JSON value projection alone. Retains the
    /// conservative node/container/string allowances; excludes the two canonical
    /// wire buffers reserved by decode/re-encode materialization. Callers must
    /// separately charge their original typed input, retained outputs and any
    /// subsequent encoding. This is not typed heap or measured allocator usage.
    pub fn json_projection_resident_bytes_estimate(&self) -> Option<u64> {
        self.materialization_resident_bytes_estimate
            .checked_sub(self.encoded_bytes_upper_bound.checked_mul(2)?)
            .filter(|bytes| {
                *bytes >= crate::materialization::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT
            })
    }

    /// Incremental structural storage when a collection already reserves the shared
    /// contract envelope. This must not replace admission of an independent contract.
    /// Synthetic profiles below the shared envelope are rejected rather than rounded up.
    pub fn materialization_bytes_without_envelope(&self) -> Option<u64> {
        self.materialization_resident_bytes_estimate
            .checked_sub(crate::materialization::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT)
    }
}

/// Requested direct-encoding workspace, distinct from typed retention and hostile decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalEncodingProfile {
    pub encoded_bytes_upper_bound: u64,
    pub resident_workspace_bytes_estimate: u64,
}

/// Profiles the direct encoder's byte buffers and sorted-map metadata: the reserved output
/// bound, plus the peak of separately held buffers (struct fields that arrive before a smaller
/// key, maps that leave strictly increasing key order, collections of unknown length) with
/// their growth and copy slack, plus peak live ancestor-map metadata and copied keys.
/// String values are borrowed. No full JSON tree is constructed by that encoder. Original
/// typed input must be charged separately.
pub fn profile_canonical_encoding<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<CanonicalEncodingProfile, CanonicalCodecError> {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: 0,
        limits,
        live_map_metadata: 0,
        peak_map_metadata: 0,
        integer_value_bounds: false,
        child_peak: 0,
    };
    counter.encoded(ENVELOPE_BYTES)?;
    value.serialize(Profiler::value(&mut counter, 0))?;
    // The encoder reserves the profiled output once and streams into it; only fields that
    // arrive before a smaller key and collections of unknown length are held separately.
    let workspace = counter
        .encoded_bytes
        .checked_add(counter.child_peak)
        .and_then(|bytes| bytes.checked_add(counter.peak_map_metadata))
        .and_then(|bytes| {
            bytes.checked_add(
                crate::materialization::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
            )
        })
        .ok_or_else(|| {
            CanonicalCodecError::resource_limit("canonical encoding profile", "workspace overflow")
        })?;
    Ok(CanonicalEncodingProfile {
        encoded_bytes_upper_bound: counter.encoded_bytes,
        resident_workspace_bytes_estimate: workspace,
    })
}

/// Profiles the same borrowed encoding with cancellation at recursive boundaries.
/// The adapter reuses the shared controlled Serde traversal and constructs no value tree.
pub fn profile_canonical_encoding_with_control<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
    control: &dyn crate::CanonicalCodecControl,
) -> Result<CanonicalEncodingProfile, CanonicalCodecError> {
    profile_canonical_encoding(&crate::controlled_serde::tracked(value, control), limits)
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
    live_map_metadata: u64,
    peak_map_metadata: u64,
    integer_value_bounds: bool,
    /// Peak bytes held in encoder buffers while encoding the value that just completed,
    /// relative to its parent's sink (see `compound::Compound::finish`).
    child_peak: u64,
}

impl Counter {
    fn encoded(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.encoded_bytes = self.encoded_bytes.checked_add(bytes).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "encoded byte count overflow",
            )
        })?;
        if self.encoded_bytes > self.limits.maximum_encoded_bytes as u64 {
            return Err(CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "projected encoded bytes exceed the contract limit",
            ));
        }
        Ok(())
    }

    fn depth(&self, depth: usize) -> Result<(), CanonicalCodecError> {
        if depth > self.limits.maximum_nesting_depth {
            Err(CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "nesting depth exceeds the contract limit",
            ))
        } else {
            Ok(())
        }
    }

    fn node(&mut self, depth: usize) -> Result<(), CanonicalCodecError> {
        self.depth(depth)?;
        self.profile.value_nodes = self.profile.value_nodes.checked_add(1).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "value count overflow",
            )
        })?;
        Ok(())
    }

    fn string(&mut self, len: usize, key: bool, depth: usize) -> Result<(), CanonicalCodecError> {
        self.depth(depth)?;
        if len > self.limits.maximum_string_bytes {
            return Err(CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "string bytes exceed the contract limit",
            ));
        }
        if key {
            // Only keys and slots on active ancestor maps stay live. Scalar
            // string values are borrowed; completed child maps become bytes.
            let slot = (2
                * (std::mem::size_of::<String>()
                    + std::mem::size_of::<Vec<u8>>()
                    + 4 * std::mem::size_of::<usize>())) as u64;
            self.live_map_metadata = self
                .live_map_metadata
                .checked_add(slot)
                .and_then(|bytes| bytes.checked_add((len as u64).checked_mul(2)?))
                .ok_or_else(|| invalid("live map metadata overflow"))?;
            self.peak_map_metadata = self.peak_map_metadata.max(self.live_map_metadata);
            self.profile.map_entries =
                self.profile.map_entries.checked_add(1).ok_or_else(|| {
                    CanonicalCodecError::resource_limit(
                        "serialized materialization profile",
                        "map entry count overflow",
                    )
                })?;
        } else {
            self.node(depth)?;
        }
        self.encoded(wire::header(len as u64))?;
        self.profile.string_bytes = self
            .profile
            .string_bytes
            .checked_add(len as u64)
            .ok_or_else(|| {
                CanonicalCodecError::resource_limit(
                    "serialized materialization profile",
                    "string byte count overflow",
                )
            })?;
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

    fn scalar(self, bytes: u64) -> Result<(), CanonicalCodecError> {
        if self.key {
            return Err(invalid("unsupported map key"));
        }
        self.counter.node(self.depth)?;
        self.counter.encoded(bytes)
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
                    .saturating_sub(1) as usize,
            ),
        };
        write!(&mut count, "{value}").map_err(|_| {
            CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "formatted string failed or exceeded the contract limit",
            )
        })?;
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
            return Err(CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "enum wrapper exceeds the collection limit",
            ));
        }
        self.counter.node(self.depth)?;
        self.counter.encoded(wire::header(1))?;
        let retained = self.counter.live_map_metadata;
        self.counter.string(variant.len(), true, self.depth)?;
        // The direct encoder writes enum wrappers directly, without a map owner.
        self.counter.live_map_metadata = retained;
        let depth = self.depth.checked_add(1).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "depth overflow",
            )
        })?;
        self.counter.depth(depth)?;
        Ok(Self::value(self.counter, depth))
    }

    fn transparent<T: Serialize + ?Sized>(self, value: &T) -> Result<(), CanonicalCodecError> {
        // Bound Rust traversal even for wrappers that vanish from the serialized value tree.
        let depth = self.depth.checked_add(1).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "serialized materialization profile",
                "depth overflow",
            )
        })?;
        self.counter.depth(depth)?;
        value.serialize(Self { depth, ..self })
    }
}

macro_rules! integer {
    ($($name:ident($ty:ty)),* $(,)?) => { $(
        fn $name(self, value: $ty) -> Result<(), Self::Error> {
            // Full-width CBOR integer: one tag byte plus its Serde type width.
            self.integer(value as i128, 1 + std::mem::size_of::<$ty>() as u64, 20)
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
        self.integer(value, 9, 40)
    }
    fn serialize_u128(self, value: u128) -> Result<(), Self::Error> {
        if !self.key && value > u64::MAX as u128 {
            return Err(invalid("integer is outside the JSON number range"));
        }
        if self.key {
            if self.counter.integer_value_bounds {
                self.counter.string(40, true, self.depth)
            } else {
                self.display(value)
            }
        } else {
            self.integer(value as i128, 9, 40)
        }
    }
    fn serialize_bool(self, value: bool) -> Result<(), Self::Error> {
        if self.key {
            self.display(value)
        } else {
            self.scalar(1)
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
        self.scalar(9)
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
        self.scalar(1)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        if self.key {
            return Err(invalid("optional map keys are unsupported"));
        }
        self.transparent(value)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        self.scalar(1)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        self.scalar(1)
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
        self.compound(Some(len), Kind::Struct)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.variant(variant)?.compound(Some(len), Kind::Struct)
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        self.display(value)
    }
}

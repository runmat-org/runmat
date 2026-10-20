use super::{
    compound::{Array, Map},
    encoding_error, invalid, State,
};
use crate::CanonicalCodecError;
use serde::{ser, Serialize};

pub(super) struct Serializer<'a, 'control> {
    pub(super) output: &'a mut Vec<u8>,
    pub(super) state: &'a mut State<'control>,
    /// Probed field names when this value is struct-shaped (see `probe`).
    pub(super) keys: Option<Vec<&'static str>>,
}
impl<'a, 'control> Serializer<'a, 'control> {
    /// Serializer for one child value, carrying that value's probed struct fields.
    pub(super) fn child<T: Serialize + ?Sized>(
        output: &'a mut Vec<u8>,
        state: &'a mut State<'control>,
        value: &T,
    ) -> Self {
        Self {
            output,
            state,
            keys: super::probe::struct_keys(value),
        }
    }
}
impl Serializer<'_, '_> {
    fn scalar(
        self,
        encode: impl FnOnce(&mut minicbor::Encoder<&mut Vec<u8>>) -> Result<(), CanonicalCodecError>,
    ) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        let before = self.output.len();
        encode(&mut minicbor::Encoder::new(&mut *self.output))?;
        self.state.charge(self.output.len() - before)
    }
    fn variant(&mut self, variant: &str) -> Result<(), CanonicalCodecError> {
        self.state.check()?;
        let before = self.output.len();
        minicbor::Encoder::new(&mut *self.output)
            .map(1)
            .and_then(|encoder| encoder.str(variant))
            .map_err(encoding_error)?;
        self.state.charge(self.output.len() - before)
    }
}
macro_rules! scalar {
    ($method:ident, $ty:ty, $encode:ident) => {
        fn $method(self, value: $ty) -> Result<(), Self::Error> {
            self.scalar(|encoder| encoder.$encode(value).map(|_| ()).map_err(encoding_error))
        }
    };
}
macro_rules! signed {
    ($method:ident, $ty:ty) => {
        fn $method(self, value: $ty) -> Result<(), Self::Error> {
            self.serialize_i64(value as i64)
        }
    };
}
macro_rules! unsigned {
    ($method:ident, $ty:ty) => {
        fn $method(self, value: $ty) -> Result<(), Self::Error> {
            self.serialize_u64(value as u64)
        }
    };
}
impl<'a, 'control> ser::Serializer for Serializer<'a, 'control> {
    type Ok = ();
    type Error = CanonicalCodecError;
    type SerializeSeq = Array<'a, 'control>;
    type SerializeTuple = Array<'a, 'control>;
    type SerializeTupleStruct = Array<'a, 'control>;
    type SerializeTupleVariant = Array<'a, 'control>;
    type SerializeMap = Map<'a, 'control>;
    type SerializeStruct = Map<'a, 'control>;
    type SerializeStructVariant = Map<'a, 'control>;
    scalar!(serialize_bool, bool, bool);
    signed!(serialize_i8, i8);
    signed!(serialize_i16, i16);
    signed!(serialize_i32, i32);
    unsigned!(serialize_u8, u8);
    unsigned!(serialize_u16, u16);
    unsigned!(serialize_u32, u32);
    scalar!(serialize_i64, i64, i64);
    scalar!(serialize_u64, u64, u64);
    fn serialize_i128(self, value: i128) -> Result<(), Self::Error> {
        if let Ok(value) = i64::try_from(value) {
            self.serialize_i64(value)
        } else if let Ok(value) = u64::try_from(value) {
            self.serialize_u64(value)
        } else {
            Err(invalid("integer outside the JSON number range"))
        }
    }
    fn serialize_u128(self, value: u128) -> Result<(), Self::Error> {
        self.serialize_u64(
            u64::try_from(value).map_err(|_| invalid("integer outside the JSON number range"))?,
        )
    }
    fn serialize_f32(self, value: f32) -> Result<(), Self::Error> {
        self.serialize_f64(value as f64)
    }
    fn serialize_f64(self, value: f64) -> Result<(), Self::Error> {
        // Keep serde_json::Value's established projection semantics, including null
        // for nonfinite floats. Domain admission owns finite physical coefficients.
        if value.is_finite() {
            self.scalar(|encoder| encoder.f64(value).map(|_| ()).map_err(encoding_error))
        } else {
            self.serialize_unit()
        }
    }
    fn serialize_char(self, value: char) -> Result<(), Self::Error> {
        self.serialize_str(value.encode_utf8(&mut [0; 4]))
    }
    fn serialize_str(self, value: &str) -> Result<(), Self::Error> {
        self.scalar(|encoder| encoder.str(value).map(|_| ()).map_err(encoding_error))
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<(), Self::Error> {
        use ser::SerializeSeq;
        let mut sequence = self.serialize_seq(Some(value.len()))?;
        for byte in value {
            sequence.serialize_element(byte)?;
        }
        sequence.end()
    }
    fn serialize_none(self) -> Result<(), Self::Error> {
        self.serialize_unit()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        self.state.check()?;
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        self.scalar(|encoder| encoder.null().map(|_| ()).map_err(encoding_error))
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        self.serialize_unit()
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
        self.serialize_some(value)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.variant(variant)?;
        value.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Array::new(self, len)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Array::new(self, Some(len))
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Array::new(self, Some(len))
    }
    fn serialize_tuple_variant(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.variant(variant)?;
        Array::new(self, Some(len))
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Map::new(self, len)
    }
    fn serialize_struct(
        self,
        _: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Map::structure(self, len)
    }
    fn serialize_struct_variant(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.variant(variant)?;
        Map::structure(self, len)
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        self.serialize_str(&value.to_string())
    }
    fn is_human_readable(&self) -> bool {
        true
    }
}

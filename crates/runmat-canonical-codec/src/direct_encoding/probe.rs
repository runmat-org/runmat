//! Field-name probe for struct-shaped values.
//!
//! Canonical maps order keys by UTF-8 bytes, while Serde emits struct fields in declaration
//! order. Knowing a struct's complete field set before its first field lets the encoder write
//! fields that already arrive in canonical order straight into the parent buffer and hold only
//! the fields that arrive early. The probe records field names and never visits field values.
use serde::{ser, Serialize};

#[derive(Debug)]
pub(super) struct Stop;
impl std::fmt::Display for Stop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("value is not struct-shaped")
    }
}
impl std::error::Error for Stop {}
impl ser::Error for Stop {
    fn custom<T: std::fmt::Display>(_: T) -> Self {
        Stop
    }
}

/// Field names of a struct or struct variant, in emission order; `None` for any other shape.
pub(super) fn struct_keys<T: Serialize + ?Sized>(value: &T) -> Option<Vec<&'static str>> {
    value.serialize(Probe).ok()
}

struct Probe;

pub(super) struct Fields(Vec<&'static str>);
impl Fields {
    fn record(&mut self, key: &'static str) {
        self.0.push(key);
    }
}
impl ser::SerializeStruct for Fields {
    type Ok = Vec<&'static str>;
    type Error = Stop;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        _: &T,
    ) -> Result<(), Stop> {
        self.record(key);
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Stop> {
        Ok(self.0)
    }
}
impl ser::SerializeStructVariant for Fields {
    type Ok = Vec<&'static str>;
    type Error = Stop;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        _: &T,
    ) -> Result<(), Stop> {
        self.record(key);
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Stop> {
        Ok(self.0)
    }
}

macro_rules! stop {
    ($($method:ident($($arg:ty),*)),* $(,)?) => { $(
        fn $method(self, $(_: $arg),*) -> Result<Self::Ok, Stop> {
            Err(Stop)
        }
    )* };
}

impl ser::Serializer for Probe {
    type Ok = Vec<&'static str>;
    type Error = Stop;
    type SerializeSeq = ser::Impossible<Self::Ok, Stop>;
    type SerializeTuple = ser::Impossible<Self::Ok, Stop>;
    type SerializeTupleStruct = ser::Impossible<Self::Ok, Stop>;
    type SerializeTupleVariant = ser::Impossible<Self::Ok, Stop>;
    type SerializeMap = ser::Impossible<Self::Ok, Stop>;
    type SerializeStruct = Fields;
    type SerializeStructVariant = Fields;

    stop!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8]),
        serialize_none(),
        serialize_unit(),
        serialize_unit_struct(&'static str),
        serialize_unit_variant(&'static str, u32, &'static str),
    );

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Stop> {
        value.serialize(self)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Stop> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Stop> {
        value.serialize(self)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Stop> {
        Err(Stop)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Stop> {
        Err(Stop)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Stop> {
        Err(Stop)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Stop> {
        Err(Stop)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Stop> {
        Err(Stop)
    }
    fn serialize_struct(self, _: &'static str, len: usize) -> Result<Self::SerializeStruct, Stop> {
        Ok(Fields(Vec::with_capacity(len)))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Stop> {
        Ok(Fields(Vec::with_capacity(len)))
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, _: &T) -> Result<Self::Ok, Stop> {
        Err(Stop)
    }
    fn is_human_readable(&self) -> bool {
        true
    }
}

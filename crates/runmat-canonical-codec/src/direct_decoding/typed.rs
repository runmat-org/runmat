use super::{access, invalid, CanonicalCodecControl, CanonicalCodecError};
use minicbor::data::Type;
use serde::de::{self, Visitor};
macro_rules! numeric {
    ($($method:ident),*) => {$(
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            let number = match self.input.datatype().map_err(invalid)? {
                Type::U8 | Type::U16 | Type::U32 | Type::U64 => serde_json::Number::from(self.input.u64().map_err(invalid)?),
                Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Int => serde_json::Number::from(self.input.i64().map_err(invalid)?),
                Type::F64 => serde_json::Number::from_f64(self.input.f64().map_err(invalid)?).ok_or_else(|| invalid("finite number required"))?,
                _ => return Err(invalid("number required")),
            };
            crate::checkpoint(self.control)?;
            de::Deserializer::$method(number, visitor).map_err(invalid)
        }
    )*};
}
macro_rules! restricted {
    ($method:ident, $($kind:pat_param)|+) => {
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            if !matches!(self.input.datatype().map_err(invalid)?, $($kind)|+) {
                return Err(invalid("value has an incompatible type"));
            }
            self.deserialize_any(visitor)
        }
    };
}
pub(super) struct Decoder<'de, 'c> {
    pub(super) input: minicbor::Decoder<'de>,
    pub(super) control: &'c dyn CanonicalCodecControl,
}
impl Decoder<'_, '_> {
    // Owned Value visitors could discard an option/newtype/enum payload. Preserve
    // that behavior while advancing the original stream cooperatively.
    fn consume_if_unread(&mut self, start: usize) -> Result<(), CanonicalCodecError> {
        if self.input.position() == start {
            de::Deserializer::deserialize_any(self, de::IgnoredAny)?;
        }
        Ok(())
    }
}
impl<'de> de::Deserializer<'de> for &mut Decoder<'de, '_> {
    type Error = CanonicalCodecError;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control)?;
        match self.input.datatype().map_err(invalid)? {
            Type::Null => {
                self.input.null().map_err(invalid)?;
                visitor.visit_unit()
            }
            Type::Bool => visitor.visit_bool(self.input.bool().map_err(invalid)?),
            Type::U8 | Type::U16 | Type::U32 | Type::U64 => {
                visitor.visit_u64(self.input.u64().map_err(invalid)?)
            }
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Int => {
                visitor.visit_i64(self.input.i64().map_err(invalid)?)
            }
            Type::F64 => visitor.visit_f64(self.input.f64().map_err(invalid)?),
            Type::String => visitor.visit_string(self.input.str().map_err(invalid)?.to_owned()),
            Type::Array => {
                let count = self
                    .input
                    .array()
                    .map_err(invalid)?
                    .ok_or_else(|| invalid("definite array required"))?;
                let mut access = access::Sequence {
                    decoder: self,
                    remaining: count,
                };
                let result = visitor.visit_seq(&mut access)?;
                if access.remaining != 0 {
                    return Err(invalid("sequence visitor left unread elements"));
                }
                Ok(result)
            }
            Type::Map => {
                let count = self
                    .input
                    .map()
                    .map_err(invalid)?
                    .ok_or_else(|| invalid("definite map required"))?;
                let mut access = access::Map {
                    decoder: self,
                    remaining: count,
                    pending: false,
                };
                let result = visitor.visit_map(&mut access)?;
                if access.remaining != 0 || access.pending {
                    return Err(invalid("map visitor left unread entries"));
                }
                Ok(result)
            }
            _ => Err(invalid("unsupported canonical value")),
        }
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control)?;
        if self.input.datatype().map_err(invalid)? == Type::Null {
            self.input.null().map_err(invalid)?;
            visitor.visit_none()
        } else {
            let start = self.input.position();
            let result = visitor.visit_some(&mut *self)?;
            self.consume_if_unread(start)?;
            Ok(result)
        }
    }
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control)?;
        let start = self.input.position();
        let result = visitor.visit_newtype_struct(&mut *self)?;
        self.consume_if_unread(start)?;
        Ok(result)
    }
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control)?;
        let payload = match self.input.datatype().map_err(invalid)? {
            Type::String => false,
            Type::Map if self.input.map().map_err(invalid)? == Some(1) => true,
            _ => return Err(invalid("enum requires a string or single-entry object")),
        };
        let name = self.input.str().map_err(invalid)?;
        let start = self.input.position();
        let result = visitor.visit_enum(access::Variant {
            decoder: &mut *self,
            name,
            payload,
        })?;
        if payload {
            self.consume_if_unread(start)?;
        }
        Ok(result)
    }
    numeric!(
        deserialize_i8,
        deserialize_i16,
        deserialize_i32,
        deserialize_i64,
        deserialize_i128,
        deserialize_u8,
        deserialize_u16,
        deserialize_u32,
        deserialize_u64,
        deserialize_u128,
        deserialize_f32,
        deserialize_f64
    );
    restricted!(deserialize_bool, Type::Bool);
    restricted!(deserialize_char, Type::String);
    restricted!(deserialize_str, Type::String);
    restricted!(deserialize_string, Type::String);
    restricted!(deserialize_bytes, Type::String | Type::Array);
    restricted!(deserialize_byte_buf, Type::String | Type::Array);
    restricted!(deserialize_unit, Type::Null);
    restricted!(deserialize_seq, Type::Array);
    restricted!(deserialize_map, Type::Map);
    restricted!(deserialize_identifier, Type::String);
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_unit(visitor)
    }
    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }
    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        if !matches!(
            self.input.datatype().map_err(invalid)?,
            Type::Array | Type::Map
        ) {
            return Err(invalid("struct requires array or object"));
        }
        self.deserialize_any(visitor)
    }
    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_any(de::IgnoredAny)?;
        visitor.visit_unit()
    }
    fn is_human_readable(&self) -> bool {
        true
    }
}

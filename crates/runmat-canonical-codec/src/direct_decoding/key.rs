//! Preserve JSON's string-key projection, including its numeric grammar. Numeric
//! keys use the existing JSON parser; no second general value decoder is invented.
use super::{invalid, CanonicalCodecError};
use serde::de::{self, Visitor};
pub(super) struct Key<'de>(pub(super) &'de str);
macro_rules! numeric {
    ($($method:ident),*) => {$(
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            if !matches!(self.0.as_bytes().first(), Some(b'0'..=b'9' | b'-')) || self.0.bytes().any(|byte| byte.is_ascii_whitespace()) { return Err(invalid("numeric map key required")); }
            let mut decoder = serde_json::Deserializer::from_str(self.0);
            let value = de::Deserializer::$method(&mut decoder, visitor).map_err(invalid)?;
            decoder.end().map_err(invalid)?;
            Ok(value)
        }
    )*};
}
impl<'de> de::Deserializer<'de> for Key<'de> {
    type Error = CanonicalCodecError;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_string(self.0.to_owned())
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
    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            "true" => visitor.visit_bool(true),
            "false" => visitor.visit_bool(false),
            _ => Err(invalid("boolean map key required")),
        }
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_some(self)
    }
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let key = de::value::StringDeserializer::<Self::Error>::new(self.0.to_owned());
        de::Deserializer::deserialize_enum(key, name, variants, visitor)
    }
    serde::forward_to_deserialize_any! { char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier ignored_any }
}

//! Keep the original deserializer and visitors, probing recursive values and collection access.
use crate::CanonicalCodecControl;
use serde::de::*;
use serde::Deserializer;
use std::fmt;

pub(super) struct TrackedDeserializer<'a, D> {
    pub(super) inner: D,
    pub(super) control: &'a dyn CanonicalCodecControl,
}
struct TrackedVisitor<'a, V> {
    inner: V,
    control: &'a dyn CanonicalCodecControl,
}
struct TrackedAccess<'a, A> {
    inner: A,
    control: &'a dyn CanonicalCodecControl,
}
struct TrackedSeed<'a, S> {
    inner: S,
    control: &'a dyn CanonicalCodecControl,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for TrackedSeed<'_, S> {
    type Value = S::Value;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.deserialize(TrackedDeserializer {
            inner: deserializer,
            control: self.control,
        })
    }
}
macro_rules! deserialize_method {
    ($($method:ident($($arg:ident: $type:ty),*);)*) => {$(
        fn $method<V: Visitor<'de>>(self, $($arg: $type,)* visitor: V) -> Result<V::Value, Self::Error> {
            crate::checkpoint(self.control).map_err(Error::custom)?;
            self.inner.$method($($arg,)* TrackedVisitor { inner: visitor, control: self.control })
        }
    )*};
}
impl<'de, D: Deserializer<'de>> Deserializer<'de> for TrackedDeserializer<'_, D> {
    type Error = D::Error;
    deserialize_method! {
        deserialize_any(); deserialize_bool(); deserialize_i8(); deserialize_i16(); deserialize_i32();
        deserialize_i64(); deserialize_i128(); deserialize_u8(); deserialize_u16(); deserialize_u32();
        deserialize_u64(); deserialize_u128(); deserialize_f32(); deserialize_f64(); deserialize_char();
        deserialize_str(); deserialize_string(); deserialize_bytes(); deserialize_byte_buf();
        deserialize_option(); deserialize_unit(); deserialize_unit_struct(name: &'static str);
        deserialize_newtype_struct(name: &'static str); deserialize_seq(); deserialize_tuple(len: usize);
        deserialize_tuple_struct(name: &'static str, len: usize); deserialize_map();
        deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_identifier(); deserialize_ignored_any();
    }
    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}
macro_rules! visit_scalar {
    ($($method:ident($type:ty);)*) => {$(
        fn $method<E: Error>(self, value: $type) -> Result<Self::Value, E> {
            crate::checkpoint(self.control).map_err(Error::custom)?;
            self.inner.$method(value)
        }
    )*};
}
impl<'de, V: Visitor<'de>> Visitor<'de> for TrackedVisitor<'_, V> {
    type Value = V::Value;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.expecting(formatter)
    }
    visit_scalar! {
        visit_bool(bool); visit_i8(i8); visit_i16(i16); visit_i32(i32); visit_i64(i64); visit_i128(i128);
        visit_u8(u8); visit_u16(u16); visit_u32(u32); visit_u64(u64); visit_u128(u128);
        visit_f32(f32); visit_f64(f64); visit_char(char); visit_str(&str); visit_borrowed_str(&'de str);
        visit_string(String); visit_bytes(&[u8]); visit_borrowed_bytes(&'de [u8]); visit_byte_buf(Vec<u8>);
    }
    fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.visit_none()
    }
    fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.visit_unit()
    }
    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.visit_some(TrackedDeserializer {
            inner: deserializer,
            control: self.control,
        })
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        self.inner.visit_newtype_struct(TrackedDeserializer {
            inner: deserializer,
            control: self.control,
        })
    }
    fn visit_seq<A: SeqAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.visit_seq(TrackedAccess {
            inner: access,
            control: self.control,
        })
    }
    fn visit_map<A: MapAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.visit_map(TrackedAccess {
            inner: access,
            control: self.control,
        })
    }
    fn visit_enum<A: EnumAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.visit_enum(TrackedAccess {
            inner: access,
            control: self.control,
        })
    }
}
impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for TrackedAccess<'_, A> {
    type Error = A::Error;
    fn next_element_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.next_element_seed(TrackedSeed {
            inner: seed,
            control: self.control,
        })
    }
    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}
impl<'de, A: MapAccess<'de>> MapAccess<'de> for TrackedAccess<'_, A> {
    type Error = A::Error;
    fn next_key_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.next_key_seed(TrackedSeed {
            inner: seed,
            control: self.control,
        })
    }
    fn next_value_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<S::Value, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.next_value_seed(TrackedSeed {
            inner: seed,
            control: self.control,
        })
    }
    fn next_entry_seed<K: DeserializeSeed<'de>, V: DeserializeSeed<'de>>(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<(K::Value, V::Value)>, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.next_entry_seed(
            TrackedSeed {
                inner: key,
                control: self.control,
            },
            TrackedSeed {
                inner: value,
                control: self.control,
            },
        )
    }
    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}
impl<'a, 'de, A: EnumAccess<'de>> EnumAccess<'de> for TrackedAccess<'a, A> {
    type Error = A::Error;
    type Variant = TrackedAccess<'a, A::Variant>;
    fn variant_seed<S: DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<(S::Value, Self::Variant), Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        let (value, variant) = self.inner.variant_seed(TrackedSeed {
            inner: seed,
            control: self.control,
        })?;
        Ok((
            value,
            TrackedAccess {
                inner: variant,
                control: self.control,
            },
        ))
    }
}
impl<'de, A: VariantAccess<'de>> VariantAccess<'de> for TrackedAccess<'_, A> {
    type Error = A::Error;
    fn unit_variant(self) -> Result<(), Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.unit_variant()
    }
    fn newtype_variant_seed<S: DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<S::Value, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.newtype_variant_seed(TrackedSeed {
            inner: seed,
            control: self.control,
        })
    }
    fn tuple_variant<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.tuple_variant(
            len,
            TrackedVisitor {
                inner: visitor,
                control: self.control,
            },
        )
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.struct_variant(
            fields,
            TrackedVisitor {
                inner: visitor,
                control: self.control,
            },
        )
    }
}

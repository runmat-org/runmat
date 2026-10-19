//! Delegate every value/compound to the original serializer, adding probes at recursive boundaries.
use super::TrackedValue;
use crate::CanonicalCodecControl;
use serde::{ser::*, Serialize, Serializer};

pub(super) struct TrackedSerializer<'a, S> {
    pub(super) inner: S,
    pub(super) control: &'a dyn CanonicalCodecControl,
}
pub(super) struct TrackedCompound<'a, S> {
    inner: S,
    control: &'a dyn CanonicalCodecControl,
}

macro_rules! scalar {
    ($($method:ident($type:ty);)*) => {$(
        fn $method(self, value: $type) -> Result<Self::Ok, Self::Error> {
            self.inner.$method(value)
        }
    )*};
}
macro_rules! compound {
    ($method:ident($($arg:ident: $type:ty),*) -> $associated:ident) => {
        fn $method(self, $($arg: $type),*) -> Result<Self::$associated, Self::Error> {
            Ok(TrackedCompound { inner: self.inner.$method($($arg),*)?, control: self.control })
        }
    };
}
impl<'a, S: Serializer> Serializer for TrackedSerializer<'a, S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = TrackedCompound<'a, S::SerializeSeq>;
    type SerializeTuple = TrackedCompound<'a, S::SerializeTuple>;
    type SerializeTupleStruct = TrackedCompound<'a, S::SerializeTupleStruct>;
    type SerializeTupleVariant = TrackedCompound<'a, S::SerializeTupleVariant>;
    type SerializeMap = TrackedCompound<'a, S::SerializeMap>;
    type SerializeStruct = TrackedCompound<'a, S::SerializeStruct>;
    type SerializeStructVariant = TrackedCompound<'a, S::SerializeStructVariant>;
    scalar! {
        serialize_bool(bool); serialize_i8(i8); serialize_i16(i16); serialize_i32(i32);
        serialize_i64(i64); serialize_i128(i128); serialize_u8(u8); serialize_u16(u16);
        serialize_u32(u32); serialize_u64(u64); serialize_u128(u128);
        serialize_f32(f32); serialize_f64(f64); serialize_char(char); serialize_str(&str);
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Self::Error> {
        // serde_json::Value represents bytes as an ordinary integer array. Probe each byte
        // rather than forwarding a potentially large uninterruptible slice conversion.
        let control = self.control;
        let mut sequence = self.inner.serialize_seq(Some(value.len()))?;
        for byte in value {
            sequence.serialize_element(&TrackedValue {
                value: byte,
                control,
            })?;
        }
        sequence.end()
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_none()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_some(&TrackedValue {
            value,
            control: self.control,
        })
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_unit()
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_unit_struct(name)
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_unit_variant(name, index, variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_newtype_struct(
            name,
            &TrackedValue {
                value,
                control: self.control,
            },
        )
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.inner.serialize_newtype_variant(
            name,
            index,
            variant,
            &TrackedValue {
                value,
                control: self.control,
            },
        )
    }
    compound!(serialize_seq(len: Option<usize>) -> SerializeSeq);
    compound!(serialize_tuple(len: usize) -> SerializeTuple);
    compound!(serialize_tuple_struct(name: &'static str, len: usize) -> SerializeTupleStruct);
    compound!(serialize_tuple_variant(name: &'static str, index: u32, variant: &'static str, len: usize) -> SerializeTupleVariant);
    compound!(serialize_map(len: Option<usize>) -> SerializeMap);
    compound!(serialize_struct(name: &'static str, len: usize) -> SerializeStruct);
    compound!(serialize_struct_variant(name: &'static str, index: u32, variant: &'static str, len: usize) -> SerializeStructVariant);
    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(
        self,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.inner.collect_str(value)
    }
}

macro_rules! elements {
    ($trait:ident, $method:ident) => {
        impl<S: $trait> $trait for TrackedCompound<'_, S> {
            type Ok = S::Ok;
            type Error = S::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
                self.inner.$method(&TrackedValue {
                    value,
                    control: self.control,
                })
            }
            fn end(self) -> Result<Self::Ok, Self::Error> {
                crate::checkpoint(self.control).map_err(Error::custom)?;
                self.inner.end()
            }
        }
    };
}
elements!(SerializeSeq, serialize_element);
elements!(SerializeTuple, serialize_element);
elements!(SerializeTupleStruct, serialize_field);
elements!(SerializeTupleVariant, serialize_field);
impl<S: SerializeMap> SerializeMap for TrackedCompound<'_, S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Self::Error> {
        self.inner.serialize_key(&TrackedValue {
            value: key,
            control: self.control,
        })
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.inner.serialize_value(&TrackedValue {
            value,
            control: self.control,
        })
    }
    fn serialize_entry<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        key: &K,
        value: &V,
    ) -> Result<(), Self::Error> {
        self.inner.serialize_entry(
            &TrackedValue {
                value: key,
                control: self.control,
            },
            &TrackedValue {
                value,
                control: self.control,
            },
        )
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        crate::checkpoint(self.control).map_err(Error::custom)?;
        self.inner.end()
    }
}
macro_rules! fields {
    ($trait:ident) => {
        impl<S: $trait> $trait for TrackedCompound<'_, S> {
            type Ok = S::Ok;
            type Error = S::Error;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<(), Self::Error> {
                self.inner.serialize_field(
                    key,
                    &TrackedValue {
                        value,
                        control: self.control,
                    },
                )
            }
            fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
                self.inner.skip_field(key)
            }
            fn end(self) -> Result<Self::Ok, Self::Error> {
                crate::checkpoint(self.control).map_err(Error::custom)?;
                self.inner.end()
            }
        }
    };
}
fields!(SerializeStruct);
fields!(SerializeStructVariant);

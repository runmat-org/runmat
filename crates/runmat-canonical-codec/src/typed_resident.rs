//! Typed resident-storage estimate from one borrowed Serde traversal.
//!
//! Callers that hold or clone typed Rust values charge this, not the JSON
//! value-tree projection, which models a tree that typed code never builds.
//! Every scalar is charged one machine word (covering field padding), every
//! heap collection or string a vector header plus contents, every enum or
//! option one tag word, and every map entry an ordered-map node allowance.
//! Allocator slack and spare collection capacity are not modeled; this is a
//! requested estimate, not measured heap.

use serde::{ser, Serialize};

use crate::CanonicalCodecError;

const WORD: u64 = 8;
const VECTOR_HEADER: u64 = 24;
const MAP_ENTRY_OVERHEAD: u64 = 32;

pub fn typed_resident_bytes_estimate<T: Serialize + ?Sized>(
    value: &T,
) -> Result<u64, CanonicalCodecError> {
    let mut total = 0;
    value.serialize(Estimator { total: &mut total })?;
    Ok(total)
}

struct Estimator<'a> {
    total: &'a mut u64,
}

fn overflow() -> CanonicalCodecError {
    CanonicalCodecError::resource_limit("typed resident estimate", "byte count overflow")
}

impl Estimator<'_> {
    fn add(&mut self, bytes: u64) -> Result<(), CanonicalCodecError> {
        *self.total = self.total.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
    fn reborrow(&mut self) -> Estimator<'_> {
        Estimator { total: self.total }
    }
}

pub struct Compound<'a> {
    total: &'a mut u64,
    map_entries: bool,
}

impl Compound<'_> {
    fn item<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        value.serialize(Estimator { total: self.total })
    }
    fn entry(&mut self) -> Result<(), CanonicalCodecError> {
        if self.map_entries {
            *self.total = self
                .total
                .checked_add(MAP_ENTRY_OVERHEAD)
                .ok_or_else(overflow)?;
        }
        Ok(())
    }
}

macro_rules! scalar {
    ($($method:ident: $ty:ty),*) => {
        $(fn $method(mut self, _: $ty) -> Result<(), CanonicalCodecError> {
            self.add(WORD)
        })*
    };
}

impl<'a> ser::Serializer for Estimator<'a> {
    type Ok = ();
    type Error = CanonicalCodecError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = Compound<'a>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    scalar!(
        serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
        serialize_i64: i64, serialize_i128: i128, serialize_u8: u8, serialize_u16: u16,
        serialize_u32: u32, serialize_u64: u64, serialize_u128: u128, serialize_f32: f32,
        serialize_f64: f64, serialize_char: char
    );

    fn serialize_str(mut self, value: &str) -> Result<(), Self::Error> {
        self.add(VECTOR_HEADER + value.len() as u64)
    }
    fn serialize_bytes(mut self, value: &[u8]) -> Result<(), Self::Error> {
        self.add(VECTOR_HEADER + value.len() as u64)
    }
    fn serialize_none(mut self) -> Result<(), Self::Error> {
        self.add(WORD)
    }
    fn serialize_some<T: Serialize + ?Sized>(mut self, value: &T) -> Result<(), Self::Error> {
        self.add(WORD)?;
        value.serialize(self.reborrow())
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        mut self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), Self::Error> {
        self.add(WORD)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        mut self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.add(WORD)?;
        value.serialize(self.reborrow())
    }
    fn serialize_seq(mut self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.add(VECTOR_HEADER)?;
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
    fn serialize_tuple_variant(
        mut self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.add(WORD)?;
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
    fn serialize_map(mut self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.add(VECTOR_HEADER)?;
        Ok(Compound {
            total: self.total,
            map_entries: true,
        })
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
    fn serialize_struct_variant(
        mut self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.add(WORD)?;
        Ok(Compound {
            total: self.total,
            map_entries: false,
        })
    }
}

macro_rules! sequence {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Compound<'_> {
            type Ok = ();
            type Error = CanonicalCodecError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
                self.item(value)
            }
            fn end(self) -> Result<(), Self::Error> {
                Ok(())
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
        self.entry()?;
        self.item(key)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.item(value)
    }
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

macro_rules! fields {
    ($trait:ident) => {
        impl ser::$trait for Compound<'_> {
            type Ok = ();
            type Error = CanonicalCodecError;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                _: &'static str,
                value: &T,
            ) -> Result<(), Self::Error> {
                self.item(value)
            }
            fn end(self) -> Result<(), Self::Error> {
                Ok(())
            }
        }
    };
}
fields!(SerializeStruct);
fields!(SerializeStructVariant);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Record {
        id: u64,
        name: String,
        values: Vec<f64>,
        tag: Option<u32>,
    }

    #[test]
    fn typed_estimate_covers_owned_heap_and_inline_words() {
        let record = Record {
            id: 7,
            name: "abc".into(),
            values: vec![1.0; 1000],
            tag: Some(3),
        };
        let estimate = typed_resident_bytes_estimate(&record).unwrap();
        let heap = record.values.capacity() * 8 + record.name.capacity();
        let inline = std::mem::size_of::<Record>();
        assert!(estimate >= (heap + inline) as u64);
        // A typed estimate stays within a small constant of the owned bytes,
        // unlike the JSON value-tree projection (tens of bytes per scalar).
        assert!(estimate <= (heap + inline) as u64 + 128);
    }

    #[test]
    fn maps_charge_an_entry_allowance() {
        let map = (0..10u64).map(|key| (key, key)).collect::<BTreeMap<_, _>>();
        assert_eq!(
            typed_resident_bytes_estimate(&map).unwrap(),
            VECTOR_HEADER + 10 * (MAP_ENTRY_OVERHEAD + 2 * WORD)
        );
    }
}

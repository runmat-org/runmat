use serde::{ser, Serialize};

use super::{invalid, CanonicalCodecError, Counter, Profiler};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    Sequence,
    Map,
}

pub(super) struct Compound<'a> {
    counter: &'a mut Counter,
    depth: usize,
    expected: Option<usize>,
    count: usize,
    kind: Kind,
    pending_value: bool,
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
        Ok(Self {
            counter,
            depth,
            expected,
            count: 0,
            kind,
            pending_value: false,
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

    fn child<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        let depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| invalid("depth overflow"))?;
        self.counter.depth(depth)?;
        value.serialize(Profiler::value(self.counter, depth))
    }

    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        if self.kind != Kind::Sequence {
            return Err(invalid("sequence element in map"));
        }
        self.item()?;
        self.child(value)
    }

    fn key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        if self.kind != Kind::Map || self.pending_value {
            return Err(invalid("invalid map key order"));
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

    fn value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalCodecError> {
        if self.kind != Kind::Map || !self.pending_value {
            return Err(invalid("map value has no key"));
        }
        self.child(value)?;
        self.pending_value = false;
        Ok(())
    }

    fn field<T: Serialize + ?Sized>(
        &mut self,
        key: &str,
        value: &T,
    ) -> Result<(), CanonicalCodecError> {
        self.key(key)?;
        self.value(value)
    }

    fn finish(self) -> Result<(), CanonicalCodecError> {
        if self.pending_value || self.expected.is_some_and(|len| len != self.count) {
            return Err(invalid(
                "incomplete collection or mismatched declared length",
            ));
        }
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
        self.value(value)
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

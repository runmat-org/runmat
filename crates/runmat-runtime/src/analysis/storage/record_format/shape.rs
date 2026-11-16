//! Records the serialized shape of persisted records: every struct and struct variant reached,
//! with all of its field names, including fields `skip_serializing_if` leaves out of the bytes.
//! Enum variants are recorded as they are reached. The fixture runs live in
//! `analysis::tests::run_record_shape`, which pins the digest of this shape against the record
//! format versions.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::ser::{self, Serialize};

use crate::analysis::contracts::AnalysisRunResult;

/// The record format versions a shape digest is pinned against.
pub(in crate::analysis) const RUN_RECORD_VERSION: &str = super::super::ARTIFACT_SCHEMA_VERSION;
pub(in crate::analysis) const FREQUENCY_RECORD_VERSIONS: [&str; 2] = [
    super::super::frequency::FREQUENCY_ARTIFACT_SCHEMA_VERSION,
    super::super::frequency_sample::ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION,
];

/// Struct name (or `Enum::Variant`) to the field names it serializes, and `enum Name` to the
/// variants reached.
#[derive(Debug, Default)]
pub(in crate::analysis) struct RecordShape(BTreeMap<String, BTreeSet<String>>);

impl RecordShape {
    pub(in crate::analysis) fn record<T: Serialize + ?Sized>(&mut self, value: &T) {
        value
            .serialize(Recorder {
                shape: self,
                owner: None,
            })
            .expect("recording a serialized shape cannot fail");
    }

    /// Records the stored run record this runtime writes for `run`, as it is written.
    pub(in crate::analysis) fn record_stored_run(&mut self, run: &AnalysisRunResult) {
        let prepared = super::super::prepare_run_artifact(run, "shape-fixture".into())
            .expect("prepare the stored run record");
        self.record(&prepared.persisted);
    }

    pub(in crate::analysis) fn contains(&self, owner: &str, field: &str) -> bool {
        self.0
            .get(owner)
            .is_some_and(|fields| fields.contains(field))
    }

    pub(in crate::analysis) fn text(&self) -> String {
        self.0
            .iter()
            .map(|(owner, fields)| {
                let fields = fields.iter().cloned().collect::<Vec<_>>().join(", ");
                format!("{owner} {{ {fields} }}\n")
            })
            .collect()
    }

    pub(in crate::analysis) fn digest(&self) -> String {
        runmat_execution::Digest::sha256(self.text().as_bytes()).to_string()
    }

    fn note(&mut self, owner: &str, field: &str) {
        self.0
            .entry(owner.to_string())
            .or_default()
            .insert(field.to_string());
    }
}

#[derive(Debug)]
struct ShapeError(String);

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

impl ser::Error for ShapeError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

struct Recorder<'a> {
    shape: &'a mut RecordShape,
    /// The struct whose fields are being serialized.
    owner: Option<String>,
}

impl Recorder<'_> {
    fn child(&mut self) -> Recorder<'_> {
        Recorder {
            shape: &mut *self.shape,
            owner: None,
        }
    }

    fn open(self, owner: String) -> Self {
        self.shape.0.entry(owner.clone()).or_default();
        Recorder {
            shape: self.shape,
            owner: Some(owner),
        }
    }

    fn variant(&mut self, name: &str, variant: &str) {
        self.shape.note(&format!("enum {name}"), variant);
    }

    fn field<T: Serialize + ?Sized>(&mut self, key: &str, value: &T) -> Result<(), ShapeError> {
        if let Some(owner) = self.owner.clone() {
            self.shape.note(&owner, key);
        }
        value.serialize(self.child())
    }

    fn skipped(&mut self, key: &str) {
        if let Some(owner) = self.owner.clone() {
            self.shape.note(&owner, key);
        }
    }
}

macro_rules! leaf {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method(self, _: $ty) -> Result<(), ShapeError> { Ok(()) })*
    };
}

impl<'a> ser::Serializer for Recorder<'a> {
    type Ok = ();
    type Error = ShapeError;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    leaf!(
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
        serialize_unit_struct(&'static str),
    );

    fn serialize_none(self) -> Result<(), ShapeError> {
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), ShapeError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), ShapeError> {
        Ok(())
    }

    fn serialize_unit_variant(
        mut self,
        name: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), ShapeError> {
        self.variant(name, variant);
        Ok(())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), ShapeError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        mut self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), ShapeError> {
        self.variant(name, variant);
        value.serialize(self.child())
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self, ShapeError> {
        Ok(self)
    }

    fn serialize_tuple(self, _: usize) -> Result<Self, ShapeError> {
        Ok(self)
    }

    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, ShapeError> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        mut self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, ShapeError> {
        self.variant(name, variant);
        Ok(self)
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self, ShapeError> {
        Ok(self)
    }

    fn serialize_struct(self, name: &'static str, _: usize) -> Result<Self, ShapeError> {
        Ok(self.open(name.to_string()))
    }

    fn serialize_struct_variant(
        mut self,
        name: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, ShapeError> {
        self.variant(name, variant);
        Ok(self.open(format!("{name}::{variant}")))
    }
}

macro_rules! elements {
    ($($trait:ident :: $method:ident),* $(,)?) => {
        $(impl ser::$trait for Recorder<'_> {
            type Ok = ();
            type Error = ShapeError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), ShapeError> {
                value.serialize(self.child())
            }
            fn end(self) -> Result<(), ShapeError> {
                Ok(())
            }
        })*
    };
}

elements!(
    SerializeSeq::serialize_element,
    SerializeTuple::serialize_element,
    SerializeTupleStruct::serialize_field,
    SerializeTupleVariant::serialize_field,
);

impl ser::SerializeMap for Recorder<'_> {
    type Ok = ();
    type Error = ShapeError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), ShapeError> {
        key.serialize(self.child())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), ShapeError> {
        value.serialize(self.child())
    }
    fn end(self) -> Result<(), ShapeError> {
        Ok(())
    }
}

impl ser::SerializeStruct for Recorder<'_> {
    type Ok = ();
    type Error = ShapeError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ShapeError> {
        self.field(key, value)
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), ShapeError> {
        self.skipped(key);
        Ok(())
    }
    fn end(self) -> Result<(), ShapeError> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Recorder<'_> {
    type Ok = ();
    type Error = ShapeError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ShapeError> {
        self.field(key, value)
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), ShapeError> {
        self.skipped(key);
        Ok(())
    }
    fn end(self) -> Result<(), ShapeError> {
        Ok(())
    }
}

use super::StructValue;
use crate::{ExactOrderedMap, OutputListStorage, SharedShape, Value};
use std::{fmt, ops::Index};

mod admission;
mod storage;
mod builder;
pub use builder::{
    StructArrayBuilder, StructArrayConstructionError, StructArrayFieldError, StructArrayQuote,
    UnboundedStructArrayBuilder,
};
mod concatenate;
mod construction;
mod fields;
mod indexing;
mod mapping;
mod mutation;
mod shape;
mod transform;

pub use admission::{StructArrayAdmission, ValidatedStructArrayParts};

#[cfg(test)]
mod mutation_tests;
#[cfg(test)]
mod tests;

pub use concatenate::StructArrayOperand;
pub(crate) use shape::{column_major_strides, total_len};

/// A homogeneous MATLAB structure array in visible column-major order.
///
/// Scalar structures use [`Value::Struct`]. Array field names are stored once;
/// each field owns one column of values in linear array order.
#[derive(Debug, Clone, PartialEq)]
pub struct StructArray {
    fields: ExactOrderedMap<OutputListStorage>,
    shape: SharedShape,
}

#[derive(Clone, Copy)]
pub struct StructElementRef<'a> {
    pub fields: StructFieldsRef<'a>,
}

impl StructElementRef<'_> {
    pub fn to_owned(self) -> StructValue {
        StructValue::from_fields_unbounded(
            self.fields
                .iter()
                .map(|(name, value)| (name.to_owned(), value.clone())),
        )
    }
}

#[derive(Clone, Copy)]
pub struct StructFieldsRef<'a> {
    fields: &'a ExactOrderedMap<OutputListStorage>,
    index: usize,
}

impl<'a> StructFieldsRef<'a> {
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    pub fn keys(self) -> impl ExactSizeIterator<Item = &'a str> + Clone {
        self.fields.keys()
    }

    pub fn get(&self, name: &str) -> Option<&'a Value> {
        self.fields
            .get(name)
            .and_then(|values| values.get(self.index))
    }

    pub fn values(self) -> impl Iterator<Item = &'a Value> + Clone {
        self.fields.values().map(move |values| &values[self.index])
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = (&'a str, &'a Value)> + Clone {
        self.fields
            .iter()
            .map(move |(name, values)| (name, &values[self.index]))
    }
}

impl Index<&str> for StructFieldsRef<'_> {
    type Output = Value;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("no field named '{name}'"))
    }
}

pub struct StructElements<'a> {
    array: &'a StructArray,
    next: usize,
}

impl<'a> Iterator for StructElements<'a> {
    type Item = StructElementRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let element = self.array.get_linear(self.next)?;
        self.next += 1;
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for StructElements<'_> {}

impl StructArray {
    pub fn column_owners(
        &self,
    ) -> impl ExactSizeIterator<Item = (&str, &OutputListStorage)> + Clone {
        self.fields
            .entries()
            .iter()
            .map(|entry| (entry.key().as_str(), entry.value()))
    }
    pub fn shape_owner(&self) -> &SharedShape {
        &self.shape
    }
    pub fn column_owner(&self, name: &str) -> Option<&OutputListStorage> {
        self.fields.get(name)
    }
    pub fn field_names(&self) -> impl ExactSizeIterator<Item = &str> + Clone {
        self.fields.keys()
    }

    pub fn field_values(&self, name: &str) -> Option<&[Value]> {
        self.fields.get(name).map(OutputListStorage::as_slice)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.fields.get_index(0).map_or_else(
            || total_len(&self.shape).expect("validated structure-array shape"),
            |(_, values)| values.len(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_linear(&self, index: usize) -> Option<StructElementRef<'_>> {
        (index < self.len()).then_some(StructElementRef {
            fields: StructFieldsRef {
                fields: &self.fields,
                index,
            },
        })
    }

    pub fn elements(&self) -> StructElements<'_> {
        StructElements {
            array: self,
            next: 0,
        }
    }
}

impl fmt::Display for StructArray {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimensions = self
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x");
        write!(formatter, "{dimensions} struct array with fields:")?;
        for field in self.fields.keys() {
            write!(formatter, "\n    {field}")?;
        }
        Ok(())
    }
}

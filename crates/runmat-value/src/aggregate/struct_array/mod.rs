use super::StructValue;
use crate::Value;
use indexmap::IndexMap;
use std::{fmt, ops::Index};

mod concatenate;
mod construction;
mod fields;
mod indexing;
mod mapping;
mod mutation;
mod shape;
mod transform;

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
    fields: IndexMap<String, Vec<Value>>,
    shape: Vec<usize>,
}

#[derive(Clone, Copy)]
pub struct StructElementRef<'a> {
    pub fields: StructFieldsRef<'a>,
}

impl StructElementRef<'_> {
    pub fn to_owned(self) -> StructValue {
        StructValue {
            fields: self
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct StructFieldsRef<'a> {
    fields: &'a IndexMap<String, Vec<Value>>,
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

    pub fn keys(self) -> impl Iterator<Item = &'a String> {
        self.fields.keys()
    }

    pub fn get(&self, name: &str) -> Option<&'a Value> {
        self.fields
            .get(name)
            .and_then(|values| values.get(self.index))
    }

    pub fn values(self) -> impl Iterator<Item = &'a Value> {
        self.fields.values().map(move |values| &values[self.index])
    }

    pub fn iter(self) -> impl Iterator<Item = (&'a String, &'a Value)> {
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
    pub fn field_names(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }

    pub fn field_values(&self, name: &str) -> Option<&[Value]> {
        self.fields.get(name).map(Vec::as_slice)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.fields.first().map_or_else(
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

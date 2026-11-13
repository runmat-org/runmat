//! Lossless constant columns remove repeated quality values without approximating them.
use super::*;
use serde::{Deserialize, Serialize};

pub(super) trait Value: Copy + Serialize {
    fn same_bits(self, other: Self) -> bool;
}
impl Value for f64 {
    fn same_bits(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}
impl Value for u64 {
    fn same_bits(self, other: Self) -> bool {
        self == other
    }
}
impl Value for [f64; 3] {
    fn same_bits(self, other: Self) -> bool {
        self.map(f64::to_bits) == other.map(f64::to_bits)
    }
}

pub(super) struct Column<'a, T> {
    rows: &'a [crate::ExactFaceMeshTriangle],
    read: fn(&crate::ExactFaceMeshTriangle) -> T,
    constant: Option<T>,
}
impl<'a, T: Value> Column<'a, T> {
    pub fn new(
        rows: &'a [crate::ExactFaceMeshTriangle],
        read: fn(&crate::ExactFaceMeshTriangle) -> T,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        let mut constant = rows.first().map(read);
        for (index, row) in rows.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            if constant.is_some_and(|value| !value.same_bits(read(row))) {
                constant = None;
                break;
            }
        }
        Ok(Self {
            rows,
            read,
            constant,
        })
    }
}
impl<T: Value> Serialize for Column<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        enum Projection<'a, T> {
            Constant(T),
            Dense(Rows<'a, T>),
        }
        struct Rows<'a, T> {
            rows: &'a [crate::ExactFaceMeshTriangle],
            read: fn(&crate::ExactFaceMeshTriangle) -> T,
        }
        impl<T: Serialize> Serialize for Rows<'_, T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeSeq;
                let mut sequence = serializer.serialize_seq(Some(self.rows.len()))?;
                for row in self.rows {
                    sequence.serialize_element(&(self.read)(row))?;
                }
                sequence.end()
            }
        }
        match self.constant {
            Some(value) => Projection::Constant(value),
            None => Projection::Dense(Rows {
                rows: self.rows,
                read: self.read,
            }),
        }
        .serialize(serializer)
    }
}
#[derive(Deserialize)]
pub(super) enum DecodedColumn<T> {
    Constant(T),
    Dense(runmat_canonical_codec::ExactVec<T>),
}
impl<T: Copy> DecodedColumn<T> {
    pub fn validate_length(&self, count: usize) -> Result<(), ExactSurfaceMeshError> {
        if matches!(self,Self::Dense(values) if values.len()!=count) {
            return Err(invalid(
                "quality column length differs from triangle inventory",
            ));
        }
        Ok(())
    }
    pub fn at(&self, index: usize) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::Dense(values) => values[index],
        }
    }
}

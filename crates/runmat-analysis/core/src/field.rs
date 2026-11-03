use serde::{Deserialize, Serialize, Serializer};

use crate::study::output::{
    ComplexRepresentation, FieldLocation, ObservationQuantity, ObservationValueShape,
};

fn is_real(representation: &ComplexRepresentation) -> bool {
    *representation == ComplexRepresentation::Real
}

fn real() -> ComplexRepresentation {
    ComplexRepresentation::Real
}

/// What the published state index of a field means.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "index")]
pub enum FieldOrdinal {
    /// The only published state of this field.
    #[default]
    Single,
    /// The n-th published state of a stepped, incremented or time-sampled solve.
    Step(#[serde(with = "runmat_types::wire_integer")] usize),
    /// The mode with this 0-based mode index (field ids may number modes from 1).
    Mode(#[serde(with = "runmat_types::wire_integer")] usize),
}

/// Typed metadata a publisher declares with every field: the physical quantity, where its
/// values live, the per-entity component layout, how a complex quantity is represented
/// (`Real` for real-valued quantities; `Complex` stores interleaved (re, im) pairs in a
/// trailing extent of 2) and the state ordinal.
/// Storage shape is `[entity_count]` for scalar fields and `[entity_count, components]` (or
/// the value shape's extents) for component fields; integration-point fields have one entity
/// per quadrature point. Interleaved complex storage adds a trailing extent of 2.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisFieldLayout {
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    #[serde(default = "real", skip_serializing_if = "is_real")]
    pub representation: ComplexRepresentation,
    #[serde(default)]
    pub ordinal: FieldOrdinal,
}

impl AnalysisFieldLayout {
    pub fn new(
        quantity: ObservationQuantity,
        location: FieldLocation,
        value_shape: ObservationValueShape,
    ) -> Self {
        Self {
            quantity,
            location,
            value_shape,
            representation: ComplexRepresentation::Real,
            ordinal: FieldOrdinal::Single,
        }
    }

    /// The quantity's canonical component layout at `location`.
    pub fn canonical(quantity: ObservationQuantity, location: FieldLocation) -> Self {
        Self::new(quantity, location, quantity.canonical_value_shape())
    }

    pub fn scalar(quantity: ObservationQuantity, location: FieldLocation) -> Self {
        Self::new(quantity, location, ObservationValueShape::Scalar)
    }

    pub fn vector3(quantity: ObservationQuantity, location: FieldLocation) -> Self {
        Self::new(quantity, location, ObservationValueShape::Vector3)
    }

    pub fn symmetric_tensor3(quantity: ObservationQuantity, location: FieldLocation) -> Self {
        Self::new(quantity, location, ObservationValueShape::SymmetricTensor3)
    }

    /// `count` components per entity with no further structure (e.g. 2D vectors).
    pub fn components(
        quantity: ObservationQuantity,
        location: FieldLocation,
        count: usize,
    ) -> Self {
        let value_shape = match count {
            1 => ObservationValueShape::Scalar,
            _ => ObservationValueShape::Tensor {
                extents: vec![count],
            },
        };
        Self::new(quantity, location, value_shape)
    }

    pub fn step(mut self, index: usize) -> Self {
        self.ordinal = FieldOrdinal::Step(index);
        self
    }

    pub fn mode(mut self, mode_number: usize) -> Self {
        self.ordinal = FieldOrdinal::Mode(mode_number);
        self
    }

    pub fn with_ordinal(mut self, ordinal: FieldOrdinal) -> Self {
        self.ordinal = ordinal;
        self
    }

    pub fn representation(mut self, representation: ComplexRepresentation) -> Self {
        self.representation = representation;
        self
    }

    /// Values stored per entity of `location` for this shape: the product of the trailing
    /// storage extents. `None` when the storage shape does not carry this layout.
    pub fn component_count(&self) -> Option<usize> {
        self.value_shape.component_count()
    }

    /// Stored f64 values per located value: 2 for interleaved complex storage, else 1.
    pub fn storage_width(&self) -> usize {
        match self.representation {
            ComplexRepresentation::Complex => 2,
            _ => 1,
        }
    }

    /// Number of located entities for a storage `shape`, checked against the declared layout.
    pub fn entity_count(&self, shape: &[usize]) -> Result<usize, String> {
        let components = self
            .component_count()
            .ok_or_else(|| "field layout has an invalid value shape".to_string())?
            * self.storage_width();
        if matches!(self.location, FieldLocation::Global | FieldLocation::Mode) {
            let total: usize = shape.iter().product();
            return if total % components == 0 {
                Ok(total / components)
            } else {
                Err(format!(
                    "{total} values do not divide into {components}-component entries"
                ))
            };
        }
        let Some((&entities, trailing)) = shape.split_first() else {
            return Err("located field has an empty storage shape".to_string());
        };
        let per_entity: usize = trailing.iter().product();
        if per_entity != components {
            return Err(format!(
                "storage shape {shape:?} does not hold {components} value(s) per {:?}",
                self.location
            ));
        }
        Ok(entities)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFieldRef {
    pub backend: String,
    pub token: String,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldValues {
    HostF64(Vec<f64>),
    DeviceRef(DeviceFieldRef),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AnalysisField {
    pub field_id: String,
    pub layout: AnalysisFieldLayout,
    pub shape: Vec<usize>,
    pub values: AnalysisFieldValues,
}

// One wire authority for owned fields and borrowed/hypothetical host sequences.
#[derive(Serialize)]
#[serde(rename = "AnalysisFieldValues", rename_all = "snake_case")]
enum BorrowedValues<'a, T: Serialize + ?Sized> {
    HostF64(&'a T),
    DeviceRef(&'a DeviceFieldRef),
}

#[derive(Serialize)]
#[serde(rename = "AnalysisField")]
struct BorrowedField<'a, T: Serialize + ?Sized> {
    field_id: &'a str,
    layout: &'a AnalysisFieldLayout,
    shape: &'a [usize],
    values: BorrowedValues<'a, T>,
}

impl Serialize for AnalysisFieldValues {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::HostF64(values) => {
                BorrowedValues::HostF64(values.as_slice()).serialize(serializer)
            }
            Self::DeviceRef(reference) => {
                BorrowedValues::<[f64]>::DeviceRef(reference).serialize(serializer)
            }
        }
    }
}

impl Serialize for AnalysisField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let values = match &self.values {
            AnalysisFieldValues::HostF64(values) => BorrowedValues::HostF64(values.as_slice()),
            AnalysisFieldValues::DeviceRef(reference) => BorrowedValues::DeviceRef(reference),
        };
        BorrowedField {
            field_id: &self.field_id,
            layout: &self.layout,
            shape: &self.shape,
            values,
        }
        .serialize(serializer)
    }
}

impl AnalysisField {
    /// Borrow the canonical host-field wire schema with caller-supplied serializable values.
    /// `values` must serialize as an f64 sequence. This is a wire/profile view, not validated
    /// numerical storage; it neither checks shape agreement nor materializes numeric values.
    pub fn host_f64_serialization<'a, T: Serialize + ?Sized>(
        field_id: &'a str,
        layout: &'a AnalysisFieldLayout,
        shape: &'a [usize],
        values: &'a T,
    ) -> impl Serialize + 'a {
        BorrowedField {
            field_id,
            layout,
            shape,
            values: BorrowedValues::HostF64(values),
        }
    }
    pub fn host_f64(
        field_id: impl Into<String>,
        layout: AnalysisFieldLayout,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            field_id: field_id.into(),
            layout,
            shape,
            values: AnalysisFieldValues::HostF64(values),
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check that the storage shape carries the declared layout.
    pub fn validate_layout(&self) -> Result<(), String> {
        self.layout
            .entity_count(&self.shape)
            .map(|_| ())
            .map_err(|error| format!("field '{}': {error}", self.field_id))
    }

    pub fn is_empty(&self) -> bool {
        self.element_count() == 0
    }

    pub fn as_host_f64(&self) -> Option<&[f64]> {
        match &self.values {
            AnalysisFieldValues::HostF64(values) => Some(values.as_slice()),
            AnalysisFieldValues::DeviceRef(_) => None,
        }
    }
}

#[cfg(test)]
mod tests;

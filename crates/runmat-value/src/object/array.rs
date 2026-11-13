use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use runmat_gc_api::{Trace, Tracer};
use runmat_types::ClassIdentity;

use crate::{
    ExactArrayStorage, HandleRef, ObjectInstance, OutputListStorage, OwnedExactArrayValues,
    SelectedObjectClassFacts, SharedClassIdentity, SharedShape, Value,
};

#[path = "array/admission.rs"]
mod admission;
pub use admission::{ObjectArrayAdmission, ValidatedObjectArrayParts};
#[path = "array/builder.rs"]
mod builder;
pub use builder::{
    ObjectArrayBuildError, ObjectArrayBuilder, ObjectArrayElementError, ObjectArrayQuote,
};

/// A homogeneous MATLAB object array.
///
/// Storage is column-major and the shape is always explicit. Elements may be
/// value objects or handle objects, but every element must have the same
/// concrete class. Scalar objects continue to use `Value::Object` and
/// `Value::HandleObject`; this type represents zero or multiple elements
/// without falling back to a cell array.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectArray {
    class_name: SharedClassIdentity,
    data: OutputListStorage,
    shape: SharedShape,
    // Empty-only exact products captured before slicing/deletion discards all
    // elements. Nonempty elements already own their own selected facts, so an
    // array-level copy would be redundant. Distinct old/new products of one
    // ClassIdentity remain distinct; no current-name lookup is licensed.
    selected_provenance: ExactArrayStorage<Rc<SelectedObjectClassFacts>>,
    unselected_provenance: bool,
}

impl ObjectArray {
    /// Explicit ordinary publication retaining existing immutable storage.
    /// This validates the same class/kind invariant as the owned admission
    /// machine; it never turns an unpublished owned Vec into a hidden clone.
    pub fn from_shared_storage_unbounded(
        class_name: impl Into<SharedClassIdentity>,
        data: OutputListStorage,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if !data.is_shallow_clone() {
            return Err("object array requires shared slot storage".into());
        }
        if shape.len() < 2
            || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(data.len())
        {
            return Err("object array shape differs from its element count".into());
        }
        let class_name = class_name.into();
        let mut kind = None;
        for child in &data {
            admission::validate_value(child, class_name.as_identity(), &mut kind)?;
        }
        Ok(Self {
            class_name,
            data,
            shape,
            selected_provenance: ExactArrayStorage::empty(),
            unselected_provenance: false,
        })
    }
    pub fn new(
        class_name: impl Into<SharedClassIdentity>,
        data: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(ObjectArrayAdmission::values(class_name, data, shape)?
            .finish_synchronously()?
            .into_array())
    }

    pub fn from_objects(
        class_name: impl Into<SharedClassIdentity>,
        objects: Vec<ObjectInstance>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(ObjectArrayAdmission::objects(class_name, objects, shape)?
            .finish_synchronously()?
            .into_array())
    }

    pub fn from_handles(
        class_name: impl Into<SharedClassIdentity>,
        handles: Vec<HandleRef>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(ObjectArrayAdmission::handles(class_name, handles, shape)?
            .finish_synchronously()?
            .into_array())
    }

    pub fn row(
        class_name: impl Into<SharedClassIdentity>,
        data: Vec<Value>,
    ) -> Result<Self, String> {
        let len = data.len();
        Self::new(class_name, data, vec![1, len])
    }

    pub fn empty(
        class_name: impl Into<SharedClassIdentity>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(class_name, Vec::new(), shape)
    }

    pub fn class_name(&self) -> &ClassIdentity {
        self.class_name.as_identity()
    }

    pub fn data(&self) -> &[Value] {
        &self.data
    }
    pub fn class_name_owner(&self) -> &SharedClassIdentity {
        &self.class_name
    }
    pub fn data_owner(&self) -> &OutputListStorage {
        &self.data
    }
    pub fn shape_owner(&self) -> &SharedShape {
        &self.shape
    }
    pub fn provenance_owner(&self) -> &ExactArrayStorage<Rc<SelectedObjectClassFacts>> {
        &self.selected_provenance
    }

    /// Replace one existing slot only when this generation is unique. Validate
    /// before mutation; no alias copy and no unchecked homogeneous-array view.
    pub fn try_replace_linear_unique(
        &mut self,
        index: usize,
        value: Value,
    ) -> Result<Value, Value> {
        if index >= self.data.len() || !self.data.is_unique() {
            return Err(value);
        }
        let mut kind = None;
        if let Some(first) = self.data.first() {
            if admission::validate_value(first, self.class_name.as_identity(), &mut kind).is_err() {
                return Err(value);
            }
        }
        if admission::validate_value(&value, self.class_name.as_identity(), &mut kind).is_err() {
            return Err(value);
        }
        Ok(std::mem::replace(
            &mut self
                .data
                .try_as_mut_slice_unique()
                .expect("checked unique object slots")[index],
            value,
        ))
    }

    pub fn selected_provenance(&self) -> &[Rc<SelectedObjectClassFacts>] {
        &self.selected_provenance
    }

    pub fn has_unselected_provenance(&self) -> bool {
        self.unselected_provenance
    }

    pub fn with_unselected_provenance(mut self) -> Self {
        if self.is_empty() {
            self.unselected_provenance = true;
        }
        self
    }

    pub fn with_selected_provenance_unbounded(
        mut self,
        provenance: impl IntoIterator<Item = Rc<SelectedObjectClassFacts>>,
    ) -> Result<Self, String> {
        let mut provenance_values = self.selected_provenance.iter().cloned().collect::<Vec<_>>();
        let mut seen = self
            .selected_provenance
            .iter()
            .map(Rc::as_ptr)
            .collect::<HashSet<_>>();
        for facts in provenance {
            if facts.class() != self.class_name.as_identity() {
                return Err("object-array selected product has a different class".into());
            }
            if !self.is_empty() {
                // The element itself is the authority until an empty-result
                // frontier captures its facts. Never duplicate that Rc here.
                continue;
            }
            if seen.insert(Rc::as_ptr(&facts)) {
                provenance_values.push(facts);
            }
        }
        self.selected_provenance = ExactArrayStorage::shared_unbounded(provenance_values);
        Ok(self)
    }

    /// Preserve a unique physical slot allocation and its retained claim.
    pub fn try_into_values(self) -> Result<OwnedExactArrayValues<Value>, Self> {
        let Self {
            class_name,
            data,
            shape,
            selected_provenance,
            unselected_provenance,
        } = self;
        match data.try_into_owned() {
            Ok(data) => Ok(data),
            Err(data) => Err(Self {
                class_name,
                data,
                shape,
                selected_provenance,
                unselected_provenance,
            }),
        }
    }
    /// Explicit ordinary materialization. Managed consumers construct admitted
    /// successor slots or retain data_owner; aliases never copy implicitly.
    pub fn into_data_unbounded(self) -> Vec<Value> {
        match self.data.try_into_owned() {
            Ok(values) => match values.try_into_unbounded_vec() {
                Ok(values) => values,
                Err(values) => values.into_iter().collect(),
            },
            Err(data) => data.iter().cloned().collect(),
        }
    }

    pub fn reshape_preserving_provenance_unbounded(
        self,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        self.try_with_shape(shape.into())
    }
    /// Replace only an already-owned shape, without allocating or copying any
    /// element/provenance slots. The runtime supplies admitted metadata.
    pub fn try_with_shape(mut self, shape: SharedShape) -> Result<Self, String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |length, axis| length.checked_mul(*axis))
            .ok_or("object array shape overflows addressable storage")?;
        if shape.len() < 2 || expected != self.data.len() {
            return Err("object array shape differs from its element count".into());
        }
        self.shape = shape;
        Ok(self)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_linear(&self, index: usize) -> Option<&Value> {
        self.data.get(index)
    }

    pub fn select_linear_unbounded(
        &self,
        indices: &[usize],
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let data = indices
            .iter()
            .map(|index| {
                self.data
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| format!("object array index {} is out of bounds", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selected = Self::new(self.class_name.clone(), data, shape)?;
        if selected.is_empty() {
            let selected = selected
                .with_selected_provenance_unbounded(self.selected_provenance.iter().cloned())?;
            Ok(if self.unselected_provenance {
                selected.with_unselected_provenance()
            } else {
                selected
            })
        } else {
            Ok(selected)
        }
    }
}

impl Trace for ObjectArray {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.data {
            value.trace(tracer);
        }
    }
}

impl fmt::Display for ObjectArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimensions = self
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x");
        write!(f, "{dimensions} {} array", self.class_name)
    }
}

#[cfg(test)]
mod tests {
    use super::ObjectArray;
    use crate::{ObjectInstance, Value};

    #[test]
    fn validates_homogeneous_column_major_storage() {
        let values = vec![
            Value::Object(ObjectInstance::new("pkg.Result")),
            Value::Object(ObjectInstance::new("pkg.Result")),
        ];
        let array = ObjectArray::new("pkg.Result", values, vec![1, 2]).unwrap();
        assert_eq!(array.shape(), &[1, 2]);
        assert_eq!(array.len(), 2);
        assert_eq!(array.class_name().display_name(), "pkg.Result");
    }

    #[test]
    fn original_generation_aliases_retain_slots_shape_and_unique_validation() {
        let values = vec![Value::Object(ObjectInstance::new("A"))];
        let slots = values.as_ptr();
        let shape = vec![1, 1];
        let dimensions = shape.as_ptr();
        let mut array = ObjectArray::new("A", values, shape).unwrap();
        assert_eq!(array.data().as_ptr(), slots);
        assert_eq!(array.shape().as_ptr(), dimensions);
        let alias = array.clone();
        assert_eq!(alias.data().as_ptr(), slots);
        assert_eq!(alias.shape().as_ptr(), dimensions);
        assert!(array
            .try_replace_linear_unique(0, Value::Object(ObjectInstance::new("A")))
            .is_err());
        drop(alias);
        assert!(array
            .try_replace_linear_unique(0, Value::Object(ObjectInstance::new("B")))
            .is_err());
        assert_eq!(array.data().as_ptr(), slots);
        assert!(array
            .try_replace_linear_unique(0, Value::Object(ObjectInstance::new("A")))
            .is_ok());
    }

    #[test]
    fn rejects_mixed_classes_and_shape_mismatch() {
        let mixed = vec![
            Value::Object(ObjectInstance::new("A")),
            Value::Object(ObjectInstance::new("B")),
        ];
        assert!(ObjectArray::new("A", mixed, vec![1, 2]).is_err());
        assert!(ObjectArray::new(
            "A",
            vec![Value::Object(ObjectInstance::new("A"))],
            vec![1, 2]
        )
        .is_err());
    }
}

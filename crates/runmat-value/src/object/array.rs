use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use runmat_gc_api::{Trace, Tracer};
use runmat_types::ClassIdentity;

use crate::{HandleRef, ObjectInstance, SelectedObjectClassFacts, Value};

#[path = "array/admission.rs"]
mod admission;
pub use admission::{ObjectArrayAdmission, ValidatedObjectArrayParts};

/// A homogeneous MATLAB object array.
///
/// Storage is column-major and the shape is always explicit. Elements may be
/// value objects or handle objects, but every element must have the same
/// concrete class. Scalar objects continue to use `Value::Object` and
/// `Value::HandleObject`; this type represents zero or multiple elements
/// without falling back to a cell array.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectArray {
    class_name: ClassIdentity,
    data: Vec<Value>,
    shape: Vec<usize>,
    // Empty-only exact products captured before slicing/deletion discards all
    // elements. Nonempty elements already own their own selected facts, so an
    // array-level copy would be redundant. Distinct old/new products of one
    // ClassIdentity remain distinct; no current-name lookup is licensed.
    selected_provenance: Vec<Rc<SelectedObjectClassFacts>>,
    unselected_provenance: bool,
}

impl ObjectArray {
    pub fn new(
        class_name: impl Into<ClassIdentity>,
        data: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(
            ObjectArrayAdmission::values(class_name.into(), data, shape)?
                .finish_synchronously()?
                .into_array(),
        )
    }

    pub fn from_objects(
        class_name: impl Into<ClassIdentity>,
        objects: Vec<ObjectInstance>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(
            ObjectArrayAdmission::objects(class_name.into(), objects, shape)?
                .finish_synchronously()?
                .into_array(),
        )
    }

    pub fn from_handles(
        class_name: impl Into<ClassIdentity>,
        handles: Vec<HandleRef>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(
            ObjectArrayAdmission::handles(class_name.into(), handles, shape)?
                .finish_synchronously()?
                .into_array(),
        )
    }

    pub fn row(class_name: impl Into<ClassIdentity>, data: Vec<Value>) -> Result<Self, String> {
        let len = data.len();
        Self::new(class_name, data, vec![1, len])
    }

    pub fn empty(class_name: impl Into<ClassIdentity>, shape: Vec<usize>) -> Result<Self, String> {
        Self::new(class_name, Vec::new(), shape)
    }

    pub fn class_name(&self) -> &ClassIdentity {
        &self.class_name
    }

    pub fn data(&self) -> &[Value] {
        &self.data
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

    pub fn with_selected_provenance(
        mut self,
        provenance: impl IntoIterator<Item = Rc<SelectedObjectClassFacts>>,
    ) -> Result<Self, String> {
        let mut seen = self
            .selected_provenance
            .iter()
            .map(Rc::as_ptr)
            .collect::<HashSet<_>>();
        for facts in provenance {
            if facts.class() != &self.class_name {
                return Err("object-array selected product has a different class".into());
            }
            if !self.is_empty() {
                // The element itself is the authority until an empty-result
                // frontier captures its facts. Never duplicate that Rc here.
                continue;
            }
            if seen.insert(Rc::as_ptr(&facts)) {
                self.selected_provenance.push(facts);
            }
        }
        Ok(self)
    }

    pub fn into_data(self) -> Vec<Value> {
        self.data
    }

    pub fn reshape_preserving_provenance(mut self, shape: Vec<usize>) -> Result<Self, String> {
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

    pub fn select_linear(&self, indices: &[usize], shape: Vec<usize>) -> Result<Self, String> {
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
            let selected =
                selected.with_selected_provenance(self.selected_provenance.iter().cloned())?;
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

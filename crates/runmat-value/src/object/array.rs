use std::fmt;

use runmat_gc_api::{Trace, Tracer};

use crate::{HandleRef, ObjectInstance, Value};

/// A homogeneous MATLAB object array.
///
/// Storage is column-major and the shape is always explicit. Elements may be
/// value objects or handle objects, but every element must have the same
/// concrete class. Scalar objects continue to use `Value::Object` and
/// `Value::HandleObject`; this type represents zero or multiple elements
/// without falling back to a cell array.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectArray {
    class_name: String,
    data: Vec<Value>,
    shape: Vec<usize>,
}

impl ObjectArray {
    pub fn new(
        class_name: impl Into<String>,
        data: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let class_name = class_name.into();
        if class_name.is_empty() {
            return Err("object array class name must not be empty".into());
        }
        if shape.len() < 2 {
            return Err("object array shape must contain at least two dimensions".into());
        }
        let element_count = shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| "object array shape overflows addressable storage".to_string())?;
        if element_count != data.len() {
            return Err(format!(
                "object array shape describes {element_count} elements but {} were supplied",
                data.len()
            ));
        }
        for value in &data {
            let element_class = match value {
                Value::Object(object) => object.class_name.as_str(),
                Value::HandleObject(handle) => handle.class_name.as_str(),
                _ => return Err("object array elements must be value or handle objects".into()),
            };
            if element_class != class_name {
                return Err(format!(
                    "object array element class '{element_class}' does not match '{class_name}'"
                ));
            }
        }
        Ok(Self {
            class_name,
            data,
            shape,
        })
    }

    pub fn from_objects(
        class_name: impl Into<String>,
        objects: Vec<ObjectInstance>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(
            class_name,
            objects.into_iter().map(Value::Object).collect(),
            shape,
        )
    }

    pub fn from_handles(
        class_name: impl Into<String>,
        handles: Vec<HandleRef>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(
            class_name,
            handles.into_iter().map(Value::HandleObject).collect(),
            shape,
        )
    }

    pub fn row(class_name: impl Into<String>, data: Vec<Value>) -> Result<Self, String> {
        let len = data.len();
        Self::new(class_name, data, vec![1, len])
    }

    pub fn empty(class_name: impl Into<String>, shape: Vec<usize>) -> Result<Self, String> {
        Self::new(class_name, Vec::new(), shape)
    }

    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    pub fn data(&self) -> &[Value] {
        &self.data
    }

    pub fn into_data(self) -> Vec<Value> {
        self.data
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
        Self::new(self.class_name.clone(), data, shape)
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
            Value::Object(ObjectInstance::new("pkg.Result".into())),
            Value::Object(ObjectInstance::new("pkg.Result".into())),
        ];
        let array = ObjectArray::new("pkg.Result", values, vec![1, 2]).unwrap();
        assert_eq!(array.shape(), &[1, 2]);
        assert_eq!(array.len(), 2);
        assert_eq!(array.class_name(), "pkg.Result");
    }

    #[test]
    fn rejects_mixed_classes_and_shape_mismatch() {
        let mixed = vec![
            Value::Object(ObjectInstance::new("A".into())),
            Value::Object(ObjectInstance::new("B".into())),
        ];
        assert!(ObjectArray::new("A", mixed, vec![1, 2]).is_err());
        assert!(ObjectArray::new(
            "A",
            vec![Value::Object(ObjectInstance::new("A".into()))],
            vec![1, 2]
        )
        .is_err());
    }
}

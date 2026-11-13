use super::*;
use std::num::NonZeroUsize;

enum Candidate {
    Values { values: Vec<Value>, index: usize },
    Objects(std::vec::IntoIter<ObjectInstance>),
    Handles(std::vec::IntoIter<HandleRef>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum StorageKind {
    Value,
    Handle,
}

impl Candidate {
    fn len(&self) -> usize {
        match self {
            Self::Values { values, .. } => values.len(),
            Self::Objects(values) => values.len(),
            Self::Handles(values) => values.len(),
        }
    }
}

/// Incremental owner for homogeneous object-array storage.
pub struct ObjectArrayAdmission {
    class_name: SharedClassIdentity,
    candidate: Candidate,
    shape: Vec<usize>,
    shape_index: usize,
    length: usize,
    header_checked: bool,
    storage_kind: Option<StorageKind>,
    data: Vec<Value>,
    complete: bool,
}

/// Sealed, once-consumed object-array parts.
#[derive(Debug)]
pub struct ValidatedObjectArrayParts {
    array: ObjectArray,
}

impl ObjectArrayAdmission {
    pub fn values(
        class_name: impl Into<SharedClassIdentity>,
        values: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(
            class_name.into(),
            Candidate::Values { values, index: 0 },
            shape,
        )
    }

    pub fn objects(
        class_name: impl Into<SharedClassIdentity>,
        values: Vec<ObjectInstance>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(
            class_name.into(),
            Candidate::Objects(values.into_iter()),
            shape,
        )
    }

    pub fn handles(
        class_name: impl Into<SharedClassIdentity>,
        values: Vec<HandleRef>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::new(
            class_name.into(),
            Candidate::Handles(values.into_iter()),
            shape,
        )
    }

    fn new(
        class_name: SharedClassIdentity,
        candidate: Candidate,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        if shape.len() < 2 {
            return Err("object array shape must contain at least two dimensions".into());
        }
        let capacity = match &candidate {
            Candidate::Values { .. } => 0,
            Candidate::Objects(values) => values.len(),
            Candidate::Handles(values) => values.len(),
        };
        Ok(Self {
            class_name,
            candidate,
            shape,
            shape_index: 0,
            length: 1,
            header_checked: false,
            storage_kind: None,
            data: Vec::with_capacity(capacity),
            complete: false,
        })
    }

    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<usize, String> {
        let mut completed = 0;
        while completed < maximum_work.get() && !self.complete {
            if self.shape_index < self.shape.len() {
                self.length = self
                    .length
                    .checked_mul(self.shape[self.shape_index])
                    .ok_or_else(|| {
                        "object array shape overflows addressable storage".to_string()
                    })?;
                self.shape_index += 1;
            } else if !self.header_checked {
                let actual = self.candidate.len();
                if self.length != actual {
                    return Err(format!(
                        "object array shape describes {} elements but {actual} were supplied",
                        self.length
                    ));
                }
                self.header_checked = true;
            } else {
                match &mut self.candidate {
                    Candidate::Values { values, index } => {
                        let Some(value) = values.get(*index) else {
                            self.complete = true;
                            completed += 1;
                            break;
                        };
                        validate_value(
                            value,
                            self.class_name.as_identity(),
                            &mut self.storage_kind,
                        )?;
                        *index += 1;
                    }
                    Candidate::Objects(values) => {
                        let Some(object) = values.next() else {
                            self.complete = true;
                            completed += 1;
                            break;
                        };
                        let value = Value::Object(object);
                        validate_value(
                            &value,
                            self.class_name.as_identity(),
                            &mut self.storage_kind,
                        )?;
                        self.data.push(value);
                    }
                    Candidate::Handles(values) => {
                        let Some(handle) = values.next() else {
                            self.complete = true;
                            completed += 1;
                            break;
                        };
                        let value = Value::HandleObject(handle);
                        validate_value(
                            &value,
                            self.class_name.as_identity(),
                            &mut self.storage_kind,
                        )?;
                        self.data.push(value);
                    }
                }
            }
            completed += 1;
        }
        Ok(completed)
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn into_validated(self) -> Result<ValidatedObjectArrayParts, Box<Self>> {
        if !self.complete {
            return Err(Box::new(self));
        }
        let data = match self.candidate {
            Candidate::Values { values, .. } => values,
            Candidate::Objects(_) | Candidate::Handles(_) => self.data,
        };
        Ok(ValidatedObjectArrayParts {
            array: ObjectArray {
                class_name: self.class_name,
                data: OutputListStorage::shared_unbounded(data),
                shape: self.shape.into(),
                selected_provenance: ExactArrayStorage::empty(),
                unselected_provenance: false,
            },
        })
    }

    pub fn finish_synchronously(mut self) -> Result<ValidatedObjectArrayParts, String> {
        let quantum = NonZeroUsize::new(usize::MAX).unwrap();
        while !self.complete {
            self.advance(quantum)?;
        }
        self.into_validated()
            .map_err(|_| "object array admission unexpectedly incomplete".to_string())
    }
}

pub(super) fn validate_value(
    value: &Value,
    class_name: &ClassIdentity,
    storage_kind: &mut Option<StorageKind>,
) -> Result<(), String> {
    let (element_class, element_kind) = match value {
        Value::Object(object) => (object.class_name(), StorageKind::Value),
        Value::HandleObject(handle) => (handle.class_name.as_identity(), StorageKind::Handle),
        _ => return Err("object array elements must be value or handle objects".into()),
    };
    if element_class != class_name {
        return Err(format!(
            "object array element class '{element_class}' does not match '{class_name}'"
        ));
    }
    if storage_kind
        .replace(element_kind)
        .is_some_and(|kind| kind != element_kind)
    {
        return Err("object array cannot mix value-object and handle-object storage".into());
    }
    if let Value::Object(object) = value {
        if let Some(facts) = object.selected_class() {
            if facts.class() != class_name {
                return Err("object-array selected element differs from its class".into());
            }
        }
    }
    Ok(())
}

impl ValidatedObjectArrayParts {
    pub fn into_array(self) -> ObjectArray {
        self.array
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(class: &str) -> Value {
        Value::Object(ObjectInstance::new(class))
    }

    #[test]
    fn stepped_and_synchronous_value_object_arrays_match() {
        let values = vec![object("A"), object("A")];
        let original_storage = values.as_ptr();
        let class = ClassIdentity::new("A").unwrap();
        let mut stepped =
            ObjectArrayAdmission::values(class.clone(), values.clone(), vec![1, 2]).unwrap();
        while !stepped.is_complete() {
            assert_eq!(stepped.advance(NonZeroUsize::new(1).unwrap()).unwrap(), 1);
        }
        let stepped = stepped.into_validated().ok().unwrap().into_array();
        let synchronous = ObjectArrayAdmission::values(class, values, vec![1, 2])
            .unwrap()
            .finish_synchronously()
            .unwrap()
            .into_array();
        assert_eq!(stepped, synchronous);
        assert_eq!(synchronous.data().as_ptr(), original_storage);
    }

    #[test]
    fn empty_and_nd_object_shapes_match_the_synchronous_constructor() {
        for (values, shape) in [
            (Vec::new(), vec![2, 0, 3]),
            (
                vec![object("A"), object("A"), object("A"), object("A")],
                vec![2, 1, 2],
            ),
        ] {
            let class = ClassIdentity::new("A").unwrap();
            let mut stepped =
                ObjectArrayAdmission::values(class.clone(), values.clone(), shape.clone()).unwrap();
            while !stepped.is_complete() {
                stepped.advance(NonZeroUsize::new(1).unwrap()).unwrap();
            }
            assert_eq!(
                stepped.into_validated().ok().unwrap().into_array(),
                ObjectArray::new(class, values, shape).unwrap()
            );
        }
    }

    #[test]
    fn shape_kind_class_and_mixed_storage_errors_keep_precedence() {
        let class = ClassIdentity::new("A").unwrap();
        let error = ObjectArrayAdmission::values(class.clone(), vec![Value::Num(1.0)], vec![1, 2])
            .unwrap()
            .finish_synchronously()
            .expect_err("shape mismatch precedes element kind");
        assert_eq!(
            error,
            "object array shape describes 2 elements but 1 were supplied"
        );

        let error = ObjectArrayAdmission::values(
            class.clone(),
            vec![object("B"), Value::Num(1.0)],
            vec![1, 2],
        )
        .unwrap()
        .finish_synchronously()
        .expect_err("first class mismatch");
        assert_eq!(error, "object array element class 'B' does not match 'A'");
    }
}

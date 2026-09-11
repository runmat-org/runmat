use std::fmt;

use crate::{validate_no_transient_sequence, TransientValueError, Value};

/// Maximum number of values carried by one execution result sequence.
pub const MAX_VALUE_SEQUENCE_OUTPUTS: usize = u16::MAX as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSequenceKind {
    Single,
    CommaSeparated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueSequenceError {
    TooManyOutputs { actual: usize, maximum: usize },
    TransientValue(TransientValueError),
}

impl fmt::Display for ValueSequenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyOutputs { actual, maximum } => write!(
                formatter,
                "an output sequence contains {actual} values, but the maximum is {maximum}"
            ),
            Self::TransientValue(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ValueSequenceError {}

impl From<TransientValueError> for ValueSequenceError {
    fn from(error: TransientValueError) -> Self {
        Self::TransientValue(error)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ValueSequenceStorage {
    Single(Value),
    CommaSeparated(Vec<Value>),
}

/// Values produced by one source expression before its surrounding context
/// selects, expands, or discards them.
///
/// A sequence is an execution carrier, not a language value. It deliberately
/// has no serialization representation and cannot be nested in a [`Value`].
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSequence(ValueSequenceStorage);

impl ValueSequence {
    /// Constructs an empty comma-separated sequence.
    pub const fn empty() -> Self {
        Self(ValueSequenceStorage::CommaSeparated(Vec::new()))
    }

    /// Constructs the result of an ordinary single-valued expression.
    pub fn single(value: Value) -> Result<Self, ValueSequenceError> {
        validate_no_transient_sequence(&value)?;
        Ok(Self(ValueSequenceStorage::Single(value)))
    }

    /// Constructs a checked comma-separated sequence.
    ///
    /// Its cardinality may be zero, one, or many; cardinality alone does not
    /// erase the distinction between a comma-separated sequence and one value.
    pub fn comma_separated(values: Vec<Value>) -> Result<Self, ValueSequenceError> {
        validate_output_count(values.len())?;
        for value in &values {
            validate_no_transient_sequence(value)?;
        }
        Ok(Self(ValueSequenceStorage::CommaSeparated(values)))
    }

    pub const fn kind(&self) -> ValueSequenceKind {
        match self.0 {
            ValueSequenceStorage::Single(_) => ValueSequenceKind::Single,
            ValueSequenceStorage::CommaSeparated(_) => ValueSequenceKind::CommaSeparated,
        }
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            ValueSequenceStorage::Single(_) => 1,
            ValueSequenceStorage::CommaSeparated(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[Value] {
        match &self.0 {
            ValueSequenceStorage::Single(value) => std::slice::from_ref(value),
            ValueSequenceStorage::CommaSeparated(values) => values,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Value> {
        self.as_slice().iter()
    }

    pub fn into_values(self) -> Vec<Value> {
        match self.0 {
            ValueSequenceStorage::Single(value) => vec![value],
            ValueSequenceStorage::CommaSeparated(values) => values,
        }
    }
}

/// Checks a requested or produced result cardinality without allocating its
/// corresponding value storage.
pub fn validate_output_count(count: usize) -> Result<(), ValueSequenceError> {
    if count > MAX_VALUE_SEQUENCE_OUTPUTS {
        return Err(ValueSequenceError::TooManyOutputs {
            actual: count,
            maximum: MAX_VALUE_SEQUENCE_OUTPUTS,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_types::ClassIdentity;

    use super::*;
    use crate::{CellArray, Closure, ObjectArray, ObjectInstance, StructArray, StructValue};

    #[test]
    fn preserves_empty_single_and_comma_separated_cardinality() {
        let empty = ValueSequence::empty();
        assert_eq!(empty.kind(), ValueSequenceKind::CommaSeparated);
        assert!(empty.is_empty());

        assert_eq!(
            ValueSequence::single(Value::Num(3.0))
                .unwrap()
                .into_values(),
            vec![Value::Num(3.0)]
        );
        assert_eq!(
            ValueSequence::comma_separated(vec![Value::Num(1.0), Value::Num(2.0)])
                .unwrap()
                .into_values(),
            vec![Value::Num(1.0), Value::Num(2.0)]
        );
    }

    #[test]
    fn enforces_the_artifact_cardinality_domain_without_allocating() {
        assert!(validate_output_count(MAX_VALUE_SEQUENCE_OUTPUTS).is_ok());
        assert_eq!(
            validate_output_count(MAX_VALUE_SEQUENCE_OUTPUTS + 1).unwrap_err(),
            ValueSequenceError::TooManyOutputs {
                actual: MAX_VALUE_SEQUENCE_OUTPUTS + 1,
                maximum: MAX_VALUE_SEQUENCE_OUTPUTS,
            }
        );
    }

    #[test]
    fn rejects_transient_sequences_nested_in_supported_aggregate_kinds() {
        let transient = || Value::OutputList(vec![Value::Num(1.0)]);
        assert_eq!(
            ValueSequence::comma_separated(vec![transient()]).unwrap_err(),
            ValueSequenceError::TransientValue(TransientValueError::LegacyOutputList)
        );
        let cell = Value::Cell(CellArray::new(vec![transient()], 1, 1).unwrap());

        let mut structure = StructValue::new();
        structure.fields.insert("value".into(), transient());
        let structure = Value::Struct(structure);

        let mut element = StructValue::new();
        element.fields.insert("value".into(), transient());
        let struct_array = Value::StructArray(
            StructArray::new(vec![element.clone(), element], vec![1, 2]).unwrap(),
        );

        let mut object = ObjectInstance::new("SequenceTestObject");
        object.properties.insert("value".into(), transient());
        let object = Value::Object(object);

        let mut array_element = ObjectInstance::new("SequenceTestObject");
        array_element.properties.insert("value".into(), transient());
        let object_array = Value::ObjectArray(
            ObjectArray::from_objects(
                ClassIdentity::from("SequenceTestObject"),
                vec![array_element.clone(), array_element],
                vec![1, 2],
            )
            .unwrap(),
        );

        let closure = Value::Closure(Closure {
            function_name: "sequence_test".into(),
            bound_function: None,
            captures: vec![transient()],
        });

        for value in [cell, structure, struct_array, object, object_array, closure] {
            assert_eq!(
                ValueSequence::single(value).unwrap_err(),
                ValueSequenceError::TransientValue(TransientValueError::LegacyOutputList)
            );
        }
    }

    #[test]
    fn ordinary_objects_remain_accepted() {
        let object = ObjectInstance {
            class_name: ClassIdentity::from("SequenceTestObject"),
            properties: HashMap::from([("value".into(), Value::Num(1.0))]),
            dynamic_properties: None,
        };
        assert!(ValueSequence::single(Value::Object(object)).is_ok());
    }
}

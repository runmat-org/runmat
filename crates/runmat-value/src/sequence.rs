use std::fmt;

use crate::{validate_no_transient_sequence, OutputListStorage, TransientValueError, Value};

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
    SharedOutputStorage,
}

impl fmt::Display for ValueSequenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyOutputs { actual, maximum } => write!(
                formatter,
                "an output sequence contains {actual} values, but the maximum is {maximum}"
            ),
            Self::TransientValue(error) => error.fmt(formatter),
            Self::SharedOutputStorage => formatter.write_str(
                "shared exact output storage cannot be materialized without separate admission",
            ),
        }
    }
}

impl std::error::Error for ValueSequenceError {}

impl From<TransientValueError> for ValueSequenceError {
    fn from(error: TransientValueError) -> Self {
        Self::TransientValue(error)
    }
}

/// Physical parts of an execution sequence. Constructing these parts does not
/// validate a sequence; admission remains in the `ValueSequence` constructors.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueSequenceStorage {
    Single(Value),
    CommaSeparated(OutputListStorage),
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
        Self(ValueSequenceStorage::CommaSeparated(
            OutputListStorage::empty(),
        ))
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
        Ok(Self(ValueSequenceStorage::CommaSeparated(
            OutputListStorage::unbounded(values),
        )))
    }

    /// Adopt the canonical output backing without copying its children or
    /// discarding an exact physical claim.
    pub fn from_output_storage(values: OutputListStorage) -> Result<Self, ValueSequenceError> {
        validate_output_count(values.len())?;
        for value in values.iter() {
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

    /// Keep at most `len` existing values without rebuilding or validating
    /// children. Shared sequences retain their original physical backing;
    /// an inline single becomes empty only when zero values are selected.
    pub fn truncate(&mut self, len: usize) {
        match &mut self.0 {
            ValueSequenceStorage::Single(_) if len == 0 => {
                self.0 = ValueSequenceStorage::CommaSeparated(OutputListStorage::empty());
            }
            ValueSequenceStorage::Single(_) => {}
            ValueSequenceStorage::CommaSeparated(values) => values.truncate(len),
        }
    }

    pub fn as_slice(&self) -> &[Value] {
        match &self.0 {
            ValueSequenceStorage::Single(value) => std::slice::from_ref(value),
            ValueSequenceStorage::CommaSeparated(values) => values,
        }
    }

    /// Whether a comma-separated sequence retains an exact claim for its
    /// physical output collection (independent of child Value claims).
    pub fn has_retained_output_claim(&self) -> bool {
        matches!(&self.0, ValueSequenceStorage::CommaSeparated(values) if values.has_retained_claim())
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Value> {
        self.as_slice().iter()
    }

    /// Transfer the admitted carrier without allocating, cloning children, or
    /// projecting a transient execution sequence into a language value.
    pub fn into_storage(self) -> ValueSequenceStorage {
        self.0
    }

    pub fn into_output_storage(self) -> OutputListStorage {
        match self.0 {
            ValueSequenceStorage::Single(value) => OutputListStorage::unbounded(vec![value]),
            ValueSequenceStorage::CommaSeparated(values) => values,
        }
    }

    pub fn into_legacy_value(self) -> Value {
        match self.0 {
            ValueSequenceStorage::Single(value) => value,
            ValueSequenceStorage::CommaSeparated(values) => Value::OutputList(values),
        }
    }

    /// Transforms owned outputs without changing their sequence kind or
    /// cardinality. Each replacement is checked before it enters the sequence.
    /// A single-valued result does not allocate a temporary output vector.
    pub fn try_map<E>(
        self,
        mut map: impl FnMut(Value) -> Result<Value, E>,
        mut sequence_error: impl FnMut(ValueSequenceError) -> E,
    ) -> Result<Self, E> {
        let storage = match self.0 {
            ValueSequenceStorage::Single(value) => {
                let value = map(value)?;
                validate_no_transient_sequence(&value)
                    .map_err(ValueSequenceError::from)
                    .map_err(&mut sequence_error)?;
                ValueSequenceStorage::Single(value)
            }
            ValueSequenceStorage::CommaSeparated(mut values) => {
                let slots = values
                    .try_as_mut_slice_unique()
                    .ok_or_else(|| sequence_error(ValueSequenceError::SharedOutputStorage))?;
                for slot in slots {
                    let value = map(std::mem::replace(slot, Value::Num(0.0)))?;
                    validate_no_transient_sequence(&value)
                        .map_err(ValueSequenceError::from)
                        .map_err(&mut sequence_error)?;
                    *slot = value;
                }
                ValueSequenceStorage::CommaSeparated(values)
            }
        };
        Ok(Self(storage))
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
    use runmat_types::ClassIdentity;

    use super::*;
    use crate::{CellArray, Closure, ObjectArray, StructArray};

    #[test]
    fn mapping_preserves_sequence_kind_and_cardinality() {
        for sequence in [
            ValueSequence::empty(),
            ValueSequence::single(Value::Num(2.0)).unwrap(),
            ValueSequence::comma_separated(vec![Value::Num(2.0)]).unwrap(),
            ValueSequence::comma_separated(vec![Value::Num(2.0), Value::Num(3.0)]).unwrap(),
        ] {
            let kind = sequence.kind();
            let expected = sequence.as_slice().to_vec();
            let mapped = sequence
                .try_map::<ValueSequenceError>(Ok, std::convert::identity)
                .unwrap();
            assert_eq!(mapped.kind(), kind);
            assert_eq!(mapped.as_slice(), expected);
        }
    }

    #[test]
    fn mapping_rejects_transient_output_values() {
        for sequence in [
            ValueSequence::single(Value::Num(2.0)).unwrap(),
            ValueSequence::comma_separated(vec![Value::Num(2.0)]).unwrap(),
        ] {
            let error = sequence
                .try_map::<ValueSequenceError>(
                    |_| Ok(Value::output_list(Vec::new())),
                    std::convert::identity,
                )
                .unwrap_err();
            assert_eq!(
                error,
                ValueSequenceError::TransientValue(TransientValueError::LegacyOutputList)
            );
        }
    }

    #[test]
    fn mapping_stops_at_the_original_failure() {
        let sequence =
            ValueSequence::comma_separated(vec![Value::Num(2.0), Value::Num(3.0)]).unwrap();
        let expected = ValueSequenceError::TooManyOutputs {
            actual: 9,
            maximum: 8,
        };
        let mut calls = 0;
        let error = sequence
            .try_map(
                |_| {
                    calls += 1;
                    Err(expected.clone())
                },
                std::convert::identity,
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(calls, 1);
    }

    #[test]
    fn mapping_uses_explicit_error_conversion_only_for_invalid_replacements() {
        #[derive(Debug, PartialEq)]
        enum MappingError {
            Callback,
            Invalid(ValueSequenceError),
        }

        let mut conversions = 0;
        let error = ValueSequence::single(Value::Num(2.0))
            .unwrap()
            .try_map(
                |_| Ok(Value::output_list(Vec::new())),
                |error| {
                    conversions += 1;
                    MappingError::Invalid(error)
                },
            )
            .unwrap_err();
        assert_eq!(conversions, 1);
        assert_eq!(
            error,
            MappingError::Invalid(ValueSequenceError::TransientValue(
                TransientValueError::LegacyOutputList
            ))
        );

        let error = ValueSequence::single(Value::Num(2.0))
            .unwrap()
            .try_map(
                |_| Err(MappingError::Callback),
                |error| {
                    conversions += 1;
                    MappingError::Invalid(error)
                },
            )
            .unwrap_err();
        assert_eq!(error, MappingError::Callback);
        assert_eq!(conversions, 1);
    }

    #[test]
    fn preserves_empty_single_and_comma_separated_cardinality() {
        let empty = ValueSequence::empty();
        assert_eq!(empty.kind(), ValueSequenceKind::CommaSeparated);
        assert!(empty.is_empty());

        assert_eq!(
            ValueSequence::single(Value::Num(3.0))
                .unwrap()
                .into_legacy_value(),
            Value::Num(3.0)
        );
        let values = ValueSequence::comma_separated(vec![Value::Num(1.0), Value::Num(2.0)])
            .unwrap()
            .into_output_storage();
        assert_eq!(values.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);
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
        let transient = || Value::output_list(vec![Value::Num(1.0)]);
        assert_eq!(
            ValueSequence::comma_separated(vec![transient()]).unwrap_err(),
            ValueSequenceError::TransientValue(TransientValueError::LegacyOutputList)
        );
        let cell = Value::Cell(CellArray::new(vec![transient()], 1, 1).unwrap());

        let mut structure = crate::UnboundedStructValueBuilder::new();
        structure.fields.insert("value".into(), transient());
        let structure = Value::Struct(structure.finish());

        let mut element = crate::UnboundedStructValueBuilder::new();
        element.fields.insert("value".into(), transient());
        let element = element.finish();
        let struct_array = Value::StructArray(
            StructArray::new(vec![element.clone(), element], vec![1, 2]).unwrap(),
        );

        let mut object = crate::UnboundedObjectBuilder::new("SequenceTestObject");
        object.insert("value".into(), transient());
        let object = Value::Object(object.finish());

        let mut array_element = crate::UnboundedObjectBuilder::new("SequenceTestObject");
        array_element.insert("value".into(), transient());
        let array_element = array_element.finish();
        let object_array = Value::ObjectArray(
            ObjectArray::from_objects(
                ClassIdentity::from("SequenceTestObject"),
                vec![array_element.clone(), array_element],
                vec![1, 2],
            )
            .unwrap(),
        );

        let closure = Value::Closure(Closure::from_data(crate::ClosureData {
            lexical_environment: None,
            function_name: "sequence_test".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: OutputListStorage::shared_unbounded(vec![transient()]),
        }));

        for value in [cell, structure, struct_array, object, object_array, closure] {
            assert_eq!(
                ValueSequence::single(value).unwrap_err(),
                ValueSequenceError::TransientValue(TransientValueError::LegacyOutputList)
            );
        }
    }

    #[test]
    fn ordinary_objects_remain_accepted() {
        let mut object = crate::UnboundedObjectBuilder::new("SequenceTestObject");
        object.insert("value".into(), Value::Num(1.0));
        assert!(ValueSequence::single(Value::Object(object.finish())).is_ok());
    }
}

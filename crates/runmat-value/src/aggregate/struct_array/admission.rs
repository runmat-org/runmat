use super::StructArray;
use crate::UnboundedOrderedMapBuilder;
use crate::Value;
use indexmap::IndexSet;
use std::num::NonZeroUsize;

#[derive(Clone, Copy)]
enum Phase {
    Shape,
    Header,
    Unique,
    Repack,
    Complete,
}

/// Incremental owner for canonical field-major structure-array storage.
pub struct StructArrayAdmission {
    field_names: Option<std::vec::IntoIter<String>>,
    unique_names: IndexSet<String>,
    field_values: Option<std::vec::IntoIter<Value>>,
    shape: Vec<usize>,
    shape_index: usize,
    length: usize,
    names: Option<indexmap::set::IntoIter<String>>,
    current: Option<(String, Vec<Value>)>,
    fields: Option<UnboundedOrderedMapBuilder<Vec<Value>>>,
    scalar_fields: Option<crate::UnboundedStructValueBuilder>,
    phase: Phase,
    value: Option<Value>,
}

/// Sealed, once-consumed completed structure value.
#[derive(Debug)]
pub struct ValidatedStructArrayParts {
    value: Value,
}

impl StructArrayAdmission {
    pub fn field_major(
        field_names: Vec<String>,
        field_values: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        if shape.len() < 2 {
            return Err("structure array shape must contain at least two dimensions".into());
        }
        Ok(Self {
            field_names: Some(field_names.into_iter()),
            unique_names: IndexSet::new(),
            field_values: Some(field_values.into_iter()),
            shape,
            shape_index: 0,
            length: 1,
            names: None,
            current: None,
            fields: None,
            scalar_fields: None,
            phase: Phase::Shape,
            value: None,
        })
    }

    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<usize, String> {
        let mut remaining = maximum_work.get();
        let mut completed = 0;
        while remaining != 0 && !matches!(self.phase, Phase::Complete) {
            match self.phase {
                Phase::Shape => {
                    self.length = self
                        .length
                        .checked_mul(self.shape[self.shape_index])
                        .ok_or_else(|| {
                            "structure array shape exceeds platform limits".to_string()
                        })?;
                    self.shape_index += 1;
                    if self.shape_index == self.shape.len() {
                        self.phase = Phase::Header;
                    }
                }
                Phase::Header => {
                    let field_count = self.field_names.as_ref().expect("field names").len();
                    let expected = field_count.checked_mul(self.length).ok_or_else(|| {
                        "structure array field storage exceeds platform limits".to_string()
                    })?;
                    let actual = self.field_values.as_ref().expect("field values").len();
                    if actual != expected {
                        return Err(format!(
                            "structure array requires {expected} field values but {actual} were supplied"
                        ));
                    }
                    self.unique_names.reserve(field_count);
                    self.phase = Phase::Unique;
                }
                Phase::Unique => {
                    let Some(name) = self.field_names.as_mut().expect("field names").next() else {
                        let field_count = self.unique_names.len();
                        self.field_names = None;
                        self.names = Some(std::mem::take(&mut self.unique_names).into_iter());
                        if self.length == 1 {
                            self.scalar_fields = Some(
                                crate::UnboundedStructValueBuilder::with_capacity(field_count),
                            );
                        } else {
                            self.fields =
                                Some(UnboundedOrderedMapBuilder::with_capacity(field_count));
                        }
                        self.phase = Phase::Repack;
                        completed += 1;
                        remaining -= 1;
                        continue;
                    };
                    if !self.unique_names.insert(name) {
                        return Err("structure array field names must be unique".into());
                    }
                }
                Phase::Repack => self.repack_one(),
                Phase::Complete => unreachable!(),
            }
            completed += 1;
            remaining -= 1;
        }
        Ok(completed)
    }

    fn repack_one(&mut self) {
        if self.current.is_none() {
            if let Some(name) = self.names.as_mut().expect("field names").next() {
                self.current = Some((name, Vec::with_capacity(self.length)));
                return;
            }
            let shape = std::mem::take(&mut self.shape);
            self.value = Some(if self.length == 1 {
                Value::Struct(self.scalar_fields.take().expect("scalar fields").finish())
            } else {
                Value::StructArray(
                    StructArray::publish_columns_unbounded(
                        self.fields.take().expect("array fields"),
                        shape,
                    )
                    .expect("validated array columns"),
                )
            });
            self.phase = Phase::Complete;
            return;
        }
        let (_, column) = self.current.as_mut().expect("current field");
        if column.len() < self.length {
            column.push(
                self.field_values
                    .as_mut()
                    .expect("field values")
                    .next()
                    .expect("validated field-major length"),
            );
        } else {
            let (name, mut column) = self.current.take().expect("completed field");
            if self.length == 1 {
                self.scalar_fields
                    .as_mut()
                    .expect("scalar fields")
                    .insert(name, column.pop().expect("validated scalar field"));
            } else {
                self.fields
                    .as_mut()
                    .expect("array fields")
                    .insert(name, column);
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    pub fn into_validated(self) -> Result<ValidatedStructArrayParts, Box<Self>> {
        if !self.is_complete() {
            return Err(Box::new(self));
        }
        Ok(ValidatedStructArrayParts {
            value: self.value.expect("completed structure value"),
        })
    }

    pub fn finish_synchronously(mut self) -> Result<ValidatedStructArrayParts, String> {
        let quantum = NonZeroUsize::new(usize::MAX).unwrap();
        while !self.is_complete() {
            self.advance(quantum)?;
        }
        self.into_validated()
            .map_err(|_| "structure array admission unexpectedly incomplete".to_string())
    }
}

impl ValidatedStructArrayParts {
    pub fn into_value(self) -> Value {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> Value {
        Value::Num(value)
    }

    #[test]
    fn stepped_and_synchronous_field_major_admission_match_for_array_and_scalar() {
        let names = vec!["left".to_string(), "right".to_string()];
        let values = vec![scalar(1.0), scalar(2.0), scalar(10.0), scalar(20.0)];
        let mut stepped =
            StructArrayAdmission::field_major(names.clone(), values.clone(), vec![1, 2]).unwrap();
        while !stepped.is_complete() {
            assert_eq!(stepped.advance(NonZeroUsize::new(1).unwrap()).unwrap(), 1);
        }
        let stepped = stepped.into_validated().ok().unwrap().into_value();
        let synchronous = StructArrayAdmission::field_major(names, values, vec![1, 2])
            .unwrap()
            .finish_synchronously()
            .unwrap()
            .into_value();
        assert_eq!(stepped, synchronous);

        let Value::Struct(scalar) = StructArrayAdmission::field_major(
            vec!["value".into()],
            vec![Value::Num(7.0)],
            vec![1, 1],
        )
        .unwrap()
        .finish_synchronously()
        .unwrap()
        .into_value() else {
            panic!("one element normalizes to scalar struct");
        };
        assert_eq!(scalar.fields["value"], Value::Num(7.0));
    }

    #[test]
    fn shape_length_overflow_and_duplicate_precedence_match_legacy_contract() {
        let error = StructArrayAdmission::field_major(
            vec!["x".into(), "x".into()],
            vec![Value::Num(1.0)],
            vec![1, 1],
        )
        .unwrap()
        .finish_synchronously()
        .expect_err("length precedes duplicate names");
        assert_eq!(
            error,
            "structure array requires 2 field values but 1 were supplied"
        );

        let error = StructArrayAdmission::field_major(
            vec!["x".into(), "x".into()],
            vec![Value::Num(1.0), Value::Num(2.0)],
            vec![1, 1],
        )
        .unwrap()
        .finish_synchronously()
        .expect_err("duplicate field names");
        assert_eq!(error, "structure array field names must be unique");

        let error =
            StructArrayAdmission::field_major(vec!["x".into()], Vec::new(), vec![usize::MAX, 2])
                .unwrap()
                .finish_synchronously()
                .expect_err("shape overflow precedes length");
        assert_eq!(error, "structure array shape exceeds platform limits");
    }

    #[test]
    fn distinct_field_admission_uses_linear_accounted_steps() {
        let field_count = 128;
        let names = (0..field_count)
            .map(|index| format!("field_{index}"))
            .collect();
        let mut admission =
            StructArrayAdmission::field_major(names, Vec::new(), vec![0, 1]).unwrap();
        let mut steps = 0;
        while !admission.is_complete() {
            steps += admission.advance(NonZeroUsize::new(1).unwrap()).unwrap();
        }
        assert!(steps <= field_count * 4 + 8);
    }

    #[test]
    fn empty_schema_zero_extent_and_nd_shapes_match_the_synchronous_constructor() {
        for (names, values, shape) in [
            (Vec::new(), Vec::new(), vec![2, 3, 2]),
            (vec!["empty".into()], Vec::new(), vec![2, 0, 3]),
            (
                vec!["value".into()],
                (0..4).map(|value| Value::Num(value as f64)).collect(),
                vec![2, 1, 2],
            ),
        ] {
            let mut stepped =
                StructArrayAdmission::field_major(names.clone(), values.clone(), shape.clone())
                    .unwrap();
            while !stepped.is_complete() {
                stepped.advance(NonZeroUsize::new(1).unwrap()).unwrap();
            }
            assert_eq!(
                stepped.into_validated().ok().unwrap().into_value(),
                StructArray::normalize_field_major(names, values, shape).unwrap()
            );
        }
    }
}

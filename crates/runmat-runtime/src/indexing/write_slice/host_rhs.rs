//! One borrowed native RHS interpretation. Scalar replication is a mapping,
//! never a repeated payload, and exact integer components stay exact.
use super::{mex, rhs_shape::AssignmentRhsLayout, RuntimeError};
use crate::indexing::integer_assignment::{ComplexIntegerAssignmentValue, IntegerAssignmentValue};
use runmat_value::{ComplexTensor, LogicalArray, NumericScalar, Tensor, Value};

enum Source<'a> {
    Scalar(NumericScalar),
    Tensor(&'a Tensor),
    Logical(&'a LogicalArray),
    Complex(f64, f64),
    ComplexTensor(&'a ComplexTensor),
}
/// Borrows the actual source throughout growth/scatter. Complete RHS class and
/// shape are validated before any destination mutation.
pub(super) struct HostRhs<'a> {
    source: Source<'a>,
    layout: AssignmentRhsLayout,
}
impl<'a> HostRhs<'a> {
    fn source(
        rhs: &'a Value,
        complex: bool,
        logical: bool,
    ) -> Result<(Source<'a>, &'a [usize], usize), RuntimeError> {
        Ok(match rhs {
            Value::Num(value) => (Source::Scalar(NumericScalar::F64(*value)), &[1, 1], 1),
            Value::Int(value) => (
                Source::Scalar(NumericScalar::from(value.clone())),
                &[1, 1],
                1,
            ),
            Value::Bool(value) => (
                Source::Scalar(NumericScalar::F64(if *value { 1.0 } else { 0.0 })),
                &[1, 1],
                1,
            ),
            Value::Tensor(value) => (Source::Tensor(value), value.shape(), value.len()),
            Value::LogicalArray(value) if logical => {
                (Source::Logical(value), &value.shape, value.data.len())
            }
            Value::Complex(real, imag) if complex => (Source::Complex(*real, *imag), &[1, 1], 1),
            Value::ComplexTensor(value) if complex => {
                (Source::ComplexTensor(value), value.shape(), value.len())
            }
            other => {
                return Err(mex(
                    "InvalidSliceAssignmentRhs",
                    format!("slice assign: unsupported RHS type {:?}", other),
                ))
            }
        })
    }
    pub(super) fn linear(rhs: &'a Value, count: usize) -> Result<Self, RuntimeError> {
        let (source, _, len) = Self::source(rhs, false, true)?;
        Ok(Self {
            source,
            layout: AssignmentRhsLayout::linear(len, count)?,
        })
    }
    pub(super) fn multidimensional(
        rhs: &'a Value,
        selection: &[usize],
    ) -> Result<Self, RuntimeError> {
        let (source, shape, len) = Self::source(rhs, false, true)?;
        Ok(Self {
            source,
            layout: AssignmentRhsLayout::multidimensional(shape, len, selection)?,
        })
    }
    pub(super) fn complex(
        rhs: &'a Value,
        selection: &[usize],
        logical: bool,
    ) -> Result<Self, RuntimeError> {
        let (source, shape, len) = Self::source(rhs, true, logical)
            .map_err(|_| mex("InvalidSliceAssignmentRhs", "rhs must be numeric or tensor"))?;
        Ok(Self {
            source,
            layout: AssignmentRhsLayout::multidimensional(shape, len, selection)?,
        })
    }
    pub(super) fn for_plan(rhs: &'a Value, plan: &super::IndexPlan) -> Result<Self, RuntimeError> {
        if plan.dims == 1 {
            Self::linear(rhs, plan.indices.len())
        } else {
            Self::multidimensional(rhs, &plan.selection_lengths)
        }
    }
    pub(super) fn complex_for_plan(
        rhs: &'a Value,
        plan: &super::IndexPlan,
    ) -> Result<Self, RuntimeError> {
        let (source, shape, len) = Self::source(rhs, true, true)?;
        let layout = if plan.dims == 1 {
            AssignmentRhsLayout::linear(len, plan.indices.len())?
        } else {
            AssignmentRhsLayout::multidimensional(shape, len, &plan.selection_lengths)?
        };
        Ok(Self { source, layout })
    }
    pub(super) fn len(&self) -> usize {
        self.layout.count()
    }
    fn components(&self, ordinal: usize) -> (NumericScalar, NumericScalar) {
        let index = self.layout.source_index(ordinal);
        let zero = NumericScalar::F64(0.0);
        match &self.source {
            Source::Scalar(value) => (*value, zero),
            Source::Tensor(value) => (
                value
                    .numeric_value_at(index)
                    .expect("validated native RHS storage"),
                zero,
            ),
            Source::Logical(value) => (
                NumericScalar::F64(if value.data[index] != 0 { 1.0 } else { 0.0 }),
                zero,
            ),
            Source::Complex(real, imag) => (NumericScalar::F64(*real), NumericScalar::F64(*imag)),
            Source::ComplexTensor(value) => value
                .numeric_value_at(index)
                .expect("validated complex RHS storage"),
        }
    }
    pub(super) fn real_at(&self, ordinal: usize) -> f64 {
        self.components(ordinal).0.materialize_f64()
    }
    /// Native destination fills can retain same-class F32 bits and signed
    /// zero; floating materialization is only an explicit legacy boundary.
    pub(super) fn numeric_at(&self, ordinal: usize) -> NumericScalar {
        self.components(ordinal).0
    }
    pub(super) fn numeric_components_at(&self, ordinal: usize) -> (NumericScalar, NumericScalar) {
        self.components(ordinal)
    }
    pub(super) fn complex_at(&self, ordinal: usize) -> (f64, f64) {
        let (real, imag) = self.components(ordinal);
        (real.materialize_f64(), imag.materialize_f64())
    }
    pub(super) fn integer_at(&self, ordinal: usize) -> IntegerAssignmentValue {
        assignment_scalar(self.components(ordinal).0)
    }
    pub(super) fn complex_integer_at(&self, ordinal: usize) -> ComplexIntegerAssignmentValue {
        let (real, imag) = self.components(ordinal);
        ComplexIntegerAssignmentValue {
            real: assignment_scalar(real),
            imag: assignment_scalar(imag),
        }
    }
}
fn assignment_scalar(value: NumericScalar) -> IntegerAssignmentValue {
    match value.into_int_value() {
        Some(value) => IntegerAssignmentValue::Exact(value),
        None => IntegerAssignmentValue::Float(value.materialize_f64()),
    }
}

// Existing resident/sparse adapters actually require contiguous floating data.
// Dense host assignment instead consumes this same borrowed interpretation.
pub(super) fn linear(rhs: &Value, count: usize) -> Result<Vec<f64>, RuntimeError> {
    let view = HostRhs::linear(rhs, count)?;
    Ok((0..view.len())
        .map(|ordinal| view.real_at(ordinal))
        .collect())
}
pub(super) fn multidimensional(rhs: &Value, selection: &[usize]) -> Result<Vec<f64>, RuntimeError> {
    let view = HostRhs::multidimensional(rhs, selection)?;
    Ok((0..view.len())
        .map(|ordinal| view.real_at(ordinal))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};

    #[test]
    fn borrowed_views_keep_native_identity_and_scalar_or_linear_order() {
        let source = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .unwrap(),
        );
        let Value::Tensor(tensor) = &source else {
            unreachable!()
        };
        let view = HostRhs::multidimensional(&source, &[2, 1]).unwrap();
        let Source::Tensor(borrowed) = &view.source else {
            panic!("native source")
        };
        assert!(std::ptr::eq(*borrowed, tensor));
        for (ordinal, expected) in [u64::MAX, 9_007_199_254_740_993].into_iter().enumerate() {
            assert!(
                matches!(view.integer_at(ordinal), IntegerAssignmentValue::Exact(IntValue::U64(value)) if value == expected)
            );
        }
        let scalar = Value::Int(IntValue::U64(u64::MAX));
        let repeated = HostRhs::multidimensional(&scalar, &[2, 3]).unwrap();
        assert_eq!(repeated.len(), 6);
        for ordinal in 0..repeated.len() {
            assert!(matches!(
                repeated.integer_at(ordinal),
                IntegerAssignmentValue::Exact(IntValue::U64(u64::MAX))
            ));
        }
        assert_eq!(
            HostRhs::multidimensional(&scalar, &[1, 0]).unwrap().len(),
            0
        );
        assert!(HostRhs::multidimensional(&source, &[2, 2]).is_err());
    }

    #[test]
    fn complex_integer_components_stay_native_without_component_batches() {
        let source = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![9_007_199_254_740_993]),
                )
                .unwrap(),
                vec![1, 1],
            )
            .unwrap(),
        );
        let view = HostRhs::complex(&source, &[2, 2], false).unwrap();
        for ordinal in 0..view.len() {
            let value = view.complex_integer_at(ordinal);
            assert!(matches!(
                value.real,
                IntegerAssignmentValue::Exact(IntValue::U64(u64::MAX))
            ));
            assert!(matches!(
                value.imag,
                IntegerAssignmentValue::Exact(IntValue::U64(9_007_199_254_740_993))
            ));
        }
    }
}

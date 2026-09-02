use runmat_value::{NumericScalar, Tensor, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::array::introspection) enum EmptySelectorPolicy {
    Allow,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::array::introspection) enum DimensionArgumentForm {
    ScalarList,
    Vector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::builtins::array::introspection) struct DimensionSelectors {
    values: Vec<u64>,
}

impl DimensionSelectors {
    pub(in crate::builtins::array::introspection) fn values(&self) -> &[u64] {
        &self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::array::introspection) enum DimensionSelectorError {
    ArgumentType,
    VectorShape,
    Empty,
    NonFinite,
    NonInteger,
    LessThanOne,
    OutOfRange,
    VectorInScalarList,
}

pub(in crate::builtins::array::introspection) fn parse_dimension_arguments(
    arguments: &[Value],
    empty_policy: EmptySelectorPolicy,
) -> Result<DimensionSelectors, DimensionSelectorError> {
    let form = if arguments.len() == 1 && matches!(arguments[0], Value::Tensor(_)) {
        DimensionArgumentForm::Vector
    } else {
        DimensionArgumentForm::ScalarList
    };
    let mut values = Vec::new();
    for argument in arguments {
        match argument {
            Value::Int(value) => values.push(parse_integer(value)?),
            Value::Num(value) => values.push(parse_float(*value)?),
            Value::Tensor(tensor) if form == DimensionArgumentForm::Vector => {
                ensure_vector(tensor)?;
                for index in 0..tensor.len() {
                    let value = tensor
                        .numeric_value_at(index)
                        .expect("numeric tensor index is in bounds");
                    values.push(parse_numeric_scalar(value)?);
                }
            }
            Value::Tensor(_) => return Err(DimensionSelectorError::VectorInScalarList),
            _ => return Err(DimensionSelectorError::ArgumentType),
        }
    }
    if values.is_empty() && empty_policy == EmptySelectorPolicy::Reject {
        return Err(DimensionSelectorError::Empty);
    }
    Ok(DimensionSelectors { values })
}

fn ensure_vector(tensor: &Tensor) -> Result<(), DimensionSelectorError> {
    let non_unit_dimensions = tensor
        .shape
        .iter()
        .filter(|dimension| **dimension > 1)
        .count();
    if non_unit_dimensions <= 1 {
        Ok(())
    } else {
        Err(DimensionSelectorError::VectorShape)
    }
}

fn parse_numeric_scalar(value: NumericScalar) -> Result<u64, DimensionSelectorError> {
    match value {
        NumericScalar::F64(value) => parse_float(value),
        NumericScalar::F32(value) => parse_float(f64::from(value)),
        value => parse_integer(
            &value
                .into_int_value()
                .expect("non-floating numeric scalar is an integer"),
        ),
    }
}

fn parse_integer(value: &runmat_value::IntValue) -> Result<u64, DimensionSelectorError> {
    let value = value
        .try_to_u64()
        .ok_or(DimensionSelectorError::LessThanOne)?;
    if value == 0 {
        return Err(DimensionSelectorError::LessThanOne);
    }
    Ok(value)
}

fn parse_float(value: f64) -> Result<u64, DimensionSelectorError> {
    if !value.is_finite() {
        return Err(DimensionSelectorError::NonFinite);
    }
    if value < 1.0 {
        return Err(DimensionSelectorError::LessThanOne);
    }
    if value.fract() != 0.0 {
        return Err(DimensionSelectorError::NonInteger);
    }
    const U64_UPPER_EXCLUSIVE: f64 = 18_446_744_073_709_551_616.0;
    if value >= U64_UPPER_EXCLUSIVE {
        return Err(DimensionSelectorError::OutOfRange);
    }
    Ok(value as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn preserves_wide_integer_selectors_without_binary64() {
        let wide = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, wide]), vec![1, 2])
            .expect("selector tensor");
        let parsed =
            parse_dimension_arguments(&[Value::Tensor(tensor)], EmptySelectorPolicy::Allow)
                .expect("selectors");
        assert_eq!(parsed.values(), &[1, wide]);
    }

    #[test]
    fn vector_in_variadic_scalar_list_is_rejected() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("selector tensor");
        assert_eq!(
            parse_dimension_arguments(
                &[Value::Num(1.0), Value::Tensor(tensor)],
                EmptySelectorPolicy::Allow,
            ),
            Err(DimensionSelectorError::VectorInScalarList)
        );
    }
}

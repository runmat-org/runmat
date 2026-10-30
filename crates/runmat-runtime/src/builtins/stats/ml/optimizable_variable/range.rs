use super::*;

#[derive(Debug, Clone)]
pub(super) enum OptimizableRange {
    Numeric {
        value: Value,
        lower: f64,
        upper: f64,
    },
    Categorical {
        value: Value,
        categories: Vec<runmat_value::Utf8Text>,
    },
}

impl OptimizableRange {
    pub(super) fn parse(value: Value, explicit_type: Option<VariableType>) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => {
                if tensor.integer_storage().is_some_and(|storage| {
                    (0..storage.len()).any(|index| {
                        storage
                            .value_at(index)
                            .is_none_or(|value| !integer_is_exact_f64(&value))
                    })
                }) {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be exactly representable as double",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                let values = tensor_helpers::tensor_values_f64(&tensor);
                Self::numeric_from_values(Value::Tensor(tensor), &values)
            }
            Value::Num(n) => Self::numeric_from_values(
                Value::Tensor(Tensor::new(vec![n], vec![1, 1]).unwrap()),
                &[n],
            ),
            Value::Int(i) => {
                if !integer_is_exact_f64(&i) {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be exactly representable as double",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                let n = i.to_f64();
                Self::numeric_from_values(
                    Value::Tensor(Tensor::new(vec![n], vec![1, 1]).unwrap()),
                    &[n],
                )
            }
            Value::StringArray(array) => {
                let categories = text_categories_from_string_array(&array)?;
                Ok(Self::Categorical {
                    value: Value::StringArray(array),
                    categories,
                })
            }
            Value::String(runmat_value::StringElement::Text(text))
                if explicit_type == Some(VariableType::Categorical) =>
            {
                check_categories(std::slice::from_ref(&text))?;
                Ok(Self::Categorical {
                    value: Value::string(text.clone()),
                    categories: vec![text],
                })
            }
            Value::Cell(cell) => {
                let categories = text_categories_from_cell(&cell)?;
                Ok(Self::Categorical {
                    value: Value::Cell(cell),
                    categories,
                })
            }
            Value::CharArray(chars) if explicit_type == Some(VariableType::Categorical) => {
                let category = char_row_to_string(&chars, "Range", &ERROR_INVALID_RANGE)?;
                check_categories(std::slice::from_ref(&category))?;
                let value = Value::Cell(
                    CellArray::new(vec![Value::string(category.clone())], 1, 1)
                        .map_err(|err| error(err, &ERROR_INVALID_RANGE))?,
                );
                Ok(Self::Categorical {
                    value,
                    categories: vec![category.into()],
                })
            }
            other => Err(error(
                format!("optimizableVariable: unsupported Range value {other:?}"),
                &ERROR_INVALID_RANGE,
            )),
        }
    }

    fn numeric_from_values(value: Value, values: &[f64]) -> BuiltinResult<Self> {
        if values.len() != 2 {
            return Err(error(
                "optimizableVariable: numeric Range must contain exactly two elements",
                &ERROR_INVALID_RANGE,
            ));
        }
        let lower = values[0];
        let upper = values[1];
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(error(
                "optimizableVariable: numeric Range bounds must be finite and increasing",
                &ERROR_INVALID_RANGE,
            ));
        }
        Ok(Self::Numeric {
            value,
            lower,
            upper,
        })
    }

    pub(super) fn inferred_type(&self) -> VariableType {
        match self {
            Self::Numeric { .. } => VariableType::Real,
            Self::Categorical { .. } => VariableType::Categorical,
        }
    }

    pub(super) fn into_value(self) -> Value {
        match self {
            Self::Numeric { value, .. } | Self::Categorical { value, .. } => value,
        }
    }

    pub(super) fn validate_for(
        &self,
        var_type: VariableType,
        transform: Transform,
    ) -> BuiltinResult<()> {
        match (self, var_type) {
            (Self::Numeric { lower, .. }, VariableType::Real) => {
                validate_real_transform(*lower, transform)
            }
            (Self::Numeric { lower, upper, .. }, VariableType::Integer) => {
                if lower.fract().abs() > f64::EPSILON || upper.fract().abs() > f64::EPSILON {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be whole numbers",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                validate_integer_transform(*lower, transform)
            }
            (Self::Numeric { .. }, VariableType::Categorical) => Err(error(
                "optimizableVariable: categorical variables require a categorical text Range",
                &ERROR_INVALID_RANGE,
            )),
            (Self::Categorical { categories, .. }, VariableType::Categorical) => {
                check_categories(categories)?;
                if transform != Transform::None {
                    return Err(error(
                        "optimizableVariable: categorical variables only support Transform 'none'",
                        &ERROR_INVALID_OPTION,
                    ));
                }
                Ok(())
            }
            (Self::Categorical { .. }, _) => Err(error(
                "optimizableVariable: real and integer variables require a two-element numeric Range",
                &ERROR_INVALID_RANGE,
            )),
        }
    }
}

fn validate_real_transform(lower: f64, transform: Transform) -> BuiltinResult<()> {
    if transform == Transform::Log && lower <= 0.0 {
        return Err(error(
            "optimizableVariable: log transform requires positive real Range bounds",
            &ERROR_INVALID_RANGE,
        ));
    }
    Ok(())
}

fn validate_integer_transform(lower: f64, transform: Transform) -> BuiltinResult<()> {
    if transform == Transform::Log && lower < 0.0 {
        return Err(error(
            "optimizableVariable: log transform requires nonnegative integer Range bounds",
            &ERROR_INVALID_RANGE,
        ));
    }
    Ok(())
}

fn text_categories_from_string_array(
    array: &StringArray,
) -> BuiltinResult<Vec<runmat_value::Utf8Text>> {
    let categories = array
        .data()
        .iter()
        .map(|value| {
            value.text_ref().cloned().ok_or_else(|| {
                error(
                    "optimizableVariable: categorical Range values must be nonempty text",
                    &ERROR_INVALID_RANGE,
                )
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    check_categories(&categories)?;
    Ok(categories)
}

fn text_categories_from_cell(cell: &CellArray) -> BuiltinResult<Vec<runmat_value::Utf8Text>> {
    let mut categories = Vec::with_capacity(cell.data().len());
    for value in cell.data() {
        categories.push(parse_text_scalar(value, "Range", &ERROR_INVALID_RANGE)?);
    }
    check_categories(&categories)?;
    Ok(categories)
}

fn check_categories(categories: &[impl AsRef<str>]) -> BuiltinResult<()> {
    if categories.is_empty() {
        return Err(error(
            "optimizableVariable: categorical Range must contain at least one category",
            &ERROR_INVALID_RANGE,
        ));
    }
    let mut seen = HashSet::with_capacity(categories.len());
    for category in categories {
        let category = category.as_ref();
        if category.trim().is_empty() {
            return Err(error(
                "optimizableVariable: categorical Range values must be nonempty text",
                &ERROR_INVALID_RANGE,
            ));
        }
        if !seen.insert(category) {
            return Err(error(
                "optimizableVariable: categorical Range values must be unique",
                &ERROR_INVALID_RANGE,
            ));
        }
    }
    Ok(())
}

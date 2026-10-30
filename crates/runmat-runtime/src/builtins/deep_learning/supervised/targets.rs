//! Public loss selection and external label parsing. Numerical target geometry
//! and one-hot lowering belong to the typed native target plan.

use runmat_value::{NumericDType, NumericScalar, Value};

use crate::BuiltinResult;

use crate::builtins::deep_learning::scalar_text;

use super::deep_learning_error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LossKind {
    Regression,
    Classification,
}

pub(super) fn parse_trainnet_loss(value: &Value) -> BuiltinResult<LossKind> {
    if matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    ) {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: custom loss function handles require autodiff infrastructure and are not supported in this training slice",
        ));
    }
    match scalar_text(value, "trainnet")?
        .to_ascii_lowercase()
        .as_str()
    {
        "mse" | "mean-squared-error" | "mean_squared_error" => Ok(LossKind::Regression),
        "crossentropy" | "cross-entropy" => Ok(LossKind::Classification),
        other => Err(deep_learning_error(
            "trainnet",
            format!("trainnet: unsupported loss function '{other}'"),
        )),
    }
}

pub(super) fn is_label_target(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => true,
        Value::Cell(cell) => cell.data().iter().all(is_label_target),
        Value::Object(object) => object.class_name().is(runmat_types::standard::CATEGORICAL),
        _ => false,
    }
}

fn decimal_label(value: f64, function: &'static str) -> BuiltinResult<String> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(deep_learning_error(
            function,
            format!("{function}: numeric classification labels must be positive integers"),
        ));
    }
    Ok(format!("{value:.0}"))
}

fn numeric_label_column_shape(shape: &[usize]) -> bool {
    matches!(shape, [1] | [_, 1])
}

pub(super) fn label_strings(
    value: &Value,
    function: &'static str,
) -> BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Int(code) => {
            if code.to_i128() < 1 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: numeric classification labels must be positive integers"),
                ));
            }
            Ok(Some(vec![code.decimal_string()]))
        }
        Value::Num(number) => Ok(Some(vec![decimal_label(*number, function)?])),
        Value::Tensor(codes)
            if codes.numeric_dtype().integer_class().is_some()
                && numeric_label_column_shape(codes.shape()) =>
        {
            let mut labels = Vec::with_capacity(codes.len());
            for index in 0..codes.len() {
                let code = codes
                    .numeric_value_at(index)
                    .and_then(NumericScalar::into_int_value)
                    .ok_or_else(|| {
                        deep_learning_error(
                            function,
                            format!("{function}: numeric classification label storage is invalid"),
                        )
                    })?;
                if code.to_i128() < 1 {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: numeric classification labels must be positive integers"
                        ),
                    ));
                }
                labels.push(code.decimal_string());
            }
            Ok(Some(labels))
        }
        Value::Tensor(codes)
            if matches!(codes.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                && numeric_label_column_shape(codes.shape()) =>
        {
            let mut labels = Vec::with_capacity(codes.len());
            for index in 0..codes.len() {
                let number = match codes.numeric_value_at(index) {
                    Some(NumericScalar::F32(number)) => f64::from(number),
                    Some(NumericScalar::F64(number)) => number,
                    _ => {
                        return Err(deep_learning_error(
                            function,
                            format!("{function}: invalid numeric label storage"),
                        ))
                    }
                };
                labels.push(decimal_label(number, function)?);
            }
            Ok(Some(labels))
        }
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Some(vec![text.to_string()])),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(Some(vec![runmat_value::decode_utf16_code_units(
                chars.data(),
            )
            .map_err(|source| {
                let mut error = deep_learning_error(function, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?]))
        }
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| scalar_text(&Value::String(text.clone()), function))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|item| scalar_text(item, function))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(Some),
        Value::Object(object) if object.class_name().is(runmat_types::standard::CATEGORICAL) => {
            let Some(Value::StringArray(categories)) = object.properties().get("Categories") else {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: categorical labels are missing Categories"),
                ));
            };
            let Some(Value::Tensor(codes)) = object.properties().get("Codes") else {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: categorical labels are missing numeric Codes"),
                ));
            };
            let mut labels = Vec::with_capacity(codes.len());
            for index in 0..codes.len() {
                let code = codes.numeric_value_at(index).ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: categorical Codes storage is invalid"),
                    )
                })?;
                let index = categorical_index(code, categories.data().len(), function)?;
                labels.push(scalar_text(
                    &Value::String(categories.data()[index].clone()),
                    function,
                )?);
            }
            Ok(Some(labels))
        }
        _ => Ok(None),
    }
}

fn categorical_index(
    code: NumericScalar,
    count: usize,
    function: &'static str,
) -> BuiltinResult<usize> {
    let index = match code {
        NumericScalar::F32(value) => {
            let value = f64::from(value);
            if !value.is_finite() || value < 1.0 || value > count as f64 || value.fract() != 0.0 {
                None
            } else {
                Some(value as usize - 1)
            }
        }
        NumericScalar::F64(value) => {
            if !value.is_finite() || value < 1.0 || value > count as f64 || value.fract() != 0.0 {
                None
            } else {
                Some(value as usize - 1)
            }
        }
        integer => integer
            .into_int_value()
            .and_then(|value| value.try_to_usize())
            .filter(|index| *index >= 1 && *index <= count)
            .map(|index| index - 1),
    };
    index.ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: categorical Codes must index known categories"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_label_columns_keep_rank_zero_and_rank_one_admission() {
        let wide = (1_u64 << 53) + 1;
        for (shape, admitted) in [
            (vec![], false),
            (vec![1], true),
            (vec![2], false),
            (vec![2, 1], true),
            (vec![2, 0], false),
            (vec![1, 1, 1], false),
        ] {
            let len = shape.iter().product::<usize>();
            let integers = (0..len)
                .map(|index| if index == 0 { wide } else { 2 })
                .collect();
            let integer = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(integers), shape.clone()).unwrap(),
            );
            let integer_labels = label_strings(&integer, "trainnet").unwrap();
            assert_eq!(
                integer_labels,
                admitted.then(|| {
                    (0..len)
                        .map(|index| {
                            if index == 0 {
                                wide.to_string()
                            } else {
                                "2".into()
                            }
                        })
                        .collect()
                })
            );

            let floats = (0..len).map(|index| (index + 1) as f64).collect();
            let float = Value::Tensor(Tensor::new(floats, shape).unwrap());
            let float_labels = label_strings(&float, "trainnet").unwrap();
            assert_eq!(
                float_labels,
                admitted.then(|| (1..=len).map(|value| value.to_string()).collect())
            );
        }
    }
}

//! Native typed output publication for one-hot encoding and decoding.
use super::*;

pub(super) fn onehot_numeric_output(
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let mut output =
        Tensor::from_numeric_storage(NumericStorage::zeros(dtype, values.len()), shape).map_err(
            |err| {
                internal(
                    EncodingBuiltin::OnehotEncode,
                    format!("onehotencode: {err}"),
                )
            },
        )?;
    for (index, value) in values.into_iter().enumerate() {
        output
            .set_numeric_assignment_at(index, NumericScalar::F64(value))
            .map_err(|err| {
                internal(
                    EncodingBuiltin::OnehotEncode,
                    format!("onehotencode: {err}"),
                )
            })?;
    }
    Ok(Value::Tensor(output))
}

pub(super) fn labels_to_value(
    name: EncodingBuiltin,
    labels: &[Label],
    shape: Vec<usize>,
    output: OutputKind,
    category_order: Option<&[Label]>,
) -> BuiltinResult<Value> {
    match output {
        OutputKind::Numeric(class) => {
            let mut output = Tensor::from_numeric_storage(
                NumericStorage::zeros(class.into(), labels.len()),
                shape,
            )
            .map_err(|err| internal(name, format!("{name}: {err}")))?;
            for (index, label) in labels.iter().enumerate() {
                let Label::Numeric(value) = label else {
                    return Err(invalid(
                        name,
                        format!("{name}: numeric output requires numeric classes"),
                    ));
                };
                output
                    .set_numeric_assignment_at(index, *value)
                    .map_err(|err| internal(name, format!("{name}: {err}")))?;
            }
            Ok(Value::Tensor(output))
        }
        OutputKind::Logical => {
            let mut data = Vec::with_capacity(labels.len());
            for label in labels {
                match label {
                    Label::Logical(value) => data.push(u8::from(*value)),
                    _ => {
                        return Err(invalid(
                            name,
                            format!("{name}: logical output requires logical classes"),
                        ))
                    }
                }
            }
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(|err| internal(name, format!("{name}: {err}")))
        }
        OutputKind::String => StringArray::new(text_values(labels), shape)
            .map(Value::StringArray)
            .map_err(|err| internal(name, format!("{name}: {err}"))),
        OutputKind::Cell => {
            let rows = shape.first().copied().unwrap_or(labels.len());
            let cols = if rows == 0 { 0 } else { labels.len() / rows };
            CellArray::new(
                text_values(labels)
                    .into_iter()
                    .map(|text| Value::CharArray(CharArray::new_row(&text)))
                    .collect(),
                rows,
                cols,
            )
            .map(Value::Cell)
            .map_err(|err| internal(name, format!("{name}: {err}")))
        }
        OutputKind::Categorical => {
            let values = text_values(labels);
            let category_values = category_order
                .map(text_values)
                .unwrap_or_else(|| dedupe_text(values.clone()));
            let value_array = Value::StringArray(
                StringArray::new(values.clone(), shape)
                    .map_err(|err| internal(name, format!("{name}: {err}")))?,
            );
            let categories = Value::StringArray(
                StringArray::new(category_values.clone(), vec![1, category_values.len()])
                    .map_err(|err| internal(name, format!("{name}: {err}")))?,
            );
            crate::builtins::table::categorical_from_args(
                crate::call::FunctionArgumentView::from_slice(&[value_array, categories]),
            )
        }
    }
}

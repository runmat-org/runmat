use super::*;
pub(super) fn labels_to_value(labels: &[Label], kind: LabelKind) -> BuiltinResult<Value> {
    match kind {
        LabelKind::Numeric => Ok(Value::Tensor(
            Tensor::new(
                labels
                    .iter()
                    .map(|label| match label {
                        Label::Numeric(value) => *value,
                        _ => f64::NAN,
                    })
                    .collect(),
                vec![labels.len(), 1],
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Integer(dtype) => {
            let values = labels
                .iter()
                .map(|label| match label {
                    Label::Integer(value) => Ok(value.clone()),
                    _ => Err(internal("confusionmat: noninteger label in integer order")),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let storage = integer_storage_from_values(dtype, values)?;
            Ok(Value::Tensor(
                Tensor::new_integer(storage, vec![labels.len(), 1])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            ))
        }
        LabelKind::Logical => Ok(Value::LogicalArray(
            LogicalArray::new(
                labels
                    .iter()
                    .map(|label| match label {
                        Label::Logical(true) => 1,
                        _ => 0,
                    })
                    .collect(),
                vec![labels.len(), 1],
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::String => Ok(Value::StringArray(
            StringArray::new(text_labels(labels), vec![labels.len(), 1])
                .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Char => Ok(Value::CharArray(char_array_from_rows(&text_labels(
            labels,
        ))?)),
        LabelKind::Cell => Ok(Value::Cell(
            CellArray::new(
                text_labels(labels)
                    .into_iter()
                    .map(|label| Value::CharArray(CharArray::new_row(&label)))
                    .collect(),
                labels.len(),
                1,
            )
            .map_err(|err| internal(format!("confusionmat: {err}")))?,
        )),
        LabelKind::Categorical => {
            let values = text_labels(labels);
            let value_array = Value::StringArray(
                StringArray::new(values.clone(), vec![labels.len(), 1])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            );
            let categories = Value::StringArray(
                StringArray::new(values, vec![1, labels.len()])
                    .map_err(|err| internal(format!("confusionmat: {err}")))?,
            );
            crate::builtins::table::categorical_from_args(
                crate::call::FunctionArgumentView::from_slice(&[value_array, categories]),
            )
        }
    }
}

pub(super) fn text_labels(labels: &[Label]) -> Vec<String> {
    labels
        .iter()
        .map(|label| match label {
            Label::Text(value) => value.clone(),
            _ => String::new(),
        })
        .collect()
}

pub(super) fn char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(
            runmat_value::decode_utf16_code_units(&array.data()[start..start + array.cols])
                .map_err(|source| internal(format!("confusionmat: {source}")))?,
        );
    }
    Ok(rows)
}

pub(super) fn char_array_from_rows(rows: &[String]) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.encode_utf16().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let units = row.encode_utf16().count();
        data.extend(row.chars());
        data.extend(std::iter::repeat_n(' ', width - units));
    }
    CharArray::from_unicode_scalars(data, rows.len(), width)
        .map_err(|err| internal(format!("confusionmat: {err}")))
}

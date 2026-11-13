use super::*;

pub(super) fn label_vector(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    labels_from_value(value, name).map_err(|err| {
        let message = err.message.clone();
        let mut builder = build_runtime_error(message)
            .with_builtin(FITCLINEAR_NAME)
            .with_source(err);
        if let Some(identifier) = ERROR_FITCLINEAR_INVALID_ARGUMENT.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })
}

pub(super) fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let (rows, columns) = classification_matrix_dimensions(&tensor);
            if tensor.shape().len() > 2 || (rows != 1 && columns != 1) {
                return Err(predict_invalid(format!("predict: {name} must be a vector")));
            }
            let storage = tensor.integer_storage().expect("checked above");
            Ok(LabelVector {
                labels: storage
                    .exact_values()
                    .into_iter()
                    .map(ClassLabel::Integer)
                    .collect(),
                kind: LabelKind::Integer(integer_class(storage)),
            })
        }
        Value::Tensor(tensor) => Ok(LabelVector {
            labels: vector_values_predict(&tensor, name)?
                .into_iter()
                .map(ClassLabel::Numeric)
                .collect(),
            kind: LabelKind::Numeric,
        }),
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(value)],
            kind: LabelKind::Numeric,
        }),
        Value::Int(value) => Ok(LabelVector {
            kind: LabelKind::Integer(integer_class_for_value(&value)),
            labels: vec![ClassLabel::Integer(value)],
        }),
        Value::String(text) => Ok(LabelVector {
            labels: vec![ClassLabel::Text(text)],
            kind: LabelKind::Text,
        }),
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array)?
                .into_iter()
                .map(|text| ClassLabel::Text(text.into()))
                .collect(),
            kind: LabelKind::Text,
        }),
        Value::StringArray(array) => {
            let (data, _shape, _rows, _cols) = array.into_parts();
            let labels = match data.try_into_owned() {
                Ok(values) => values.into_iter().map(ClassLabel::Text).collect(),
                Err(values) => values.iter().cloned().map(ClassLabel::Text).collect(),
            };
            Ok(LabelVector {
                labels,
                kind: LabelKind::Text,
            })
        },
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data().len());
            for item in cell.data().iter() {
                labels.push(ClassLabel::Text(match item {
                    Value::String(text) => text.clone(),
                    Value::StringArray(array) if array.data().len() == 1 => array.data()[0].clone(),
                    _ => predict_scalar_text(item)?.ok_or_else(|| {
                        predict_invalid(format!("predict: {name} cell entries must be text"))
                    })?.into(),
                }));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Text,
            })
        }
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Logical(value)],
            kind: LabelKind::Logical,
        }),
        Value::LogicalArray(array) => Ok(LabelVector {
            labels: array
                .data
                .into_iter()
                .map(|value| ClassLabel::Logical(value != 0))
                .collect(),
            kind: LabelKind::Logical,
        }),
        other => Err(predict_invalid(format!(
            "predict: {name} must be numeric, logical, string, character, or cellstr labels; got {other:?}"
        ))),
    }
}

pub(super) fn validate_class_names(
    names: Vec<ClassLabel>,
    kind: LabelKind,
) -> BuiltinResult<Vec<ClassLabel>> {
    if names.len() != 2 {
        return Err(fitclinear_invalid(
            "fitclinear: ClassNames must contain exactly two values",
        ));
    }
    let mut out = Vec::new();
    for name in names {
        if is_missing_label(&name) {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must not contain missing values",
            ));
        }
        if label_kind(&name) != kind {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must have the same type as the response",
            ));
        }
        if out.iter().any(|existing| same_label(existing, &name)) {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must contain unique values",
            ));
        }
        out.push(name);
    }
    Ok(out)
}

pub(super) fn unique_labels(labels: &[ClassLabel], kind: LabelKind) -> Vec<ClassLabel> {
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    match kind {
        LabelKind::Numeric => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::Integer(_) => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Integer(a), ClassLabel::Integer(b)) => integer_cmp(a, b),
            _ => Ordering::Equal,
        }),
        LabelKind::Text => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Text(a), ClassLabel::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }),
        LabelKind::Logical => out.sort_by_key(|label| match label {
            ClassLabel::Logical(value) => *value,
            _ => false,
        }),
    }
    out
}

fn label_kind(label: &ClassLabel) -> LabelKind {
    match label {
        ClassLabel::Numeric(_) => LabelKind::Numeric,
        ClassLabel::Integer(value) => LabelKind::Integer(integer_class_for_value(value)),
        ClassLabel::Text(_) => LabelKind::Text,
        ClassLabel::Logical(_) => LabelKind::Logical,
    }
}

pub(super) fn label_kind_name(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Numeric => "numeric",
        LabelKind::Integer(class) => integer_class_name(class),
        LabelKind::Text => "text",
        LabelKind::Logical => "logical",
    }
}

pub(super) fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (ClassLabel::Integer(a), ClassLabel::Integer(b)) => a == b,
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

pub(super) fn is_missing_label(label: &ClassLabel) -> bool {
    matches!(label, ClassLabel::Numeric(value) if value.is_nan())
        || matches!(label, ClassLabel::Text(value) if value.is_missing())
}

fn integer_class(storage: &IntegerStorage) -> IntegerClass {
    storage.integer_class()
}

fn integer_class_for_value(value: &IntValue) -> IntegerClass {
    value.integer_class()
}

fn integer_class_name(class: IntegerClass) -> &'static str {
    class.class_name()
}

fn integer_cmp(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

pub(super) fn integer_storage(
    class: IntegerClass,
    values: Vec<IntValue>,
) -> Result<IntegerStorage, String> {
    match class {
        IntegerClass::Int8 => collect_integer_values(values, "int8", |v| match v {
            IntValue::I8(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I8),
        IntegerClass::Int16 => collect_integer_values(values, "int16", |v| match v {
            IntValue::I16(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I16),
        IntegerClass::Int32 => collect_integer_values(values, "int32", |v| match v {
            IntValue::I32(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I32),
        IntegerClass::Int64 => collect_integer_values(values, "int64", |v| match v {
            IntValue::I64(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::I64),
        IntegerClass::UInt8 => collect_integer_values(values, "uint8", |v| match v {
            IntValue::U8(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U8),
        IntegerClass::UInt16 => collect_integer_values(values, "uint16", |v| match v {
            IntValue::U16(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U16),
        IntegerClass::UInt32 => collect_integer_values(values, "uint32", |v| match v {
            IntValue::U32(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U32),
        IntegerClass::UInt64 => collect_integer_values(values, "uint64", |v| match v {
            IntValue::U64(v) => Some(v),
            _ => None,
        })
        .map(IntegerStorage::U64),
    }
}

fn collect_integer_values<T>(
    values: Vec<IntValue>,
    expected: &str,
    convert: impl Fn(IntValue) -> Option<T>,
) -> Result<Vec<T>, String> {
    values
        .into_iter()
        .map(|value| {
            let actual = value.class_name();
            convert(value).ok_or_else(|| {
                format!("integer label invariant expected {expected}, found {actual}")
            })
        })
        .collect()
}

pub(super) fn labels_value(labels: &[ClassLabel], kind: LabelKind) -> BuiltinResult<Value> {
    predicted_labels_value(
        labels,
        kind,
        &(0..labels.len()).collect::<Vec<_>>(),
        labels.len(),
        1,
    )
}

pub(super) fn predicted_labels_value(
    labels: &[ClassLabel],
    kind: LabelKind,
    indices: &[usize],
    rows: usize,
    cols: usize,
) -> BuiltinResult<Value> {
    let shape = vec![rows, cols];
    match kind {
        LabelKind::Numeric => Ok(Value::Tensor(
            Tensor::new(
                indices
                    .iter()
                    .map(|idx| match labels.get(*idx) {
                        Some(ClassLabel::Numeric(value)) => *value,
                        _ => f64::NAN,
                    })
                    .collect(),
                shape,
            )
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
        )),
        LabelKind::Integer(class) => {
            let values = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Integer(value)) => Ok(value.clone()),
                    Some(_) => Err(predict_internal(
                        "predict: integer class metadata does not match stored labels",
                    )),
                    None => Err(predict_internal(
                        "predict: predicted class index is out of range",
                    )),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Value::Tensor(
                Tensor::new_integer(
                    integer_storage(class, values).map_err(predict_internal)?,
                    shape,
                )
                .map_err(|err| predict_internal(format!("predict: {err}")))?,
            ))
        }
        LabelKind::Text => Ok(Value::StringArray(
            StringArray::from_elements_unbounded(
                indices
                    .iter()
                    .map(|idx| match labels.get(*idx) {
                        Some(ClassLabel::Text(value)) => value.clone(),
                        _ => StringElement::default(),
                    })
                    .collect(),
                shape,
            )
            .map_err(predict_internal)?,
        )),
        LabelKind::Logical => Ok(Value::LogicalArray(
            LogicalArray::new(
                indices
                    .iter()
                    .map(|idx| match labels.get(*idx) {
                        Some(ClassLabel::Logical(value)) if *value => 1,
                        _ => 0,
                    })
                    .collect(),
                shape,
            )
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
        )),
    }
}

pub(super) fn score_transform(spec: &FitSpec) -> ScoreTransform {
    spec.options
        .score_transform
        .unwrap_or(match spec.options.learner {
            Learner::Logistic => ScoreTransform::Logit,
            Learner::Svm => ScoreTransform::None,
        })
}

pub(super) fn score_transform_property(object: &ObjectInstance) -> BuiltinResult<ScoreTransform> {
    match object.properties().get("ScoreTransform") {
        Some(Value::String(runmat_value::StringElement::Text(text)))
            if text.eq_ignore_ascii_case("none") =>
        {
            Ok(ScoreTransform::None)
        }
        Some(Value::String(runmat_value::StringElement::Text(text)))
            if text.eq_ignore_ascii_case("logit") =>
        {
            Ok(ScoreTransform::Logit)
        }
        _ => Err(predict_invalid(
            "predict: model has unsupported ScoreTransform",
        )),
    }
}

pub(super) fn string_array_property(
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<Vec<String>> {
    match object.properties().get(name) {
        Some(Value::StringArray(array)) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    predict_invalid("predict: model text metadata must be nonmissing")
                })
            })
            .collect(),
        Some(Value::String(runmat_value::StringElement::Text(text))) => {
            Ok(vec![text.as_str().to_owned()])
        }
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

pub(super) fn tensor_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Tensor> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| predict_internal(format!("predict: {err}"))),
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

pub(super) fn numeric_vector_property(
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<Vec<f64>> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor)) => vector_values_predict(tensor, name),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

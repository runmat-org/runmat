use runmat_value::{ComplexTensor, IntValue, ObjectInstance, Tensor, Value};

use runmat_builtins::fea_contract::identities;

use crate::analysis::{
    analysis_field_page_by_run_id_op, AnalysisFieldPageResult, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection, ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT,
};
use crate::operations::OperationContext;
use crate::BuiltinResult;

use super::{
    builtin_error, empty_double_value, ensure_value_classes_registered, operation_error,
    usize_from_value, ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION, FIELD_PAGE_NAME,
};

pub(super) async fn create_page_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 3 {
        return Err(builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INPUT,
            "fea.fieldPage requires a field and optional one-based start and count",
        ));
    }
    let Value::Object(field) = &args[0] else {
        return Err(builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INPUT,
            "fea.fieldPage requires an fea.Field",
        ));
    };
    if !field.class_name().is(identities::FEA_FIELD_CLASS) {
        return Err(builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INPUT,
            format!("expected fea.Field, got {}", field.class_name()),
        ));
    }
    let run_id = required_nonempty_string(field, "run_id")?;
    let start = args
        .get(1)
        .map(|value| usize_from_value(FIELD_PAGE_NAME, value))
        .transpose()?
        .unwrap_or(1);
    let count = args
        .get(2)
        .map(|value| usize_from_value(FIELD_PAGE_NAME, value))
        .transpose()?
        .unwrap_or(ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT);
    if start == 0 || count == 0 {
        return Err(builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INPUT,
            "field page start and count must be positive integers",
        ));
    }
    let selection = selection_from_field_object(FIELD_PAGE_NAME, field)?;
    let page = super::artifact_access::read(FIELD_PAGE_NAME, || {
        analysis_field_page_by_run_id_op(
            run_id,
            selection,
            AnalysisFieldRequestOptions {
                offset: Some(start - 1),
                limit: Some(count),
            },
            OperationContext::new(None, None),
        )
        .map_err(|error| operation_error(FIELD_PAGE_NAME, &ERROR_OPERATION, error))
    })
    .await?
    .data;
    page_to_value(
        page,
        field
            .properties()
            .get("sample")
            .cloned()
            .unwrap_or_else(empty_double_value),
    )
}

pub(super) fn selection_from_field_object(
    builtin: &'static str,
    field: &ObjectInstance,
) -> BuiltinResult<AnalysisFieldSelection> {
    let field_id = required_nonempty_string_for_builtin(builtin, field, "field_id")?;
    let sample = match field.properties().get("sample") {
        Some(value @ Value::Object(sample))
            if sample.class_name().is(identities::FEA_FREQUENCY_CLASS) =>
        {
            let hertz = super::values::quantity_scalar(
                builtin,
                value,
                super::values::FREQUENCY_CLASS,
                "hertz",
            )?;
            if !hertz.is_finite() || hertz <= 0.0 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.Field frequency sample must be finite and positive",
                ));
            }
            Some(runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(hertz))
        }
        Some(value @ Value::Object(sample))
            if sample.class_name().is(identities::FEA_TIME_CLASS) =>
        {
            let seconds = super::values::quantity_scalar(
                builtin,
                value,
                super::values::TIME_CLASS,
                "seconds",
            )?;
            if !seconds.is_finite() {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.Field time sample must be finite",
                ));
            }
            Some(runmat_analysis_fea::results::field::RecoveredFieldSample::TimeSeconds(seconds))
        }
        Some(Value::Tensor(tensor)) if tensor.is_empty() => None,
        None => None,
        Some(_) => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "fea.Field has an unsupported or malformed sample coordinate",
            ))
        }
    };
    Ok(AnalysisFieldSelection::new(field_id, sample))
}

fn page_to_value(page: AnalysisFieldPageResult, sample: Value) -> BuiltinResult<Value> {
    ensure_value_classes_registered();
    let start = page.offset.checked_add(1).ok_or_else(|| {
        builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INTERNAL,
            "field page start exceeds the public indexing range",
        )
    })?;
    let end = page.offset.checked_add(page.count).ok_or_else(|| {
        builtin_error(
            FIELD_PAGE_NAME,
            &ERROR_INTERNAL,
            "field page end exceeds the structural indexing range",
        )
    })?;
    let has_more = end < page.total_count;
    let next_start = if has_more {
        Some(start.checked_add(page.count).ok_or_else(|| {
            builtin_error(
                FIELD_PAGE_NAME,
                &ERROR_INTERNAL,
                "next field page start exceeds the public indexing range",
            )
        })?)
    } else {
        None
    };
    let representation = match &page.values {
        AnalysisFieldPageValues::Real(_) => {
            runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::Real
        }
        AnalysisFieldPageValues::ComplexInterleaved(_) => {
            runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::ComplexInterleaved
        }
    };
    let values = page_values_to_value(page.values, page.count)?;
    let mut object = crate::builtins::fea::object_api::ordinary_builder(
        FIELD_PAGE_NAME,
        super::classes::FEA_FIELD_PAGE_CLASS,
    )?;
    object.insert("run_id".into(), Value::String(page.run_id.into()));
    object.insert("field_id".into(), Value::String(page.field_id.into()));
    object.insert("sample".into(), sample);
    object.insert("values".into(), values);
    object.insert("start".into(), exact_usize_value(start)?);
    object.insert("count".into(), exact_usize_value(page.count)?);
    object.insert("total_count".into(), exact_usize_value(page.total_count)?);
    object.insert(
        "next_start".into(),
        next_start
            .map(exact_usize_value)
            .transpose()?
            .unwrap_or_else(empty_double_value),
    );
    object.insert("has_more".into(), Value::Bool(has_more));
    object.insert(
        "numeric_representation".into(),
        numeric_representation_value(Some(representation)),
    );
    Ok(Value::Object(object.finish()))
}

pub(super) fn numeric_representation_value(
    representation: Option<
        runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation,
    >,
) -> Value {
    match representation {
        Some(runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::Real) => {
            Value::String("real".into())
        }
        Some(
            runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::ComplexInterleaved,
        ) => Value::String("complex".into()),
        None => empty_double_value(),
    }
}

fn page_values_to_value(values: AnalysisFieldPageValues, count: usize) -> BuiltinResult<Value> {
    match values {
        AnalysisFieldPageValues::Real(values) => {
            if values.len() != count {
                return Err(builtin_error(
                    FIELD_PAGE_NAME,
                    &ERROR_INTERNAL,
                    "real field page value count differs from its logical count",
                ));
            }
            Tensor::new(values, vec![count, 1])
                .map(Value::Tensor)
                .map_err(|error| builtin_error(FIELD_PAGE_NAME, &ERROR_INTERNAL, error))
        }
        AnalysisFieldPageValues::ComplexInterleaved(values) => {
            let storage_count = count.checked_mul(2).ok_or_else(|| {
                builtin_error(
                    FIELD_PAGE_NAME,
                    &ERROR_INTERNAL,
                    "complex field page storage count overflows",
                )
            })?;
            if values.len() != storage_count {
                return Err(builtin_error(
                    FIELD_PAGE_NAME,
                    &ERROR_INTERNAL,
                    "complex field page storage count differs from twice its logical count",
                ));
            }
            let pairs = values
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect();
            ComplexTensor::new(pairs, vec![count, 1])
                .map(Value::ComplexTensor)
                .map_err(|error| builtin_error(FIELD_PAGE_NAME, &ERROR_INTERNAL, error))
        }
    }
}

fn exact_usize_value(value: usize) -> BuiltinResult<Value> {
    u64::try_from(value)
        .map(IntValue::U64)
        .map(Value::Int)
        .map_err(|_| {
            builtin_error(
                FIELD_PAGE_NAME,
                &ERROR_INTERNAL,
                "field page integer exceeds the portable uint64 range",
            )
        })
}

fn required_nonempty_string<'a>(
    object: &'a ObjectInstance,
    property: &str,
) -> BuiltinResult<&'a str> {
    required_nonempty_string_for_builtin(FIELD_PAGE_NAME, object, property)
}

fn required_nonempty_string_for_builtin<'a>(
    builtin: &'static str,
    object: &'a ObjectInstance,
    property: &str,
) -> BuiltinResult<&'a str> {
    match object.properties().get(property) {
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty() => {
            Ok(value)
        }
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("fea.Field is missing a non-empty {property}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};

    #[test]
    fn page_values_preserve_real_and_complex_numeric_semantics() {
        let Value::Tensor(real) =
            page_values_to_value(AnalysisFieldPageValues::Real(vec![1.0, 2.0]), 2)
                .expect("real page")
        else {
            panic!("real page must be a tensor")
        };
        assert_eq!(real.shape(), vec![2, 1]);
        assert_eq!(real.materialize_f64(), vec![1.0, 2.0]);

        let Value::ComplexTensor(complex) = page_values_to_value(
            AnalysisFieldPageValues::ComplexInterleaved(vec![1.0, -2.0, 3.0, 4.0]),
            2,
        )
        .expect("complex page") else {
            panic!("complex page must be a complex tensor")
        };
        assert_eq!(complex.shape(), vec![2, 1]);
        assert_eq!(complex.materialize_f64(), vec![(1.0, -2.0), (3.0, 4.0)]);
    }

    #[test]
    fn page_values_reject_storage_count_mismatches() {
        assert!(page_values_to_value(AnalysisFieldPageValues::Real(vec![1.0]), 2).is_err());
        assert!(page_values_to_value(
            AnalysisFieldPageValues::ComplexInterleaved(vec![1.0, 2.0, 3.0]),
            2,
        )
        .is_err());
    }

    #[test]
    fn complex_page_exposes_one_based_progress_without_flattening_phase() {
        let mut descriptor = crate::analysis::AnalysisFieldDescriptor::from_field(
            &runmat_analysis_core::AnalysisField::host_f64(
                "structural.displacement",
                AnalysisFieldLayout::scalar(
                    ObservationQuantity::Displacement,
                    FieldLocation::Global,
                ),
                vec![4],
                vec![0.0; 4],
            ),
        );
        descriptor.numeric_representation = Some(
            runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::ComplexInterleaved,
        );
        let sample = super::super::values::quantity_to_value(
            FIELD_PAGE_NAME,
            super::super::values::FREQUENCY_CLASS,
            "hertz",
            40.0,
        )
        .expect("frequency sample");
        let page = page_to_value(
            AnalysisFieldPageResult {
                run_id: "run-frequency".into(),
                field_id: "structural.displacement".into(),
                descriptor,
                values: AnalysisFieldPageValues::ComplexInterleaved(vec![1.0, -2.0, 3.0, 4.0]),
                offset: 1,
                count: 2,
                total_count: 4,
                truncated: true,
            },
            sample,
        )
        .expect("complex field page");
        let Value::Object(page) = page else {
            panic!("expected page object")
        };
        assert_eq!(
            page.properties().get("start"),
            Some(&Value::Int(IntValue::U64(2)))
        );
        assert_eq!(
            page.properties().get("next_start"),
            Some(&Value::Int(IntValue::U64(4)))
        );
        assert_eq!(page.properties().get("has_more"), Some(&Value::Bool(true)));
        assert_eq!(
            page.properties().get("numeric_representation"),
            Some(&Value::String("complex".into()))
        );
        let Some(Value::ComplexTensor(values)) = page.properties().get("values") else {
            panic!("expected native complex values")
        };
        assert_eq!(values.materialize_f64(), vec![(1.0, -2.0), (3.0, 4.0)]);
    }
}

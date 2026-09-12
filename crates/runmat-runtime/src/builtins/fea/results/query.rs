use crate::analysis::analysis_results_by_run_id_op;
use crate::analysis::analysis_run_study_op;
use crate::analysis::AnalysisResultsQuery;
use crate::analysis::AnalysisStudySpec;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    FEA_PAYLOAD_JSON_PROPERTY, FEA_RESULTS_CLASS, FEA_RUN_ID_CONTEXT_PROPERTY,
    FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS, FEA_STUDY_CONTEXT_JSON_PROPERTY,
    FEA_STUDY_SPEC_JSON_PROPERTY, RESULTS_NAME, RUN_NAME,
};
use crate::builtins::fea::errors::{builtin_error, builtin_error_with_source, operation_error};
use crate::builtins::fea::geometry::{object_json_property, scalar_string};
use crate::builtins::fea::integer_serialization::serializable_to_object_value;
use crate::builtins::fea::options_json::expect_name_value_tail;
use crate::builtins::fea::value_decode::{
    exact_bool_from_value, one_based_usize_vec_from_value, string_vec_from_value,
};
use crate::operations::OperationContext;
use crate::BuiltinResult;
use runmat_value::ObjectInstance;
use runmat_value::Value;
use std::collections::HashSet;

pub(in crate::builtins::fea) fn results_query_from_args(
    args: &[Value],
) -> BuiltinResult<AnalysisResultsQuery> {
    let mut query = AnalysisResultsQuery::default();
    let mut seen = HashSet::new();
    for pair in expect_name_value_tail(RESULTS_NAME, args)? {
        let canonical = match pair.key.as_str() {
            "includefields" | "fields" => "includefields",
            "includefieldvalues" | "fieldvalues" => "includefieldvalues",
            other => other,
        };
        if !seen.insert(canonical.to_string()) {
            return Err(builtin_error(
                RESULTS_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.results option `{canonical}`"),
            ));
        }
        match pair.key.as_str() {
            "includefields" | "fields" => {
                query.include_fields = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includefieldvalues" | "fieldvalues" => {
                query.include_field_values = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includediagnostics" => {
                query.include_diagnostics = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "diagnosticcodes" => {
                query.diagnostic_codes = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includemodalresults" => {
                query.include_modal_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "modeindices" => {
                query.mode_indices = one_based_usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includetransientresults" => {
                query.include_transient_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "transientsnapshotindices" => {
                query.transient_snapshot_indices =
                    one_based_usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includenonlinearresults" => {
                query.include_nonlinear_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includeelectromagneticresults" => {
                query.include_electromagnetic_results =
                    exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    RESULTS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.results option `{other}`"),
                ));
            }
        }
    }
    Ok(query)
}

pub(in crate::builtins::fea) fn run_id_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.class_name == FEA_RUN_RESULT_CLASS => {
            run_id_from_object(object).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.RunResult does not contain a run_id; sweep results expose run_entries",
                )
            })
        }
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            scalar_string(value, builtin, &ERROR_INPUT)
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected run id string or fea.RunResult; got {other:?}"),
        )),
    }
}

pub(in crate::builtins::fea) fn run_id_from_object(object: &ObjectInstance) -> Option<String> {
    object
        .properties
        .get(FEA_RUN_ID_CONTEXT_PROPERTY)
        .or_else(|| object.properties.get("run_id"))
        .or_else(|| object.properties.get("runId"))
        .and_then(|value| match value {
            Value::String(run_id) => Some(run_id.clone()),
            _ => None,
        })
}

pub(in crate::builtins::fea) fn results_data_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisResultsData> {
    match value {
        Value::Object(object) if object.class_name == FEA_RESULTS_CLASS => {
            object_json_property(builtin, object, FEA_PAYLOAD_JSON_PROPERTY, &ERROR_INPUT)
        }
        _ => {
            let run_id = run_id_from_value(builtin, value)?;
            analysis_results_by_run_id_op(
                &run_id,
                AnalysisResultsQuery::default(),
                OperationContext::new(None, None),
            )
            .map(|envelope| envelope.data)
            .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))
        }
    }
}

pub(in crate::builtins::fea) fn run_study_result_to_object(
    spec: &AnalysisStudySpec,
) -> BuiltinResult<Value> {
    let envelope = analysis_run_study_op(spec, OperationContext::new(None, None))
        .map_err(|err| operation_error(RUN_NAME, &ERROR_OPERATION, err))?;
    let mut object = serializable_to_object_value(
        RUN_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_RESULT_CLASS,
        &envelope.data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )?;
    object.properties.insert(
        FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    object.properties.insert(
        "run_id".to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    object.properties.insert(
        "runId".to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    insert_study_context(&mut object, spec)?;
    Ok(Value::Object(object))
}

pub(in crate::builtins::fea) fn insert_study_context(
    object: &mut ObjectInstance,
    spec: &AnalysisStudySpec,
) -> BuiltinResult<()> {
    let json = serde_json::to_string(spec).map_err(|err| {
        builtin_error_with_source(RUN_NAME, &ERROR_INTERNAL, err.to_string(), err)
    })?;
    object.properties.insert(
        FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
        Value::String(json),
    );
    Ok(())
}

pub(in crate::builtins::fea) fn copy_study_context_property(
    source: &Value,
    target: &mut ObjectInstance,
) {
    if let Some(json) = study_context_json_from_value(source) {
        target.properties.insert(
            FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
            Value::String(json),
        );
    }
}

pub(in crate::builtins::fea) fn copy_run_id_context_property(
    source: &Value,
    target: &mut ObjectInstance,
) {
    if let Some(run_id) = run_id_context_from_value(source) {
        target.properties.insert(
            FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
            Value::String(run_id.clone()),
        );
        target
            .properties
            .entry("run_id".to_string())
            .or_insert(Value::String(run_id.clone()));
        target
            .properties
            .entry("runId".to_string())
            .or_insert(Value::String(run_id));
    }
}

pub(in crate::builtins::fea) fn study_context_json_from_value(value: &Value) -> Option<String> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name == FEA_STUDY_CLASS {
        if let Some(Value::String(json)) = object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY) {
            return Some(json.clone());
        }
    }
    object
        .properties
        .get(FEA_STUDY_CONTEXT_JSON_PROPERTY)
        .and_then(|value| match value {
            Value::String(json) => Some(json.clone()),
            _ => None,
        })
}

pub(in crate::builtins::fea) fn study_context_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisStudySpec> {
    let Some(json) = study_context_json_from_value(value) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin}: FEA plot requires study geometry context; pass a fea.RunResult from fea.run(study), a derived fea.Results/fea.Field, or call fea.plot(study, runId, fieldId)"),
        ));
    };
    serde_json::from_str(&json)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INPUT, err.to_string(), err))
}

pub(in crate::builtins::fea) fn run_id_context_from_value(value: &Value) -> Option<String> {
    let Value::Object(object) = value else {
        return None;
    };
    run_id_from_object(object)
}

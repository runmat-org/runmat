use crate::analysis::analysis_results_by_run_id_op;
use crate::analysis::AnalysisFieldDescriptor;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    FEA_PAYLOAD_JSON_PROPERTY, FEA_RESULTS_CLASS, FEA_RUN_ID_CONTEXT_PROPERTY, FIELD_NAME,
    RESULTS_NAME,
};
use crate::builtins::fea::errors::{builtin_error, operation_error};
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::integer_serialization::serializable_to_object_preserving_integers;
use crate::builtins::fea::results::field::{field_to_object, find_descriptor, find_field};
use crate::builtins::fea::results::query::{
    copy_run_id_context_property, copy_study_context_property, results_data_from_value,
    results_query_from_args, run_id_from_value,
};
use crate::operations::OperationContext;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_results_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RESULTS_NAME,
            &ERROR_INPUT,
            "fea.results requires a run id or fea.RunResult",
        ));
    }
    if let Value::Object(object) = &args[0] {
        if object.class_name == FEA_RESULTS_CLASS && args.len() == 1 {
            return Ok(args[0].clone());
        }
    }
    let run_id = run_id_from_value(RESULTS_NAME, &args[0])?;
    let query = results_query_from_args(&args[1..])?;
    let envelope = analysis_results_by_run_id_op(&run_id, query, OperationContext::new(None, None))
        .map_err(|err| operation_error(RESULTS_NAME, &ERROR_OPERATION, err))?;
    let mut public_data = envelope.data;
    for index in &mut public_data.summary.available_mode_indices {
        *index = index.checked_add(1).ok_or_else(|| {
            builtin_error(
                RESULTS_NAME,
                &ERROR_INTERNAL,
                "available mode index cannot be represented at the one-based public boundary",
            )
        })?;
    }
    let value = serializable_to_object_preserving_integers(
        RESULTS_NAME,
        &ERROR_INTERNAL,
        FEA_RESULTS_CLASS,
        &public_data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &[
            "shape",
            "element_count",
            "component_count",
            "size_bytes",
            "solver_host_sync_count",
            "field_count",
            "total_elements",
            "mode_count",
            "available_mode_indices",
            "snapshot_count",
            "increment_count",
            "failed_increment_count",
            "max_nonlinear_iteration_count",
            "nonlinear_line_search_backtracks",
            "nonlinear_max_backtracks_per_increment",
            "nonlinear_tangent_rebuild_count",
            "nonlinear_iteration_spike_count",
            "nonlinear_convergence_stall_count",
            "nonlinear_backtrack_burst_count",
            "prep_calibration_fingerprint",
            "prep_acceptance_fingerprint",
            "thermo_coupling_fingerprint",
            "electro_thermal_coupling_fingerprint",
            "iteration_counts",
            "failed_increments",
            "line_search_backtracks",
            "max_line_search_backtracks_per_increment",
            "tangent_rebuild_count",
            "iteration_spike_count",
            "convergence_stall_count",
            "backtrack_burst_count",
        ],
    )?;
    let Value::Object(mut object) = value else {
        unreachable!("integer-preserving FEA result serialization returns an object")
    };
    object
        .properties
        .insert("run_id".to_string(), Value::String(run_id.clone()));
    object.properties.insert(
        FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
        Value::String(run_id),
    );
    copy_study_context_property(&args[0], &mut object);
    Ok(Value::Object(object))
}

pub(in crate::builtins::fea) fn create_field_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            "fea.field requires results/run input and field id",
        ));
    }
    let field_id = scalar_string(&args[1], FIELD_NAME, &ERROR_INPUT)?;
    let results = results_data_from_value(FIELD_NAME, &args[0])?;
    let field = find_field(results.fields.into_iter(), &field_id).ok_or_else(|| {
        builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            format!("FEA field `{field_id}` was not found in results"),
        )
    })?;
    let descriptor = find_descriptor(results.field_descriptors.iter(), &field_id)
        .cloned()
        .unwrap_or_else(|| AnalysisFieldDescriptor::from_field(&field));
    let mut object = field_to_object(&field, &descriptor)?;
    copy_study_context_property(&args[0], &mut object);
    copy_run_id_context_property(&args[0], &mut object);
    Ok(Value::Object(object))
}

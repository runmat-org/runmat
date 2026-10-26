use super::*;

#[runtime_builtin(
    name = "fea.stepResult",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::run_results"
)]
pub async fn step_result(run: Value, step_id: String) -> BuiltinResult<Value> {
    let aggregate: crate::analysis::AnalysisStudyRunData =
        object_payload("fea.stepResult", &run, FEA_RUN_RESULT_CLASS)?;
    let step = aggregate
        .steps
        .into_iter()
        .find(|step| step.procedure_step.step_id.0 == step_id)
        .ok_or_else(|| {
            builtin_error(
                "fea.stepResult",
                &ERROR_INPUT,
                format!("FEA run has no procedure step `{step_id}`"),
            )
        })?;
    let mut object = serializable_to_object_value(
        "fea.stepResult",
        &ERROR_INTERNAL,
        FEA_STEP_RUN_RESULT_CLASS,
        &step,
    )?;
    copy_study_context_property(&run, &mut object)?;
    Ok(Value::Object(object))
}

/// Wraps an aggregate run payload as the public `fea.RunResult` object, as
/// `fea.run` does, for tests that exercise step selection.
#[cfg(test)]
pub(crate) fn run_result_object_for_tests(data: &crate::analysis::AnalysisStudyRunData) -> Value {
    serializable_to_object_value("fea.run", &ERROR_INTERNAL, FEA_RUN_RESULT_CLASS, data)
        .map(Value::Object)
        .expect("run payload should convert to fea.RunResult")
}

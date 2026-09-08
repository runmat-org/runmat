use super::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "run",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Aggregate fea.RunResult returned by fea.run.",
    },
    BuiltinParamDescriptor {
        name: "stepId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact authored procedure-step id.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "stepRun = fea.stepResult(run, stepId)",
    inputs: &INPUTS,
    outputs: &OUT_ANY,
}];

pub(super) const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.stepResult",
    category = "fea",
    summary = "Select one typed procedure-step result from an aggregate FEA run.",
    keywords = "fea,run,result,step,sequence",
    descriptor(crate::builtins::fea::run_results::DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    copy_study_context_property(&run, &mut object);
    Ok(Value::Object(object))
}

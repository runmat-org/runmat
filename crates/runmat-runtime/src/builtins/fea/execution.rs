use std::sync::Once;

use runmat_analysis_core::{FormulationRequirement, ProcedureStep};
use runmat_analysis_fea::ComputeBackend;

use super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, object_or_cell, register_immutable_class, to_typed_value, VALUE_FIELD,
};

const BACKEND_CLASS: &str = "fea.ComputeBackend";
const REQUIREMENT_CLASS: &str = "fea.FormulationRequirement";
const SOLVER_OPTIONS_CLASS: &str = "fea.SolverOptions";
const EXECUTION_STEP_CLASS: &str = "fea.ExecutionStep";
const EXECUTION_PLAN_CLASS: &str = "fea.ExecutionPlan";

const EXECUTION_STEP_FIELDS: &[&str] = &["step_id", "backend", "requirement", "solver_options"];
const EXECUTION_PLAN_FIELDS: &[&str] = &["steps"];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const STEP_INPUT: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "Typed family-specific procedure step.",
)];
const DEFAULT_PLAN_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "sequence",
        BuiltinParamType::Any,
        "Typed analysis sequence.",
    ),
    required("backend", BuiltinParamType::Any, "Typed compute backend."),
    required(
        "requirement",
        BuiltinParamType::Any,
        "Typed formulation requirement.",
    ),
];
const STEP_EXECUTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required("step", BuiltinParamType::Any, "Typed procedure step."),
    required("backend", BuiltinParamType::Any, "Typed compute backend."),
    required(
        "requirement",
        BuiltinParamType::Any,
        "Typed formulation requirement.",
    ),
    required(
        "solverOptions",
        BuiltinParamType::Any,
        "Typed solver options compatible with the procedure step.",
    ),
];
const PLAN_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "sequence",
        BuiltinParamType::Any,
        "Typed analysis sequence.",
    ),
    required(
        "steps",
        BuiltinParamType::Any,
        "Typed execution step or cell array.",
    ),
];

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUT_ANY,
        }];
        const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    BACKEND_CPU_DESCRIPTOR,
    BACKEND_CPU_SIGNATURES,
    "backend = fea.ComputeBackend.cpu()",
    &NO_INPUTS
);
descriptor!(
    BACKEND_GPU_DESCRIPTOR,
    BACKEND_GPU_SIGNATURES,
    "backend = fea.ComputeBackend.gpu()",
    &NO_INPUTS
);
descriptor!(
    REQUIREMENT_PRODUCTION_DESCRIPTOR,
    REQUIREMENT_PRODUCTION_SIGNATURES,
    "requirement = fea.FormulationRequirement.production()",
    &NO_INPUTS
);
descriptor!(
    REQUIREMENT_EXPERIMENTAL_DESCRIPTOR,
    REQUIREMENT_EXPERIMENTAL_SIGNATURES,
    "requirement = fea.FormulationRequirement.experimental()",
    &NO_INPUTS
);
descriptor!(
    SOLVER_DEFAULTS_DESCRIPTOR,
    SOLVER_DEFAULTS_SIGNATURES,
    "options = fea.SolverOptions.defaults(step)",
    &STEP_INPUT
);
descriptor!(
    EXECUTION_STEP_DESCRIPTOR,
    EXECUTION_STEP_SIGNATURES,
    "execution = fea.ExecutionStep.new(step, backend, requirement, solverOptions)",
    &STEP_EXECUTION_INPUTS
);
descriptor!(
    EXECUTION_PLAN_DESCRIPTOR,
    EXECUTION_PLAN_SIGNATURES,
    "plan = fea.ExecutionPlan.new(sequence, steps)",
    &PLAN_INPUTS
);
descriptor!(
    EXECUTION_DEFAULT_PLAN_DESCRIPTOR,
    EXECUTION_DEFAULT_PLAN_SIGNATURES,
    "plan = fea.ExecutionPlan.defaults(sequence, backend, requirement)",
    &DEFAULT_PLAN_INPUTS
);

const STRUCTURAL_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Execution authoring accepts typed immutable study values only.",
};

macro_rules! choice {
    ($function:ident, $name:literal, $descriptor:ident, $class:expr, $value:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed FEA execution policy value.", keywords = "fea,execution,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::execution")]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($name, $class, $value)
        }
    };
}

choice!(
    backend_cpu,
    "fea.ComputeBackend.cpu",
    BACKEND_CPU_DESCRIPTOR,
    BACKEND_CLASS,
    ComputeBackend::Cpu
);
choice!(
    backend_gpu,
    "fea.ComputeBackend.gpu",
    BACKEND_GPU_DESCRIPTOR,
    BACKEND_CLASS,
    ComputeBackend::Gpu
);
choice!(
    requirement_production,
    "fea.FormulationRequirement.production",
    REQUIREMENT_PRODUCTION_DESCRIPTOR,
    REQUIREMENT_CLASS,
    FormulationRequirement::Production
);
choice!(
    requirement_experimental,
    "fea.FormulationRequirement.experimental",
    REQUIREMENT_EXPERIMENTAL_DESCRIPTOR,
    REQUIREMENT_CLASS,
    FormulationRequirement::Experimental
);

fn procedure_step(name: &'static str, value: &Value) -> BuiltinResult<ProcedureStep> {
    let mut steps = super::modeling::step_vec_from_value(name, value)?;
    if steps.len() != 1 {
        return Err(builtin_error(
            name,
            &ERROR_INPUT,
            "expected exactly one typed procedure step",
        ));
    }
    Ok(steps.remove(0))
}

#[runtime_builtin(
    name = "fea.SolverOptions.defaults",
    category = "fea",
    summary = "Create solver options selected by a typed procedure step.",
    keywords = "fea,execution,solver,options,typed",
    descriptor(SOLVER_DEFAULTS_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn solver_defaults(step: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SolverOptions.defaults";
    let step = procedure_step(NAME, &step)?;
    let options = AnalysisSolverOptions::defaults_for(&step)
        .map_err(|message| builtin_error(NAME, &ERROR_INPUT, message))?;
    to_typed_value(NAME, SOLVER_OPTIONS_CLASS, options)
}

#[runtime_builtin(
    name = "fea.ExecutionStep.new",
    category = "fea",
    summary = "Bind a typed procedure step to backend, maturity, and solver options.",
    keywords = "fea,execution,step,backend,solver,typed",
    descriptor(EXECUTION_STEP_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn execution_step(
    step: Value,
    backend: Value,
    requirement: Value,
    options: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ExecutionStep.new";
    let step = procedure_step(NAME, &step)?;
    let options: AnalysisSolverOptions = from_typed_value(NAME, SOLVER_OPTIONS_CLASS, &options)?;
    if !options.supports(&step) {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "solver options do not support the supplied procedure step",
        ));
    }
    let execution = AnalysisExecutionStep {
        step_id: step.step_id,
        backend: from_typed_value(NAME, BACKEND_CLASS, &backend)?,
        requirement: from_typed_value(NAME, REQUIREMENT_CLASS, &requirement)?,
        solver_options: options,
    };
    serializable_to_object(NAME, &ERROR_INTERNAL, EXECUTION_STEP_CLASS, &execution)
}

#[runtime_builtin(
    name = "fea.ExecutionPlan.new",
    category = "fea",
    summary = "Create a typed execution plan for an analysis sequence.",
    keywords = "fea,execution,plan,sequence,typed",
    descriptor(EXECUTION_PLAN_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn execution_plan(sequence: Value, steps: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ExecutionPlan.new";
    let sequence = super::sequences::sequence_from_value(NAME, &sequence)?;
    let plan = AnalysisExecutionPlan {
        steps: object_or_cell(NAME, &steps, EXECUTION_STEP_CLASS)?,
    };
    plan.validate_against(&sequence)
        .map_err(|message| builtin_error(NAME, &ERROR_INPUT, message))?;
    serializable_to_object(NAME, &ERROR_INTERNAL, EXECUTION_PLAN_CLASS, &plan)
}

#[runtime_builtin(
    name = "fea.ExecutionPlan.defaults",
    category = "fea",
    summary = "Create default solver options for every step in a typed sequence.",
    keywords = "fea,execution,plan,sequence,defaults,typed",
    descriptor(EXECUTION_DEFAULT_PLAN_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn execution_plan_defaults(
    sequence: Value,
    backend: Value,
    requirement: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ExecutionPlan.defaults";
    let sequence = super::sequences::sequence_from_value(NAME, &sequence)?;
    let plan = AnalysisExecutionPlan::for_sequence_with_requirement(
        &sequence,
        from_typed_value(NAME, BACKEND_CLASS, &backend)?,
        from_typed_value(NAME, REQUIREMENT_CLASS, &requirement)?,
    )
    .map_err(|message| builtin_error(NAME, &ERROR_INPUT, message))?;
    serializable_to_object(NAME, &ERROR_INTERNAL, EXECUTION_PLAN_CLASS, &plan)
}

pub(super) fn execution_plan_from_value(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisExecutionPlan> {
    object_payload(name, value, EXECUTION_PLAN_CLASS)
}

pub(super) fn ensure_execution_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            BACKEND_CLASS,
            "runmat.fea.compute-backend",
            VALUE_FIELD,
            &[
                ("cpu", "fea.ComputeBackend.cpu"),
                ("gpu", "fea.ComputeBackend.gpu"),
            ],
        );
        register_immutable_class(
            REQUIREMENT_CLASS,
            "runmat.fea.formulation-requirement",
            VALUE_FIELD,
            &[
                ("production", "fea.FormulationRequirement.production"),
                ("experimental", "fea.FormulationRequirement.experimental"),
            ],
        );
        register_immutable_class(
            SOLVER_OPTIONS_CLASS,
            "runmat.fea.solver-options",
            VALUE_FIELD,
            &[("defaults", "fea.SolverOptions.defaults")],
        );
        register_immutable_class(
            EXECUTION_STEP_CLASS,
            "runmat.fea.execution-step",
            EXECUTION_STEP_FIELDS,
            &[("new", "fea.ExecutionStep.new")],
        );
        register_immutable_class(
            EXECUTION_PLAN_CLASS,
            "runmat.fea.execution-plan",
            EXECUTION_PLAN_FIELDS,
            &[
                ("new", "fea.ExecutionPlan.new"),
                ("defaults", "fea.ExecutionPlan.defaults"),
            ],
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::CellArray;

    #[tokio::test]
    async fn typed_execution_plan_covers_each_sequence_step() {
        super::super::ensure_value_classes_registered();
        let first = super::super::steps::structural_static_builtin("first".into())
            .await
            .unwrap();
        let second = super::super::steps::structural_static_builtin("second".into())
            .await
            .unwrap();
        let sequence = super::super::sequences::sequence(vec![
            "sequence".into(),
            Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
        ])
        .await
        .unwrap();
        let plan = execution_plan_defaults(
            sequence.clone(),
            backend_cpu().await.unwrap(),
            requirement_experimental().await.unwrap(),
        )
        .await
        .unwrap();
        let plan = execution_plan_from_value("test", &plan).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert!(plan
            .steps
            .iter()
            .all(|step| step.requirement == FormulationRequirement::Experimental));
    }

    #[tokio::test]
    async fn execution_step_rejects_solver_options_for_another_procedure() {
        super::super::ensure_value_classes_registered();
        let static_step = super::super::steps::structural_static_builtin("static".into())
            .await
            .unwrap();
        let modal_step = super::super::steps::structural_modal_builtin("modal".into())
            .await
            .unwrap();
        let static_options = solver_defaults(static_step).await.unwrap();

        let error = execution_step(
            modal_step,
            backend_cpu().await.unwrap(),
            requirement_production().await.unwrap(),
            static_options,
        )
        .await
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("solver options do not support the supplied procedure step"));
    }

    #[tokio::test]
    async fn execution_plan_rejects_missing_sequence_steps() {
        super::super::ensure_value_classes_registered();
        let first = super::super::steps::structural_static_builtin("first".into())
            .await
            .unwrap();
        let second = super::super::steps::structural_static_builtin("second".into())
            .await
            .unwrap();
        let sequence = super::super::sequences::sequence(vec![
            "sequence".into(),
            Value::Cell(CellArray::new(vec![first.clone(), second], 1, 2).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
        ])
        .await
        .unwrap();
        let first_options = solver_defaults(first.clone()).await.unwrap();
        let first_execution = execution_step(
            first,
            backend_cpu().await.unwrap(),
            requirement_production().await.unwrap(),
            first_options,
        )
        .await
        .unwrap();

        let error = execution_plan(sequence, first_execution).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("exactly one entry for each sequence procedure step"));
    }

    #[test]
    fn study_rejects_retired_string_execution_options() {
        for option in ["Backend", "Requirement", "FormulationRequirement"] {
            let error = super::super::modeling::StudyConstructorOptions::parse(&[
                Value::String(option.to_string()),
                Value::String("cpu".to_string()),
            ])
            .unwrap_err();
            assert!(
                error.to_string().contains("unsupported fea.study option"),
                "{option} should reject through the breaking typed boundary: {error}"
            );
        }
    }
}

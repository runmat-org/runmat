use std::sync::Once;

use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::ComputeBackend;

use super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, object_or_cell, register_immutable_runtime_classes,
    register_immutable_schemas, to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};

use runmat_builtins::fea_contract::policy_types::classes;

const BACKEND_CLASS: &str = classes::COMPUTE_BACKEND.display_name();
const SOLVER_OPTIONS_CLASS: &str = classes::SOLVER_OPTIONS.display_name();
const EXECUTION_STEP_CLASS: &str = classes::EXECUTION_STEP.display_name();
const EXECUTION_PLAN_CLASS: &str = classes::EXECUTION_PLAN.display_name();

const EXECUTION_STEP_FIELDS: &[&str] = &[
    "step_id",
    "backend",
    "accelerator_admission",
    "solver_options",
];
const EXECUTION_PLAN_FIELDS: &[&str] = &["steps"];

macro_rules! choice {
    ($function:ident, $name:literal, $class:expr, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::execution")]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($name, $class, $value)
        }
    };
}

choice!(
    backend_cpu,
    "fea.ComputeBackend.cpu",
    BACKEND_CLASS,
    ComputeBackend::Cpu
);
choice!(
    backend_gpu,
    "fea.ComputeBackend.gpu",
    BACKEND_CLASS,
    ComputeBackend::Gpu
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
    binding_variant = "default",
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn execution_step(
    step: Value,
    backend: Value,
    options: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ExecutionStep.new";
    // `strict` is an optional trailing logical; omitted means fallback to the CPU reference.
    let strict = match (rest.len(), rest.first()) {
        (0, _) => false,
        (1, Some(strict)) => super::logical_from_value(NAME, strict)?,
        _ => {
            return Err(builtin_error(
                NAME,
                &ERROR_INPUT,
                "fea.ExecutionStep.new accepts at most four arguments",
            ))
        }
    };
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
        // `strict = true` is `accelerators = required`: a GPU step that cannot run on the
        // accelerator fails closed instead of falling back to cpu_reference.
        accelerator_admission: if strict {
            runmat_analysis_core::AcceleratorAdmissionPolicy::Required
        } else {
            runmat_analysis_core::AcceleratorAdmissionPolicy::FallbackToCpu
        },
        solver_options: options,
    };
    serializable_to_object(NAME, &ERROR_INTERNAL, EXECUTION_STEP_CLASS, &execution)
}

#[runtime_builtin(
    name = "fea.ExecutionPlan.new",
    binding_variant = "default",
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::execution"
)]
pub async fn execution_plan_defaults(sequence: Value, backend: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ExecutionPlan.defaults";
    let sequence = super::sequences::sequence_from_value(NAME, &sequence)?;
    let plan = AnalysisExecutionPlan::for_sequence(
        &sequence,
        from_typed_value(NAME, BACKEND_CLASS, &backend)?,
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

const EXECUTION_STUDY_CLASSES: [ImmutableStudyClass; 4] = [
    ImmutableStudyClass {
        class_name: BACKEND_CLASS,
        type_identity: "runmat.fea.compute-backend",
        fields: VALUE_FIELD,
        methods: &[
            ("cpu", "fea.ComputeBackend.cpu"),
            ("gpu", "fea.ComputeBackend.gpu"),
        ],
    },
    ImmutableStudyClass {
        class_name: SOLVER_OPTIONS_CLASS,
        type_identity: "runmat.fea.solver-options",
        fields: VALUE_FIELD,
        methods: &[("defaults", "fea.SolverOptions.defaults")],
    },
    ImmutableStudyClass {
        class_name: EXECUTION_STEP_CLASS,
        type_identity: "runmat.fea.execution-step",
        fields: EXECUTION_STEP_FIELDS,
        methods: &[("new", "fea.ExecutionStep.new")],
    },
    ImmutableStudyClass {
        class_name: EXECUTION_PLAN_CLASS,
        type_identity: "runmat.fea.execution-plan",
        fields: EXECUTION_PLAN_FIELDS,
        methods: &[
            ("new", "fea.ExecutionPlan.new"),
            ("defaults", "fea.ExecutionPlan.defaults"),
        ],
    },
];

pub(super) fn ensure_execution_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(classes::COMPUTE_BACKEND);
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &EXECUTION_STUDY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&EXECUTION_STUDY_CLASSES));
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
        let sequence = super::super::sequences::sequence(
            crate::builtins::common::test_support::builtin_arguments(vec![
                "sequence".into(),
                Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            ]),
        )
        .await
        .unwrap();
        let plan = execution_plan_defaults(sequence.clone(), backend_cpu().await.unwrap())
            .await
            .unwrap();
        let plan = execution_plan_from_value("test", &plan).unwrap();
        assert_eq!(plan.steps.len(), 2);
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
            static_options,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .await
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("solver options do not support the supplied procedure step"));
    }

    #[tokio::test]
    async fn execution_step_strict_argument_selects_the_required_admission_policy() {
        super::super::ensure_value_classes_registered();
        let step = super::super::steps::structural_static_builtin("static".into())
            .await
            .unwrap();
        let options = solver_defaults(step.clone()).await.unwrap();
        let fallback = runmat_analysis_core::AcceleratorAdmissionPolicy::FallbackToCpu;
        let required = runmat_analysis_core::AcceleratorAdmissionPolicy::Required;
        // Omitting `strict` keeps three-argument calls working and selects the fallback.
        for (rest, expected) in [
            (Vec::new(), fallback),
            (vec![Value::Bool(false)], fallback),
            (vec![Value::Bool(true)], required),
        ] {
            let value = execution_step(
                step.clone(),
                backend_gpu().await.unwrap(),
                options.clone(),
                crate::builtins::common::test_support::builtin_arguments(rest),
            )
            .await
            .unwrap();
            let execution: AnalysisExecutionStep =
                object_payload("test", &value, EXECUTION_STEP_CLASS).unwrap();
            assert_eq!(execution.accelerator_admission, expected);
            assert_eq!(execution.backend, ComputeBackend::Gpu);
        }
        for rest in [
            vec![Value::String("yes".into())],
            vec![Value::Bool(true), Value::Bool(true)],
        ] {
            assert!(execution_step(
                step.clone(),
                backend_gpu().await.unwrap(),
                options.clone(),
                crate::builtins::common::test_support::builtin_arguments(rest),
            )
            .await
            .is_err());
        }
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
        let sequence = super::super::sequences::sequence(
            crate::builtins::common::test_support::builtin_arguments(vec![
                "sequence".into(),
                Value::Cell(CellArray::new(vec![first.clone(), second], 1, 2).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
                Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()),
            ]),
        )
        .await
        .unwrap();
        let first_options = solver_defaults(first.clone()).await.unwrap();
        let first_execution = execution_step(
            first,
            backend_cpu().await.unwrap(),
            first_options,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
            let error = super::super::modeling::StudyConstructorOptions::parse(
                crate::call::FunctionArgumentView::from_slice(&[
                    Value::String(option.to_string().into()),
                    Value::String("cpu".to_string().into()),
                ]),
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("unsupported fea.study option"),
                "{option} should reject through the breaking typed boundary: {error}"
            );
        }
    }
}

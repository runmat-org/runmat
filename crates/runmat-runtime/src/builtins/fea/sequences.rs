use std::{collections::HashMap, sync::Once};

use runmat_analysis_core::{
    AnalysisSequence, IndependentDomain, ProcedureStep, ProcedureStepId, RestartPolicy,
    SequenceBranch, SequenceBranchId, SequenceDependency, StateId, StateInput, StateOutput,
    StateSource, StateTransition, TransitionId,
};

use super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::observations::study_outputs;
use crate::builtins::fea::study_policies::sequence as policy;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_runtime_classes,
    register_immutable_schemas, to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};
use crate::class_registry::declaration::static_methods;
use runmat_builtins::fea_contract::study_types::{
    classes::{
        FEA_ANALYSIS_SEQUENCE_CLASS, FEA_INDEPENDENT_DOMAIN_CLASS, FEA_PROCEDURE_STEP_CLASS,
        FEA_RESTART_POLICY_CLASS, FEA_SEQUENCE_BRANCH_CLASS, FEA_SEQUENCE_DEPENDENCY_CLASS,
        FEA_STATE_INPUT_CLASS, FEA_STATE_OUTPUT_CLASS, FEA_STATE_TRANSITION_CLASS,
    },
    descriptors::{
        analysis_sequence::*, independent_domain::*, procedure_step::*, restart_policy::*,
        sequence_branch::*, sequence_dependency::*, state_input::*, state_output::*,
        state_transition::*,
    },
};

pub(super) const INDEPENDENT_DOMAIN_CLASS: &str = FEA_INDEPENDENT_DOMAIN_CLASS.display_name();
pub(super) const RESTART_POLICY_CLASS: &str = FEA_RESTART_POLICY_CLASS.display_name();
pub(super) const STATE_INPUT_CLASS: &str = FEA_STATE_INPUT_CLASS.display_name();
pub(super) const STATE_OUTPUT_CLASS: &str = FEA_STATE_OUTPUT_CLASS.display_name();
pub(super) const DEPENDENCY_CLASS: &str = FEA_SEQUENCE_DEPENDENCY_CLASS.display_name();
pub(super) const TRANSITION_CLASS: &str = FEA_STATE_TRANSITION_CLASS.display_name();
pub(super) const BRANCH_CLASS: &str = FEA_SEQUENCE_BRANCH_CLASS.display_name();
pub(super) const SEQUENCE_CLASS: &str = FEA_ANALYSIS_SEQUENCE_CLASS.display_name();
pub(super) const PROCEDURE_STEP_CLASS: &str = FEA_PROCEDURE_STEP_CLASS.display_name();

const STATE_INPUT_FIELDS: &[&str] = &["input_id", "required_kind", "source", "use_as", "conflict"];
const STATE_OUTPUT_FIELDS: &[&str] = &["state_id", "kind", "completeness", "persistence"];
const DEPENDENCY_FIELDS: &[&str] = &["predecessor", "successor", "reason"];
const TRANSITION_FIELDS: &[&str] = &[
    "transition_id",
    "producer_step",
    "consumer_step",
    "state_id",
    "method",
    "topology",
    "reference",
];
const BRANCH_FIELDS: &[&str] = &["branch_id", "producer_step", "state_ids", "consumer_steps"];
const SEQUENCE_FIELDS: &[&str] = &[
    "sequence_id",
    "steps",
    "dependencies",
    "transitions",
    "branches",
];

#[runtime_builtin(
    name = "fea.IndependentDomain.staticDomain",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_static() -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_STATIC_DOMAIN_DESCRIPTOR,
        "fea.IndependentDomain.staticDomain",
    )?;
    to_typed_value(
        "fea.IndependentDomain.staticDomain",
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::Static,
    )
}

fn numeric_values(builtin: &'static str, value: &Value, field: &str) -> BuiltinResult<Vec<f64>> {
    let values = match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        Value::Num(value) => vec![*value],
        Value::Int(value) => vec![value.to_f64()],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{field} must be a host numeric scalar or array; got {other:?}"),
            ))
        }
    };
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{field} must contain finite values"),
        ));
    }
    Ok(values)
}

#[runtime_builtin(
    name = "fea.IndependentDomain.timeSeconds",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_time(
    origin: Value,
    start: Value,
    end: Value,
    interval_count: Value,
) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_TIME_SECONDS_DESCRIPTOR,
        "fea.IndependentDomain.timeSeconds",
    )?;
    const NAME: &str = "fea.IndependentDomain.timeSeconds";
    let interval_count = usize_from_value(NAME, &interval_count)?;
    if interval_count == 0 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "intervalCount must be positive",
        ));
    }
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::TimeSeconds {
            origin: values::finite_numeric_scalar(NAME, &origin, "origin")?,
            start: values::finite_numeric_scalar(NAME, &start, "start")?,
            end: values::finite_numeric_scalar(NAME, &end, "end")?,
            interval_count,
        },
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.frequencyHertz",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_frequency(values: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_FREQUENCY_HERTZ_DESCRIPTOR,
        "fea.IndependentDomain.frequencyHertz",
    )?;
    const NAME: &str = "fea.IndependentDomain.frequencyHertz";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::FrequencyHertz(numeric_values(NAME, &values, "values")?),
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.loadFactor",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_load(start: Value, end: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_LOAD_FACTOR_DESCRIPTOR,
        "fea.IndependentDomain.loadFactor",
    )?;
    const NAME: &str = "fea.IndependentDomain.loadFactor";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::LoadFactor {
            start: values::finite_numeric_scalar(NAME, &start, "start")?,
            end: values::finite_numeric_scalar(NAME, &end, "end")?,
        },
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.eigenvalue",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_eigenvalue(count: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_EIGENVALUE_DESCRIPTOR,
        "fea.IndependentDomain.eigenvalue",
    )?;
    const NAME: &str = "fea.IndependentDomain.eigenvalue";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::Eigenvalue {
            requested_count: usize_from_value(NAME, &count)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.studyParameter",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_parameter(id: String, values: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_STUDY_PARAMETER_DESCRIPTOR,
        "fea.IndependentDomain.studyParameter",
    )?;
    const NAME: &str = "fea.IndependentDomain.studyParameter";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::StudyParameter {
            parameter_id: nonempty_id(NAME, "parameter id", id)?,
            canonical_values: numeric_values(NAME, &values, "values")?,
        },
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.responseSpectra",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_response_spectra(spectra: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_INDEPENDENT_DOMAIN_RESPONSE_SPECTRA_DESCRIPTOR,
        "fea.IndependentDomain.responseSpectra",
    )?;
    const NAME: &str = "fea.IndependentDomain.responseSpectra";
    let spectra = super::response_spectra::response_spectrum_vec_from_value(NAME, &spectra)?;
    if spectra.is_empty() {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "spectra must contain at least one typed response spectrum",
        ));
    }
    let spectrum_ids = spectra
        .into_iter()
        .map(|spectrum| spectrum.spectrum_id)
        .collect::<Vec<_>>();
    if spectrum_ids
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != spectrum_ids.len()
    {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "spectra must have unique typed response-spectrum identities",
        ));
    }
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::ResponseSpectra { spectrum_ids },
    )
}

#[runtime_builtin(
    name = "fea.StateOutput.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn state_output(
    id: String,
    kind: Value,
    completeness: Value,
    persistence: Value,
) -> BuiltinResult<Value> {
    study_outputs(&FEA_STATE_OUTPUT_NEW_DESCRIPTOR, "fea.StateOutput.new")?;
    const NAME: &str = "fea.StateOutput.new";
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        STATE_OUTPUT_CLASS,
        &StateOutput {
            state_id: StateId(nonempty_id(NAME, "state id", id)?),
            kind: policy::state_kind(NAME, &kind)?,
            completeness: policy::state_completeness(NAME, &completeness)?,
            persistence: policy::state_persistence(NAME, &persistence)?,
        },
    )
}

fn build_state_input(
    name: &'static str,
    id: String,
    kind: &Value,
    source: StateSource,
    use_as: &Value,
    conflict: &Value,
) -> BuiltinResult<Value> {
    serializable_to_object(
        name,
        &ERROR_INTERNAL,
        STATE_INPUT_CLASS,
        &StateInput {
            input_id: nonempty_id(name, "state input id", id)?,
            required_kind: policy::state_kind(name, kind)?,
            source,
            use_as: policy::state_use(name, use_as)?,
            conflict: policy::initial_conflict(name, conflict)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.StateInput.initialConditions",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn state_input_initial(
    id: String,
    kind: Value,
    use_as: Value,
    conflict: Value,
) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_STATE_INPUT_INITIAL_CONDITIONS_DESCRIPTOR,
        "fea.StateInput.initialConditions",
    )?;
    build_state_input(
        "fea.StateInput.initialConditions",
        id,
        &kind,
        StateSource::InitialConditions,
        &use_as,
        &conflict,
    )
}

#[runtime_builtin(
    name = "fea.StateInput.upstream",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn state_input_upstream(
    id: String,
    kind: Value,
    producer: String,
    state_id: String,
    use_as: Value,
    conflict: Value,
) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_STATE_INPUT_UPSTREAM_DESCRIPTOR,
        "fea.StateInput.upstream",
    )?;
    const NAME: &str = "fea.StateInput.upstream";
    let source = StateSource::Upstream {
        producer_step: ProcedureStepId(nonempty_id(NAME, "producer step id", producer)?),
        state_id: StateId(nonempty_id(NAME, "state id", state_id)?),
    };
    build_state_input(NAME, id, &kind, source, &use_as, &conflict)
}

#[runtime_builtin(
    name = "fea.SequenceDependency.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn dependency(
    predecessor: String,
    successor: String,
    reason: Value,
) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_SEQUENCE_DEPENDENCY_NEW_DESCRIPTOR,
        "fea.SequenceDependency.new",
    )?;
    const NAME: &str = "fea.SequenceDependency.new";
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        DEPENDENCY_CLASS,
        &SequenceDependency {
            predecessor: ProcedureStepId(nonempty_id(NAME, "predecessor step id", predecessor)?),
            successor: ProcedureStepId(nonempty_id(NAME, "successor step id", successor)?),
            reason: policy::dependency_reason(NAME, &reason)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.StateTransition.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn transition(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    study_outputs(
        &FEA_STATE_TRANSITION_NEW_DESCRIPTOR,
        "fea.StateTransition.new",
    )?;
    const NAME: &str = "fea.StateTransition.new";
    if args.len() != 7 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected id, producerStepId, consumerStepId, stateId, method, topology, and reference",
        ));
    }
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        TRANSITION_CLASS,
        &StateTransition {
            transition_id: TransitionId(nonempty_id(
                NAME,
                "transition id",
                scalar_string(&args[0], NAME, &ERROR_INPUT)?,
            )?),
            producer_step: ProcedureStepId(nonempty_id(
                NAME,
                "producer step id",
                scalar_string(&args[1], NAME, &ERROR_INPUT)?,
            )?),
            consumer_step: ProcedureStepId(nonempty_id(
                NAME,
                "consumer step id",
                scalar_string(&args[2], NAME, &ERROR_INPUT)?,
            )?),
            state_id: StateId(nonempty_id(
                NAME,
                "state id",
                scalar_string(&args[3], NAME, &ERROR_INPUT)?,
            )?),
            method: policy::transition_method(NAME, &args[4])?,
            topology: policy::topology_transition(NAME, &args[5])?,
            reference: policy::reference_state(NAME, &args[6])?,
        },
    )
}

#[runtime_builtin(
    name = "fea.SequenceBranch.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn branch(
    id: String,
    producer: String,
    states: Value,
    consumers: Value,
) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_SEQUENCE_BRANCH_NEW_DESCRIPTOR,
        "fea.SequenceBranch.new",
    )?;
    const NAME: &str = "fea.SequenceBranch.new";
    let states = string_vec_from_value(NAME, &states)?
        .into_iter()
        .map(StateId)
        .collect();
    let consumers = string_vec_from_value(NAME, &consumers)?
        .into_iter()
        .map(ProcedureStepId)
        .collect();
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        BRANCH_CLASS,
        &SequenceBranch {
            branch_id: SequenceBranchId(nonempty_id(NAME, "branch id", id)?),
            producer_step: ProcedureStepId(nonempty_id(NAME, "producer step id", producer)?),
            state_ids: states,
            consumer_steps: consumers,
        },
    )
}

#[runtime_builtin(
    name = "fea.RestartPolicy.fresh",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_fresh() -> BuiltinResult<Value> {
    study_outputs(
        &FEA_RESTART_POLICY_FRESH_DESCRIPTOR,
        "fea.RestartPolicy.fresh",
    )?;
    to_typed_value(
        "fea.RestartPolicy.fresh",
        RESTART_POLICY_CLASS,
        RestartPolicy::Fresh,
    )
}
#[runtime_builtin(
    name = "fea.RestartPolicy.continueFromCompletedState",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_completed(producer: String, state: String) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_RESTART_POLICY_CONTINUE_FROM_COMPLETED_STATE_DESCRIPTOR,
        "fea.RestartPolicy.continueFromCompletedState",
    )?;
    const NAME: &str = "fea.RestartPolicy.continueFromCompletedState";
    to_typed_value(
        NAME,
        RESTART_POLICY_CLASS,
        RestartPolicy::ContinueFromCompletedState {
            producer_step: ProcedureStepId(nonempty_id(NAME, "producer step id", producer)?),
            state_id: StateId(nonempty_id(NAME, "state id", state)?),
        },
    )
}
#[runtime_builtin(
    name = "fea.RestartPolicy.resumeCheckpoint",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_checkpoint(checkpoint: String, digest: String) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_RESTART_POLICY_RESUME_CHECKPOINT_DESCRIPTOR,
        "fea.RestartPolicy.resumeCheckpoint",
    )?;
    const NAME: &str = "fea.RestartPolicy.resumeCheckpoint";
    to_typed_value(
        NAME,
        RESTART_POLICY_CLASS,
        RestartPolicy::ResumeCheckpoint {
            checkpoint_id: nonempty_id(NAME, "checkpoint id", checkpoint)?,
            required_state_digest: nonempty_id(NAME, "required state digest", digest)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.AnalysisSequence.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn sequence(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    study_outputs(
        &FEA_ANALYSIS_SEQUENCE_NEW_DESCRIPTOR,
        "fea.AnalysisSequence.new",
    )?;
    const NAME: &str = "fea.AnalysisSequence.new";
    if args.len() != 5 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected id, steps, dependencies, transitions, and branches",
        ));
    }
    let steps = super::modeling::step_vec_from_value(NAME, &args[1])?;
    let sequence = AnalysisSequence {
        sequence_id: nonempty_id(
            NAME,
            "sequence id",
            scalar_string(&args[0], NAME, &ERROR_INPUT)?,
        )?,
        steps,
        dependencies: object_or_cell(NAME, &args[2], DEPENDENCY_CLASS)?,
        transitions: object_or_cell(NAME, &args[3], TRANSITION_CLASS)?,
        branches: object_or_cell(NAME, &args[4], BRANCH_CLASS)?,
    };
    serializable_to_object(NAME, &ERROR_INTERNAL, SEQUENCE_CLASS, &sequence)
}

#[runtime_builtin(
    name = "fea.ProcedureStep.configure",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn configure_step(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    study_outputs(
        &FEA_PROCEDURE_STEP_CONFIGURE_DESCRIPTOR,
        "fea.ProcedureStep.configure",
    )?;
    const NAME: &str = "fea.ProcedureStep.configure";
    if args.len() != 8 {
        return Err(builtin_error(NAME, &ERROR_INPUT, "expected step, independentDomain, activations, stateInputs, stateOutputs, observations, accuracy, and restart"));
    }
    let Value::Object(object) = &args[0] else {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "step must be a typed FEA procedure step",
        ));
    };
    let class_name = match object.class_name().display_name() {
        FEA_STRUCTURAL_STEP_CLASS => FEA_STRUCTURAL_STEP_CLASS,
        FEA_THERMAL_STEP_CLASS => FEA_THERMAL_STEP_CLASS,
        FEA_ELECTROMAGNETIC_STEP_CLASS => FEA_ELECTROMAGNETIC_STEP_CLASS,
        FEA_ACOUSTIC_STEP_CLASS => FEA_ACOUSTIC_STEP_CLASS,
        FEA_FLUID_STEP_CLASS => FEA_FLUID_STEP_CLASS,
        FEA_THERMOMECHANICAL_STEP_CLASS => FEA_THERMOMECHANICAL_STEP_CLASS,
        FEA_ELECTROTHERMAL_STEP_CLASS => FEA_ELECTROTHERMAL_STEP_CLASS,
        FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS => FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
        FEA_FLUID_STRUCTURE_STEP_CLASS => FEA_FLUID_STRUCTURE_STEP_CLASS,
        other => {
            return Err(builtin_error(
                NAME,
                &ERROR_INPUT,
                format!("unsupported FEA step class {other}"),
            ))
        }
    };
    let mut step: ProcedureStep = object_payload(NAME, &args[0], class_name)?;
    require_step_class(NAME, class_name, &step)?;
    step.independent_domain = independent_domain(NAME, &args[1])?;
    step.activations = super::activations::activations(NAME, &args[2])?;
    step.state_inputs = state_inputs(NAME, &args[3])?;
    step.state_outputs = state_outputs(NAME, &args[4])?;
    step.observations = super::observations::observation_requests(NAME, &args[5])?;
    step.accuracy = super::accuracy::accuracy_plan(NAME, &args[6])?;
    step.restart = restart_policy(NAME, &args[7])?;
    serializable_to_object(NAME, &ERROR_INTERNAL, class_name, &step)
}

pub(super) fn independent_domain(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<IndependentDomain> {
    from_typed_value(name, INDEPENDENT_DOMAIN_CLASS, value)
}
pub(super) fn restart_policy(name: &'static str, value: &Value) -> BuiltinResult<RestartPolicy> {
    from_typed_value(name, RESTART_POLICY_CLASS, value)
}
pub(super) fn state_inputs(name: &'static str, value: &Value) -> BuiltinResult<Vec<StateInput>> {
    object_or_cell(name, value, STATE_INPUT_CLASS)
}
pub(super) fn state_outputs(name: &'static str, value: &Value) -> BuiltinResult<Vec<StateOutput>> {
    object_or_cell(name, value, STATE_OUTPUT_CLASS)
}
pub(super) fn sequence_from_value(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisSequence> {
    object_payload(name, value, SEQUENCE_CLASS)
}

const SEQUENCE_STUDY_CLASSES: [ImmutableStudyClass; 8] = [
    ImmutableStudyClass {
        class_name: INDEPENDENT_DOMAIN_CLASS,
        type_identity: "runmat.fea.independent-domain",
        fields: VALUE_FIELD,
        methods: &[
            ("staticDomain", "fea.IndependentDomain.staticDomain"),
            ("timeSeconds", "fea.IndependentDomain.timeSeconds"),
            ("frequencyHertz", "fea.IndependentDomain.frequencyHertz"),
            ("loadFactor", "fea.IndependentDomain.loadFactor"),
            ("eigenvalue", "fea.IndependentDomain.eigenvalue"),
            ("studyParameter", "fea.IndependentDomain.studyParameter"),
            ("responseSpectra", "fea.IndependentDomain.responseSpectra"),
        ],
    },
    ImmutableStudyClass {
        class_name: RESTART_POLICY_CLASS,
        type_identity: "runmat.fea.restart-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("fresh", "fea.RestartPolicy.fresh"),
            (
                "continueFromCompletedState",
                "fea.RestartPolicy.continueFromCompletedState",
            ),
            ("resumeCheckpoint", "fea.RestartPolicy.resumeCheckpoint"),
        ],
    },
    ImmutableStudyClass {
        class_name: STATE_INPUT_CLASS,
        type_identity: "runmat.fea.state-input",
        fields: STATE_INPUT_FIELDS,
        methods: &[
            ("initialConditions", "fea.StateInput.initialConditions"),
            ("upstream", "fea.StateInput.upstream"),
        ],
    },
    ImmutableStudyClass {
        class_name: STATE_OUTPUT_CLASS,
        type_identity: "runmat.fea.state-output",
        fields: STATE_OUTPUT_FIELDS,
        methods: &[("new", "fea.StateOutput.new")],
    },
    ImmutableStudyClass {
        class_name: DEPENDENCY_CLASS,
        type_identity: "runmat.fea.sequence-dependency",
        fields: DEPENDENCY_FIELDS,
        methods: &[("new", "fea.SequenceDependency.new")],
    },
    ImmutableStudyClass {
        class_name: TRANSITION_CLASS,
        type_identity: "runmat.fea.state-transition",
        fields: TRANSITION_FIELDS,
        methods: &[("new", "fea.StateTransition.new")],
    },
    ImmutableStudyClass {
        class_name: BRANCH_CLASS,
        type_identity: "runmat.fea.sequence-branch",
        fields: BRANCH_FIELDS,
        methods: &[("new", "fea.SequenceBranch.new")],
    },
    ImmutableStudyClass {
        class_name: SEQUENCE_CLASS,
        type_identity: "runmat.fea.analysis-sequence",
        fields: SEQUENCE_FIELDS,
        methods: &[("new", "fea.AnalysisSequence.new")],
    },
];

pub(super) fn ensure_sequence_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(FEA_INDEPENDENT_DOMAIN_CLASS);
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &SEQUENCE_STUDY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_immutable_runtime_classes(&SEQUENCE_STUDY_CLASSES);
        register_value_class(
            PROCEDURE_STEP_CLASS,
            HashMap::new(),
            static_methods([("configure", "fea.ProcedureStep.configure")]),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::{activations, observations};
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};
    use runmat_value::CellArray;

    fn cell(values: Vec<Value>) -> Value {
        let columns = values.len();
        Value::Cell(CellArray::new(values, 1, columns).expect("valid row cell"))
    }

    #[test]
    fn independent_domain_class_exposes_every_static_constructor() {
        super::super::ensure_value_classes_registered();
        let class = crate::class_registry::get_class(
            &runmat_types::ClassIdentity::new(INDEPENDENT_DOMAIN_CLASS).expect("class identity"),
        )
        .expect("registered fea.IndependentDomain");
        for builtin in [
            "fea.IndependentDomain.staticDomain",
            "fea.IndependentDomain.responseSpectra",
        ] {
            assert!(
                class
                    .methods
                    .values()
                    .any(|method| method.function_name == builtin),
                "{builtin} is a static method of fea.IndependentDomain"
            );
        }
    }

    #[tokio::test]
    async fn time_domain_requires_and_retains_a_positive_interval_count() {
        let domain = domain_time(
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(2.0),
            Value::Num(20.0),
        )
        .await
        .unwrap();
        let decoded: IndependentDomain =
            from_typed_value("test", INDEPENDENT_DOMAIN_CLASS, &domain).unwrap();
        assert_eq!(
            decoded,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 2.0,
                interval_count: 20,
            }
        );
        assert!(domain_time(
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(2.0),
            Value::Num(0.0),
        )
        .await
        .is_err());
    }

    #[tokio::test]
    async fn response_spectrum_domain_retains_typed_spectrum_id() {
        let spectrum = runmat_analysis_core::ResponseSpectrum {
            spectrum_id: runmat_analysis_core::ResponseSpectrumId("launch".into()),
            abscissa: runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Frequency, _>::new(
                    vec![10.0, 20.0],
                ),
            ),
            ordinates: runmat_analysis_core::NumericalDataSource::authored(
                vec![1.0, 2.0],
                "sha256:launch",
            ),
            damping: runmat_analysis_core::SpectrumDamping::Ratio(0.05),
            response: runmat_analysis_core::SpectrumResponse::AbsoluteAcceleration,
            envelope: runmat_analysis_core::SpectrumEnvelope::Maximax,
            interpolation: runmat_analysis_core::SpectrumInterpolation::LinearFrequency,
            extrapolation: runmat_analysis_core::SpectrumExtrapolation::Reject,
            direction: runmat_analysis_core::SpectrumDirection {
                frame: runmat_analysis_core::CoordinateFrame::Global,
                vector: [0.0, 0.0, 1.0],
            },
            modal_combination: runmat_analysis_core::ModalCombination::CompleteQuadraticCombination,
            directional_combination: runmat_analysis_core::DirectionalCombination::Srss,
            truncation: runmat_analysis_core::ModalTruncationPolicy {
                minimum_cumulative_mass_ratio: 0.9,
                correction: runmat_analysis_core::ModalTruncationCorrection::MissingMass {
                    zero_period_acceleration: 9.0,
                },
            },
            rigid_response: runmat_analysis_core::RigidResponsePolicy::Disabled,
            construction: runmat_analysis_core::SpectrumConstruction::AuthoredDesign,
            evidence: runmat_analysis_core::SpectrumEvidence {
                sampling_adequacy_ratio: 1.0,
                max_interpolation_error: None,
            },
        };
        let typed = super::super::serializable_to_object(
            "test",
            &ERROR_INTERNAL,
            super::super::response_spectra::RESPONSE_SPECTRUM_CLASS,
            &spectrum,
        )
        .unwrap();
        let domain = domain_response_spectra(typed).await.unwrap();
        let decoded: IndependentDomain =
            from_typed_value("test", INDEPENDENT_DOMAIN_CLASS, &domain).unwrap();
        assert_eq!(
            decoded,
            IndependentDomain::ResponseSpectra {
                spectrum_ids: vec![runmat_analysis_core::ResponseSpectrumId("launch".into())],
            }
        );
    }

    #[tokio::test]
    async fn configured_sequence_round_trips_with_typed_study_contracts() {
        super::super::ensure_value_classes_registered();

        let object = activations::boundary("fixed".into())
            .await
            .expect("boundary reference");
        let lifecycle = activations::activate().await.expect("activation lifecycle");
        let activation = activations::activation(object, lifecycle)
            .await
            .expect("activation");

        let state = state_output(
            "equilibrium".into(),
            policy::state_structural_primary()
                .await
                .expect("state kind"),
            policy::completeness_complete()
                .await
                .expect("state completeness"),
            policy::persistence_canonical()
                .await
                .expect("state persistence"),
        )
        .await
        .expect("state output");

        let target = observations::target_whole().await.expect("target");
        let storage = observations::storage(
            observations::precision_real64().await.expect("precision"),
            observations::compression_none().await.expect("compression"),
            observations::retention_canonical()
                .await
                .expect("retention"),
            super::super::study_policies::observation::partial_chunks()
                .await
                .expect("partial access"),
        )
        .await
        .expect("storage");
        let observation = observations::field(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("displacement".into()),
                super::super::study_policies::observation::quantity_displacement()
                    .await
                    .expect("quantity"),
                target,
                super::super::study_policies::observation::location_node()
                    .await
                    .expect("source location"),
                super::super::study_policies::observation::location_node()
                    .await
                    .expect("output location"),
                super::super::study_policies::observation::recovery_native()
                    .await
                    .expect("recovery"),
                super::super::study_policies::observation::component_all()
                    .await
                    .expect("component"),
                super::super::constraints::support::frame::coordinate_frame_global_builtin(None)
                    .await
                    .expect("frame"),
                super::super::study_policies::observation::sign_native()
                    .await
                    .expect("sign"),
                super::super::study_policies::observation::complex_real()
                    .await
                    .expect("complex representation"),
                observations::sampling_final().await.expect("sampling"),
                observations::reduction_none().await.expect("reduction"),
                storage,
                super::super::study_policies::observation::purpose_visualization()
                    .await
                    .expect("purpose"),
                observations::cost(crate::builtins::common::test_support::builtin_arguments(
                    vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()],
                ))
                .await
                .expect("cost"),
            ]),
        )
        .await
        .expect("observation");

        let step = super::super::steps::structural_static_builtin(Value::String("static".into()))
            .await
            .expect("structural step");
        let configured = configure_step(crate::builtins::common::test_support::builtin_arguments(
            vec![
                step,
                domain_static().await.expect("domain"),
                cell(vec![activation]),
                cell(Vec::new()),
                cell(vec![state]),
                cell(vec![observation]),
                super::super::accuracy::baseline_plan("acceptance".into())
                    .await
                    .expect("accuracy"),
                restart_fresh().await.expect("restart"),
            ],
        ))
        .await
        .expect("configured step");
        let configured_for_conflict = configured.clone();
        let value = sequence(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("production".into()),
                cell(vec![configured]),
                cell(Vec::new()),
                cell(Vec::new()),
                cell(Vec::new()),
            ],
        ))
        .await
        .expect("sequence");

        let decoded =
            decode_inline_value(&encode_inline_value(&value).expect("encode")).expect("decode");
        let sequence = sequence_from_value("test", &decoded).expect("typed sequence");
        assert_eq!(sequence.sequence_id, "production");
        assert_eq!(sequence.steps[0].activations.len(), 1);
        assert_eq!(sequence.steps[0].state_outputs.len(), 1);
        assert_eq!(sequence.steps[0].observations.len(), 1);
        assert!(sequence.steps[0].observations[0]
            .spec()
            .expect("field observation")
            .reduction
            .is_none());

        let directory = tempfile::tempdir().expect("temporary geometry directory");
        let path = directory.path().join("triangle.stl");
        std::fs::write(
            &path,
            "solid triangle\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid triangle\n",
        )
        .expect("write geometry fixture");
        let geometry = crate::builtins::geometry::geometry_load_builtin(
            Value::String(path.to_string_lossy().into_owned().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .await
        .expect("load geometry fixture");
        let model_value = super::super::modeling::create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                geometry.clone(),
                Value::String("Sequence".into()),
                value.clone(),
            ]),
        )
        .expect("compose model with explicit sequence");
        let model: AnalysisModel =
            object_payload("test", &model_value, FEA_MODEL_CLASS).expect("decode typed model");
        assert_eq!(model.sequence.sequence_id, "production");

        let conflict = super::super::modeling::create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                geometry,
                Value::String("Sequence".into()),
                value,
                Value::String("Steps".into()),
                cell(vec![configured_for_conflict]),
            ]),
        )
        .expect_err("sequence and steps must remain mutually exclusive");
        assert!(conflict.message.contains("mutually exclusive"));
    }
}

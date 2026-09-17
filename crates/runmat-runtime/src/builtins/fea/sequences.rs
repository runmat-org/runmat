use std::{collections::HashMap, sync::Once};

use runmat_analysis_core::{
    AnalysisSequence, IndependentDomain, ProcedureStep, ProcedureStepId, RestartPolicy,
    SequenceBranch, SequenceBranchId, SequenceDependency, StateId, StateInput, StateOutput,
    StateSource, StateTransition, TransitionId,
};
use runmat_types::MemberAccess;

use super::*;
use crate::builtins::fea::study_policies::sequence as policy;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_class, to_typed_value,
    VALUE_FIELD,
};

pub(super) const INDEPENDENT_DOMAIN_CLASS: &str = "fea.IndependentDomain";
pub(super) const RESTART_POLICY_CLASS: &str = "fea.RestartPolicy";
pub(super) const STATE_INPUT_CLASS: &str = "fea.StateInput";
pub(super) const STATE_OUTPUT_CLASS: &str = "fea.StateOutput";
pub(super) const DEPENDENCY_CLASS: &str = "fea.SequenceDependency";
pub(super) const TRANSITION_CLASS: &str = "fea.StateTransition";
pub(super) const BRANCH_CLASS: &str = "fea.SequenceBranch";
pub(super) const SEQUENCE_CLASS: &str = "fea.AnalysisSequence";
pub(super) const PROCEDURE_STEP_CLASS: &str = "fea.ProcedureStep";

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
const DOMAIN_TIME_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "origin",
        BuiltinParamType::NumericScalar,
        "Total-time origin in seconds.",
    ),
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Step start time in seconds.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Step end time in seconds.",
    ),
];
const DOMAIN_VALUES_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite domain coordinates.",
)];
const DOMAIN_RANGE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required("start", BuiltinParamType::NumericScalar, "Domain start."),
    required("end", BuiltinParamType::NumericScalar, "Domain end."),
];
const DOMAIN_COUNT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "count",
    BuiltinParamType::NumericScalar,
    "Positive requested mode count.",
)];
const DOMAIN_PARAMETER_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Typed study-parameter identifier.",
    ),
    required(
        "values",
        BuiltinParamType::NumericArray,
        "Finite canonical parameter values.",
    ),
];
const DOMAIN_SPECTRA_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "spectra",
    BuiltinParamType::Any,
    "One or more typed fea.ResponseSpectrum values consumed by the procedure.",
)];
const DOMAIN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "count or coordinates",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "Mode counts remain exact; physical coordinates cross once into canonical binary64 storage.",
}];
const DOMAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "independent-domain construction",
        inputs: &DOMAIN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed domains are immutable host authoring values.",
    }];
const STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This typed sequence constructor accepts identifiers and typed FEA objects only.",
};

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    DOMAIN_STATIC_DESCRIPTOR,
    DOMAIN_STATIC_SIGNATURES,
    "domain = fea.IndependentDomain.staticDomain()",
    &NO_INPUTS
);
descriptor!(
    DOMAIN_TIME_DESCRIPTOR,
    DOMAIN_TIME_SIGNATURES,
    "domain = fea.IndependentDomain.timeSeconds(origin, start, end)",
    &DOMAIN_TIME_INPUTS
);
descriptor!(
    DOMAIN_FREQUENCY_DESCRIPTOR,
    DOMAIN_FREQUENCY_SIGNATURES,
    "domain = fea.IndependentDomain.frequencyHertz(values)",
    &DOMAIN_VALUES_INPUTS
);
descriptor!(
    DOMAIN_LOAD_DESCRIPTOR,
    DOMAIN_LOAD_SIGNATURES,
    "domain = fea.IndependentDomain.loadFactor(start, end)",
    &DOMAIN_RANGE_INPUTS
);
descriptor!(
    DOMAIN_EIGENVALUE_DESCRIPTOR,
    DOMAIN_EIGENVALUE_SIGNATURES,
    "domain = fea.IndependentDomain.eigenvalue(count)",
    &DOMAIN_COUNT_INPUTS
);
descriptor!(
    DOMAIN_PARAMETER_DESCRIPTOR,
    DOMAIN_PARAMETER_SIGNATURES,
    "domain = fea.IndependentDomain.studyParameter(id, values)",
    &DOMAIN_PARAMETER_INPUTS
);
descriptor!(
    DOMAIN_SPECTRA_DESCRIPTOR,
    DOMAIN_SPECTRA_SIGNATURES,
    "domain = fea.IndependentDomain.responseSpectra(spectra)",
    &DOMAIN_SPECTRA_INPUTS
);

#[runtime_builtin(
    name = "fea.IndependentDomain.staticDomain",
    category = "fea",
    summary = "Create a static independent domain.",
    keywords = "fea,step,domain,static,typed",
    descriptor(DOMAIN_STATIC_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_static() -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Create a typed time domain in seconds.",
    keywords = "fea,step,domain,time,typed",
    descriptor(DOMAIN_TIME_DESCRIPTOR),
    integer_capabilities(DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_time(origin: Value, start: Value, end: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.IndependentDomain.timeSeconds";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::TimeSeconds {
            origin: values::finite_numeric_scalar(NAME, &origin, "origin")?,
            start: values::finite_numeric_scalar(NAME, &start, "start")?,
            end: values::finite_numeric_scalar(NAME, &end, "end")?,
        },
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.frequencyHertz",
    category = "fea",
    summary = "Create a typed frequency domain in hertz.",
    keywords = "fea,step,domain,frequency,typed",
    descriptor(DOMAIN_FREQUENCY_DESCRIPTOR),
    integer_capabilities(DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_frequency(values: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.IndependentDomain.frequencyHertz";
    to_typed_value(
        NAME,
        INDEPENDENT_DOMAIN_CLASS,
        IndependentDomain::FrequencyHertz(numeric_values(NAME, &values, "values")?),
    )
}

#[runtime_builtin(
    name = "fea.IndependentDomain.loadFactor",
    category = "fea",
    summary = "Create a typed load-factor domain.",
    keywords = "fea,step,domain,load,factor,typed",
    descriptor(DOMAIN_LOAD_DESCRIPTOR),
    integer_capabilities(DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_load(start: Value, end: Value) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Create a typed eigenvalue domain.",
    keywords = "fea,step,domain,eigenvalue,modes,typed",
    descriptor(DOMAIN_EIGENVALUE_DESCRIPTOR),
    integer_capabilities(DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_eigenvalue(count: Value) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Create a typed parameter-sampling domain.",
    keywords = "fea,step,domain,parameter,sweep,typed",
    descriptor(DOMAIN_PARAMETER_DESCRIPTOR),
    integer_capabilities(DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_parameter(id: String, values: Value) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Bind a modal-response procedure to typed response spectra.",
    keywords = "fea,step,domain,response,spectrum,typed",
    descriptor(DOMAIN_SPECTRA_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn domain_response_spectra(spectra: Value) -> BuiltinResult<Value> {
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

const STATE_OUTPUT_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Published state identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed fea.StateKind."),
    required(
        "completeness",
        BuiltinParamType::Any,
        "Typed fea.StateCompleteness.",
    ),
    required(
        "persistence",
        BuiltinParamType::Any,
        "Typed fea.StatePersistence.",
    ),
];
descriptor!(
    STATE_OUTPUT_DESCRIPTOR,
    STATE_OUTPUT_SIGNATURES,
    "output = fea.StateOutput.new(id, kind, completeness, persistence)",
    &STATE_OUTPUT_INPUTS
);
#[runtime_builtin(
    name = "fea.StateOutput.new",
    category = "fea",
    summary = "Declare a typed procedure-state output.",
    keywords = "fea,sequence,state,output,typed",
    descriptor(STATE_OUTPUT_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn state_output(
    id: String,
    kind: Value,
    completeness: Value,
    persistence: Value,
) -> BuiltinResult<Value> {
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

const INITIAL_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "State-input identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed required state kind."),
    required("useAs", BuiltinParamType::Any, "Typed state-use policy."),
    required(
        "conflict",
        BuiltinParamType::Any,
        "Typed initial-state conflict policy.",
    ),
];
const UPSTREAM_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "State-input identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed required state kind."),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer procedure-step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Producer state identifier.",
    ),
    required("useAs", BuiltinParamType::Any, "Typed state-use policy."),
    required(
        "conflict",
        BuiltinParamType::Any,
        "Typed initial-state conflict policy.",
    ),
];
descriptor!(
    STATE_INPUT_INITIAL_DESCRIPTOR,
    STATE_INPUT_INITIAL_SIGNATURES,
    "input = fea.StateInput.initialConditions(id, kind, useAs, conflict)",
    &INITIAL_INPUTS
);
descriptor!(
    STATE_INPUT_UPSTREAM_DESCRIPTOR,
    STATE_INPUT_UPSTREAM_SIGNATURES,
    "input = fea.StateInput.upstream(id, kind, producerStepId, stateId, useAs, conflict)",
    &UPSTREAM_INPUTS
);

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
    category = "fea",
    summary = "Consume authored initial conditions as typed state.",
    keywords = "fea,sequence,state,input,initial,typed",
    descriptor(STATE_INPUT_INITIAL_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn state_input_initial(
    id: String,
    kind: Value,
    use_as: Value,
    conflict: Value,
) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Consume a typed state from an upstream procedure step.",
    keywords = "fea,sequence,state,input,upstream,typed",
    descriptor(STATE_INPUT_UPSTREAM_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
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
    const NAME: &str = "fea.StateInput.upstream";
    let source = StateSource::Upstream {
        producer_step: ProcedureStepId(nonempty_id(NAME, "producer step id", producer)?),
        state_id: StateId(nonempty_id(NAME, "state id", state_id)?),
    };
    build_state_input(NAME, id, &kind, source, &use_as, &conflict)
}

const DEPENDENCY_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "predecessorStepId",
        BuiltinParamType::StringScalar,
        "Predecessor step identifier.",
    ),
    required(
        "successorStepId",
        BuiltinParamType::StringScalar,
        "Successor step identifier.",
    ),
    required(
        "reason",
        BuiltinParamType::Any,
        "Typed sequence-dependency reason.",
    ),
];
descriptor!(
    DEPENDENCY_DESCRIPTOR,
    DEPENDENCY_SIGNATURES,
    "dependency = fea.SequenceDependency.new(predecessorStepId, successorStepId, reason)",
    &DEPENDENCY_INPUTS
);
#[runtime_builtin(
    name = "fea.SequenceDependency.new",
    category = "fea",
    summary = "Declare a typed semantic dependency between procedure steps.",
    keywords = "fea,sequence,dependency,typed",
    descriptor(DEPENDENCY_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn dependency(
    predecessor: String,
    successor: String,
    reason: Value,
) -> BuiltinResult<Value> {
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

const TRANSITION_INPUTS: [BuiltinParamDescriptor; 7] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Transition identifier.",
    ),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer step identifier.",
    ),
    required(
        "consumerStepId",
        BuiltinParamType::StringScalar,
        "Consumer step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Transferred state identifier.",
    ),
    required(
        "method",
        BuiltinParamType::Any,
        "Typed state-transition method.",
    ),
    required(
        "topology",
        BuiltinParamType::Any,
        "Typed topology-transition policy.",
    ),
    required(
        "reference",
        BuiltinParamType::Any,
        "Typed reference-state policy.",
    ),
];
descriptor!(TRANSITION_DESCRIPTOR, TRANSITION_SIGNATURES, "transition = fea.StateTransition.new(id, producerStepId, consumerStepId, stateId, method, topology, reference)", &TRANSITION_INPUTS);
#[runtime_builtin(
    name = "fea.StateTransition.new",
    category = "fea",
    summary = "Declare a typed state transition between procedure steps.",
    keywords = "fea,sequence,state,transition,typed",
    descriptor(TRANSITION_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn transition(args: Vec<Value>) -> BuiltinResult<Value> {
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

const BRANCH_INPUTS: [BuiltinParamDescriptor; 4] = [
    required("id", BuiltinParamType::StringScalar, "Branch identifier."),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer step identifier.",
    ),
    required(
        "stateIds",
        BuiltinParamType::Any,
        "State identifier or cell array of identifiers.",
    ),
    required(
        "consumerStepIds",
        BuiltinParamType::Any,
        "Consumer identifier or cell array of identifiers.",
    ),
];
descriptor!(
    BRANCH_DESCRIPTOR,
    BRANCH_SIGNATURES,
    "branch = fea.SequenceBranch.new(id, producerStepId, stateIds, consumerStepIds)",
    &BRANCH_INPUTS
);
#[runtime_builtin(
    name = "fea.SequenceBranch.new",
    category = "fea",
    summary = "Declare immutable downstream consumers of a published state.",
    keywords = "fea,sequence,branch,state,typed",
    descriptor(BRANCH_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn branch(
    id: String,
    producer: String,
    states: Value,
    consumers: Value,
) -> BuiltinResult<Value> {
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

const RESTART_COMPLETED_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Completed producer step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Complete producer state identifier.",
    ),
];
const RESTART_CHECKPOINT_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "checkpointId",
        BuiltinParamType::StringScalar,
        "Checkpoint identifier.",
    ),
    required(
        "requiredStateDigest",
        BuiltinParamType::StringScalar,
        "Required checkpoint-state digest.",
    ),
];
descriptor!(
    RESTART_FRESH_DESCRIPTOR,
    RESTART_FRESH_SIGNATURES,
    "policy = fea.RestartPolicy.fresh()",
    &NO_INPUTS
);
descriptor!(
    RESTART_COMPLETED_DESCRIPTOR,
    RESTART_COMPLETED_SIGNATURES,
    "policy = fea.RestartPolicy.continueFromCompletedState(producerStepId, stateId)",
    &RESTART_COMPLETED_INPUTS
);
descriptor!(
    RESTART_CHECKPOINT_DESCRIPTOR,
    RESTART_CHECKPOINT_SIGNATURES,
    "policy = fea.RestartPolicy.resumeCheckpoint(checkpointId, requiredStateDigest)",
    &RESTART_CHECKPOINT_INPUTS
);
#[runtime_builtin(
    name = "fea.RestartPolicy.fresh",
    category = "fea",
    summary = "Start a procedure from fresh state.",
    keywords = "fea,sequence,restart,fresh,typed",
    descriptor(RESTART_FRESH_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_fresh() -> BuiltinResult<Value> {
    to_typed_value(
        "fea.RestartPolicy.fresh",
        RESTART_POLICY_CLASS,
        RestartPolicy::Fresh,
    )
}
#[runtime_builtin(
    name = "fea.RestartPolicy.continueFromCompletedState",
    category = "fea",
    summary = "Continue from a complete immutable state.",
    keywords = "fea,sequence,restart,state,typed",
    descriptor(RESTART_COMPLETED_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_completed(producer: String, state: String) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Resume a procedure from a digest-fenced checkpoint.",
    keywords = "fea,sequence,restart,checkpoint,typed",
    descriptor(RESTART_CHECKPOINT_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn restart_checkpoint(checkpoint: String, digest: String) -> BuiltinResult<Value> {
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

const SEQUENCE_INPUTS: [BuiltinParamDescriptor; 5] = [
    required("id", BuiltinParamType::StringScalar, "Sequence identifier."),
    required(
        "steps",
        BuiltinParamType::Any,
        "Typed procedure step or cell array.",
    ),
    required(
        "dependencies",
        BuiltinParamType::Any,
        "Typed dependency or cell array.",
    ),
    required(
        "transitions",
        BuiltinParamType::Any,
        "Typed transition or cell array.",
    ),
    required(
        "branches",
        BuiltinParamType::Any,
        "Typed branch or cell array.",
    ),
];
descriptor!(
    SEQUENCE_DESCRIPTOR,
    SEQUENCE_SIGNATURES,
    "sequence = fea.AnalysisSequence.new(id, steps, dependencies, transitions, branches)",
    &SEQUENCE_INPUTS
);
#[runtime_builtin(
    name = "fea.AnalysisSequence.new",
    category = "fea",
    summary = "Create an explicit semantic procedure sequence.",
    keywords = "fea,analysis,sequence,state,transition,typed",
    descriptor(SEQUENCE_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn sequence(args: Vec<Value>) -> BuiltinResult<Value> {
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

const CONFIGURE_STEP_INPUTS: [BuiltinParamDescriptor; 8] = [
    required(
        "step",
        BuiltinParamType::Any,
        "Typed family-specific procedure step.",
    ),
    required(
        "independentDomain",
        BuiltinParamType::Any,
        "Typed independent domain.",
    ),
    required(
        "activations",
        BuiltinParamType::Any,
        "Typed step-object activation or cell array.",
    ),
    required(
        "stateInputs",
        BuiltinParamType::Any,
        "Typed state input or cell array.",
    ),
    required(
        "stateOutputs",
        BuiltinParamType::Any,
        "Typed state output or cell array.",
    ),
    required(
        "observations",
        BuiltinParamType::Any,
        "Typed observation or cell array.",
    ),
    required("accuracy", BuiltinParamType::Any, "Typed accuracy plan."),
    required("restart", BuiltinParamType::Any, "Typed restart policy."),
];
descriptor!(CONFIGURE_STEP_DESCRIPTOR, CONFIGURE_STEP_SIGNATURES, "step = fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, stateOutputs, observations, accuracy, restart)", &CONFIGURE_STEP_INPUTS);

#[runtime_builtin(
    name = "fea.ProcedureStep.configure",
    category = "fea",
    summary = "Configure a family-specific procedure step with typed study contracts.",
    keywords = "fea,procedure,step,sequence,observation,accuracy,state,typed",
    descriptor(CONFIGURE_STEP_DESCRIPTOR),
    integer_audit(STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::sequences"
)]
pub async fn configure_step(args: Vec<Value>) -> BuiltinResult<Value> {
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
    let class_name = match object.class_name.as_str() {
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

pub(super) fn ensure_sequence_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            INDEPENDENT_DOMAIN_CLASS,
            "runmat.fea.independent-domain",
            VALUE_FIELD,
            &[
                ("staticDomain", "fea.IndependentDomain.staticDomain"),
                ("timeSeconds", "fea.IndependentDomain.timeSeconds"),
                ("frequencyHertz", "fea.IndependentDomain.frequencyHertz"),
                ("loadFactor", "fea.IndependentDomain.loadFactor"),
                ("eigenvalue", "fea.IndependentDomain.eigenvalue"),
                ("studyParameter", "fea.IndependentDomain.studyParameter"),
            ],
        );
        register_immutable_class(
            RESTART_POLICY_CLASS,
            "runmat.fea.restart-policy",
            VALUE_FIELD,
            &[
                ("fresh", "fea.RestartPolicy.fresh"),
                (
                    "continueFromCompletedState",
                    "fea.RestartPolicy.continueFromCompletedState",
                ),
                ("resumeCheckpoint", "fea.RestartPolicy.resumeCheckpoint"),
            ],
        );
        register_immutable_class(
            STATE_INPUT_CLASS,
            "runmat.fea.state-input",
            STATE_INPUT_FIELDS,
            &[
                ("initialConditions", "fea.StateInput.initialConditions"),
                ("upstream", "fea.StateInput.upstream"),
            ],
        );
        register_immutable_class(
            STATE_OUTPUT_CLASS,
            "runmat.fea.state-output",
            STATE_OUTPUT_FIELDS,
            &[("new", "fea.StateOutput.new")],
        );
        register_immutable_class(
            DEPENDENCY_CLASS,
            "runmat.fea.sequence-dependency",
            DEPENDENCY_FIELDS,
            &[("new", "fea.SequenceDependency.new")],
        );
        register_immutable_class(
            TRANSITION_CLASS,
            "runmat.fea.state-transition",
            TRANSITION_FIELDS,
            &[("new", "fea.StateTransition.new")],
        );
        register_immutable_class(
            BRANCH_CLASS,
            "runmat.fea.sequence-branch",
            BRANCH_FIELDS,
            &[("new", "fea.SequenceBranch.new")],
        );
        register_immutable_class(
            SEQUENCE_CLASS,
            "runmat.fea.analysis-sequence",
            SEQUENCE_FIELDS,
            &[("new", "fea.AnalysisSequence.new")],
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: PROCEDURE_STEP_CLASS.into(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::from([(
                "configure".into(),
                crate::class_registry::RuntimeMethod {
                    name: "configure".into(),
                    is_static: true,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: "fea.ProcedureStep.configure".into(),
                    implicit_class_argument: None,
                },
            )]),
        });
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

    #[tokio::test]
    async fn response_spectrum_domain_retains_typed_spectrum_id() {
        let spectrum = runmat_analysis_core::ResponseSpectrum {
            spectrum_id: runmat_analysis_core::ResponseSpectrumId("launch".into()),
            abscissa: runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(vec![10.0, 20.0]),
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
        let observation = observations::field(vec![
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
            super::super::constraints::support::frame::coordinate_frame_global_builtin()
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
            observations::cost(vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()])
                .await
                .expect("cost"),
        ])
        .await
        .expect("observation");

        let step = super::super::steps::structural_static_builtin(Value::String("static".into()))
            .await
            .expect("structural step");
        let configured = configure_step(vec![
            step,
            domain_static().await.expect("domain"),
            cell(vec![activation]),
            cell(Vec::new()),
            cell(vec![state]),
            cell(vec![observation]),
            super::super::accuracy::experimental_plan("acceptance".into())
                .await
                .expect("accuracy"),
            restart_fresh().await.expect("restart"),
        ])
        .await
        .expect("configured step");
        let configured_for_conflict = configured.clone();
        let value = sequence(vec![
            Value::String("production".into()),
            cell(vec![configured]),
            cell(Vec::new()),
            cell(Vec::new()),
            cell(Vec::new()),
        ])
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
        let geometry = crate::builtins::geometry::geometry_load_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )])
        .await
        .expect("load geometry fixture");
        let model_value = super::super::modeling::create_model_object_from_args(vec![
            Value::String("model".into()),
            geometry.clone(),
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Defaults".into()),
            Value::String("None".into()),
            Value::String("Sequence".into()),
            value.clone(),
        ])
        .expect("compose model with explicit sequence");
        let model: AnalysisModel =
            object_payload("test", &model_value, FEA_MODEL_CLASS).expect("decode typed model");
        assert_eq!(model.sequence.sequence_id, "production");

        let conflict = super::super::modeling::create_model_object_from_args(vec![
            Value::String("model".into()),
            geometry,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Defaults".into()),
            Value::String("None".into()),
            Value::String("Sequence".into()),
            value,
            Value::String("Steps".into()),
            cell(vec![configured_for_conflict]),
        ])
        .expect_err("sequence and steps must remain mutually exclusive");
        assert!(conflict.message.contains("mutually exclusive"));
    }
}

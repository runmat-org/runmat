use super::*;

const SPEC_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed observation quantity.",
    ),
    required("target", BuiltinParamType::Any, "Typed observation target."),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed source field location.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed output field location.",
    ),
    required("recovery", BuiltinParamType::Any, "Typed recovery policy."),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed component or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed coordinate frame."),
    required("sign", BuiltinParamType::Any, "Typed sign convention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed complex representation.",
    ),
    required("sampling", BuiltinParamType::Any, "Typed sampling policy."),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed reduction policy.",
    ),
    required("storage", BuiltinParamType::Any, "Typed storage policy."),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed observation purpose.",
    ),
    required(
        "cost",
        BuiltinParamType::Any,
        "Typed observation cost estimate.",
    ),
];
descriptor!(FIELD_DESCRIPTOR, FIELD_SIGNATURES, "observation = fea.Observation.field(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(HISTORY_DESCRIPTOR, HISTORY_SIGNATURES, "observation = fea.Observation.history(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(PROBE_DESCRIPTOR, PROBE_SIGNATURES, "observation = fea.Observation.probe(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(REDUCTION_OBSERVATION_DESCRIPTOR, REDUCTION_OBSERVATION_SIGNATURES, "observation = fea.Observation.reduction(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);

fn canonical_shape(quantity: ObservationQuantity) -> ObservationValueShape {
    quantity.canonical_value_shape()
}

enum ObservationForm {
    Field,
    History,
    Probe,
    Reduction,
}
fn observation(
    name: &'static str,
    args: Vec<Value>,
    form: ObservationForm,
) -> BuiltinResult<Value> {
    if args.len() != 15 {
        return Err(builtin_error(
            name,
            &ERROR_INPUT,
            "expected the 15 typed observation arguments",
        ));
    }
    let quantity = policy::quantity(name, &args[1])?;
    let target: TypedStudyValue<ObservationTarget> = object_payload(name, &args[2], TARGET_CLASS)?;
    let sampling: TypedStudyValue<SamplingPolicy> =
        object_payload(name, &args[10], SAMPLING_CLASS)?;
    let spec = ObservationSpec {
        observation_id: ObservationId(nonempty_id(
            name,
            "observation id",
            scalar_string(&args[0], name, &ERROR_INPUT)?,
        )?),
        quantity,
        value_shape: canonical_shape(quantity),
        target: target.value,
        source_location: policy::field_location(name, &args[3])?,
        output_location: policy::field_location(name, &args[4])?,
        recovery: policy::recovery(name, &args[5])?,
        component: policy::component(name, &args[6])?,
        frame: constraints::support::coordinate_frame_from_value(name, &args[7])?,
        sign_convention: policy::sign(name, &args[8])?,
        complex_representation: policy::complex_representation(name, &args[9])?,
        sampling: sampling.value,
        reduction: from_typed_value(name, REDUCTION_CLASS, &args[11])?,
        storage: object_payload(name, &args[12], STORAGE_CLASS)?,
        purpose: policy::purpose(name, &args[13])?,
        cost: object_payload(name, &args[14], COST_CLASS)?,
    };
    let request = match form {
        ObservationForm::Field => ObservationRequest::Field(spec),
        ObservationForm::History => ObservationRequest::History(spec),
        ObservationForm::Probe => ObservationRequest::Probe(spec),
        ObservationForm::Reduction => ObservationRequest::Reduction(spec),
    };
    to_typed_value(name, OBSERVATION_CLASS, request)
}
macro_rules! observation_builtin {
    ($function:ident, $name:literal, $descriptor:ident, $form:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed scientific observation request.", keywords = "fea,observation,field,history,probe,reduction,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(args: Vec<Value>) -> BuiltinResult<Value> { observation($name, args, $form) }
    };
}
observation_builtin!(
    field,
    "fea.Observation.field",
    FIELD_DESCRIPTOR,
    ObservationForm::Field
);
observation_builtin!(
    history,
    "fea.Observation.history",
    HISTORY_DESCRIPTOR,
    ObservationForm::History
);
observation_builtin!(
    probe,
    "fea.Observation.probe",
    PROBE_DESCRIPTOR,
    ObservationForm::Probe
);
observation_builtin!(
    reduction_observation,
    "fea.Observation.reduction",
    REDUCTION_OBSERVATION_DESCRIPTOR,
    ObservationForm::Reduction
);

const DIAGNOSTIC_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Diagnostic observation identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed diagnostic kind."),
    required("sampling", BuiltinParamType::Any, "Typed sampling policy."),
    required(
        "retention",
        BuiltinParamType::Any,
        "Typed retention policy.",
    ),
];
descriptor!(
    DIAGNOSTIC_DESCRIPTOR,
    DIAGNOSTIC_SIGNATURES,
    "observation = fea.Observation.diagnostic(id, kind, sampling, retention)",
    &DIAGNOSTIC_INPUTS
);
#[runtime_builtin(
    name = "fea.Observation.diagnostic",
    category = "fea",
    summary = "Create a typed diagnostic observation request.",
    keywords = "fea,observation,diagnostic,typed",
    descriptor(DIAGNOSTIC_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn diagnostic(
    id: String,
    kind: Value,
    sampling: Value,
    retention: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.diagnostic";
    let sampling: TypedStudyValue<SamplingPolicy> =
        object_payload(NAME, &sampling, SAMPLING_CLASS)?;
    let retention: TypedStudyValue<RetentionPolicy> =
        object_payload(NAME, &retention, RETENTION_CLASS)?;
    to_typed_value(
        NAME,
        OBSERVATION_CLASS,
        ObservationRequest::Diagnostic(DiagnosticObservation {
            observation_id: ObservationId(nonempty_id(NAME, "observation id", id)?),
            diagnostic: policy::diagnostic_kind(NAME, &kind)?,
            sampling: sampling.value,
            retention: retention.value,
        }),
    )
}

const CADENCE_ACCEPTED_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interval",
    BuiltinParamType::NumericScalar,
    "Positive accepted-increment interval.",
)];
descriptor!(
    CADENCE_ACCEPTED_DESCRIPTOR,
    CADENCE_ACCEPTED_SIGNATURES,
    "cadence = fea.MonitorCadence.acceptedIncrements(interval)",
    &CADENCE_ACCEPTED_INPUTS
);
descriptor!(
    CADENCE_WALL_DESCRIPTOR,
    CADENCE_WALL_SIGNATURES,
    "cadence = fea.MonitorCadence.wallSeconds(interval)",
    &ONE_VALUE
);
#[runtime_builtin(
    name = "fea.MonitorCadence.acceptedIncrements",
    category = "fea",
    summary = "Evaluate a monitor every bounded number of accepted increments.",
    keywords = "fea,observation,monitor,cadence,typed",
    descriptor(CADENCE_ACCEPTED_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn cadence_accepted(interval: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.MonitorCadence.acceptedIncrements",
        CADENCE_CLASS,
        MonitorCadence::AcceptedIncrements {
            interval: usize_from_value("fea.MonitorCadence.acceptedIncrements", &interval)?,
        },
    )
}
#[runtime_builtin(
    name = "fea.MonitorCadence.wallSeconds",
    category = "fea",
    summary = "Evaluate a monitor at a wall-clock cadence.",
    keywords = "fea,observation,monitor,cadence,typed",
    descriptor(CADENCE_WALL_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn cadence_wall(interval: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorCadence.wallSeconds";
    to_typed_value(
        NAME,
        CADENCE_CLASS,
        MonitorCadence::WallSeconds {
            interval: values::finite_numeric_scalar(NAME, &interval, "interval")?,
        },
    )
}

const STOP_THRESHOLD_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed threshold quantity.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite threshold in canonical units.",
    ),
];
const STOP_CHANGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed threshold quantity.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite change threshold in canonical units.",
    ),
    required(
        "consecutive",
        BuiltinParamType::NumericScalar,
        "Positive consecutive-sample count.",
    ),
];
descriptor!(
    STOP_NONE_DESCRIPTOR,
    STOP_NONE_SIGNATURES,
    "condition = fea.MonitorStopCondition.none()",
    &NO_INPUTS
);
descriptor!(
    STOP_LESS_DESCRIPTOR,
    STOP_LESS_SIGNATURES,
    "condition = fea.MonitorStopCondition.lessThan(quantity, canonicalValue)",
    &STOP_THRESHOLD_INPUTS
);
descriptor!(
    STOP_GREATER_DESCRIPTOR,
    STOP_GREATER_SIGNATURES,
    "condition = fea.MonitorStopCondition.greaterThan(quantity, canonicalValue)",
    &STOP_THRESHOLD_INPUTS
);
descriptor!(STOP_CHANGE_DESCRIPTOR, STOP_CHANGE_SIGNATURES, "condition = fea.MonitorStopCondition.absoluteChangeBelow(quantity, canonicalValue, consecutive)", &STOP_CHANGE_INPUTS);
#[runtime_builtin(
    name = "fea.MonitorStopCondition.none",
    category = "fea",
    summary = "Create a progress-only monitor with no stop condition.",
    keywords = "fea,observation,monitor,stop,typed",
    descriptor(STOP_NONE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_none() -> BuiltinResult<Value> {
    to_typed_value::<Option<MonitorStopCondition>>(
        "fea.MonitorStopCondition.none",
        STOP_CLASS,
        None,
    )
}
fn threshold(
    name: &'static str,
    quantity: &Value,
    value: &Value,
) -> BuiltinResult<ObservationThreshold> {
    Ok(ObservationThreshold {
        quantity: policy::quantity(name, quantity)?,
        canonical_value: values::finite_numeric_scalar(name, value, "canonicalValue")?,
    })
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.lessThan",
    category = "fea",
    summary = "Stop when a scalar observation is below a typed threshold.",
    keywords = "fea,observation,monitor,stop,threshold,typed",
    descriptor(STOP_LESS_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_less(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.lessThan";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::LessThan {
            threshold: threshold(NAME, &quantity, &value)?,
        }),
    )
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.greaterThan",
    category = "fea",
    summary = "Stop when a scalar observation exceeds a typed threshold.",
    keywords = "fea,observation,monitor,stop,threshold,typed",
    descriptor(STOP_GREATER_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_greater(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.greaterThan";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::GreaterThan {
            threshold: threshold(NAME, &quantity, &value)?,
        }),
    )
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.absoluteChangeBelow",
    category = "fea",
    summary = "Stop after a scalar observation remains below a change threshold.",
    keywords = "fea,observation,monitor,stop,change,typed",
    descriptor(STOP_CHANGE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_change(
    quantity: Value,
    value: Value,
    consecutive: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.absoluteChangeBelow";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::AbsoluteChangeBelow {
            threshold: threshold(NAME, &quantity, &value)?,
            consecutive: usize_from_value(NAME, &consecutive)?,
        }),
    )
}

const MONITOR_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Monitor observation identifier.",
    ),
    required(
        "sourceObservationId",
        BuiltinParamType::StringScalar,
        "Scalar source observation identifier.",
    ),
    required("cadence", BuiltinParamType::Any, "Typed monitor cadence."),
    required(
        "stop",
        BuiltinParamType::Any,
        "Typed optional stop condition.",
    ),
];
descriptor!(
    MONITOR_DESCRIPTOR,
    MONITOR_SIGNATURES,
    "observation = fea.Observation.monitor(id, sourceObservationId, cadence, stop)",
    &MONITOR_INPUTS
);
#[runtime_builtin(
    name = "fea.Observation.monitor",
    category = "fea",
    summary = "Create a bounded scalar solver monitor.",
    keywords = "fea,observation,monitor,solver,typed",
    descriptor(MONITOR_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn monitor(
    id: String,
    source: String,
    cadence: Value,
    stop: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.monitor";
    let cadence: TypedStudyValue<MonitorCadence> = object_payload(NAME, &cadence, CADENCE_CLASS)?;
    let stop: TypedStudyValue<Option<MonitorStopCondition>> =
        object_payload(NAME, &stop, STOP_CLASS)?;
    to_typed_value(
        NAME,
        OBSERVATION_CLASS,
        ObservationRequest::Monitor(MonitorObservation {
            observation_id: ObservationId(nonempty_id(NAME, "monitor id", id)?),
            source_observation_id: ObservationId(nonempty_id(
                NAME,
                "source observation id",
                source,
            )?),
            cadence: cadence.value,
            stop: stop.value,
        }),
    )
}

pub(crate) fn observation_requests(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ObservationRequest>> {
    let wrapped: Vec<TypedStudyValue<ObservationRequest>> =
        object_or_cell(name, value, OBSERVATION_CLASS)?;
    Ok(wrapped.into_iter().map(|value| value.value).collect())
}

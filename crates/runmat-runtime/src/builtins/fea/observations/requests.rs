use super::*;
use runmat_builtins::fea_contract::study_types::descriptors::{
    monitor_cadence::*, monitor_stop_condition::*, observation::*,
};

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
        reduction: optional_typed_value(name, REDUCTION_CLASS, &args[11])?,
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
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::requests")]
        pub async fn $function(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
            study_outputs(&$descriptor, $name)?;
            observation($name, args, $form)
        }
    };
}
observation_builtin!(
    field,
    "fea.Observation.field",
    FEA_OBSERVATION_FIELD_DESCRIPTOR,
    ObservationForm::Field
);
observation_builtin!(
    history,
    "fea.Observation.history",
    FEA_OBSERVATION_HISTORY_DESCRIPTOR,
    ObservationForm::History
);
observation_builtin!(
    probe,
    "fea.Observation.probe",
    FEA_OBSERVATION_PROBE_DESCRIPTOR,
    ObservationForm::Probe
);
observation_builtin!(
    reduction_observation,
    "fea.Observation.reduction",
    FEA_OBSERVATION_REDUCTION_DESCRIPTOR,
    ObservationForm::Reduction
);

#[runtime_builtin(
    name = "fea.Observation.diagnostic",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn diagnostic(
    id: String,
    kind: Value,
    sampling: Value,
    retention: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.diagnostic";
    study_outputs(&FEA_OBSERVATION_DIAGNOSTIC_DESCRIPTOR, NAME)?;
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

#[runtime_builtin(
    name = "fea.MonitorCadence.acceptedIncrements",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn cadence_accepted(interval: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_MONITOR_CADENCE_ACCEPTED_INCREMENTS_DESCRIPTOR,
        "fea.MonitorCadence.acceptedIncrements",
    )?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn cadence_wall(interval: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorCadence.wallSeconds";
    study_outputs(&FEA_MONITOR_CADENCE_WALL_SECONDS_DESCRIPTOR, NAME)?;
    to_typed_value(
        NAME,
        CADENCE_CLASS,
        MonitorCadence::WallSeconds {
            interval: values::finite_numeric_scalar(NAME, &interval, "interval")?,
        },
    )
}

#[runtime_builtin(
    name = "fea.MonitorStopCondition.none",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn stop_none() -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.none";
    study_outputs(&FEA_MONITOR_STOP_CONDITION_NONE_DESCRIPTOR, NAME)?;
    absent_typed_value(NAME, STOP_CLASS)
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn stop_less(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.lessThan";
    study_outputs(&FEA_MONITOR_STOP_CONDITION_LESS_THAN_DESCRIPTOR, NAME)?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn stop_greater(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.greaterThan";
    study_outputs(&FEA_MONITOR_STOP_CONDITION_GREATER_THAN_DESCRIPTOR, NAME)?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn stop_change(
    quantity: Value,
    value: Value,
    consecutive: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.absoluteChangeBelow";
    study_outputs(
        &FEA_MONITOR_STOP_CONDITION_ABSOLUTE_CHANGE_BELOW_DESCRIPTOR,
        NAME,
    )?;
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::AbsoluteChangeBelow {
            threshold: threshold(NAME, &quantity, &value)?,
            consecutive: usize_from_value(NAME, &consecutive)?,
        }),
    )
}

#[runtime_builtin(
    name = "fea.Observation.monitor",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::requests"
)]
pub async fn monitor(
    id: String,
    source: String,
    cadence: Value,
    stop: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.monitor";
    study_outputs(&FEA_OBSERVATION_MONITOR_DESCRIPTOR, NAME)?;
    let cadence: TypedStudyValue<MonitorCadence> = object_payload(NAME, &cadence, CADENCE_CLASS)?;
    let stop = optional_typed_value::<MonitorStopCondition>(NAME, STOP_CLASS, &stop)?;
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
            stop,
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

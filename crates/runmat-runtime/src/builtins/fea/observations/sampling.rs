use super::*;
use runmat_builtins::fea_contract::study_types::descriptors::{
    observation_event::*, sampling_policy::*,
};

macro_rules! sampling_no_arg {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function() -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value($name, SAMPLING_CLASS, $value)
        }
    };
}
sampling_no_arg!(
    sampling_final,
    "fea.SamplingPolicy.finalSample",
    FEA_SAMPLING_POLICY_FINAL_SAMPLE_DESCRIPTOR,
    SamplingPolicy::Final
);
sampling_no_arg!(
    sampling_increment,
    "fea.SamplingPolicy.everyAcceptedIncrement",
    FEA_SAMPLING_POLICY_EVERY_ACCEPTED_INCREMENT_DESCRIPTOR,
    SamplingPolicy::EveryAcceptedIncrement
);
macro_rules! sampling_values {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function(values: Value) -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value(
                $name,
                SAMPLING_CLASS,
                ($wrap)(numeric_values($name, &values)?),
            )
        }
    };
}
sampling_values!(
    sampling_time,
    "fea.SamplingPolicy.timeSeconds",
    FEA_SAMPLING_POLICY_TIME_SECONDS_DESCRIPTOR,
    SamplingPolicy::TimeSeconds
);
sampling_values!(
    sampling_frequency,
    "fea.SamplingPolicy.frequencyHertz",
    FEA_SAMPLING_POLICY_FREQUENCY_HERTZ_DESCRIPTOR,
    SamplingPolicy::FrequencyHertz
);
sampling_values!(
    sampling_load,
    "fea.SamplingPolicy.loadFactors",
    FEA_SAMPLING_POLICY_LOAD_FACTORS_DESCRIPTOR,
    SamplingPolicy::LoadFactors
);
sampling_values!(
    sampling_study_parameter,
    "fea.SamplingPolicy.studyParameterValues",
    FEA_SAMPLING_POLICY_STUDY_PARAMETER_VALUES_DESCRIPTOR,
    SamplingPolicy::StudyParameterValues
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.modeIndices",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_modes(values: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SamplingPolicy.modeIndices";
    study_outputs(&FEA_SAMPLING_POLICY_MODE_INDICES_DESCRIPTOR, NAME)?;
    // Mode indices are one-based in the analysis contract (validation rejects 0
    // and bounds the last index by the requested mode count), so the public
    // one-based selection is stored as given after rejecting non-positive values.
    let indices = usize_vec_from_value(NAME, &values)?;
    if indices.contains(&0) {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "mode indices are one-based and must be positive",
        ));
    }
    to_typed_value(NAME, SAMPLING_CLASS, SamplingPolicy::ModeIndices(indices))
}
macro_rules! sampling_interval {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function(start: Value, end: Value, interval: Value) -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value(
                $name,
                SAMPLING_CLASS,
                ($wrap)(
                    values::finite_numeric_scalar($name, &start, "start")?,
                    values::finite_numeric_scalar($name, &end, "end")?,
                    values::finite_numeric_scalar($name, &interval, "interval")?,
                ),
            )
        }
    };
}
sampling_interval!(
    sampling_time_interval,
    "fea.SamplingPolicy.timeIntervalSeconds",
    FEA_SAMPLING_POLICY_TIME_INTERVAL_SECONDS_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::TimeIntervalSeconds {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_frequency_interval,
    "fea.SamplingPolicy.frequencyIntervalHertz",
    FEA_SAMPLING_POLICY_FREQUENCY_INTERVAL_HERTZ_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::FrequencyIntervalHertz {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_load_interval,
    "fea.SamplingPolicy.loadFactorInterval",
    FEA_SAMPLING_POLICY_LOAD_FACTOR_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::LoadFactorInterval {
        start,
        end,
        interval
    }
);

macro_rules! event_choice {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function() -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value($name, EVENT_CLASS, $value)
        }
    };
}
event_choice!(
    event_first_yield,
    "fea.ObservationEvent.firstYield",
    FEA_OBSERVATION_EVENT_FIRST_YIELD_DESCRIPTOR,
    ObservationEvent::Solver(SolverObservationEvent::FirstYield)
);
event_choice!(
    event_contact,
    "fea.ObservationEvent.contactStatusChange",
    FEA_OBSERVATION_EVENT_CONTACT_STATUS_CHANGE_DESCRIPTOR,
    ObservationEvent::Solver(SolverObservationEvent::ContactStatusChange)
);
event_choice!(
    event_peak,
    "fea.ObservationEvent.peakResponse",
    FEA_OBSERVATION_EVENT_PEAK_RESPONSE_DESCRIPTOR,
    ObservationEvent::Series(SeriesObservationEvent::PeakResponse)
);
event_choice!(
    event_failure,
    "fea.ObservationEvent.convergenceFailure",
    FEA_OBSERVATION_EVENT_CONVERGENCE_FAILURE_DESCRIPTOR,
    ObservationEvent::Solver(SolverObservationEvent::ConvergenceFailure)
);
event_choice!(
    event_accuracy,
    "fea.ObservationEvent.accuracyCriterionCrossing",
    FEA_OBSERVATION_EVENT_ACCURACY_CRITERION_CROSSING_DESCRIPTOR,
    ObservationEvent::Accuracy(AccuracyObservationEvent::CriterionCrossing)
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.event",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_event(event: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_SAMPLING_POLICY_EVENT_DESCRIPTOR,
        "fea.SamplingPolicy.event",
    )?;
    let event: TypedStudyValue<ObservationEvent> =
        object_payload("fea.SamplingPolicy.event", &event, EVENT_CLASS)?;
    to_typed_value(
        "fea.SamplingPolicy.event",
        SAMPLING_CLASS,
        SamplingPolicy::Event(event.value),
    )
}
#[runtime_builtin(
    name = "fea.SamplingPolicy.adaptive",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_adaptive(
    minimum: Value,
    maximum: Value,
    relative: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SamplingPolicy.adaptive";
    study_outputs(&FEA_SAMPLING_POLICY_ADAPTIVE_DESCRIPTOR, NAME)?;
    to_typed_value(
        NAME,
        SAMPLING_CLASS,
        SamplingPolicy::Adaptive {
            minimum_interval: values::finite_numeric_scalar(NAME, &minimum, "minimumInterval")?,
            maximum_samples: usize_from_value(NAME, &maximum)?,
            relative_change: values::finite_numeric_scalar(NAME, &relative, "relativeChange")?,
        },
    )
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    fn event_payload(value: &Value) -> ObservationEvent {
        object_payload::<TypedStudyValue<ObservationEvent>>("test", value, EVENT_CLASS)
            .unwrap()
            .value
    }

    #[test]
    fn event_constructors_preserve_their_domain_owner() {
        assert_eq!(
            event_payload(&block_on(event_first_yield()).unwrap()),
            ObservationEvent::Solver(SolverObservationEvent::FirstYield)
        );
        assert_eq!(
            event_payload(&block_on(event_peak()).unwrap()),
            ObservationEvent::Series(SeriesObservationEvent::PeakResponse)
        );
        assert_eq!(
            event_payload(&block_on(event_accuracy()).unwrap()),
            ObservationEvent::Accuracy(AccuracyObservationEvent::CriterionCrossing)
        );
    }

    #[test]
    fn sampling_event_keeps_the_typed_event() {
        let event = block_on(event_contact()).unwrap();
        let policy = block_on(sampling_event(event)).unwrap();
        let payload =
            object_payload::<TypedStudyValue<SamplingPolicy>>("test", &policy, SAMPLING_CLASS)
                .unwrap();
        assert_eq!(
            payload.value,
            SamplingPolicy::Event(ObservationEvent::Solver(
                SolverObservationEvent::ContactStatusChange,
            ))
        );
    }
}

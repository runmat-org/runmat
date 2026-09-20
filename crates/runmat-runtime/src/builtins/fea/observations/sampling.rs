use super::*;

macro_rules! sampling_descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        descriptor!($descriptor, $signatures, $label, $inputs);
    };
}
sampling_descriptor!(
    SAMPLING_FINAL_DESCRIPTOR,
    SAMPLING_FINAL_SIGNATURES,
    "policy = fea.SamplingPolicy.finalSample()",
    &NO_INPUTS
);
sampling_descriptor!(
    SAMPLING_INCREMENT_DESCRIPTOR,
    SAMPLING_INCREMENT_SIGNATURES,
    "policy = fea.SamplingPolicy.everyAcceptedIncrement()",
    &NO_INPUTS
);
sampling_descriptor!(
    SAMPLING_TIME_DESCRIPTOR,
    SAMPLING_TIME_SIGNATURES,
    "policy = fea.SamplingPolicy.timeSeconds(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_TIME_INTERVAL_DESCRIPTOR,
    SAMPLING_TIME_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.timeIntervalSeconds(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_FREQUENCY_DESCRIPTOR,
    SAMPLING_FREQUENCY_SIGNATURES,
    "policy = fea.SamplingPolicy.frequencyHertz(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_FREQUENCY_INTERVAL_DESCRIPTOR,
    SAMPLING_FREQUENCY_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.frequencyIntervalHertz(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_LOAD_DESCRIPTOR,
    SAMPLING_LOAD_SIGNATURES,
    "policy = fea.SamplingPolicy.loadFactors(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_LOAD_INTERVAL_DESCRIPTOR,
    SAMPLING_LOAD_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.loadFactorInterval(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_STUDY_PARAMETER_DESCRIPTOR,
    SAMPLING_STUDY_PARAMETER_SIGNATURES,
    "policy = fea.SamplingPolicy.studyParameterValues(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_MODES_DESCRIPTOR,
    SAMPLING_MODES_SIGNATURES,
    "policy = fea.SamplingPolicy.modeIndices(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_EVENT_DESCRIPTOR,
    SAMPLING_EVENT_SIGNATURES,
    "policy = fea.SamplingPolicy.event(event)",
    &[required(
        "event",
        BuiltinParamType::Any,
        "Typed observation event."
    )]
);
sampling_descriptor!(
    SAMPLING_ADAPTIVE_DESCRIPTOR,
    SAMPLING_ADAPTIVE_SIGNATURES,
    "policy = fea.SamplingPolicy.adaptive(minimumInterval, maximumSamples, relativeChange)",
    &ADAPTIVE
);

macro_rules! sampling_no_arg {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed observation sampling policy.", keywords = "fea,observation,sampling,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, SAMPLING_CLASS, $value) }
    };
}
sampling_no_arg!(
    sampling_final,
    "fea.SamplingPolicy.finalSample",
    SAMPLING_FINAL_DESCRIPTOR,
    SamplingPolicy::Final
);
sampling_no_arg!(
    sampling_increment,
    "fea.SamplingPolicy.everyAcceptedIncrement",
    SAMPLING_INCREMENT_DESCRIPTOR,
    SamplingPolicy::EveryAcceptedIncrement
);
macro_rules! sampling_values {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed observation sampling policy.", keywords = "fea,observation,sampling,typed", descriptor($descriptor), integer_capabilities(NUMERIC_CAPABILITIES), builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function(values: Value) -> BuiltinResult<Value> { to_typed_value($name, SAMPLING_CLASS, ($wrap)(numeric_values($name, &values)?)) }
    };
}
sampling_values!(
    sampling_time,
    "fea.SamplingPolicy.timeSeconds",
    SAMPLING_TIME_DESCRIPTOR,
    SamplingPolicy::TimeSeconds
);
sampling_values!(
    sampling_frequency,
    "fea.SamplingPolicy.frequencyHertz",
    SAMPLING_FREQUENCY_DESCRIPTOR,
    SamplingPolicy::FrequencyHertz
);
sampling_values!(
    sampling_load,
    "fea.SamplingPolicy.loadFactors",
    SAMPLING_LOAD_DESCRIPTOR,
    SamplingPolicy::LoadFactors
);
sampling_values!(
    sampling_study_parameter,
    "fea.SamplingPolicy.studyParameterValues",
    SAMPLING_STUDY_PARAMETER_DESCRIPTOR,
    SamplingPolicy::StudyParameterValues
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.modeIndices",
    category = "fea",
    summary = "Sample selected one-based mode indices.",
    keywords = "fea,observation,sampling,modes,typed",
    descriptor(SAMPLING_MODES_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_modes(values: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.SamplingPolicy.modeIndices",
        SAMPLING_CLASS,
        SamplingPolicy::ModeIndices(one_based_usize_vec_from_value(
            "fea.SamplingPolicy.modeIndices",
            &values,
        )?),
    )
}
macro_rules! sampling_interval {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed interval sampling policy.", keywords = "fea,observation,sampling,interval,typed", descriptor($descriptor), integer_capabilities(NUMERIC_CAPABILITIES), builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function(start: Value, end: Value, interval: Value) -> BuiltinResult<Value> {
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
    SAMPLING_TIME_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::TimeIntervalSeconds {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_frequency_interval,
    "fea.SamplingPolicy.frequencyIntervalHertz",
    SAMPLING_FREQUENCY_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::FrequencyIntervalHertz {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_load_interval,
    "fea.SamplingPolicy.loadFactorInterval",
    SAMPLING_LOAD_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::LoadFactorInterval {
        start,
        end,
        interval
    }
);

macro_rules! event_choice {
    ($function:ident, $name:literal, $descriptor:ident, $signatures:ident, $label:literal, $value:expr) => {
        descriptor!($descriptor, $signatures, $label, &NO_INPUTS);
        #[runtime_builtin(name = $name, category = "fea", summary = "Select a typed observation event.", keywords = "fea,observation,event,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations::sampling")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, EVENT_CLASS, $value) }
    };
}
event_choice!(
    event_first_yield,
    "fea.ObservationEvent.firstYield",
    EVENT_FIRST_YIELD_DESCRIPTOR,
    EVENT_FIRST_YIELD_SIGNATURES,
    "event = fea.ObservationEvent.firstYield()",
    ObservationEvent::Solver(SolverObservationEvent::FirstYield)
);
event_choice!(
    event_contact,
    "fea.ObservationEvent.contactStatusChange",
    EVENT_CONTACT_DESCRIPTOR,
    EVENT_CONTACT_SIGNATURES,
    "event = fea.ObservationEvent.contactStatusChange()",
    ObservationEvent::Solver(SolverObservationEvent::ContactStatusChange)
);
event_choice!(
    event_peak,
    "fea.ObservationEvent.peakResponse",
    EVENT_PEAK_DESCRIPTOR,
    EVENT_PEAK_SIGNATURES,
    "event = fea.ObservationEvent.peakResponse()",
    ObservationEvent::Series(SeriesObservationEvent::PeakResponse)
);
event_choice!(
    event_failure,
    "fea.ObservationEvent.convergenceFailure",
    EVENT_FAILURE_DESCRIPTOR,
    EVENT_FAILURE_SIGNATURES,
    "event = fea.ObservationEvent.convergenceFailure()",
    ObservationEvent::Solver(SolverObservationEvent::ConvergenceFailure)
);
event_choice!(
    event_accuracy,
    "fea.ObservationEvent.accuracyCriterionCrossing",
    EVENT_ACCURACY_DESCRIPTOR,
    EVENT_ACCURACY_SIGNATURES,
    "event = fea.ObservationEvent.accuracyCriterionCrossing()",
    ObservationEvent::Accuracy(AccuracyObservationEvent::CriterionCrossing)
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.event",
    category = "fea",
    summary = "Sample when a typed observation event occurs.",
    keywords = "fea,observation,sampling,event,typed",
    descriptor(SAMPLING_EVENT_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_event(event: Value) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Create bounded adaptive observation sampling.",
    keywords = "fea,observation,sampling,adaptive,typed",
    descriptor(SAMPLING_ADAPTIVE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::sampling"
)]
pub async fn sampling_adaptive(
    minimum: Value,
    maximum: Value,
    relative: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SamplingPolicy.adaptive";
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

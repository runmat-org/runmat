//! Signatures for the `fea.SamplingPolicy` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "policy",
    BuiltinParamType::Any,
    "Immutable fea.SamplingPolicy value selecting which solution samples an observation retains.",
)];

const FINAL_SAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.finalSample()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_FINAL_SAMPLE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FINAL_SAMPLE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const EVERY_ACCEPTED_INCREMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.everyAcceptedIncrement()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_EVERY_ACCEPTED_INCREMENT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&EVERY_ACCEPTED_INCREMENT_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const TIME_SECONDS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite sample times in seconds.",
)];
const TIME_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.timeSeconds(values)",
    &TIME_SECONDS_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_TIME_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&TIME_SECONDS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const TIME_INTERVAL_SECONDS_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Interval start in seconds.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Interval end in seconds.",
    ),
    required(
        "interval",
        BuiltinParamType::NumericScalar,
        "Positive spacing in seconds.",
    ),
];
const TIME_INTERVAL_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.timeIntervalSeconds(start, end, interval)",
    &TIME_INTERVAL_SECONDS_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_TIME_INTERVAL_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&TIME_INTERVAL_SECONDS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const FREQUENCY_HERTZ_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite sample frequencies in hertz.",
)];
const FREQUENCY_HERTZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.frequencyHertz(values)",
    &FREQUENCY_HERTZ_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_FREQUENCY_HERTZ_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FREQUENCY_HERTZ_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const FREQUENCY_INTERVAL_HERTZ_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Interval start in hertz.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Interval end in hertz.",
    ),
    required(
        "interval",
        BuiltinParamType::NumericScalar,
        "Positive spacing in hertz.",
    ),
];
const FREQUENCY_INTERVAL_HERTZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.frequencyIntervalHertz(start, end, interval)",
    &FREQUENCY_INTERVAL_HERTZ_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_FREQUENCY_INTERVAL_HERTZ_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FREQUENCY_INTERVAL_HERTZ_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const LOAD_FACTORS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite load factors (dimensionless).",
)];
const LOAD_FACTORS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.loadFactors(values)",
    &LOAD_FACTORS_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_LOAD_FACTORS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LOAD_FACTORS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const LOAD_FACTOR_INTERVAL_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Interval start in load-factor units.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Interval end in load-factor units.",
    ),
    required(
        "interval",
        BuiltinParamType::NumericScalar,
        "Positive spacing in load-factor units.",
    ),
];
const LOAD_FACTOR_INTERVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.loadFactorInterval(start, end, interval)",
    &LOAD_FACTOR_INTERVAL_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_LOAD_FACTOR_INTERVAL_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LOAD_FACTOR_INTERVAL_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const STUDY_PARAMETER_VALUES_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite canonical study-parameter values.",
)];
const STUDY_PARAMETER_VALUES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.studyParameterValues(values)",
    &STUDY_PARAMETER_VALUES_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_STUDY_PARAMETER_VALUES_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&STUDY_PARAMETER_VALUES_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const MODE_INDICES_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Positive one-based mode indices.",
)];
const MODE_INDICES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.modeIndices(values)",
    &MODE_INDICES_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_MODE_INDICES_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&MODE_INDICES_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const EVENT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "event",
    BuiltinParamType::Any,
    "Typed fea.ObservationEvent.",
)];
const EVENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.event(event)",
    &EVENT_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_EVENT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&EVENT_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const ADAPTIVE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "minimumInterval",
        BuiltinParamType::NumericScalar,
        "Smallest spacing between samples, in domain units.",
    ),
    required(
        "maximumSamples",
        BuiltinParamType::NumericScalar,
        "Upper bound on retained samples.",
    ),
    required(
        "relativeChange",
        BuiltinParamType::NumericScalar,
        "Relative change that triggers a new sample.",
    ),
];
const ADAPTIVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.SamplingPolicy.adaptive(minimumInterval, maximumSamples, relativeChange)",
    &ADAPTIVE_INPUTS,
    &OUTPUT,
)];
pub const FEA_SAMPLING_POLICY_ADAPTIVE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&ADAPTIVE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

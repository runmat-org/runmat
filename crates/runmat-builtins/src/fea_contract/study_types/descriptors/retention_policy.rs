//! Signatures for the `fea.RetentionPolicy` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::Any,
    "Immutable fea.RetentionPolicy value deciding how long and how many samples are kept.",
)];

const CANONICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.RetentionPolicy.canonical()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_RETENTION_POLICY_CANONICAL_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CANONICAL_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const UNTIL_DEPENDENTS_COMPLETE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.RetentionPolicy.untilDependentsComplete()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_RETENTION_POLICY_UNTIL_DEPENDENTS_COMPLETE_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &UNTIL_DEPENDENTS_COMPLETE_SIGNATURES,
    &FEA_STUDY_TYPE_ERRORS,
);

const LATEST_SAMPLES_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "count",
    BuiltinParamType::NumericScalar,
    "Number of most recent samples to keep.",
)];
const LATEST_SAMPLES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.RetentionPolicy.latestSamples(count)",
    &LATEST_SAMPLES_INPUTS,
    &OUTPUT,
)];
pub const FEA_RETENTION_POLICY_LATEST_SAMPLES_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LATEST_SAMPLES_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const EXPLICIT_SAMPLES_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "maximumCount",
    BuiltinParamType::NumericScalar,
    "Upper bound on retained samples.",
)];
const EXPLICIT_SAMPLES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.RetentionPolicy.explicitSamples(maximumCount)",
    &EXPLICIT_SAMPLES_INPUTS,
    &OUTPUT,
)];
pub const FEA_RETENTION_POLICY_EXPLICIT_SAMPLES_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&EXPLICIT_SAMPLES_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

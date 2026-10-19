//! Signatures for the `fea.IndependentDomain` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "domain",
    BuiltinParamType::Any,
    "Immutable fea.IndependentDomain value.",
)];

const STATIC_DOMAIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.staticDomain()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_STATIC_DOMAIN_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&STATIC_DOMAIN_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const TIME_SECONDS_INPUTS: [BuiltinParamDescriptor; 4] = [
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
    required(
        "intervalCount",
        BuiltinParamType::NumericScalar,
        "Positive number of equal time intervals.",
    ),
];
const TIME_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.timeSeconds(origin, start, end, intervalCount)",
    &TIME_SECONDS_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_TIME_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&TIME_SECONDS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const FREQUENCY_HERTZ_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite frequencies in hertz.",
)];
const FREQUENCY_HERTZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.frequencyHertz(values)",
    &FREQUENCY_HERTZ_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_FREQUENCY_HERTZ_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FREQUENCY_HERTZ_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const LOAD_FACTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Starting load factor.",
    ),
    required("end", BuiltinParamType::NumericScalar, "Final load factor."),
];
const LOAD_FACTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.loadFactor(start, end)",
    &LOAD_FACTOR_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_LOAD_FACTOR_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LOAD_FACTOR_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const EIGENVALUE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "count",
    BuiltinParamType::NumericScalar,
    "Requested number of eigenpairs.",
)];
const EIGENVALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.eigenvalue(count)",
    &EIGENVALUE_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_EIGENVALUE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&EIGENVALUE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const STUDY_PARAMETER_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const STUDY_PARAMETER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.studyParameter(id, values)",
    &STUDY_PARAMETER_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_STUDY_PARAMETER_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&STUDY_PARAMETER_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const RESPONSE_SPECTRA_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "spectra",
    BuiltinParamType::Any,
    "One fea.ResponseSpectrum or a cell array of them.",
)];
const RESPONSE_SPECTRA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "domain = fea.IndependentDomain.responseSpectra(spectra)",
    &RESPONSE_SPECTRA_INPUTS,
    &OUTPUT,
)];
pub const FEA_INDEPENDENT_DOMAIN_RESPONSE_SPECTRA_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&RESPONSE_SPECTRA_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

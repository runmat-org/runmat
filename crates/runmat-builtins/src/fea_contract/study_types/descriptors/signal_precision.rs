//! Signatures for the `fea.SignalPrecision` choices.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::Any,
    "Immutable fea.SignalPrecision choice for stored observation samples.",
)];

const REAL32_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.SignalPrecision.real32()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SIGNAL_PRECISION_REAL32_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&REAL32_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const REAL64_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.SignalPrecision.real64()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SIGNAL_PRECISION_REAL64_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&REAL64_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const COMPLEX64_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.SignalPrecision.complex64()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SIGNAL_PRECISION_COMPLEX64_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&COMPLEX64_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const COMPLEX128_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.SignalPrecision.complex128()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_SIGNAL_PRECISION_COMPLEX128_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&COMPLEX128_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

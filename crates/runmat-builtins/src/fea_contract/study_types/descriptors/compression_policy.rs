//! Signatures for the `fea.CompressionPolicy` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::Any,
    "Immutable fea.CompressionPolicy value for stored observation samples.",
)];

const LOSSLESS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.CompressionPolicy.lossless()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_COMPRESSION_POLICY_LOSSLESS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LOSSLESS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const NONE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.CompressionPolicy.none()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_COMPRESSION_POLICY_NONE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NONE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const ERROR_BOUNDED_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "maximumRelativeError",
    BuiltinParamType::NumericScalar,
    "Maximum relative error, a fraction in (0, 1).",
)];
const ERROR_BOUNDED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "value = fea.CompressionPolicy.errorBounded(maximumRelativeError)",
    &ERROR_BOUNDED_INPUTS,
    &OUTPUT,
)];
pub const FEA_COMPRESSION_POLICY_ERROR_BOUNDED_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&ERROR_BOUNDED_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

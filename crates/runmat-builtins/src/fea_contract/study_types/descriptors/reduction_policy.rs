//! Signatures for the `fea.ReductionPolicy` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "policy",
    BuiltinParamType::Any,
    "Immutable fea.ReductionPolicy value (absent or a spatial/series/tie triple).",
)];

const NONE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.ReductionPolicy.none()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_REDUCTION_POLICY_NONE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NONE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const NEW_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "spatial",
        BuiltinParamType::Any,
        "Typed fea.SpatialReduction.",
    ),
    required(
        "series",
        BuiltinParamType::Any,
        "Typed fea.SeriesReduction.",
    ),
    required(
        "extremaTies",
        BuiltinParamType::Any,
        "Typed fea.ExtremaTiePolicy.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.ReductionPolicy.new(spatial, series, extremaTies)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_REDUCTION_POLICY_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

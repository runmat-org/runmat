//! Signature for `fea.ObservationStorage.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "storage",
    BuiltinParamType::Any,
    "Immutable fea.ObservationStorage value.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "precision",
        BuiltinParamType::Any,
        "Typed fea.SignalPrecision.",
    ),
    required(
        "compression",
        BuiltinParamType::Any,
        "Typed fea.CompressionPolicy.",
    ),
    required(
        "retention",
        BuiltinParamType::Any,
        "Typed fea.RetentionPolicy.",
    ),
    required(
        "partialAccess",
        BuiltinParamType::Any,
        "Typed fea.PartialAccessPolicy.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "storage = fea.ObservationStorage.new(precision, compression, retention, partialAccess)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_STORAGE_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

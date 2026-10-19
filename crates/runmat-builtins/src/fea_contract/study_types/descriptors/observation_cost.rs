//! Signature for `fea.ObservationCost.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "cost",
    BuiltinParamType::Any,
    "Immutable fea.ObservationCost upper-bound estimate.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "operationCount",
        BuiltinParamType::NumericScalar,
        "Estimated arithmetic operation count.",
    ),
    required(
        "peakResidentBytes",
        BuiltinParamType::NumericScalar,
        "Estimated peak resident memory in bytes.",
    ),
    required(
        "storedBytes",
        BuiltinParamType::NumericScalar,
        "Estimated stored bytes.",
    ),
    required(
        "transferBytes",
        BuiltinParamType::NumericScalar,
        "Estimated bytes moved between processes or devices.",
    ),
    required(
        "collectiveCount",
        BuiltinParamType::NumericScalar,
        "Estimated number of collective operations.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("cost = fea.ObservationCost.new(operationCount, peakResidentBytes, storedBytes, transferBytes, collectiveCount)", &NEW_INPUTS, &OUTPUT)];
pub const FEA_OBSERVATION_COST_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

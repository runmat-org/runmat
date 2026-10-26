//! Signatures and errors for the `fea.ObservationTarget` constructors.

use crate::fea_contract::study_types::{descriptor, required, signature, NO_INPUTS};
use crate::*;

pub const FEA_OBSERVATION_TARGET_ERROR_REGION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.OBSERVATION_TARGET.REGION",
    identifier: Some("RunMat:geometry:region:InvalidRegion"),
    when: "A region target receives a value that is not a valid geometry.Region.",
    message: "geometry.region: invalid region selection",
};
const FEA_OBSERVATION_TARGET_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FEA_OBSERVATION_TARGET_ERROR_REGION,
    crate::fea_contract::study_types::ERROR_INPUT,
    crate::fea_contract::study_types::ERROR_INTERNAL,
    crate::fea_contract::study_types::FEA_STUDY_TYPE_ERROR_OUTPUTS,
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "Immutable fea.ObservationTarget value naming where an observation is evaluated.",
)];

const BOUNDARY_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "Typed geometry.Region whose boundary faces are observed.",
)];
const BOUNDARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.boundary(region)",
    &BOUNDARY_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_BOUNDARY_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&BOUNDARY_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const VOLUME_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "Typed geometry.Region whose volume cells are observed.",
)];
const VOLUME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.volume(region)",
    &VOLUME_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_VOLUME_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&VOLUME_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const POINT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "position",
    BuiltinParamType::Any,
    "Typed fea.Position in meters.",
)];
const POINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.point(position)",
    &POINT_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_POINT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&POINT_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const PATH_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "positions",
    BuiltinParamType::NumericArray,
    "N-by-3 path positions in meters, N at least two.",
)];
const PATH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.path(positions)",
    &PATH_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_PATH_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&PATH_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const INTERFACE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Identifier of an interface declared in the model.",
)];
const INTERFACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.interface(id)",
    &INTERFACE_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_INTERFACE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&INTERFACE_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const PORT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Identifier of a port declared in the model.",
)];
const PORT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.port(id)",
    &PORT_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_PORT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&PORT_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const CONNECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Identifier of a connector declared in the model.",
)];
const CONNECTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.connector(id)",
    &CONNECTOR_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_CONNECTOR_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CONNECTOR_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const CONSTRAINT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Identifier of a constraint declared in the model.",
)];
const CONSTRAINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.constraint(id)",
    &CONSTRAINT_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_CONSTRAINT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CONSTRAINT_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

const WHOLE_MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "target = fea.ObservationTarget.wholeModel()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_TARGET_WHOLE_MODEL_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&WHOLE_MODEL_SIGNATURES, &FEA_OBSERVATION_TARGET_ERRORS);

//! Signatures, parameters, and integer facts for `fea.PropertyRegionTarget` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const BEAM_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "geometry.Region selected from the model geometry.",
)];
const BEAM_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const BEAM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.beam(region)",
    inputs: &BEAM_INPUTS,
    outputs: &BEAM_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_BEAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BEAM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const SHELL_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "geometry.Region selected from the model geometry.",
)];
const SHELL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const SHELL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.shell(region)",
    inputs: &SHELL_INPUTS,
    outputs: &SHELL_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_SHELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const SOLID_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "geometry.Region selected from the model geometry.",
)];
const SOLID_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const SOLID_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.solid(region)",
    inputs: &SOLID_INPUTS,
    outputs: &SOLID_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_SOLID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SOLID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CELL_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "geometry.Region selected from the model geometry.",
)];
const CELL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const CELL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.cell(region)",
    inputs: &CELL_INPUTS,
    outputs: &CELL_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ALL_BEAMS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "geometry",
    BuiltinParamType::Any,
    "geometry.Asset whose compatible domains are selected.",
)];
const ALL_BEAMS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const ALL_BEAMS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.allBeams(geometry)",
    inputs: &ALL_BEAMS_INPUTS,
    outputs: &ALL_BEAMS_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_ALL_BEAMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALL_BEAMS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ALL_SHELLS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "geometry",
    BuiltinParamType::Any,
    "geometry.Asset whose compatible domains are selected.",
)];
const ALL_SHELLS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const ALL_SHELLS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.allShells(geometry)",
    inputs: &ALL_SHELLS_INPUTS,
    outputs: &ALL_SHELLS_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_ALL_SHELLS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALL_SHELLS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ALL_SOLIDS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "geometry",
    BuiltinParamType::Any,
    "geometry.Asset whose compatible domains are selected.",
)];
const ALL_SOLIDS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const ALL_SOLIDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.allSolids(geometry)",
    inputs: &ALL_SOLIDS_INPUTS,
    outputs: &ALL_SOLIDS_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_ALL_SOLIDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALL_SOLIDS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ALL_CELLS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "geometry",
    BuiltinParamType::Any,
    "geometry.Asset whose compatible domains are selected.",
)];
const ALL_CELLS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.PropertyRegionTarget object.",
)];
const ALL_CELLS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.PropertyRegionTarget.allCells(geometry)",
    inputs: &ALL_CELLS_INPUTS,
    outputs: &ALL_CELLS_OUTPUTS,
}];
pub const FEA_PROPERTY_REGION_TARGET_ALL_CELLS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALL_CELLS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

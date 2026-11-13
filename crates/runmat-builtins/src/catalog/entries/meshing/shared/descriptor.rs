//! Descriptors and error contracts for the meshing builtins.

use crate::*;

const MESH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mesh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Jointly validated meshing.Mesh object.",
}];
const MESH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mesh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "meshing.Mesh from meshing.generate or meshing.load.",
}];
const GENERATE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset loaded from exact CAD (STEP, IGES, or BREP).",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "TargetSize, MaximumDeviation, MaximumGradingRatio, ElementOrder, MaximumElements, or Seed.",
    },
];
const LOAD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "meshPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Canonical solver-mesh artifact path.",
    },
    BuiltinParamDescriptor {
        name: "evidencePath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Matching canonical meshing-evidence artifact path.",
    },
];
const SAVE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "mesh",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "meshing.Mesh to persist.",
    },
    BuiltinParamDescriptor {
        name: "meshPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination solver-mesh artifact path.",
    },
    BuiltinParamDescriptor {
        name: "evidencePath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination meshing-evidence path; must differ from meshPath.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Force (logical, default false) to replace existing destinations.",
    },
];
const SAVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "saved",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical true once both artifacts are published.",
}];
const DISPLAY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mesh",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "meshing.Mesh to display.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Display options: Edges, Regions, and Section.",
    },
];
const FIGURE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "figure",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle of the figure that shows the solver mesh.",
}];

const fn inspect_output(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const GENERATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mesh = meshing.generate(geometry, Name, Value, ...)",
    inputs: &GENERATE_INPUTS,
    outputs: &MESH_OUTPUT,
}];
const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mesh = meshing.load(meshPath, evidencePath)",
    inputs: &LOAD_INPUTS,
    outputs: &MESH_OUTPUT,
}];
const SAVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "saved = meshing.save(mesh, meshPath, evidencePath, Name, Value, ...)",
    inputs: &SAVE_INPUTS,
    outputs: &SAVE_OUTPUT,
}];
const NODES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "nodes = meshing.nodes(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "nodes",
        "Struct with ids (uint64) and coordinates (meters).",
    )],
}];
const ELEMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "elements = meshing.elements(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "elements",
        "Struct with ids, connectivity (uint64), order, and region_ids.",
    )],
}];
const BOUNDARY_FACES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "faces = meshing.boundaryFaces(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "faces",
        "Struct with ids, connectivity, roles, adjacent_element_ids, and provenance.",
    )],
}];
const REGIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "regions = meshing.regions(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "regions",
        "1-by-R cell of structs with id, kind, assembly_path, and element_count.",
    )],
}];
const QUALITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "quality = meshing.quality(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "quality",
        "Struct of per-element corner quality vectors and summary statistics.",
    )],
}];
const EVIDENCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "evidence = meshing.evidence(mesh)",
    inputs: &MESH_INPUT,
    outputs: &[inspect_output(
        "evidence",
        "meshing.Evidence object with stage, resource, and identity records.",
    )],
}];
const DISPLAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = mesh(mesh, Name, Value, ...)",
    inputs: &DISPLAY_INPUTS,
    outputs: &FIGURE_OUTPUT,
}];

pub const MESHING_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHING.INVALID",
    identifier: Some("RunMat:meshing:InvalidInput"),
    when: "A meshing object, path, option, artifact, or evidence contract is invalid, or meshing cannot produce a result that satisfies the request.",
    message: "meshing operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [MESHING_ERROR_INVALID];

const fn public_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
    errors: &'static [BuiltinErrorDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors,
    }
}

pub const MESHING_GENERATE_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&GENERATE_SIGNATURES, &ERRORS);
pub const MESHING_LOAD_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&LOAD_SIGNATURES, &ERRORS);
pub const MESHING_SAVE_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&SAVE_SIGNATURES, &ERRORS);
pub const MESHING_NODES_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&NODES_SIGNATURES, &ERRORS);
pub const MESHING_ELEMENTS_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&ELEMENTS_SIGNATURES, &ERRORS);
pub const MESHING_BOUNDARY_FACES_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&BOUNDARY_FACES_SIGNATURES, &ERRORS);
pub const MESHING_REGIONS_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&REGIONS_SIGNATURES, &ERRORS);
pub const MESHING_QUALITY_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&QUALITY_SIGNATURES, &ERRORS);
pub const MESHING_EVIDENCE_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&EVIDENCE_SIGNATURES, &ERRORS);
pub const MESHING_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DISPLAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};

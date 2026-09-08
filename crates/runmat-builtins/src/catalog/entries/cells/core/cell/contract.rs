use crate::*;

use super::signatures::SIGNATURES;

pub const CELL_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cell-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the cell like-prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CellLikeExtension"),
};
pub const CELL_GPU_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cell-gpu-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident GPU size controls for cell are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CellGpuSizeExtension"),
};
pub const CELL_EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[CELL_LIKE_EXTENSION, CELL_GPU_SIZE_EXTENSION];

pub const CELL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INVALID_INPUT",
    identifier: Some("RunMat:cell:InvalidInput"),
    when: "The argument or option form is invalid.",
    message: "cell: invalid input arguments",
};
pub const CELL_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INVALID_SIZE",
    identifier: Some("RunMat:cell:InvalidSize"),
    when: "A requested size is invalid or cannot be represented.",
    message: "cell: invalid size arguments",
};
pub const CELL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INTERNAL",
    identifier: Some("RunMat:cell:Internal"),
    when: "Cell allocation encounters an inconsistent internal shape.",
    message: "cell: internal allocation error",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    CELL_ERROR_INVALID_INPUT,
    CELL_ERROR_INVALID_SIZE,
    CELL_ERROR_INTERNAL,
];
pub const CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const SIZE_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "n_or_size_dimensions",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Every integer class is decoded directly as an exact structural size control.",
}];
pub const CELL_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "C = cell(n), cell(sz), or cell(sz1, ..., szN)",
        inputs: SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Signed negative sizes become zero; oversized shapes reject before allocation.",
    }];

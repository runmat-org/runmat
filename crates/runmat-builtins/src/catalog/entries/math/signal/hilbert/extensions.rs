use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const HILBERT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with integer signal data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertIntegerDataExtension"),
};
pub const HILBERT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with logical signal data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertLogicalDataExtension"),
};
pub const HILBERT_TYPED_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-typed-integer-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with a typed-integer FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertTypedIntegerLengthExtension"),
};
pub const HILBERT_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hilbert-logical-length",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hilbert with a logical FFT length is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HilbertLogicalLengthExtension"),
    };
pub const HILBERT_ZERO_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-zero-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with a zero FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertZeroLengthExtension"),
};
pub const HILBERT_EMPTY_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-empty-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with an empty FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertEmptyLengthExtension"),
};
pub const HILBERT_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with N-D input beyond a matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertNdInputExtension"),
};
pub const HILBERT_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    HILBERT_INTEGER_DATA_EXTENSION,
    HILBERT_LOGICAL_DATA_EXTENSION,
    HILBERT_TYPED_LENGTH_EXTENSION,
    HILBERT_LOGICAL_LENGTH_EXTENSION,
    HILBERT_ZERO_LENGTH_EXTENSION,
    HILBERT_EMPTY_LENGTH_EXTENSION,
    HILBERT_ND_INPUT_EXTENSION,
];

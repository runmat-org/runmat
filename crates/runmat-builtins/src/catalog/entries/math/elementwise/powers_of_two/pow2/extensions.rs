use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const POW2_INTEGER_UNARY_EXPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pow2-integer-unary-exponent",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pow2(E) accepts fixed-width integer exponents in RunMat mode",
        error_identifier: Some("RunMat:compatibility:Pow2IntegerUnaryExponentExtension"),
    };
pub const POW2_INTEGER_SIGNIFICAND_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pow2-integer-significand",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pow2(F,E) accepts fixed-width integer significands in RunMat mode",
        error_identifier: Some("RunMat:compatibility:Pow2IntegerSignificandExtension"),
    };
pub const POW2_INTEGER_BINARY_EXPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pow2-integer-binary-exponent",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pow2(F,E) accepts fixed-width integer exponents in RunMat mode",
        error_identifier: Some("RunMat:compatibility:Pow2IntegerBinaryExponentExtension"),
    };

pub const POW2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    POW2_INTEGER_UNARY_EXPONENT_EXTENSION,
    POW2_INTEGER_SIGNIFICAND_EXTENSION,
    POW2_INTEGER_BINARY_EXPONENT_EXTENSION,
];

use crate::*;
pub const DECOMPOSITION_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-nonfloating-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "decomposition with a non-single/non-double coefficient matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DecompositionNonfloatingInputExtension"),
    };

pub const DECOMPOSITION_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "decomposition with a gpuArray argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DecompositionGpuInputExtension"),
    };

pub const DECOMPOSITION_INTEGER_CHECK_CONDITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-integer-check-condition",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "decomposition with an integer CheckCondition value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DecompositionIntegerCheckConditionExtension"),
    };

pub const DECOMPOSITION_INTEGER_RANK_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-integer-rank-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "decomposition with an integer RankTolerance value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DecompositionIntegerRankToleranceExtension"),
    };

pub const DECOMPOSITION_INTEGER_SCALE_FACTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-integer-scale-factor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "scaling a decomposition by a real typed-integer scalar is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DecompositionIntegerScaleFactorExtension"),
    };

pub const DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "decomposition-complex-integer-scale-factor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "scaling a decomposition by a paired complex typed-integer scalar is a RunMat extension",
        error_identifier: Some(
            "RunMat:compatibility:DecompositionComplexIntegerScaleFactorExtension",
        ),
    };

pub const DECOMPOSITION_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DECOMPOSITION_NONFLOATING_INPUT_EXTENSION,
    DECOMPOSITION_GPU_INPUT_EXTENSION,
    DECOMPOSITION_INTEGER_CHECK_CONDITION_EXTENSION,
    DECOMPOSITION_INTEGER_RANK_TOLERANCE_EXTENSION,
];

pub const DECOMPOSITION_METHOD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DECOMPOSITION_NONFLOATING_INPUT_EXTENSION];

pub const DECOMPOSITION_SCALAR_TRANSFORM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DECOMPOSITION_NONFLOATING_INPUT_EXTENSION,
    DECOMPOSITION_GPU_INPUT_EXTENSION,
];

pub const DECOMPOSITION_SCALE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DECOMPOSITION_NONFLOATING_INPUT_EXTENSION,
    DECOMPOSITION_GPU_INPUT_EXTENSION,
    DECOMPOSITION_INTEGER_SCALE_FACTOR_EXTENSION,
    DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR_EXTENSION,
];

pub(super) const DECOMPOSITION_INTEGER_COEFFICIENT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer coefficient classes are stored exactly behind the RunMat compatibility gate and are promoted to double only at the solve boundary.",
    }];

pub(super) const DECOMPOSITION_INTEGER_OBJECT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dA",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The decomposition object contains an exact RunMat-only integer coefficient matrix.",
    }];

pub(super) const DECOMPOSITION_INTEGER_CHECK_CONDITION: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CheckCondition",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB documents CheckCondition as logical. RunMat mode additionally accepts an integer scalar and interprets zero as false and nonzero as true.",
    }];

pub(super) const DECOMPOSITION_INTEGER_RANK_TOLERANCE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "RankTolerance",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer RankTolerance is gated and must be exactly representable before the binary64 option boundary.",
    }];

pub(super) const DECOMPOSITION_INTEGER_SCALE_FACTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "scaleFactor",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A real typed-integer scale or divisor is gated independently and must be exactly representable before conversion to decomposition scale metadata.",
    }];

pub(super) const DECOMPOSITION_COMPLEX_INTEGER_SCALE_FACTOR: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "complexScaleFactor",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Paired real and imaginary typed-integer components are gated and each must be exactly representable before conversion to complex scale metadata.",
    }];

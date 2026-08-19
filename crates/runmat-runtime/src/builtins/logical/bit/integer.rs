//! MATLAB-compatible integer bitwise function builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericDType, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::sparse::{
    checked_sparse_result_len, map_sparse_real_values,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BITAND_NAME: &str = "bitand";
const BITCMP_NAME: &str = "bitcmp";
const BITGET_NAME: &str = "bitget";
const BITOR_NAME: &str = "bitor";
const BITSET_NAME: &str = "bitset";
const BITXOR_NAME: &str = "bitxor";
const BITSHIFT_NAME: &str = "bitshift";
const IDIVIDE_NAME: &str = "idivide";
const SWAPBYTES_NAME: &str = "swapbytes";

pub const SWAPBYTES_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "swapbytes-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow host fallback for explicit gpuArray input to swapbytes",
        error_identifier: Some("RunMat:compatibility:SwapbytesExplicitGpuInputExtension"),
    };
pub const SWAPBYTES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [SWAPBYTES_EXPLICIT_GPU_EXTENSION];
const SWAPBYTES_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every native integer class is documented; byte reversal preserves class and shape exactly.",
    }];
pub const SWAPBYTES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = swapbytes(integer_X)",
        inputs: &SWAPBYTES_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Each element's native byte sequence is reversed directly in authoritative storage; 8-bit classes are unchanged. Automatic residency gathers transparently, while explicit gpuArray fallback is independently gated.",
    }];

const BITAND_SINGLE_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitand-single-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitand with single-precision input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitandSingleInputExtension"),
};

const BITAND_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitand-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "bitand with a resident input outside the documented uint8/uint16/uint32 GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitandGpuUndocumentedInputExtension"),
    };

const BITAND_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitand-gpu-assumedtype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitand with resident input and assumedtype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitandGpuAssumedTypeExtension"),
};

pub const BITAND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    BITAND_SINGLE_INPUT_EXTENSION,
    BITAND_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    BITAND_GPU_ASSUMED_TYPE_EXTENSION,
];

const BITOR_SINGLE_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitor-single-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitor with single-precision input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitorSingleInputExtension"),
};

const BITOR_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitor-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "bitor with a resident input outside the documented uint8/uint16/uint32 GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitorGpuUndocumentedInputExtension"),
    };

const BITOR_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitor-gpu-assumedtype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitor with resident input and assumedtype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitorGpuAssumedTypeExtension"),
};

pub const BITOR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    BITOR_SINGLE_INPUT_EXTENSION,
    BITOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    BITOR_GPU_ASSUMED_TYPE_EXTENSION,
];

const BITXOR_SINGLE_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitxor-single-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitxor with single-precision input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitxorSingleInputExtension"),
};
const BITXOR_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitxor-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "bitxor with a resident input outside the documented uint8/uint16/uint32 GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitxorGpuUndocumentedInputExtension"),
    };
const BITXOR_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitxor-gpu-assumedtype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitxor with resident input and assumedtype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitxorGpuAssumedTypeExtension"),
};
pub const BITXOR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    BITXOR_SINGLE_INPUT_EXTENSION,
    BITXOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    BITXOR_GPU_ASSUMED_TYPE_EXTENSION,
];

const DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "direct-bit-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "A direct bit function with a resident input outside its documented GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DirectBitGpuUndocumentedInputExtension"),
    };
const DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "direct-bit-gpu-assumedtype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "A direct bit function with resident input and assumedtype is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DirectBitGpuAssumedTypeExtension"),
    };
pub const DIRECT_BIT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION,
];

const BITAND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A accepts every built-in integer class; an integer array may be paired with a scalar double, and assumedtype must match an integer input's native class.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "B accepts every built-in integer class; an integer array may be paired with a scalar double, and assumedtype must match an integer input's native class.",
    },
];

pub const BITAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitand(A, B)",
        inputs: &BITAND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Same-class integer inputs preserve their class and use exact two's-complement bit patterns with compatible-size expansion. The documented interactive GPU subset is uint8, uint16, and uint32, including an integer array paired with scalar double; host fallback restores the exact result to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitand(A, B, assumedtype)",
        inputs: &BITAND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "assumedtype selects the signed or unsigned bit width for double inputs and must equal the native class of integer inputs. Public GPU-array support excludes assumedtype; RunMat mode retains it as a gated gather-and-restore extension.",
    },
];

const BITOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A accepts every built-in integer class; an integer array may be paired with a scalar double, and assumedtype must match an integer input's native class.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "B accepts every built-in integer class; an integer array may be paired with a scalar double, and assumedtype must match an integer input's native class.",
    },
];

pub const BITOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitor(A, B)",
        inputs: &BITOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Same-class integer inputs preserve their class and use exact two's-complement bit patterns with compatible-size expansion. The documented interactive GPU subset is uint8, uint16, and uint32, including an integer array paired with scalar double; host fallback restores the exact result to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitor(A, B, assumedtype)",
        inputs: &BITOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "assumedtype selects the signed or unsigned bit width for double inputs and must equal the native class of integer inputs. Public GPU-array support excludes assumedtype; RunMat mode retains it as a gated gather-and-restore extension.",
    },
];

pub const BITXOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitxor(A, B)",
        inputs: &BITAND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Same-class integer inputs preserve their class and use exact two's-complement bit patterns with compatible-size expansion. The documented interactive GPU subset is uint8, uint16, and uint32 and restores exact output to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitxor(A, B, assumedtype)",
        inputs: &BITAND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "assumedtype selects the signed or unsigned bit width for double inputs and must equal the native class of integer inputs. Public GPU-array support excludes assumedtype; RunMat mode retains it as a gated gather-and-restore extension.",
    },
];

const BITCMP_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every native integer class is complemented within its own signed or unsigned storage width.",
    }];
pub const BITCMP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bitcmp(integer_A)",
        inputs: &BITCMP_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host execution covers all eight classes. Documented interactive GPU input is uint8, uint16, or uint32; exact fallback restores resident output to the owner.",
    }];

const BITGET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every native integer data class supplies exact signed or unsigned bit storage and determines output class and residency.",
    },
    BuiltinIntegerInputCapability {
        name: "bit",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for finite positive one-based bit positions; this control does not select output class or residency.",
    },
];
pub const BITGET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = bitget(integer_A, integer_bit)",
        inputs: &BITGET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "A and bit use scalar expansion or exactly matching nonscalar sizes. Supported resident data uses exact fallback and returns to A's owner; bit is a structural control.",
    }];

const BITSET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every native integer data class supplies exact signed or unsigned bit storage and determines output class and residency.",
    },
    BuiltinIntegerInputCapability {
        name: "bit",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for finite positive one-based bit positions; this control does not select output class or residency.",
    },
    BuiltinIntegerInputCapability {
        name: "V",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for zero-or-one replacement controls; V does not select output class or residency.",
    },
];
pub const BITSET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bitset(integer_A, integer_bit, integer_V)",
        inputs: &BITSET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "A, bit, and V use scalar expansion or exactly matching nonscalar sizes. Supported resident data uses exact fallback and returns to A's owner; bit and V are controls.",
    }];

const BITSHIFT_SINGLE_VALUE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitshift-single-value-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitshift with single-precision A is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitshiftSingleValueInputExtension"),
};

const BITSHIFT_SINGLE_COUNT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitshift-single-count-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitshift with single-precision k is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitshiftSingleCountInputExtension"),
};

const BITSHIFT_LOGICAL_VALUE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitshift-logical-value-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitshift with logical A is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitshiftLogicalValueInputExtension"),
};

const BITSHIFT_LOGICAL_COUNT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bitshift-logical-count-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bitshift with logical k is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BitshiftLogicalCountInputExtension"),
};

const BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "bitshift with resident input outside the documented non-64-bit integer-array GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftGpuUndocumentedInputExtension"),
    };

const BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-gpu-assumedtype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with resident input and assumedtype is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftGpuAssumedTypeExtension"),
    };

pub const BITSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    BITSHIFT_SINGLE_VALUE_EXTENSION,
    BITSHIFT_SINGLE_COUNT_EXTENSION,
    BITSHIFT_LOGICAL_VALUE_EXTENSION,
    BITSHIFT_LOGICAL_COUNT_EXTENSION,
    BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION,
];

const BITSHIFT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts double or every built-in integer class and determines the output class; signed right shifts preserve the sign bit.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "k independently accepts double or every built-in integer class; A and k are scalar or exactly the same size, and the class of k does not affect output class.",
    },
];

pub const BITSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "intout = bitshift(A, k)",
        inputs: &BITSHIFT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positive counts shift left and truncate overflow bits; negative counts shift right with arithmetic sign extension for signed A. The documented interactive GPU domain requires at least one integer-array input, excludes signed A and every 64-bit integer input, and restores exact output to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "intout = bitshift(A, k, assumedtype)",
        inputs: &BITSHIFT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "assumedtype selects the signed or unsigned bit width for double A and must equal typed-integer A. Public GPU-array support excludes assumedtype; RunMat mode retains resident calls as a gated gather-and-restore extension.",
    },
];

const IDIVIDE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "A is an integer array or a compatible integer-valued scalar double.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "B is an integer array or a compatible integer-valued scalar double.",
    },
];

pub const IDIVIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = idivide(A, B, roundingMode)",
        inputs: &IDIVIDE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::EvidenceOpen,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Supports fix, floor, ceil, and round modes; exact division is retained through host fallback, while division-by-zero and signed minimum divided by -1 remain named evidence questions.",
    }];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bitwise numeric result.",
}];

const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right integer-valued input.",
    },
];

const BINARY_INPUTS_ASSUMED_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double inputs.",
    },
];

const BITSHIFT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift count; positive shifts left and negative shifts right.",
    },
];

const BITSHIFT_INPUTS_ASSUMED_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift count; positive shifts left and negative shifts right.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];

const BITGET_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
];

const BITGET_INPUTS_ASSUMED_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];

const BITSET_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
];

const BITSET_INPUTS_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Zero clears the bit; any finite nonzero value sets it.",
    },
];

const BITSET_INPUTS_ASSUMED_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];

const BITSET_INPUTS_VALUE_ASSUMED_TYPE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "bit",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bit position.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Zero clears the bit; any finite nonzero value sets it.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];

const IDIVIDE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
];

const IDIVIDE_INPUTS_ROUNDING: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
    BuiltinParamDescriptor {
        name: "rounding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"fix\""),
        description: "Rounding mode: \"fix\", \"floor\", \"ceil\", or \"round\".",
    },
];

const SWAPBYTES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric scalar or array whose element byte order is reversed.",
}];

const BITCMP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integer-valued input whose bits are complemented.",
}];

const BITCMP_INPUTS_ASSUMED_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input whose bits are complemented.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];

const BITAND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitand(A, B)",
        inputs: &BINARY_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitand(A, B, assumedtype)",
        inputs: &BINARY_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITOR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitor(A, B)",
        inputs: &BINARY_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitor(A, B, assumedtype)",
        inputs: &BINARY_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITCMP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitcmp(A)",
        inputs: &BITCMP_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitcmp(A, assumedtype)",
        inputs: &BITCMP_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITXOR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitxor(A, B)",
        inputs: &BINARY_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitxor(A, B, assumedtype)",
        inputs: &BINARY_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitshift(A, K)",
        inputs: &BITSHIFT_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitshift(A, K, assumedtype)",
        inputs: &BITSHIFT_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITGET_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = bitget(A, bit)",
        inputs: &BITGET_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = bitget(A, bit, assumedtype)",
        inputs: &BITGET_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const BITSET_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit)",
        inputs: &BITSET_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, V)",
        inputs: &BITSET_INPUTS_VALUE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, assumedtype)",
        inputs: &BITSET_INPUTS_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitset(A, bit, V, assumedtype)",
        inputs: &BITSET_INPUTS_VALUE_ASSUMED_TYPE,
        outputs: &OUTPUT,
    },
];

const IDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B)",
        inputs: &IDIVIDE_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B, rounding)",
        inputs: &IDIVIDE_INPUTS_ROUNDING,
        outputs: &OUTPUT,
    },
];

const SWAPBYTES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = swapbytes(X)",
    inputs: &SWAPBYTES_INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.INVALID_INPUT",
    identifier: Some("RunMat:bitwise:InvalidInput"),
    when: "Inputs are not finite integer-valued numeric, logical, or gatherable gpuArray values.",
    message: "bitwise operation: invalid input",
};

const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.SIZE_MISMATCH",
    identifier: Some("RunMat:bitwise:SizeMismatch"),
    when: "Input shapes violate the operation's expansion rule: binary bitwise functions use compatible-size expansion, while bit positions/counts require scalar expansion or exactly matching nonscalar sizes.",
    message: "bitwise operation: array sizes are not compatible",
};

const ERROR_DIVIDE_BY_ZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.DIVIDE_BY_ZERO",
    identifier: Some("RunMat:idivide:DivideByZero"),
    when: "The divisor contains zero.",
    message: "idivide: division by zero",
};

const ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.OVERFLOW",
    identifier: Some("RunMat:idivide:Overflow"),
    when: "A rounded quotient cannot be represented in the dividend integer class.",
    message: "idivide: quotient overflows output class",
};

const BITWISE_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_SIZE_MISMATCH];

const IDIVIDE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_INPUT,
    ERROR_SIZE_MISMATCH,
    ERROR_DIVIDE_BY_ZERO,
    ERROR_OVERFLOW,
];

const SWAPBYTES_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const BITAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITCMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITCMP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITXOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITXOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITGET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITGET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const BITSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITSET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BITWISE_ERRORS,
};

pub const IDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IDIVIDE_ERRORS,
};

pub const SWAPBYTES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SWAPBYTES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SWAPBYTES_ERRORS,
};

#[runtime_builtin(
    name = "bitand",
    category = "logical/bit",
    summary = "Compute bitwise AND for integer-valued scalars and arrays.",
    keywords = "bitand,bitwise,and,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITAND_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::BITAND_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITAND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    public_binary_bitwise_builtin(
        BITAND_NAME,
        args,
        &BITAND_SINGLE_INPUT_EXTENSION,
        &BITAND_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &BITAND_GPU_ASSUMED_TYPE_EXTENSION,
        |a, b| a & b,
    )
    .await
}

async fn public_binary_bitwise_builtin(
    name: &'static str,
    args: Vec<Value>,
    single_extension: &BuiltinExtensionDescriptor,
    gpu_undocumented_input_extension: &BuiltinExtensionDescriptor,
    gpu_assumed_type_extension: &BuiltinExtensionDescriptor,
    operation: impl Fn(u64, u64) -> u64,
) -> BuiltinResult<Value> {
    enforce_public_binary_bitwise_compatibility(
        name,
        &args,
        single_extension,
        gpu_undocumented_input_extension,
        gpu_assumed_type_extension,
    )?;
    let output_source = args.iter().find_map(|value| {
        let Value::GpuTensor(handle) = value else {
            return None;
        };
        Some(handle.clone())
    });
    let logical_output = args.len() >= 2 && args[..2].iter().all(is_logical_bitwise_value);
    let result = binary_bitwise_from_args(name, args, operation).await?;
    let result = if logical_output {
        bitwise_result_as_logical(name, result)?
    } else {
        result
    };
    restore_binary_bitwise_gpu_result(name, result, output_source.as_ref())
}

fn enforce_public_binary_bitwise_compatibility(
    name: &str,
    args: &[Value],
    single_extension: &BuiltinExtensionDescriptor,
    gpu_undocumented_input_extension: &BuiltinExtensionDescriptor,
    gpu_assumed_type_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if !(2..=3).contains(&args.len()) {
        return Ok(());
    }
    if args.iter().take(2).any(is_single_bitwise_value) {
        crate::compatibility::ensure_builtin_extension_enabled(single_extension, name)?;
    }
    let has_gpu_input = args
        .iter()
        .take(2)
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if has_gpu_input && args.len() == 3 {
        crate::compatibility::ensure_builtin_extension_enabled(gpu_assumed_type_extension, name)?;
    }
    if args.iter().take(2).any(|value| {
        matches!(value, Value::GpuTensor(handle) if !matches!(
            runmat_accelerate_api::handle_integer_type(handle),
            Some(
                runmat_accelerate_api::IntegerElementType::U8
                    | runmat_accelerate_api::IntegerElementType::U16
                    | runmat_accelerate_api::IntegerElementType::U32
            )
        ))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            gpu_undocumented_input_extension,
            name,
        )?;
    }
    Ok(())
}

fn is_single_bitwise_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::SparseTensor(sparse) => sparse.numeric_dtype() == Some(NumericDType::F32),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

fn is_logical_bitwise_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn bitwise_integer_class(value: &Value) -> Option<IntegerClass> {
    match value {
        Value::Int(value) => Some(IntegerClass::from_int(value)),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerClass::from_storage),
        Value::SparseTensor(sparse) => sparse.integer_storage().map(IntegerClass::from_storage),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).map(|class| match class {
                runmat_accelerate_api::IntegerElementType::I8 => IntegerClass::I8,
                runmat_accelerate_api::IntegerElementType::I16 => IntegerClass::I16,
                runmat_accelerate_api::IntegerElementType::I32 => IntegerClass::I32,
                runmat_accelerate_api::IntegerElementType::I64 => IntegerClass::I64,
                runmat_accelerate_api::IntegerElementType::U8 => IntegerClass::U8,
                runmat_accelerate_api::IntegerElementType::U16 => IntegerClass::U16,
                runmat_accelerate_api::IntegerElementType::U32 => IntegerClass::U32,
                runmat_accelerate_api::IntegerElementType::U64 => IntegerClass::U64,
            })
        }
        _ => None,
    }
}

fn bitwise_result_as_logical(name: &'static str, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Tensor(tensor) => LogicalArray::new(
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            tensor.shape,
        )
        .map(Value::LogicalArray)
        .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error)),
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("internal logical result had unsupported value {other:?}"),
        )),
    }
}

fn restore_binary_bitwise_gpu_result(
    name: &'static str,
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    let provider = runmat_accelerate_api::provider_for_handle(source)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| {
            error_with_detail(
                name,
                &ERROR_INVALID_INPUT,
                "no acceleration provider is registered for resident output",
            )
        })?;
    let (tensor, logical) = match value {
        Value::Int(value) => (
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?,
            false,
        ),
        Value::Num(value) => (
            Tensor::new(vec![value], vec![1, 1])
                .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?,
            false,
        ),
        Value::Tensor(tensor) => (tensor, false),
        Value::Bool(value) => (
            Tensor::new(vec![f64::from(u8::from(value))], vec![1, 1])
                .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?,
            true,
        ),
        Value::LogicalArray(array) => (
            tensor::logical_to_tensor(&array)
                .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?,
            true,
        ),
        other => {
            return Err(error_with_detail(
                name,
                &ERROR_INVALID_INPUT,
                format!("cannot restore resident result {other:?}"),
            ))
        }
    };
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?;
    Ok(if logical {
        gpu_helpers::logical_gpu_value(handle)
    } else {
        gpu_helpers::resident_gpu_value(handle)
    })
}

#[runtime_builtin(
    name = "bitcmp",
    category = "logical/bit",
    summary = "Compute the bitwise complement of integer-valued scalars and arrays.",
    keywords = "bitcmp,bitwise,complement,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITCMP_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::DIRECT_BIT_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITCMP_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitcmp_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(BITCMP_NAME, &args, true)?;
    let (value, assumed) = unary_args(BITCMP_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitcmp(sparse, assumed);
    }
    let input = bit_buffer_from(BITCMP_NAME, value, assumed).await?;
    let mask = input.compute_class.map_or(u64::MAX, IntegerClass::bit_mask);
    let result = value_from_bits_with_classes(
        input.data.into_iter().map(|bits| !bits & mask).collect(),
        input.shape,
        input.compute_class,
        input.output_class,
        BITCMP_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITCMP_NAME, result, output_source.as_ref())
}

fn sparse_bitcmp(
    sparse: runmat_value::SparseTensor,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITCMP_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitcmp",
        ));
    }
    let class = assumed;
    let mask = class.map_or(u64::MAX, IntegerClass::bit_mask);
    map_sparse_real_values(&sparse, BITCMP_NAME, |value| {
        let bits = double_to_bits(BITCMP_NAME, value, class)?;
        let result = !bits & mask;
        Ok(match class {
            Some(class) => int_value_to_i128(&class.value_from_bits(result)) as f64,
            None => result as f64,
        })
    })
}

#[runtime_builtin(
    name = "bitget",
    category = "logical/bit",
    summary = "Get one-based bit positions from integer-valued scalars and arrays.",
    keywords = "bitget,bitwise,bit,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITGET_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::DIRECT_BIT_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITGET_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitget_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(BITGET_NAME, &args, false)?;
    let (value, bit, assumed) = value_bit_args(BITGET_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitget(sparse, bit, assumed).await;
    }
    let input = bit_buffer_from(BITGET_NAME, value, assumed).await?;
    let positions = shift_buffer_from(bit).await?;
    let plan = scalar_or_exact_size_plan(&input.shape, &positions.shape)
        .map_err(|err| error_with_detail(BITGET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_index, bit_index) in plan.iter() {
        let position = positions.data[bit_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITGET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        data.push((input.data[input_index] >> (position as u32 - 1)) & 1);
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        input.compute_class,
        input.output_class,
        BITGET_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITGET_NAME, result, output_source.as_ref())
}

async fn sparse_bitget(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITGET_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitget",
        ));
    }
    let positions = shift_buffer_from(bit).await?;
    let class = assumed;
    let width = class.map_or(64, IntegerClass::bit_width);
    let sparse_shape = sparse.shape();
    let plan = scalar_or_exact_size_plan(&sparse_shape, &positions.shape)
        .map_err(|err| error_with_detail(BITGET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITGET_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, sparse_index, bit_index) in plan.iter() {
        let position = positions.data[bit_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITGET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let bits = sparse
            .get(row, col)
            .map(|value| double_to_bits(BITGET_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        data.push((bits >> (position as u32 - 1)) & 1);
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITGET_NAME,
        output_shape.len() == 2,
    )
}

#[runtime_builtin(
    name = "bitset",
    category = "logical/bit",
    summary = "Set or clear one-based bit positions in integer-valued scalars and arrays.",
    keywords = "bitset,bitwise,bit,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITSET_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::DIRECT_BIT_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITSET_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitset_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let output_source = args.first().and_then(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    enforce_direct_bit_gpu_compatibility(BITSET_NAME, &args, false)?;
    let (value, bit, value_to_set, assumed) = bitset_args(args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitset(sparse, bit, value_to_set, assumed).await;
    }
    let input = bit_buffer_from(BITSET_NAME, value, assumed).await?;
    let positions = shift_buffer_from(bit).await?;
    let values = match value_to_set {
        Some(value) => bit_value_buffer_from(value).await?,
        None => BitValueBuffer {
            data: vec![true],
            shape: vec![1, 1],
        },
    };
    let input_positions = scalar_or_exact_size_plan(&input.shape, &positions.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let input_position_indices = input_positions
        .iter()
        .map(|(_, input_index, position_index)| (input_index, position_index))
        .collect::<Vec<_>>();
    let plan = scalar_or_exact_size_plan(input_positions.output_shape(), &values.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let width = input.compute_class.map_or(64, IntegerClass::bit_width);
    let mut data = Vec::with_capacity(plan.len());
    for (_, input_position_index, value_index) in plan.iter() {
        let (input_index, position_index) = input_position_indices[input_position_index];
        let position = positions.data[position_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITSET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let mask = 1_u64 << (position as u32 - 1);
        let current = input.data[input_index];
        data.push(if values.data[value_index] {
            current | mask
        } else {
            current & !mask
        });
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        input.compute_class,
        input.output_class,
        BITSET_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITSET_NAME, result, output_source.as_ref())
}

async fn sparse_bitset(
    sparse: runmat_value::SparseTensor,
    bit: Value,
    value_to_set: Option<Value>,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitset",
        ));
    }
    let positions = shift_buffer_from(bit).await?;
    let values = match value_to_set {
        Some(value) => bit_value_buffer_from(value).await?,
        None => BitValueBuffer {
            data: vec![true],
            shape: vec![1, 1],
        },
    };
    let class = assumed;
    let width = class.map_or(64, IntegerClass::bit_width);
    let sparse_shape = sparse.shape();
    let input_positions = scalar_or_exact_size_plan(&sparse_shape, &positions.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    checked_sparse_result_len(input_positions.output_shape(), BITSET_NAME)?;
    let input_position_indices = input_positions
        .iter()
        .map(|(_, sparse_index, position_index)| (sparse_index, position_index))
        .collect::<Vec<_>>();
    let plan = scalar_or_exact_size_plan(input_positions.output_shape(), &values.shape)
        .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITSET_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    let mut implicit_result_nonzero = false;
    for (_, input_position_index, value_index) in plan.iter() {
        let (sparse_index, position_index) = input_position_indices[input_position_index];
        let position = positions.data[position_index];
        if !(1..=i128::from(width)).contains(&position) {
            return Err(error_with_detail(
                BITSET_NAME,
                &ERROR_INVALID_INPUT,
                format!("bit position {position} must be between 1 and {width}"),
            ));
        }
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let stored = sparse.get(row, col);
        let current = stored
            .map(|value| double_to_bits(BITSET_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        let mask = 1_u64 << (position as u32 - 1);
        let result = if values.data[value_index] {
            current | mask
        } else {
            current & !mask
        };
        implicit_result_nonzero |= stored.is_none() && result != 0;
        data.push(result);
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITSET_NAME,
        output_shape.len() == 2 && !implicit_result_nonzero,
    )
}

#[runtime_builtin(
    name = "bitor",
    category = "logical/bit",
    summary = "Compute bitwise OR for integer-valued scalars and arrays.",
    keywords = "bitor,bitwise,or,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITOR_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::BITOR_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitor_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    public_binary_bitwise_builtin(
        BITOR_NAME,
        args,
        &BITOR_SINGLE_INPUT_EXTENSION,
        &BITOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &BITOR_GPU_ASSUMED_TYPE_EXTENSION,
        |a, b| a | b,
    )
    .await
}

#[runtime_builtin(
    name = "bitxor",
    category = "logical/bit",
    summary = "Compute bitwise XOR for integer-valued scalars and arrays.",
    keywords = "bitxor,bitwise,xor,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITXOR_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::BITXOR_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITXOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitxor_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    public_binary_bitwise_builtin(
        BITXOR_NAME,
        args,
        &BITXOR_SINGLE_INPUT_EXTENSION,
        &BITXOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &BITXOR_GPU_ASSUMED_TYPE_EXTENSION,
        |a, b| a ^ b,
    )
    .await
}

fn enforce_direct_bit_gpu_compatibility(
    name: &str,
    args: &[Value],
    restricted_integer_classes: bool,
) -> BuiltinResult<()> {
    let Some(Value::GpuTensor(source)) = args.first() else {
        return Ok(());
    };
    if args.last().is_some_and(|value| {
        matches!(value, Value::String(_))
            && ((name == BITCMP_NAME && args.len() == 2) || args.len() >= 3)
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIRECT_BIT_GPU_ASSUMED_TYPE_EXTENSION,
            name,
        )?;
    }
    let source_class = runmat_accelerate_api::handle_integer_type(source);
    let outside_domain = if restricted_integer_classes {
        !matches!(
            source_class,
            Some(
                runmat_accelerate_api::IntegerElementType::U8
                    | runmat_accelerate_api::IntegerElementType::U16
                    | runmat_accelerate_api::IntegerElementType::U32
            )
        )
    } else {
        source_class.is_none()
            && !args
                .iter()
                .skip(1)
                .any(|value| bitwise_integer_class(value).is_some())
    };
    if outside_domain {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIRECT_BIT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
            name,
        )?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "bitshift",
    category = "logical/bit",
    summary = "Shift integer-valued scalars and arrays left or right by bit counts.",
    keywords = "bitshift,bitwise,shift,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITSHIFT_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::BITSHIFT_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::BITSHIFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitshift_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    enforce_bitshift_compatibility(&args)?;
    let output_source = args.iter().find_map(|value| {
        let Value::GpuTensor(handle) = value else {
            return None;
        };
        Some(handle.clone())
    });
    let (value, shift, assumed) = value_bit_args(BITSHIFT_NAME, args)?;
    if let Value::SparseTensor(sparse) = value {
        return sparse_bitshift(sparse, shift, assumed).await;
    }
    let left = bit_buffer_from(BITSHIFT_NAME, value, assumed).await?;
    let shifts = shift_buffer_from(shift).await?;
    let plan = scalar_or_exact_size_plan(&left.shape, &shifts.shape)
        .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(apply_shift(
            left.data[idx_a],
            shifts.data[idx_b],
            left.compute_class,
        ));
    }
    let result = value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        left.compute_class,
        left.output_class,
        BITSHIFT_NAME,
    )?;
    restore_binary_bitwise_gpu_result(BITSHIFT_NAME, result, output_source.as_ref())
}

fn enforce_bitshift_compatibility(args: &[Value]) -> BuiltinResult<()> {
    if !(2..=3).contains(&args.len()) {
        return Ok(());
    }
    if is_single_bitwise_value(&args[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BITSHIFT_SINGLE_VALUE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_single_bitwise_value(&args[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BITSHIFT_SINGLE_COUNT_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_logical_bitwise_value(&args[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BITSHIFT_LOGICAL_VALUE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if is_logical_bitwise_value(&args[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BITSHIFT_LOGICAL_COUNT_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    let has_gpu_input = args
        .iter()
        .take(2)
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if has_gpu_input && args.len() == 3 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION,
            BITSHIFT_NAME,
        )?;
    }
    if has_gpu_input {
        let value_class = bitwise_integer_class(&args[0]);
        let count_class = bitwise_integer_class(&args[1]);
        let outside_public_gpu_domain = value_class.is_some_and(IntegerClass::is_signed)
            || value_class.is_some_and(|class| class.bit_width() == 64)
            || count_class.is_some_and(|class| class.bit_width() == 64)
            || (value_class.is_none() && count_class.is_none())
            || args
                .iter()
                .take(2)
                .any(|value| matches!(value, Value::SparseTensor(_)))
            || args.iter().take(2).any(is_single_bitwise_value)
            || args.iter().take(2).any(is_logical_bitwise_value);
        if outside_public_gpu_domain {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
                BITSHIFT_NAME,
            )?;
        }
    }
    Ok(())
}

async fn sparse_bitshift(
    sparse: runmat_value::SparseTensor,
    shift: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() {
        return Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            "typed sparse integer storage is a RunMat extension and is not supported by bitshift",
        ));
    }
    let shifts = shift_buffer_from(shift).await?;
    let class = assumed;
    let sparse_shape = sparse.shape();
    let plan = scalar_or_exact_size_plan(&sparse_shape, &shifts.shape)
        .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, BITSHIFT_NAME)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, sparse_index, shift_index) in plan.iter() {
        let row = sparse_index % sparse.rows;
        let col = sparse_index / sparse.rows;
        let bits = sparse
            .get(row, col)
            .map(|value| double_to_bits(BITSHIFT_NAME, value, class))
            .transpose()?
            .unwrap_or(0);
        data.push(apply_shift(bits, shifts.data[shift_index], class));
    }
    sparse_or_full_from_bits(
        data,
        output_shape.clone(),
        class,
        None,
        BITSHIFT_NAME,
        output_shape.len() == 2,
    )
}

#[runtime_builtin(
    name = "idivide",
    category = "math/elementwise",
    summary = "Integer division with MATLAB-compatible rounding modes.",
    keywords = "idivide,integer,division,rounding",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::IDIVIDE_DESCRIPTOR),
    integer_capabilities(crate::builtins::logical::bit::integer::IDIVIDE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn idivide_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "expected two integer inputs and an optional rounding mode",
        ));
    }
    let output_source = gpu_helpers::select_resident_output_source(
        args.iter().take(2).filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        IDIVIDE_NAME,
    )?;
    let mut iter = args.into_iter();
    let left = idivide_buffer_from(iter.next().expect("A")).await?;
    let right = idivide_buffer_from(iter.next().expect("B")).await?;
    let output_class = idivide_output_class(&left, &right)?;
    let rounding = match iter.next() {
        Some(value) => RoundingMode::parse(&value)?,
        None => RoundingMode::Fix,
    };
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| error_with_detail(IDIVIDE_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let mut out = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        let divisor = right.data[idx_b];
        if divisor == 0 {
            return Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_DIVIDE_BY_ZERO,
                "divisor contains zero",
            ));
        }
        let quotient = rounded_integer_divide(left.data[idx_a], divisor, rounding);
        out.push(quotient);
    }
    let result = value_from_integer_data(
        out,
        plan.output_shape().to_vec(),
        output_class,
        IDIVIDE_NAME,
    )?;
    restore_binary_bitwise_gpu_result(IDIVIDE_NAME, result, output_source.as_ref())
}

#[runtime_builtin(
    name = "swapbytes",
    category = "math/elementwise",
    summary = "Reverse byte order of numeric values.",
    keywords = "swapbytes,byte order,endian,numeric",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::SWAPBYTES_DESCRIPTOR),
    extensions(crate::builtins::logical::bit::integer::SWAPBYTES_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::bit::integer::SWAPBYTES_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn swapbytes_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SWAPBYTES_EXPLICIT_GPU_EXTENSION,
            SWAPBYTES_NAME,
        )?;
    }
    let gathered = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|err| error_with_detail(SWAPBYTES_NAME, &ERROR_INVALID_INPUT, err.message()))?;
    match gathered {
        Value::Num(value) => Ok(Value::Num(f64::from_bits(value.to_bits().swap_bytes()))),
        Value::Int(value) => Ok(Value::Int(swap_int_value(value))),
        Value::Tensor(tensor) => swap_tensor_bytes(tensor),
        other => Err(error_with_detail(
            SWAPBYTES_NAME,
            &ERROR_INVALID_INPUT,
            format!("unsupported input {other:?}"),
        )),
    }
}

async fn binary_bitwise(
    name: &'static str,
    lhs: Value,
    rhs: Value,
    assumed: Option<IntegerClass>,
    op: impl Fn(u64, u64) -> u64,
) -> BuiltinResult<Value> {
    if matches!(&lhs, Value::SparseTensor(_)) || matches!(&rhs, Value::SparseTensor(_)) {
        return sparse_binary_bitwise(name, lhs, rhs, assumed, op).await;
    }
    let left = bit_buffer_from(name, lhs, assumed).await?;
    let right = bit_buffer_from(name, rhs, assumed).await?;
    let output_class = binary_output_class(name, &left, &right)?;
    let compute_class = assumed.or(output_class);
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| error_with_detail(name, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(op(left.data[idx_a], right.data[idx_b]));
    }
    value_from_bits_with_classes(
        data,
        plan.output_shape().to_vec(),
        compute_class,
        output_class,
        name,
    )
}

enum SparseBitwiseOperand {
    Sparse(runmat_value::SparseTensor),
    Dense(BitBuffer),
}

impl SparseBitwiseOperand {
    fn metadata(&self) -> BitBuffer {
        match self {
            Self::Sparse(sparse) => BitBuffer {
                data: Vec::new(),
                shape: sparse.shape(),
                compute_class: None,
                output_class: None,
                is_scalar: sparse.rows == 1 && sparse.cols == 1,
            },
            Self::Dense(buffer) => BitBuffer {
                data: Vec::new(),
                shape: buffer.shape.clone(),
                compute_class: buffer.compute_class,
                output_class: buffer.output_class,
                is_scalar: buffer.is_scalar,
            },
        }
    }

    fn value_at(
        &self,
        name: &'static str,
        assumed: Option<IntegerClass>,
        index: usize,
    ) -> BuiltinResult<(u64, bool)> {
        match self {
            Self::Sparse(sparse) => {
                let row = index % sparse.rows;
                let col = index / sparse.rows;
                match sparse.get(row, col) {
                    Some(value) => Ok((double_to_bits(name, value, assumed)?, false)),
                    None => Ok((0, true)),
                }
            }
            Self::Dense(buffer) => Ok((buffer.data[index], false)),
        }
    }
}

async fn sparse_bitwise_operand_from(
    name: &'static str,
    value: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<SparseBitwiseOperand> {
    match value {
        Value::SparseTensor(sparse) => {
            if sparse.integer_storage().is_some() {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "typed sparse integer storage is a RunMat extension and is not supported by bitwise operations",
                ));
            }
            Ok(SparseBitwiseOperand::Sparse(sparse))
        }
        value => bit_buffer_from(name, value, assumed)
            .await
            .map(SparseBitwiseOperand::Dense),
    }
}

async fn sparse_binary_bitwise(
    name: &'static str,
    lhs: Value,
    rhs: Value,
    assumed: Option<IntegerClass>,
    op: impl Fn(u64, u64) -> u64,
) -> BuiltinResult<Value> {
    let left = sparse_bitwise_operand_from(name, lhs, assumed).await?;
    let right = sparse_bitwise_operand_from(name, rhs, assumed).await?;
    let left_meta = left.metadata();
    let right_meta = right.metadata();
    let output_class = binary_output_class(name, &left_meta, &right_meta)?;
    let compute_class = assumed.or(output_class);
    let plan = BroadcastPlan::new(&left_meta.shape, &right_meta.shape)
        .map_err(|err| error_with_detail(name, &ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    checked_sparse_result_len(&output_shape, name)?;

    let can_return_sparse = output_class.is_none() && output_shape.len() == 2;
    let mut bits = Vec::with_capacity(plan.len());
    let mut implicit_result_nonzero = false;
    for (_, left_index, right_index) in plan.iter() {
        let (left_bits, left_implicit) = left.value_at(name, compute_class, left_index)?;
        let (right_bits, right_implicit) = right.value_at(name, compute_class, right_index)?;
        let result = op(left_bits, right_bits);
        // A sparse/sparse result can store any position present in either
        // input; only positions absent from both operands are necessarily
        // implicit. With one dense operand, every implicit sparse position
        // must stay zero or the result has to become full.
        let result_is_implicit = match (&left, &right) {
            (SparseBitwiseOperand::Sparse(_), SparseBitwiseOperand::Sparse(_)) => {
                left_implicit && right_implicit
            }
            (SparseBitwiseOperand::Sparse(_), SparseBitwiseOperand::Dense(_)) => left_implicit,
            (SparseBitwiseOperand::Dense(_), SparseBitwiseOperand::Sparse(_)) => right_implicit,
            (SparseBitwiseOperand::Dense(_), SparseBitwiseOperand::Dense(_)) => false,
        };
        implicit_result_nonzero |= result_is_implicit && result != 0;
        bits.push(result);
    }

    sparse_or_full_from_bits(
        bits,
        output_shape,
        compute_class,
        output_class,
        name,
        can_return_sparse && !implicit_result_nonzero,
    )
}

fn sparse_or_full_from_bits(
    bits: Vec<u64>,
    output_shape: Vec<usize>,
    compute_class: Option<IntegerClass>,
    output_class: Option<IntegerClass>,
    name: &'static str,
    return_sparse: bool,
) -> BuiltinResult<Value> {
    if return_sparse {
        let rows = output_shape[0];
        let cols = output_shape[1];
        let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in 0..cols {
            for row in 0..rows {
                let value = bits[row + col * rows];
                if value != 0 {
                    row_indices.push(row);
                    values.push(bits_to_double(value, compute_class));
                }
            }
            col_ptrs.push(values.len());
        }
        return runmat_value::SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
            .map(Value::SparseTensor)
            .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err));
    }

    value_from_bits_with_classes(bits, output_shape, compute_class, output_class, name)
}

fn bits_to_double(bits: u64, compute_class: Option<IntegerClass>) -> f64 {
    match compute_class {
        Some(class) => int_value_to_i128(&class.value_from_bits(bits)) as f64,
        None => bits as f64,
    }
}

async fn binary_bitwise_from_args(
    name: &'static str,
    args: Vec<Value>,
    op: impl Fn(u64, u64) -> u64,
) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A, B, and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let lhs = args.next().expect("A");
    let rhs = args.next().expect("B");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    binary_bitwise(name, lhs, rhs, assumed, op).await
}

fn unary_args(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(Value, Option<IntegerClass>)> {
    if !(1..=2).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    Ok((value, assumed))
}

fn value_bit_args(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(Value, Value, Option<IntegerClass>)> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A, bit, and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let bit = args.next().expect("bit");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    Ok((value, bit, assumed))
}

fn bitset_args(
    args: Vec<Value>,
) -> BuiltinResult<(Value, Value, Option<Value>, Option<IntegerClass>)> {
    if !(2..=4).contains(&args.len()) {
        return Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "expected A, bit, optional V, and optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let bit = args.next().expect("bit");
    let third = args.next();
    let fourth = args.next();
    match (third, fourth) {
        (None, None) => Ok((value, bit, None, None)),
        (Some(third), None) if is_text_value(&third) => Ok((
            value,
            bit,
            None,
            Some(parse_assumed_type(BITSET_NAME, third)?),
        )),
        (Some(third), None) => Ok((value, bit, Some(third), None)),
        (Some(third), Some(fourth)) => Ok((
            value,
            bit,
            Some(third),
            Some(parse_assumed_type(BITSET_NAME, fourth)?),
        )),
        (None, Some(_)) => unreachable!("fourth argument requires third argument"),
    }
}

fn is_text_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn parse_assumed_type(name: &'static str, value: Value) -> BuiltinResult<IntegerClass> {
    let Some(keyword) = keyword_of(&value) else {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "assumedtype must be an integer class name",
        ));
    };
    IntegerClass::parse_name(&keyword).ok_or_else(|| {
        error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("unsupported assumedtype {keyword:?}"),
        )
    })
}

struct BitBuffer {
    data: Vec<u64>,
    shape: Vec<usize>,
    /// `None` represents MATLAB double/logical bit operands, which use the
    /// documented unsigned-64 interpretation and return a double result.
    compute_class: Option<IntegerClass>,
    output_class: Option<IntegerClass>,
    is_scalar: bool,
}

struct ShiftBuffer {
    data: Vec<i128>,
    shape: Vec<usize>,
}

struct BitValueBuffer {
    data: Vec<bool>,
    shape: Vec<usize>,
}

fn scalar_or_exact_size_plan(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<BroadcastPlan, String> {
    let scalar = |shape: &[usize]| {
        shape
            .iter()
            .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))
            == Some(1)
    };
    let same_size = || {
        let rank = lhs_shape.len().max(rhs_shape.len()).max(2);
        let lhs = crate::builtins::common::broadcast::align_shape(lhs_shape, rank);
        let rhs = crate::builtins::common::broadcast::align_shape(rhs_shape, rank);
        lhs == rhs
    };
    if !scalar(lhs_shape) && !scalar(rhs_shape) && !same_size() {
        return Err("inputs must be scalar or have exactly the same size".to_string());
    }
    let rank = lhs_shape.len().max(rhs_shape.len());
    let lhs_canonical = crate::builtins::common::broadcast::align_shape(lhs_shape, rank);
    let rhs_canonical = crate::builtins::common::broadcast::align_shape(rhs_shape, rank);
    BroadcastPlan::new(&lhs_canonical, &rhs_canonical)
}

async fn bit_buffer_from(
    name: &'static str,
    value: Value,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<BitBuffer> {
    match value {
        Value::Num(value) => Ok(BitBuffer {
            data: vec![double_to_bits(name, value, assumed)?],
            shape: vec![1, 1],
            compute_class: assumed,
            output_class: None,
            is_scalar: true,
        }),
        Value::Bool(value) => Ok(BitBuffer {
            data: vec![double_to_bits(name, f64::from(value), assumed)?],
            shape: vec![1, 1],
            compute_class: assumed,
            output_class: None,
            is_scalar: true,
        }),
        Value::Int(value) => Ok(BitBuffer {
            data: vec![int_to_bits(&value)],
            shape: vec![1, 1],
            compute_class: require_assumed_class(name, IntegerClass::from_int(&value), assumed)?,
            output_class: Some(IntegerClass::from_int(&value)),
            is_scalar: true,
        }),
        Value::Tensor(tensor) => tensor_to_bit_buffer(name, tensor, assumed),
        Value::LogicalArray(array) => Ok(BitBuffer {
            data: array.data.into_iter().map(|v| u64::from(v != 0)).collect(),
            shape: array.shape,
            compute_class: assumed,
            output_class: None,
            is_scalar: false,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
            tensor_to_bit_buffer(name, tensor, assumed)
        }
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: unsupported input {other:?}"),
        )),
    }
}

async fn shift_buffer_from(value: Value) -> BuiltinResult<ShiftBuffer> {
    match value {
        Value::Bool(value) => Ok(ShiftBuffer {
            data: vec![i128::from(value)],
            shape: vec![1, 1],
        }),
        Value::Num(value) => Ok(ShiftBuffer {
            data: vec![double_to_shift(value)?],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(ShiftBuffer {
            data: vec![int_value_to_i128(&value)],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_shift_buffer(tensor),
        Value::LogicalArray(array) => Ok(ShiftBuffer {
            data: array.data.into_iter().map(|v| i128::from(v != 0)).collect(),
            shape: array.shape,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_INVALID_INPUT, err))?;
            tensor_to_shift_buffer(tensor)
        }
        other => Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            format!("bitshift: unsupported shift input {other:?}"),
        )),
    }
}

async fn bit_value_buffer_from(value: Value) -> BuiltinResult<BitValueBuffer> {
    match value {
        Value::Bool(value) => Ok(BitValueBuffer {
            data: vec![value],
            shape: vec![1, 1],
        }),
        Value::Num(value) => Ok(BitValueBuffer {
            data: vec![finite_nonzero_bit_value(value)?],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(BitValueBuffer {
            data: vec![int_value_to_i128(&value) != 0],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_bit_value_buffer(tensor),
        Value::LogicalArray(array) => Ok(BitValueBuffer {
            data: array.data.into_iter().map(|value| value != 0).collect(),
            shape: array.shape,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(BITSET_NAME, &ERROR_INVALID_INPUT, err))?;
            tensor_to_bit_value_buffer(tensor)
        }
        other => Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            format!("bitset: unsupported bit value {other:?}"),
        )),
    }
}

fn tensor_to_bit_value_buffer(tensor: Tensor) -> BuiltinResult<BitValueBuffer> {
    let shape = tensor.shape.clone();
    let data = match tensor.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(|value| int_value_to_i128(value) != 0)
            .collect(),
        None => tensor::tensor_into_values_f64(tensor)
            .into_iter()
            .map(finite_nonzero_bit_value)
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    Ok(BitValueBuffer { data, shape })
}

fn finite_nonzero_bit_value(value: f64) -> BuiltinResult<bool> {
    if value.is_finite() {
        Ok(value != 0.0)
    } else {
        Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "bit values must be finite numeric or logical values",
        ))
    }
}

fn tensor_to_shift_buffer(tensor: Tensor) -> BuiltinResult<ShiftBuffer> {
    let shape = tensor.shape.clone();
    let data = match tensor.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(int_value_to_i128)
            .collect(),
        None => tensor::tensor_into_values_f64(tensor)
            .into_iter()
            .map(double_to_shift)
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    Ok(ShiftBuffer { data, shape })
}

fn tensor_to_bit_buffer(
    name: &'static str,
    tensor: Tensor,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<BitBuffer> {
    let is_scalar = tensor::element_count(&tensor.shape) == 1;
    let shape = tensor.shape.clone();
    let (data, native_class, output_class) = match tensor.integer_storage() {
        Some(storage) => (
            storage.exact_values().iter().map(int_to_bits).collect(),
            Some(IntegerClass::from_storage(storage)),
            Some(IntegerClass::from_storage(storage)),
        ),
        None => {
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "integer tensor is missing authoritative native storage",
                ));
            }
            let data = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| double_to_bits(name, value, assumed))
                .collect::<BuiltinResult<Vec<_>>>()?;
            (data, None, None)
        }
    };
    let compute_class = match native_class {
        Some(class) => require_assumed_class(name, class, assumed)?,
        None => assumed,
    };
    Ok(BitBuffer {
        data,
        shape,
        compute_class,
        output_class,
        is_scalar,
    })
}

fn value_from_bits_with_classes(
    data: Vec<u64>,
    shape: Vec<usize>,
    compute_class: Option<IntegerClass>,
    output_class: Option<IntegerClass>,
    name: &'static str,
) -> BuiltinResult<Value> {
    match output_class {
        Some(class) => {
            let values = data
                .into_iter()
                .map(|bits| class.value_from_bits(bits))
                .collect::<Vec<_>>();
            value_from_integer_values(values, shape, class, name)
        }
        None => {
            let double_value = |bits| match compute_class {
                Some(class) => int_value_to_i128(&class.value_from_bits(bits)) as f64,
                None => bits as f64,
            };
            if data.len() == 1 && tensor::element_count(&shape) == 1 {
                Ok(Value::Num(double_value(data[0])))
            } else {
                Tensor::new(data.into_iter().map(double_value).collect(), shape)
                    .map(Value::Tensor)
                    .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
            }
        }
    }
}

fn binary_output_class(
    name: &'static str,
    left: &BitBuffer,
    right: &BitBuffer,
) -> BuiltinResult<Option<IntegerClass>> {
    match (left.output_class, right.output_class) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Ok(Some(lhs)),
        (Some(_), Some(_)) => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "integer operands must have the same class unless the other operand is a scalar double",
        )),
        (Some(class), None) if right.is_scalar => Ok(Some(class)),
        (None, Some(class)) if left.is_scalar => Ok(Some(class)),
        (Some(_), None) | (None, Some(_)) => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "an integer array can only be combined with a scalar double",
        )),
        (None, None) => Ok(None),
    }
}

fn apply_shift(value: u64, shift: i128, class: Option<IntegerClass>) -> u64 {
    let width = class.map_or(64, IntegerClass::bit_width);
    let mask = class.map_or(u64::MAX, IntegerClass::bit_mask);
    let value = value & mask;
    if shift >= width as i128 {
        return 0;
    }
    if shift <= -(width as i128) {
        return if class.is_some_and(IntegerClass::is_signed) && value & (1_u64 << (width - 1)) != 0
        {
            mask
        } else {
            0
        };
    }
    if shift >= 0 {
        return (value << shift as u32) & mask;
    }

    let amount = (-shift) as u32;
    let shifted = value >> amount;
    if class.is_some_and(IntegerClass::is_signed) && value & (1_u64 << (width - 1)) != 0 {
        (shifted | (mask << (width - amount))) & mask
    } else {
        shifted
    }
}

fn double_to_u64(name: &'static str, value: f64) -> BuiltinResult<u64> {
    if !value.is_finite()
        || value.fract() != 0.0
        || !(0.0..18_446_744_073_709_551_616.0).contains(&value)
    {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: input values must be finite nonnegative integers smaller than 2^64"),
        ));
    }
    Ok(value as u64)
}

fn double_to_bits(
    name: &'static str,
    value: f64,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<u64> {
    let Some(class) = assumed else {
        return double_to_u64(name, value);
    };
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "input values must be finite integers",
        ));
    }
    let (min, max) = class.range();
    let fits = match class {
        IntegerClass::I64 => (-(2_f64.powi(63))..2_f64.powi(63)).contains(&value),
        IntegerClass::U64 => (0.0..2_f64.powi(64)).contains(&value),
        _ => (min as f64..=max as f64).contains(&value),
    };
    if !fits {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("input value {value} is outside assumedtype range"),
        ));
    }
    Ok(int_to_bits(&class.int_from_i128(value as i128)))
}

fn require_assumed_class(
    name: &'static str,
    native_class: IntegerClass,
    assumed: Option<IntegerClass>,
) -> BuiltinResult<Option<IntegerClass>> {
    if let Some(assumed) = assumed {
        if assumed != native_class {
            return Err(error_with_detail(
                name,
                &ERROR_INVALID_INPUT,
                "assumedtype must match the class of integer inputs",
            ));
        }
    }
    Ok(Some(native_class))
}

fn double_to_shift(value: f64) -> BuiltinResult<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            "bitshift: shift counts must be finite integers",
        ));
    }
    Ok(value.clamp(-128.0, 128.0) as i128)
}

fn int_to_bits(value: &IntValue) -> u64 {
    match value {
        IntValue::I8(value) => *value as u8 as u64,
        IntValue::I16(value) => *value as u16 as u64,
        IntValue::I32(value) => *value as u32 as u64,
        IntValue::I64(value) => *value as u64,
        IntValue::U8(value) => *value as u64,
        IntValue::U16(value) => *value as u64,
        IntValue::U32(value) => *value as u64,
        IntValue::U64(value) => *value,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntegerClass {
    fn parse_name(name: &str) -> Option<Self> {
        match name {
            "int8" => Some(Self::I8),
            "int16" => Some(Self::I16),
            "int32" => Some(Self::I32),
            "int64" => Some(Self::I64),
            "uint8" => Some(Self::U8),
            "uint16" => Some(Self::U16),
            "uint32" => Some(Self::U32),
            "uint64" => Some(Self::U64),
            _ => None,
        }
    }

    fn from_int(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => Self::I8,
            IntValue::I16(_) => Self::I16,
            IntValue::I32(_) => Self::I32,
            IntValue::I64(_) => Self::I64,
            IntValue::U8(_) => Self::U8,
            IntValue::U16(_) => Self::U16,
            IntValue::U32(_) => Self::U32,
            IntValue::U64(_) => Self::U64,
        }
    }

    fn from_storage(storage: &IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(_) => Self::I8,
            IntegerStorage::I16(_) => Self::I16,
            IntegerStorage::I32(_) => Self::I32,
            IntegerStorage::I64(_) => Self::I64,
            IntegerStorage::U8(_) => Self::U8,
            IntegerStorage::U16(_) => Self::U16,
            IntegerStorage::U32(_) => Self::U32,
            IntegerStorage::U64(_) => Self::U64,
        }
    }

    fn bit_width(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }

    fn bit_mask(self) -> u64 {
        match self.bit_width() {
            64 => u64::MAX,
            width => (1_u64 << width) - 1,
        }
    }

    fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    fn range(self) -> (i128, i128) {
        match self {
            Self::I8 => (i8::MIN as i128, i8::MAX as i128),
            Self::I16 => (i16::MIN as i128, i16::MAX as i128),
            Self::I32 => (i32::MIN as i128, i32::MAX as i128),
            Self::I64 => (i64::MIN as i128, i64::MAX as i128),
            Self::U8 => (0, u8::MAX as i128),
            Self::U16 => (0, u16::MAX as i128),
            Self::U32 => (0, u32::MAX as i128),
            Self::U64 => (0, u64::MAX as i128),
        }
    }

    fn int_from_i128(self, value: i128) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(value as i8),
            Self::I16 => IntValue::I16(value as i16),
            Self::I32 => IntValue::I32(value as i32),
            Self::I64 => IntValue::I64(value as i64),
            Self::U8 => IntValue::U8(value as u8),
            Self::U16 => IntValue::U16(value as u16),
            Self::U32 => IntValue::U32(value as u32),
            Self::U64 => IntValue::U64(value as u64),
        }
    }

    fn validate(self, value: i128, name: &'static str) -> BuiltinResult<()> {
        let (min, max) = self.range();
        if (min..=max).contains(&value) {
            Ok(())
        } else {
            Err(error_with_detail(
                name,
                &ERROR_OVERFLOW,
                format!("value {value} is outside output class range"),
            ))
        }
    }

    fn value_from_bits(self, bits: u64) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(bits as u8 as i8),
            Self::I16 => IntValue::I16(bits as u16 as i16),
            Self::I32 => IntValue::I32(bits as u32 as i32),
            Self::I64 => IntValue::I64(bits as i64),
            Self::U8 => IntValue::U8(bits as u8),
            Self::U16 => IntValue::U16(bits as u16),
            Self::U32 => IntValue::U32(bits as u32),
            Self::U64 => IntValue::U64(bits),
        }
    }
}

struct IntegerBuffer {
    data: Vec<i128>,
    shape: Vec<usize>,
    class: IntegerClass,
}

struct IdivideBuffer {
    data: Vec<i128>,
    shape: Vec<usize>,
    class: Option<IntegerClass>,
}

async fn integer_buffer_from(name: &'static str, value: Value) -> BuiltinResult<IntegerBuffer> {
    match value {
        Value::Int(value) => Ok(IntegerBuffer {
            data: vec![int_value_to_i128(&value)],
            shape: vec![1, 1],
            class: IntegerClass::from_int(&value),
        }),
        Value::Tensor(tensor) => tensor_to_integer_buffer(name, tensor),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
            tensor_to_integer_buffer(name, tensor)
        }
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("unsupported integer input {other:?}"),
        )),
    }
}

fn tensor_to_integer_buffer(name: &'static str, tensor: Tensor) -> BuiltinResult<IntegerBuffer> {
    let shape = tensor.shape.clone();
    if let Some(storage) = tensor.integer_storage() {
        return Ok(IntegerBuffer {
            data: storage
                .exact_values()
                .iter()
                .map(int_value_to_i128)
                .collect(),
            shape,
            class: IntegerClass::from_storage(storage),
        });
    }
    match tensor.numeric_dtype() {
        NumericDType::F32 | NumericDType::F64 => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "dense inputs must use an integer class",
        )),
        _ => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "integer tensor is missing authoritative native storage",
        )),
    }
}

async fn idivide_buffer_from(value: Value) -> BuiltinResult<IdivideBuffer> {
    match value {
        Value::Num(value) => {
            let integer = scalar_double_integer(value)?;
            Ok(IdivideBuffer {
                data: vec![integer],
                shape: vec![1, 1],
                class: None,
            })
        }
        other => {
            let buffer = integer_buffer_from(IDIVIDE_NAME, other).await?;
            Ok(IdivideBuffer {
                data: buffer.data,
                shape: buffer.shape,
                class: Some(buffer.class),
            })
        }
    }
}

fn idivide_output_class(
    left: &IdivideBuffer,
    right: &IdivideBuffer,
) -> BuiltinResult<IntegerClass> {
    match (left.class, right.class) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Ok(lhs),
        (Some(class), None) | (None, Some(class)) => {
            if matches!(class, IntegerClass::I64 | IntegerClass::U64) {
                return Err(error_with_detail(
                    IDIVIDE_NAME,
                    &ERROR_INVALID_INPUT,
                    "scalar double operands are not supported with int64 or uint64",
                ));
            }
            Ok(class)
        }
        (Some(_), Some(_)) => Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "integer inputs must have matching classes unless one input is a scalar double",
        )),
        (None, None) => Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "at least one input must be an integer class",
        )),
    }
}

fn scalar_double_integer(value: f64) -> BuiltinResult<i128> {
    if value.is_finite() && value.fract() == 0.0 {
        Ok(value as i128)
    } else {
        Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "scalar double operands must be finite integer-valued values",
        ))
    }
}

fn value_from_integer_data(
    data: Vec<i128>,
    shape: Vec<usize>,
    class: IntegerClass,
    name: &'static str,
) -> BuiltinResult<Value> {
    let values = data
        .into_iter()
        .map(|value| {
            class.validate(value, name)?;
            Ok(class.int_from_i128(value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    value_from_integer_values(values, shape, class, name)
}

fn value_from_integer_values(
    values: Vec<IntValue>,
    shape: Vec<usize>,
    class: IntegerClass,
    name: &'static str,
) -> BuiltinResult<Value> {
    if values.len() == 1 && tensor::element_count(&shape) == 1 {
        return Ok(Value::Int(values[0].clone()));
    }
    let prototype = IntegerStorage::from_scalar(class.value_from_bits(0));
    let storage = prototype
        .from_exact_values_like(values)
        .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => *value as i128,
        IntValue::I16(value) => *value as i128,
        IntValue::I32(value) => *value as i128,
        IntValue::I64(value) => *value as i128,
        IntValue::U8(value) => *value as i128,
        IntValue::U16(value) => *value as i128,
        IntValue::U32(value) => *value as i128,
        IntValue::U64(value) => *value as i128,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundingMode {
    Fix,
    Floor,
    Ceil,
    Round,
}

impl RoundingMode {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let Some(keyword) = keyword_of(value) else {
            return Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_INVALID_INPUT,
                "rounding mode must be text",
            ));
        };
        match keyword.as_str() {
            "fix" => Ok(Self::Fix),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            "round" => Ok(Self::Round),
            _ => Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_INVALID_INPUT,
                format!("unsupported rounding mode '{keyword}'"),
            )),
        }
    }
}

fn rounded_integer_divide(dividend: i128, divisor: i128, mode: RoundingMode) -> i128 {
    let quotient = dividend / divisor;
    let remainder = dividend % divisor;
    if remainder == 0 {
        return quotient;
    }
    match mode {
        RoundingMode::Fix => quotient,
        RoundingMode::Floor => {
            if (dividend < 0) != (divisor < 0) {
                quotient - 1
            } else {
                quotient
            }
        }
        RoundingMode::Ceil => {
            if (dividend < 0) == (divisor < 0) {
                quotient + 1
            } else {
                quotient
            }
        }
        RoundingMode::Round => round_quotient_away_from_zero(dividend, divisor),
    }
}

fn round_quotient_away_from_zero(dividend: i128, divisor: i128) -> i128 {
    let sign = if (dividend < 0) == (divisor < 0) {
        1
    } else {
        -1
    };
    let numerator = dividend.unsigned_abs();
    let denominator = divisor.unsigned_abs();
    let mut quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder.saturating_mul(2) >= denominator {
        quotient += 1;
    }
    (quotient as i128) * sign
}

fn swap_int_value(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value),
        IntValue::I16(value) => IntValue::I16(value.swap_bytes()),
        IntValue::I32(value) => IntValue::I32(value.swap_bytes()),
        IntValue::I64(value) => IntValue::I64(value.swap_bytes()),
        IntValue::U8(value) => IntValue::U8(value),
        IntValue::U16(value) => IntValue::U16(value.swap_bytes()),
        IntValue::U32(value) => IntValue::U32(value.swap_bytes()),
        IntValue::U64(value) => IntValue::U64(value.swap_bytes()),
    }
}

fn swap_tensor_bytes(tensor: Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        let swapped = swap_integer_storage(storage);
        return Tensor::new_integer(swapped, tensor.shape)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(SWAPBYTES_NAME, &ERROR_INVALID_INPUT, err));
    }
    let dtype = tensor.numeric_dtype();
    let shape = tensor.shape.clone();
    let data = tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .map(|value| swap_tensor_scalar(value, dtype))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Tensor::new_with_dtype(data, shape, dtype)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(SWAPBYTES_NAME, &ERROR_INVALID_INPUT, err))
}

fn swap_integer_storage(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => IntegerStorage::I8(values.clone()),
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.iter().map(|value| value.swap_bytes()).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.iter().map(|value| value.swap_bytes()).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.iter().map(|value| value.swap_bytes()).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(values.clone()),
        IntegerStorage::U16(values) => {
            IntegerStorage::U16(values.iter().map(|value| value.swap_bytes()).collect())
        }
        IntegerStorage::U32(values) => {
            IntegerStorage::U32(values.iter().map(|value| value.swap_bytes()).collect())
        }
        IntegerStorage::U64(values) => {
            IntegerStorage::U64(values.iter().map(|value| value.swap_bytes()).collect())
        }
    }
}

fn swap_tensor_scalar(value: f64, dtype: NumericDType) -> BuiltinResult<f64> {
    Ok(match dtype {
        NumericDType::F64 => f64::from_bits(value.to_bits().swap_bytes()),
        NumericDType::F32 => f32::from_bits((value as f32).to_bits().swap_bytes()) as f64,
        NumericDType::I8 => {
            validate_signed_scalar(value, i8::MIN as f64, i8::MAX as f64)?;
            value
        }
        NumericDType::I16 => {
            validate_signed_scalar(value, i16::MIN as f64, i16::MAX as f64)?;
            f64::from((value as i16).swap_bytes())
        }
        NumericDType::I32 => {
            validate_signed_scalar(value, i32::MIN as f64, i32::MAX as f64)?;
            (value as i32).swap_bytes() as f64
        }
        NumericDType::I64 => {
            validate_signed_scalar(value, i64::MIN as f64, i64::MAX as f64)?;
            (value as i64).swap_bytes() as f64
        }
        NumericDType::U8 => {
            validate_unsigned_scalar(value, u8::MAX as f64)?;
            value
        }
        NumericDType::U16 => {
            validate_unsigned_scalar(value, u16::MAX as f64)?;
            f64::from((value as u16).swap_bytes())
        }
        NumericDType::U32 => {
            validate_unsigned_scalar(value, u32::MAX as f64)?;
            (value as u32).swap_bytes() as f64
        }
        NumericDType::U64 => {
            validate_unsigned_scalar(value, u64::MAX as f64)?;
            (value as u64).swap_bytes() as f64
        }
    })
}

fn validate_unsigned_scalar(value: f64, max: f64) -> BuiltinResult<()> {
    if value.is_finite() && value.fract() == 0.0 && (0.0..=max).contains(&value) {
        Ok(())
    } else {
        Err(error_with_detail(
            SWAPBYTES_NAME,
            &ERROR_INVALID_INPUT,
            "integer tensor values must be finite and within dtype range",
        ))
    }
}

fn validate_signed_scalar(value: f64, min: f64, max: f64) -> BuiltinResult<()> {
    if value.is_finite() && value.fract() == 0.0 && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(error_with_detail(
            SWAPBYTES_NAME,
            &ERROR_INVALID_INPUT,
            "integer tensor values must be finite and within dtype range",
        ))
    }
}

fn error_with_detail(
    name: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let message = format!("{}: {}", error.message, detail);
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "integer_tests.rs"]
mod integer_tests;

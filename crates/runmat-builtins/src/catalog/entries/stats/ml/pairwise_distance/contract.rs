use crate::*;

const OUTPUT_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise distances in double precision.",
};
const OUTPUT_I: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "One-based row indices in X for selected distances.",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};
const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second observation matrix with observations in rows.",
};
const PDIST_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distance metric, optional metric parameter, or CacheSize.",
};
const PDIST2_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distance metric and parameter, CacheSize, or Smallest/Largest selection.",
};
const OUTPUTS_D: [BuiltinParamDescriptor; 1] = [OUTPUT_D];
const OUTPUTS_D_I: [BuiltinParamDescriptor; 2] = [OUTPUT_D, OUTPUT_I];
const INPUTS_X: [BuiltinParamDescriptor; 1] = [X];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [X, PDIST_OPTIONS];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [X, Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [X, Y, PDIST2_OPTIONS];

const PDIST_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "D = pdist(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUTS_D,
    },
    BuiltinSignatureDescriptor {
        label: "D = pdist(X, Distance, DistParameter)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUTS_D,
    },
];
const PDIST2_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "D = pdist2(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_D,
    },
    BuiltinSignatureDescriptor {
        label: "[D,I] = pdist2(X, Y, Distance, DistParameter, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_D_I,
    },
];
pub const PDIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::distance_errors::DISTANCE_ERRORS,
};
pub const PDIST2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDIST2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::distance_errors::DISTANCE_ERRORS,
};

pub const PDIST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pdist-integer-observation-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pdist with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PdistIntegerObservationDataExtension"),
};
pub const PDIST_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pdist-integer-numeric-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "pdist with a typed-integer distance parameter or cache size is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PdistIntegerNumericControlExtension"),
    };
pub const PDIST_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PDIST_INTEGER_DATA_EXTENSION,
    PDIST_INTEGER_CONTROL_EXTENSION,
];
pub const PDIST2_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pdist2-integer-observation-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pdist2 with typed-integer X or Y observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pdist2IntegerObservationDataExtension"),
};
pub const PDIST2_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pdist2-integer-numeric-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "pdist2 with a typed-integer distance parameter or cache size is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Pdist2IntegerNumericControlExtension"),
    };
pub const PDIST2_INTEGER_SELECTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pdist2-integer-selection-count",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pdist2 with a typed-integer Smallest or Largest count is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Pdist2IntegerSelectionCountExtension"),
    };
pub const PDIST2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    PDIST2_INTEGER_DATA_EXTENSION,
    PDIST2_INTEGER_CONTROL_EXTENSION,
    PDIST2_INTEGER_SELECTION_EXTENSION,
];

const PDIST_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer observations require exact binary64 conversion at the distance boundary.",
}];
const PDIST_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "DistParameter or CacheSize",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed numeric controls cross a checked floating-point boundary.",
}];
pub const PDIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = pdist(integer_X, ...)",
        inputs: &PDIST_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer observations are admitted before materialization and converted exactly at the binary64 metric boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "D = pdist(X, Distance, integer_DistParameter_or_CacheSize)",
        inputs: &PDIST_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed metric parameters and cache size are separately gated and exactly converted.",
    },
];
const PDIST2_DATA: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed X observations require exact binary64 conversion.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed Y observations require exact binary64 conversion.",
    },
];
const PDIST2_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "DistParameter or CacheSize",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed distance parameters and cache size cross a checked floating-point boundary.",
}];
const PDIST2_SELECTION: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Smallest or Largest",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed selection counts are decoded exactly as structural counts.",
}];
pub const PDIST2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = pdist2(integer_X, integer_Y, ...)",
        inputs: &PDIST2_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed observations are gated before materialization and exactly converted at the binary64 distance boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "D = pdist2(X, Y, Distance, integer_DistParameter_or_CacheSize)",
        inputs: &PDIST2_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed numeric controls are separately gated and exactly converted.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[D,I] = pdist2(X, Y, Smallest/Largest=integer_K)",
        inputs: &PDIST2_SELECTION,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Native integer counts are decoded exactly; selected distances and indices remain double.",
    },
];

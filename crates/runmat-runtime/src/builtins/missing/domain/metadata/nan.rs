use super::super::*;
use super::core::NANAWARE_SIGNATURES;

descriptor!(
    NAN_AWARE_DESCRIPTOR,
    NANAWARE_SIGNATURES,
    BuiltinOutputMode::Fixed
);

macro_rules! integer_extension {
    ($descriptor:ident, $extensions:ident, $id:literal, $description:literal, $error:literal) => {
        pub(in crate::builtins::missing::domain) const $descriptor: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: $id,
                mode: BuiltinExtensionMode::RunMatOnly,
                description: $description,
                error_identifier: Some($error),
            };
        pub const $extensions: [BuiltinExtensionDescriptor; 1] = [$descriptor];
    };
}

integer_extension!(
    NANMEAN_INTEGER_EXTENSION,
    NANMEAN_EXTENSIONS,
    "nanmean-typed-integer-input",
    "nanmean with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanmeanTypedIntegerInputExtension"
);
integer_extension!(
    NANSUM_INTEGER_EXTENSION,
    NANSUM_EXTENSIONS,
    "nansum-typed-integer-input",
    "nansum with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NansumTypedIntegerInputExtension"
);
integer_extension!(
    NANMIN_INTEGER_EXTENSION,
    NANMIN_EXTENSIONS,
    "nanmin-typed-integer-input",
    "nanmin with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanminTypedIntegerInputExtension"
);
integer_extension!(
    NANMEDIAN_INTEGER_EXTENSION,
    NANMEDIAN_EXTENSIONS,
    "nanmedian-typed-integer-input",
    "nanmedian with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanmedianTypedIntegerInputExtension"
);
integer_extension!(
    NANSTD_INTEGER_CONTROL_EXTENSION,
    NANSTD_EXTENSIONS,
    "nanstd-typed-integer-control",
    "nanstd with a typed-integer normalization or dimension input is a RunMat extension",
    "RunMat:compatibility:NanstdTypedIntegerControlExtension"
);
integer_extension!(
    NANVAR_INTEGER_CONTROL_EXTENSION,
    NANVAR_EXTENSIONS,
    "nanvar-typed-integer-control",
    "nanvar with a typed-integer normalization or dimension input is a RunMat extension",
    "RunMat:compatibility:NanvarTypedIntegerControlExtension"
);

pub(in crate::builtins::missing::domain) const MOVMAD_GPU_LARGE_WINDOW_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "movmad-gpu-large-window",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "movmad with a window longer than 31 on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MovmadGpuLargeWindowExtension"),
};

pub const MOVMAD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MOVMAD_GPU_LARGE_WINDOW_EXTENSION];

const MOVMAD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Moving median absolute deviation accepts every real integer input class and returns double deviation values.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Count windows accept exact positive typed-integer lengths or integer-valued floating lengths.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional positive scalar dimension accepts exact typed integers or integer-valued floating values.",
    },
];

pub const MOVMAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "M = movmad(A, k, dim, nanflag)",
        inputs: &MOVMAD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "On the currently supported scalar-window surface, integer observations materialize once into the double median-absolute-deviation domain; supported resident inputs gather and re-upload double output, while GPU windows longer than 31 are separately mode-gated.",
    }];

const RUNMAT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_control",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer data, pairwise operands, and dimension controls are accepted only with the builtin's declared RunMat extension; documented single- and double-valued forms remain available in MATLAB-compatible mode.",
    }];

const REJECTED_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed-integer data is rejected before host or provider reduction.",
}];

const RUNMAT_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "normalization_or_dimension",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer normalization or dimension controls are accepted only with the builtin's declared RunMat extension; integer-valued double controls remain documented-compatible.",
    }];

pub const NANMEAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmean(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmean by routing typed-integer forms through mean(...,\"omitnan\"); default/double output is double, native output preserves the input class, and modern mean-only options remain extension syntax.",
    }];

pub const NANSUM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nansum(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nansum by routing typed-integer forms through sum(...,\"omitnan\"); default/double output is double, native output preserves the input class with saturating accumulation, and modern sum-only options remain extension syntax.",
    }];

pub const NANMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmin(A, args...) or nanmin(A, B)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmin with exact typed-integer reduction and compatible pairwise forms; omit-NaN resident reductions use host fallback, while pairwise execution follows min's same-class-or-scalar-double rules.",
    }];

pub const NANMEDIAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmedian(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmedian by routing typed-integer forms through median(...,\"omitnan\"); exact host reduction preserves all eight classes and resident fallback output is re-uploaded.",
    }];

pub const NANSTD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanstd(A, args...) with typed-integer A",
        inputs: &REJECTED_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer data is unsupported in both compatibility modes and is rejected before provider dispatch.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanstd(A, flag_or_dimension)",
        inputs: &RUNMAT_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "With floating data, RunMat accepts exact typed-integer normalization and dimension controls only when the nanstd typed-integer-control extension is enabled.",
    },
];

pub const NANVAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(A, args...) with typed-integer A",
        inputs: &REJECTED_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer data is unsupported in both compatibility modes and is rejected before provider dispatch.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(A, normalization_or_dimension)",
        inputs: &RUNMAT_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "With floating data, RunMat accepts exact typed-integer normalization and dimension controls only when the nanvar typed-integer-control extension is enabled; non-scalar weighted variance remains separately unsupported.",
    },
];

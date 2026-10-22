macro_rules! moving_integer_metadata {
    (
        $form:literal,
        $domain:ident,
        $output:ident,
        $backend:ident,
        $notes:literal
    ) => {
        const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 4] = [
            BuiltinIntegerInputCapability {
                name: "A",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Ordinary real arrays accept all eight integer classes; typed complex integers remain outside the documented real integer domain.",
            },
            BuiltinIntegerInputCapability {
                name: "k_or_window",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Count windows accept exact positive typed-integer lengths or exact nonnegative backward/forward lengths.",
            },
            BuiltinIntegerInputCapability {
                name: "dim",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "The optional positive scalar dimension is decoded exactly from typed integer or integer-valued floating storage.",
            },
            BuiltinIntegerInputCapability {
                name: "endpoint_fill_or_sample_points",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Numeric endpoint fills and sample-point vectors accept integer classes; exact endpoint fills stay typed for native integer outputs, while sample locations intentionally enter the floating positional domain.",
            },
        ];

        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: $form,
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::$domain,
                output_class: BuiltinIntegerOutputClassRule::$output,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::$backend,
                overload: BuiltinIntegerOverloadKind::Multiple,
                notes: $notes,
            }];
    };
}
pub(in crate::catalog::entries::math::reduction) use moving_integer_metadata;

macro_rules! moving_gpu_sample_points_extension {
    ($id:literal, $description:literal, $identifier:literal) => {
        pub const GPU_SAMPLE_POINTS_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: $id,
                mode: BuiltinExtensionMode::RunMatOnly,
                description: $description,
                error_identifier: Some($identifier),
            };
    };
}
pub(in crate::catalog::entries::math::reduction) use moving_gpu_sample_points_extension;

macro_rules! moving_stdvar_integer_metadata {
    ($name:literal, $form:literal) => {
        const REJECTED_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "A",
                classes: &[],
                availability: BuiltinIntegerInputAvailability::Rejected,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "The documented moving-dispersion data domain is single, double, or logical; every typed-integer data class is rejected before host or provider dispatch.",
            }];

        const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "normalization_or_dimension",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "RunMat mode accepts exact host typed-integer normalization and dimension controls; MATLAB-compatible modes retain the documented floating control surface.",
            }];

        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
            BuiltinIntegerCapabilityDescriptor {
                form: $form,
                inputs: &REJECTED_INTEGER_DATA_INPUT,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: BuiltinIntegerOutputClassRule::NotApplicable,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::HostAndGpu,
                overload: BuiltinIntegerOverloadKind::Multiple,
                notes: "Integer data is invalid in every compatibility mode, including resident integer data, and no integer output contract exists.",
            },
            BuiltinIntegerCapabilityDescriptor {
                form: concat!("M = ", $name, "(A, k, normalization_or_dimension, ...) with floating A"),
                inputs: &INTEGER_CONTROL_INPUT,
                computation_domain: BuiltinIntegerComputationDomain::Structural,
                output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::HostOnly,
                overload: BuiltinIntegerOverloadKind::StructuralParameter,
                notes: "Typed-integer controls are a mode-gated RunMat extension; values are decoded exactly on the host, while the floating data reduction may still execute on a provider.",
            },
        ];
    };
}
pub(in crate::catalog::entries::math::reduction) use moving_stdvar_integer_metadata;

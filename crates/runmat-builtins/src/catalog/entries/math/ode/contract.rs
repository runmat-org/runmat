macro_rules! define_ode_integer_contract {
    ($builtin:literal, $identifier:literal) => {
        pub const INTEGER_TSPAN_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-tspan"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer tspan is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerTspanExtension")),
            };
        pub const INTEGER_Y0_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-y0"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer y0 is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerY0Extension")),
            };
        pub const INTEGER_OPTION_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-option"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with native-class integer numeric options is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerOptionExtension")),
            };
        pub const INTEGER_CALLBACK_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-integer-callback-result"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with a native-class integer derivative result is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "IntegerCallbackExtension")),
            };
        pub const LOGICAL_NUMERIC_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-logical-numeric"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " with logical solver data is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "LogicalNumericExtension")),
            };
        pub const RESIDENT_INPUT_EXTENSION: crate::BuiltinExtensionDescriptor =
            crate::BuiltinExtensionDescriptor {
                id: concat!($builtin, "-resident-input"),
                mode: crate::BuiltinExtensionMode::RunMatOnly,
                description: concat!($builtin, " host fallback for explicit gpuArray values is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $identifier, "ResidentInputExtension")),
            };
        pub const EXTENSIONS: [crate::BuiltinExtensionDescriptor; 6] = [
            INTEGER_TSPAN_EXTENSION,
            INTEGER_Y0_EXTENSION,
            INTEGER_OPTION_EXTENSION,
            INTEGER_CALLBACK_EXTENSION,
            LOGICAL_NUMERIC_EXTENSION,
            RESIDENT_INPUT_EXTENSION,
        ];
        const INTEGER_TSPAN_INPUT: [crate::BuiltinIntegerInputCapability; 1] =
            [crate::BuiltinIntegerInputCapability {
                name: "tspan",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "The compatibility target documents single and double tspan; integer time points are gated and must cross the binary64 solver boundary exactly.",
            }];
        const INTEGER_Y0_INPUT: [crate::BuiltinIntegerInputCapability; 1] =
            [crate::BuiltinIntegerInputCapability {
                name: "y0",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "The compatibility target documents single and double initial conditions; integer states are gated and must cross the binary64 solver boundary exactly.",
            }];
        const INTEGER_OPTION_INPUT: [crate::BuiltinIntegerInputCapability; 1] =
            [crate::BuiltinIntegerInputCapability {
                name: "numeric option field",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Recognized typed numeric ODE controls are gated before selective gather; floating controls convert only when exact and MaxSteps/Refine remain structural.",
            }];
        const INTEGER_CALLBACK_INPUT: [crate::BuiltinIntegerInputCapability; 1] =
            [crate::BuiltinIntegerInputCapability {
                name: "odefun result",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "The compatibility target requires single or double derivative output; RunMat gates integer derivatives and checks every floating conversion.",
            }];
        pub const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 4] = [
            crate::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(odefun, integer_tspan, y0, ___)"), inputs: &INTEGER_TSPAN_INPUT, computation_domain: crate::BuiltinIntegerComputationDomain::FloatingPoint, output_class: crate::BuiltinIntegerOutputClassRule::Double, overflow: crate::BuiltinIntegerOverflowRule::Error, backend: crate::BuiltinIntegerBackendRule::GatherFallback, overload: crate::BuiltinIntegerOverloadKind::Multiple, notes: "Integer time points are a checked RunMat-only input to the host double integration loop." },
            crate::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(odefun, tspan, integer_y0, ___)"), inputs: &INTEGER_Y0_INPUT, computation_domain: crate::BuiltinIntegerComputationDomain::FloatingPoint, output_class: crate::BuiltinIntegerOutputClassRule::Double, overflow: crate::BuiltinIntegerOverflowRule::Error, backend: crate::BuiltinIntegerBackendRule::GatherFallback, overload: crate::BuiltinIntegerOverloadKind::Multiple, notes: "Integer initial states are a checked RunMat-only input and produce floating solver outputs." },
            crate::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(___, options_with_integer_field)"), inputs: &INTEGER_OPTION_INPUT, computation_domain: crate::BuiltinIntegerComputationDomain::FunctionSpecific, output_class: crate::BuiltinIntegerOutputClassRule::Double, overflow: crate::BuiltinIntegerOverflowRule::Error, backend: crate::BuiltinIntegerBackendRule::GatherFallback, overload: crate::BuiltinIntegerOverloadKind::StructuralParameter, notes: "Integer option fields are independently gated; tolerances and step sizes cross exactly while the RunMat MaxSteps extension remains an exact count." },
            crate::BuiltinIntegerCapabilityDescriptor { form: concat!("[t,y] = ", $builtin, "(integer_returning_odefun, ___)"), inputs: &INTEGER_CALLBACK_INPUT, computation_domain: crate::BuiltinIntegerComputationDomain::FloatingPoint, output_class: crate::BuiltinIntegerOutputClassRule::Double, overflow: crate::BuiltinIntegerOverflowRule::Error, backend: crate::BuiltinIntegerBackendRule::GatherFallback, overload: crate::BuiltinIntegerOverloadKind::Multiple, notes: "Integer derivative values are a RunMat-only callback extension and enter the solver only after exact representability checks." },
        ];
    };
}
pub(super) use define_ode_integer_contract;

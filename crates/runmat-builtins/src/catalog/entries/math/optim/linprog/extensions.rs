use crate::*;

macro_rules! linprog_integer_extension {
    ($name:ident, $id:literal, $role:literal, $identifier:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!(
                "linprog with a typed-integer ",
                $role,
                " input is a RunMat extension"
            ),
            error_identifier: Some($identifier),
        };
    };
}
linprog_integer_extension!(
    LINPROG_INTEGER_F_EXTENSION,
    "linprog-integer-f",
    "f",
    "RunMat:compatibility:LinprogIntegerFExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_A_EXTENSION,
    "linprog-integer-a",
    "A",
    "RunMat:compatibility:LinprogIntegerAExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_B_EXTENSION,
    "linprog-integer-b",
    "b",
    "RunMat:compatibility:LinprogIntegerBExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_AEQ_EXTENSION,
    "linprog-integer-aeq",
    "Aeq",
    "RunMat:compatibility:LinprogIntegerAeqExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_BEQ_EXTENSION,
    "linprog-integer-beq",
    "beq",
    "RunMat:compatibility:LinprogIntegerBeqExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_LB_EXTENSION,
    "linprog-integer-lb",
    "lb",
    "RunMat:compatibility:LinprogIntegerLbExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_UB_EXTENSION,
    "linprog-integer-ub",
    "ub",
    "RunMat:compatibility:LinprogIntegerUbExtension"
);
linprog_integer_extension!(
    LINPROG_INTEGER_OPTION_EXTENSION,
    "linprog-integer-option",
    "numeric option",
    "RunMat:compatibility:LinprogIntegerOptionExtension"
);
pub const LINPROG_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linprog-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linprog with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinprogExplicitGpuInputExtension"),
};
pub const LINPROG_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    LINPROG_INTEGER_F_EXTENSION,
    LINPROG_INTEGER_A_EXTENSION,
    LINPROG_INTEGER_B_EXTENSION,
    LINPROG_INTEGER_AEQ_EXTENSION,
    LINPROG_INTEGER_BEQ_EXTENSION,
    LINPROG_INTEGER_LB_EXTENSION,
    LINPROG_INTEGER_UB_EXTENSION,
    LINPROG_GPU_INPUT_EXTENSION,
    LINPROG_INTEGER_OPTION_EXTENSION,
];
const LINPROG_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 7] = [
    BuiltinIntegerInputCapability {
        name: "f",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Public linprog requires double f; RunMat mode admits exact-binary64 typed integers.",
    },
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public A is single or double; typed integers are independently gated.",
    },
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public b is single or double; typed integers are independently gated.",
    },
    BuiltinIntegerInputCapability {
        name: "Aeq",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public Aeq is double; typed integers are independently gated.",
    },
    BuiltinIntegerInputCapability {
        name: "beq",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public beq is single or double; typed integers are independently gated.",
    },
    BuiltinIntegerInputCapability {
        name: "lb",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public lb is double; typed integers are independently gated.",
    },
    BuiltinIntegerInputCapability {
        name: "ub",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public ub is double; typed integers are independently gated.",
    },
];
const LINPROG_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "MaxIterations / MaxIter",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The exact nonnegative iteration limit is never converted through binary64 and must fit the backend's unsigned 32-bit iteration range.",
}];
pub const LINPROG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] =
    [BuiltinIntegerCapabilityDescriptor { form: "linprog(integer-valued solver inputs)", inputs: &LINPROG_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Every typed-integer role is a separately gated RunMat extension. Values must be exactly representable before crossing the host binary64 solver boundary." },BuiltinIntegerCapabilityDescriptor {
    form: "linprog(f,A,b,Aeq,beq,lb,ub,integer_options)", inputs: &LINPROG_INTEGER_OPTION_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Typed integer iteration limits are independently gated and retained as exact structural counts, not solver floating operands.",
}];

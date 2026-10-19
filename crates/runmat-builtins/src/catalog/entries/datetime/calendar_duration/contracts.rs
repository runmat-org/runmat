use crate::*;

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or logical array to construct calendar units, or a calendarDuration array to extract whole units.",
}];
const DURATION_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Calendar duration for numeric input; double whole-unit counts for calendarDuration input.",
}];
const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical scalar indicating whether X is a calendarDuration array.",
}];

macro_rules! unit_descriptor {
    ($name:ident, $label:literal) => {
        pub(super) const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &INPUT,
                outputs: &DURATION_OUTPUT,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

unit_descriptor!(CALDAYS, "C = caldays(X)");
unit_descriptor!(CALWEEKS, "C = calweeks(X)");
unit_descriptor!(CALMONTHS, "C = calmonths(X)");
unit_descriptor!(CALQUARTERS, "C = calquarters(X)");
unit_descriptor!(CALYEARS, "C = calyears(X)");

const COMPONENTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar years.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar months.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar days.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole hours.",
    },
    BuiltinParamDescriptor {
        name: "MI",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole minutes.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Seconds, possibly fractional.",
    },
];
const COMPONENTS_THREE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar years.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar months.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Whole calendar days.",
    },
];
const CONSTRUCTOR_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three- or six-column numeric component matrix. A numeric day array or existing calendarDuration is a RunMat-only extension.",
}];
pub(super) const CALENDARDURATION: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "C = calendarDuration()",
            inputs: &[],
            outputs: &DURATION_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "C = calendarDuration(Y,M,D)",
            inputs: &COMPONENTS_THREE,
            outputs: &DURATION_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "C = calendarDuration(Y,M,D,H,MI,S)",
            inputs: &COMPONENTS,
            outputs: &DURATION_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "C = calendarDuration(X)",
            inputs: &CONSTRUCTOR_INPUT,
            outputs: &DURATION_OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub(super) const ISCALENDARDURATION: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tf = iscalendarduration(X)",
        inputs: &INPUT,
        outputs: &LOGICAL_OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const INTEGER_UNIT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight fixed-width classes are accepted as whole calendar-unit counts; exact native storage is checked before conversion to the host calendar representation.",
}];
pub(super) const UNIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "calendarUnit(integer_X)",
        inputs: &INTEGER_UNIT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Construction returns calendarDuration; extraction from a calendarDuration returns double. Values outside the supported civil-calendar range reject before floating conversion.",
    }];
pub(super) const CONSTRUCTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "calendarDuration(integer_components_or_matrix)",
        inputs: &INTEGER_UNIT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Fixed-width calendar components are checked in native storage before the supported host calendar boundary; numeric matrix and component forms preserve row/component shape.",
    }];
pub(super) const PREDICATE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "iscalendarduration(integer_X)",
        inputs: &INTEGER_UNIT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The predicate inspects the value's class without converting or reading the integer payload and returns false for integer values.",
    }];

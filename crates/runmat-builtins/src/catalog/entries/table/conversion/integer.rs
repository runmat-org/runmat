use crate::*;
const TABLE_VARIABLE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer variables contained in T",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table variables retain authoritative class and payload while the host container is converted.",
    }];
const TABLE2STRUCT_INTEGER_TO_SCALAR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ToScalar",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public selector is logical; RunMat mode additionally accepts an exact typed-integer scalar as a truth value.",
    }];
pub const TABLE2ARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = table2array(T_with_integer_variables)",
        inputs: &TABLE_VARIABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Same-class integer variables concatenate without conversion; mixed compatible numeric variables follow table concatenation dominance and assignment conversion without routing integer values through binary64.",
    }];
pub const TABLE2CELL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = table2cell(T_with_integer_variables)",
        inputs: &TABLE_VARIABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Each cell receives the exact same-class integer scalar or row slice from its source variable, including wide int64 and uint64 values.",
    }];
pub const TABLE2STRUCT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "S = table2struct(T_with_integer_variables, ToScalar=tf)",
        inputs: &TABLE_VARIABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Row structs contain exact typed scalars; scalar-struct fields retain the entire original integer variable unchanged.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = table2struct(T, ToScalar=integer_tf)",
        inputs: &TABLE2STRUCT_INTEGER_TO_SCALAR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Compatibility mode rejects before resident provider access; RunMat mode gathers an admitted scalar control and interprets zero as false and nonzero as true.",
    },
];

const STRUCT2TABLE_INTEGER_FIELDS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "struct field values",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Compatible same-class integer fields become homogeneous table variables.",
    }];
const STRUCT2TABLE_INTEGER_AS_ARRAY: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "AsArray",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A typed integer scalar is accepted only when its exact value is zero or one.",
    }];
pub const STRUCT2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = struct2table(S) with integer fields",
        inputs: &STRUCT2TABLE_INTEGER_FIELDS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Scalar-struct fields pass through unchanged. Compatible struct-array fields are concatenated into exact homogeneous integer variables; incompatible values remain cells.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = struct2table(S, 'AsArray', integer_tf)",
        inputs: &STRUCT2TABLE_INTEGER_AS_ARRAY,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The option is parsed from authoritative integer storage; values other than exact zero or one reject.",
    },
];

const CELL2TABLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "C contents",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "All eight integer classes may occur in cell contents. Compatible one-row values in a column are vertically concatenated into an integer table variable; scalar doubles may coexist except with int64 or uint64.",
    }];
pub const CELL2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "T = cell2table(C, Name,Value...) with integer cell contents",
        inputs: &CELL2TABLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Homogeneous same-class integer columns preserve exact authoritative storage. Mixed integer columns use the leftmost integer class and saturating assignment; scalar doubles use that conversion except with int64/uint64. Incompatible columns remain cells. Nested resident objects are preserved without eager gather; resident numeric arrays are currently incompatible rather than concatenated into a resident variable.",
    }];

const ARRAY2TABLE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented homogeneous array domain includes all eight real integer classes.",
    }];

pub const ARRAY2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "T = array2table(integer_A, Name,Value...)",
        inputs: &ARRAY2TABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each column becomes a table variable with A's exact authoritative integer storage and class. Interactive resident GPU input is a mode-gated RunMat extension that gathers before table construction.",
    }];

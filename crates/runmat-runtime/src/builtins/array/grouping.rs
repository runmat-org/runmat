//! MATLAB-compatible array grouping, binning, and grouped-apply builtins.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericScalar,
    NumericStorage, ObjectInstance, SparseTensor, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::{
    categorical_label_at, is_tabular_object, select_rows, table_from_columns, table_height,
    table_variable_names_from_object, table_variables, value_row_count,
};
use crate::{
    build_runtime_error, call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult,
    RuntimeError,
};

const MAX_MATERIALIZED_ELEMENTS: usize = 50_000_000;

const FINDGROUPS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "findgroups on GPU-resident grouping data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsResidentInputExtension"),
    };
const FINDGROUPS_MATRIX_COLUMNS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-matrix-as-columns",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "findgroups matrix inputs interpreted as grouping columns are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsMatrixColumnsExtension"),
    };
const FINDGROUPS_TABLE_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-table-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "findgroups(T,selector) is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsTableSelectorExtension"),
    };
const FINDGROUPS_TIMETABLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "findgroups-timetable-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "findgroups on timetable input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FindgroupsTimetableExtension"),
};
pub const FINDGROUPS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FINDGROUPS_RESIDENT_INPUT_EXTENSION,
    FINDGROUPS_MATRIX_COLUMNS_EXTENSION,
    FINDGROUPS_TABLE_SELECTOR_EXTENSION,
    FINDGROUPS_TIMETABLE_EXTENSION,
];

const FINDGROUPS_INTEGER_VECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is a documented numeric grouping vector and is compared from authoritative storage.",
    }];
const FINDGROUPS_INTEGER_MULTI_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1,...,AN",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Each integer grouping role retains its own class while exact tuples are sorted lexicographically.",
    }];
const FINDGROUPS_INTEGER_TABLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer table variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table variables are exact grouping roles and retain their class in TID.",
    }];
pub const FINDGROUPS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[G,ID] = findgroups(integer_A)",
        inputs: &FINDGROUPS_INTEGER_VECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "G is double with A's orientation; ID preserves A's integer class and exact values, including values above flintmax.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[G,ID1,...,IDN] = findgroups(integer_A1,...,integer_AN)",
        inputs: &FINDGROUPS_INTEGER_MULTI_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "G is double and each ID output preserves the corresponding input class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[G,TID] = findgroups(T_with_integer_variables)",
        inputs: &FINDGROUPS_INTEGER_TABLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "G is double; TID preserves variable names, integer classes, and exact group identifiers.",
    },
];

const GRP2IDX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "s",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented grouping-vector inputs and are compared from authoritative storage.",
    }];
pub const GRP2IDX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[g,gN,gL] = grp2idx(integer_s)",
        inputs: &GRP2IDX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "g is a double group-index column, gN is cellstr, and gL preserves s's exact integer class. Documented resident inputs currently use the runtime gather fallback.",
    }];

const GROUPCOUNTS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupcounts-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "groupcounts on interactive resident GPU data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupcountsResidentInputExtension"),
    };
pub const GROUPCOUNTS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [GROUPCOUNTS_RESIDENT_INPUT_EXTENSION];

const GROUPCOUNTS_INTEGER_ARRAY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer numeric grouping columns are compared exactly and their unique group labels preserve class.",
    }];
const GROUPCOUNTS_INTEGER_TABLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer grouping table variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer-valued table grouping variables retain exact class and values in the output table.",
    }];
pub const GROUPCOUNTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B,BG,BP] = groupcounts(integer_A)",
        inputs: &GROUPCOUNTS_INTEGER_ARRAY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Unbinned grouping and BG are exact and BG preserves A's class; B and BP are double count and percentage outputs. [integer-audit-open] Documented integer groupbins remain unresolved and keep this name in the quantitative audit queue.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = groupcounts(T,groupvars) with integer grouping variables",
        inputs: &GROUPCOUNTS_INTEGER_TABLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "For the unbinned form, output grouping variables preserve exact integer class while GroupCount and Percent are double. [integer-audit-open] Documented integer groupbins remain unresolved and keep this name in the quantitative audit queue.",
    },
];

const DISCRETIZE_INTEGER_X_EDGES_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented and explicit-edge comparisons use authoritative values.",
    },
    BuiltinIntegerInputCapability {
        name: "edges",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer edge classes are documented; increasing edges are validated without an f64 mirror.",
    },
];
const DISCRETIZE_INTEGER_VALUES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer replacement values preserve class and use exact zero for out-of-range or missing X.",
    }];
pub const DISCRETIZE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = discretize(integer_X,integer_edges,___)",
        inputs: &DISCRETIZE_INTEGER_X_EDGES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Explicit-edge bin assignment is exact across mixed integer and floating classes; default bin numbers are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = discretize(X,edges,integer_values,___)",
        inputs: &DISCRETIZE_INTEGER_VALUES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Replacement output preserves the values vector's numeric class; missing assignments are zero for integer values.",
    },
];

const OUTPUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Builtin outputs.",
}];

const INPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "MATLAB-compatible arguments.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = groupingBuiltin(args...)",
    inputs: &INPUT_VARIADIC,
    outputs: &OUTPUT_ANY,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.INVALID_INPUT",
    identifier: Some("RunMat:grouping:InvalidInput"),
    when: "Inputs are malformed, have incompatible lengths, or request unsupported grouped output.",
    message: "grouping builtin: invalid input",
};

const ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.CALLBACK",
    identifier: Some("RunMat:grouping:CallbackFailed"),
    when: "A grouped callback fails or returns incompatible outputs.",
    message: "grouping builtin: callback failed",
};

const ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.TOO_LARGE",
    identifier: Some("RunMat:grouping:TooLarge"),
    when: "The requested dense or combinatorial output exceeds RunMat's materialization limit.",
    message: "grouping builtin: output is too large",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_CALLBACK, ERROR_TOO_LARGE];
const COMBINATIONS_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_TOO_LARGE];

pub const GROUPING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const DISCRETIZE_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bin numbers or replacement values with the shape of X.",
}];
const DISCRETIZE_OUTPUT_Y_EDGES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin numbers or replacement values with the shape of X.",
    },
    BuiltinParamDescriptor {
        name: "E",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Computed bin-edge row vector.",
    },
];
const DISCRETIZE_INPUTS_X_EDGES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric input values to assign to bins.",
    },
    BuiltinParamDescriptor {
        name: "edges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Monotonically increasing numeric bin edges.",
    },
];
const DISCRETIZE_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric input values to assign to bins.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer number of equal-width bins.",
    },
];
const DISCRETIZE_INPUTS_VALUES: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric input values to assign to bins.",
    },
    BuiltinParamDescriptor {
        name: "edges_or_N",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Explicit numeric edges or a positive scalar bin count.",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One replacement value for each bin.",
    },
];
const DISCRETIZE_INPUTS_INCLUDED_EDGE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric input values to assign to bins.",
    },
    BuiltinParamDescriptor {
        name: "edges_or_N",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Explicit numeric edges or a positive scalar bin count.",
    },
    BuiltinParamDescriptor {
        name: "IncludedEdge",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal option name \"IncludedEdge\".",
    },
    BuiltinParamDescriptor {
        name: "side",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Either \"left\" or \"right\".",
    },
];
const DISCRETIZE_INPUTS_X_N_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric input values to assign to bins.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer number of equal-width bins.",
    },
    BuiltinParamDescriptor {
        name: "arguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional replacement values and IncludedEdge name-value pair.",
    },
];
const DISCRETIZE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Y = discretize(X, edges)",
        inputs: &DISCRETIZE_INPUTS_X_EDGES,
        outputs: &DISCRETIZE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(X, N)",
        inputs: &DISCRETIZE_INPUTS_X_N,
        outputs: &DISCRETIZE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(___, values)",
        inputs: &DISCRETIZE_INPUTS_VALUES,
        outputs: &DISCRETIZE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(___, \"IncludedEdge\", side)",
        inputs: &DISCRETIZE_INPUTS_INCLUDED_EDGE,
        outputs: &DISCRETIZE_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, E] = discretize(X, N, ___)",
        inputs: &DISCRETIZE_INPUTS_X_N_REST,
        outputs: &DISCRETIZE_OUTPUT_Y_EDGES,
    },
];
const DISCRETIZE_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_TOO_LARGE];
pub const DISCRETIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DISCRETIZE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISCRETIZE_ERRORS,
};

const COMBINATIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host table whose variables preserve the corresponding input classes.",
}];
const COMBINATIONS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more input arrays; each is linearized in column-major order.",
}];
const COMBINATIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = combinations(A1, A2, ..., An)",
    inputs: &COMBINATIONS_INPUTS,
    outputs: &COMBINATIONS_OUTPUT,
}];
pub const COMBINATIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMBINATIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COMBINATIONS_ERRORS,
};

pub(crate) const COMBINATIONS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "combinations-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "combinations with a resident input and host-table output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CombinationsResidentInputExtension"),
    };
pub const COMBINATIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [COMBINATIONS_RESIDENT_INPUT_EXTENSION];

const COMBINATIONS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1...An",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every public integer class is accepted and each output table variable retains its corresponding input class and exact values.",
    }];
pub const COMBINATIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "T = combinations(integer_A1, ..., integer_An)",
        inputs: &COMBINATIONS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host inputs are repeated from authoritative native storage without an f64 mirror. A resident input is a gated RunMat extension because the required table result is host-resident.",
    }];

const ACCUMARRAY_IND_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ind",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive one-based output subscripts.",
};
const ACCUMARRAY_DATA_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar or vector data to accumulate.",
};
const ACCUMARRAY_SIZE_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("[]"),
    description: "Positive output-size vector or [].",
};
const ACCUMARRAY_FUN_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("@sum"),
    description: "Scalar-returning group function or [].",
};
const ACCUMARRAY_FILL_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fillval",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("0"),
    description: "Scalar fill matching the group-function output class or [].",
};
const ACCUMARRAY_SPARSE_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "issparse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("false"),
    description: "Logical or numeric scalar 0 or 1 selecting sparse output.",
};
const ACCUMARRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Accumulated full or sparse array.",
}];
const ACCUMARRAY_INPUTS_2: [BuiltinParamDescriptor; 2] =
    [ACCUMARRAY_IND_PARAM, ACCUMARRAY_DATA_PARAM];
const ACCUMARRAY_INPUTS_3: [BuiltinParamDescriptor; 3] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
];
const ACCUMARRAY_INPUTS_4: [BuiltinParamDescriptor; 4] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
];
const ACCUMARRAY_INPUTS_5: [BuiltinParamDescriptor; 5] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
    ACCUMARRAY_FILL_PARAM,
];
const ACCUMARRAY_INPUTS_6: [BuiltinParamDescriptor; 6] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
    ACCUMARRAY_FILL_PARAM,
    ACCUMARRAY_SPARSE_PARAM,
];
const ACCUMARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data)",
        inputs: &ACCUMARRAY_INPUTS_2,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz)",
        inputs: &ACCUMARRAY_INPUTS_3,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun)",
        inputs: &ACCUMARRAY_INPUTS_4,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun, fillval)",
        inputs: &ACCUMARRAY_INPUTS_5,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun, fillval, issparse)",
        inputs: &ACCUMARRAY_INPUTS_6,
        outputs: &ACCUMARRAY_OUTPUT,
    },
];
pub const ACCUMARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACCUMARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ACCUMARRAY_STRUCTURAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "ind",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive one-based subscripts accept every integer class or exactly integral floating values and are parsed without an f64 intermediary.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Explicit positive output dimensions accept every integer class or exactly integral floating values within platform and materialization bounds.",
    },
];

const ACCUMARRAY_DEFAULT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scalar or vector data accepts every integer class; scalar data expands in its native class before groups are formed.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The default integer-data sum returns double, so its fill value must also be double rather than a typed integer.",
    },
];

const ACCUMARRAY_CUSTOM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scalar or vector data accepts every integer class and reaches the group computation as a native-class column vector.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer fill value is valid when it has the same class as the scalar returned by the group computation.",
    },
];

const ACCUMARRAY_SPARSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse accumarray output requires double input data; typed-integer data is rejected before accumulation.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse accumarray output requires a double zero or omitted fill; typed-integer fills are rejected.",
    },
    BuiltinIntegerInputCapability {
        name: "issparse",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The sparse selector accepts logical or numeric scalar 0 or 1 in every integer class and rejects every other numeric value.",
    },
];

pub const ACCUMARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(integer_ind, data, integer_sz)",
        inputs: &ACCUMARRAY_STRUCTURAL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Subscripts and explicit sizes remain exact through bounds checks and column-major linearization; oversized or unmaterializable results reject deterministically.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, [])",
        inputs: &ACCUMARRAY_DEFAULT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "The default sum follows sum's default integer rule and returns double; interactive GPU accumarray does not admit integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, fun, integer_fillval)",
        inputs: &ACCUMARRAY_CUSTOM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Each group reaches fun as a native-class column vector; B has the class returned by fun, and any integer fill must match that class exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, fun, integer_fillval, true)",
        inputs: &ACCUMARRAY_SPARSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented sparse form is double-only for both data and group results and permits only an omitted or double-zero fill.",
    },
];

#[derive(Clone, Debug)]
enum Atom {
    Missing,
    Logical(bool),
    Number(f64),
    Integer(IntValue),
    CalendarDuration(f64, f64),
    Text(String),
}

impl Atom {
    fn rank(&self) -> u8 {
        match self {
            Self::Logical(_) => 0,
            Self::Number(_) => 1,
            Self::Integer(_) => 2,
            Self::CalendarDuration(_, _) => 3,
            Self::Text(_) => 4,
            Self::Missing => 5,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Missing => "<missing>".to_string(),
            Self::Logical(flag) => {
                if *flag {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            Self::Number(value) => format_key_number(*value),
            Self::Integer(value) => format_integer_key(value),
            Self::CalendarDuration(months, days) => format!("{months}mo {days}d"),
            Self::Text(text) => text.clone(),
        }
    }
}

impl PartialEq for Atom {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Atom {}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = self.rank().cmp(&other.rank());
        if rank != Ordering::Equal {
            return rank;
        }
        match (self, other) {
            (Self::Missing, Self::Missing) => Ordering::Equal,
            (Self::Logical(a), Self::Logical(b)) => a.cmp(b),
            (Self::Number(a), Self::Number(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Self::Integer(a), Self::Integer(b)) => compare_integer_values(a, b),
            (Self::CalendarDuration(am, ad), Self::CalendarDuration(bm, bd)) => am
                .partial_cmp(bm)
                .unwrap_or(Ordering::Equal)
                .then_with(|| ad.partial_cmp(bd).unwrap_or(Ordering::Equal)),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

fn compare_integer_values(left: &IntValue, right: &IntValue) -> Ordering {
    let left = integer_sign_and_magnitude(left);
    let right = integer_sign_and_magnitude(right);
    match (left.0, right.0) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => left.1.cmp(&right.1),
        (true, true) => right.1.cmp(&left.1),
    }
}

fn integer_sign_and_magnitude(value: &IntValue) -> (bool, u64) {
    match value {
        IntValue::I8(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I16(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I32(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I64(value) => (*value < 0, value.unsigned_abs()),
        IntValue::U8(value) => (false, *value as u64),
        IntValue::U16(value) => (false, *value as u64),
        IntValue::U32(value) => (false, *value as u64),
        IntValue::U64(value) => (false, *value),
    }
}

fn format_integer_key(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

#[derive(Clone)]
struct GroupColumn {
    name: String,
    value: Value,
    rows: usize,
}

struct Grouping {
    ids: Vec<f64>,
    keys: Vec<Vec<Atom>>,
    first_rows: Vec<usize>,
    row_groups: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct GroupOptions {
    include_missing: bool,
}

impl GroupOptions {
    fn parse(args: &[Value], context: &str) -> BuiltinResult<Self> {
        let mut include_missing = true;
        let mut idx = 0usize;
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(grouping_error(format!(
                    "{context}: name-value options must be provided in pairs"
                )));
            }
            let name = scalar_text(&args[idx], context)?;
            if name.eq_ignore_ascii_case("IncludeMissingGroups") {
                include_missing = binary_bool_scalar(&args[idx + 1], "IncludeMissingGroups")?;
            } else if name.eq_ignore_ascii_case("IncludeEmptyGroups") {
                let include_empty = binary_bool_scalar(&args[idx + 1], "IncludeEmptyGroups")?;
                if include_empty {
                    return Err(grouping_error(format!(
                        "{context}: IncludeEmptyGroups=true is not supported until categorical level expansion is implemented"
                    )));
                }
            } else {
                return Err(grouping_error(format!(
                    "{context}: unsupported option '{name}'"
                )));
            }
            idx += 2;
        }
        Ok(Self { include_missing })
    }
}

fn grouping_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn callback_error(message: impl Into<String>, source: Option<RuntimeError>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_CALLBACK.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(source) = source {
        builder = builder.with_source(source);
    }
    builder.build()
}

fn too_large_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_TOO_LARGE.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "findgroups",
    category = "array/grouping",
    summary = "Find groups and return group numbers.",
    keywords = "findgroups,groups,grouping,table,categorical",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    extensions(crate::builtins::array::grouping::FINDGROUPS_EXTENSIONS),
    integer_capabilities(crate::builtins::array::grouping::FINDGROUPS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn findgroups_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_findgroups_extensions(&first, &rest)?;
    let mut args = Vec::with_capacity(rest.len() + 1);
    args.push(gather_if_needed_async(&first).await?);
    for value in rest {
        args.push(gather_if_needed_async(&value).await?);
    }
    let (columns, table_mode, output_shape) = findgroups_columns(args)?;
    let grouping = build_grouping(&columns)?;
    let outputs = findgroups_outputs(&columns, &grouping, table_mode, output_shape)?;
    multi_output(outputs)
}

#[runtime_builtin(
    name = "grp2idx",
    category = "array/grouping",
    summary = "Create an index vector from a grouping variable.",
    keywords = "grp2idx,groups,index,categorical,statistics",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::grouping::GRP2IDX_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn grp2idx_builtin(value: Value) -> BuiltinResult<Value> {
    let resident_input = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let resident_owner = resident_input
        .as_ref()
        .and_then(runmat_accelerate_api::provider_for_handle);
    let value = gather_if_needed_async(&value).await?;
    let columns = columns_from_group_value("G", value, true)?;
    if columns.len() != 1 {
        return Err(grouping_error(
            "grp2idx: expected one grouping vector, not a matrix of grouping columns",
        ));
    }
    let grouping = build_grouping(&columns)?;
    let g = Tensor::new(grouping.ids.clone(), vec![grouping.ids.len(), 1])
        .map(Value::Tensor)
        .map_err(grouping_error)?;
    let names = grouping
        .keys
        .iter()
        .map(|key| key.first().map(Atom::label).unwrap_or_default())
        .collect::<Vec<_>>();
    let gn_values = names
        .into_iter()
        .map(|name| {
            let chars = name.chars().collect::<Vec<_>>();
            let cols = chars.len();
            CharArray::new(chars, 1, cols)
                .map(Value::CharArray)
                .map_err(grouping_error)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let gn =
        Value::Cell(CellArray::new(gn_values, grouping.keys.len(), 1).map_err(grouping_error)?);
    let gl = group_label_outputs(&columns, &grouping)?
        .into_iter()
        .next()
        .ok_or_else(|| grouping_error("grp2idx: missing group-level output"))?;
    if let (Some(provider), Some(prototype)) = (resident_owner, resident_input.as_ref()) {
        let (g, gl) = restore_grp2idx_outputs(provider, prototype, g, gl)?;
        return multi_output(vec![g, gn, gl]);
    }
    multi_output(vec![g, gn, gl])
}

fn restore_grp2idx_outputs(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    host_g: Value,
    host_gl: Value,
) -> BuiltinResult<(Value, Value)> {
    let host_outputs = [host_g, host_gl];
    let mut restored = Vec::with_capacity(host_outputs.len());
    for host_value in host_outputs.iter().cloned() {
        let protected = std::iter::once(prototype.clone())
            .chain(restored.iter().filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }))
            .collect::<Vec<_>>();
        let output =
            match crate::builtins::math::trigonometry::inverse_helpers::upload_value_like_protected(
                provider, host_value, "grp2idx", prototype, &protected,
            ) {
                Ok(output) => output,
                Err(_) => {
                    free_grp2idx_outputs(&restored, prototype);
                    return Ok((host_outputs[0].clone(), host_outputs[1].clone()));
                }
            };
        let valid = match &output {
            Value::GpuTensor(handle) => {
                !same_grp2idx_handle(handle, prototype)
                    && restored.iter().all(|prior| match prior {
                        Value::GpuTensor(prior) => !same_grp2idx_handle(handle, prior),
                        _ => false,
                    })
            }
            _ => false,
        };
        if !valid {
            if let Value::GpuTensor(handle) = &output {
                free_grp2idx_handle_if_fresh(handle, prototype, &restored);
            }
            free_grp2idx_outputs(&restored, prototype);
            return Ok((host_outputs[0].clone(), host_outputs[1].clone()));
        }
        restored.push(output);
    }
    Ok((restored.remove(0), restored.remove(0)))
}

fn same_grp2idx_handle(
    left: &runmat_accelerate_api::GpuTensorHandle,
    right: &runmat_accelerate_api::GpuTensorHandle,
) -> bool {
    left.device_id == right.device_id && left.buffer_id == right.buffer_id
}

fn free_grp2idx_handle_if_fresh(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    protected: &[Value],
) {
    if same_grp2idx_handle(handle, prototype)
        || protected.iter().any(|value| match value {
            Value::GpuTensor(protected) => same_grp2idx_handle(handle, protected),
            _ => false,
        })
    {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
        let _ = owner.free(handle);
    }
}

fn free_grp2idx_outputs(outputs: &[Value], prototype: &runmat_accelerate_api::GpuTensorHandle) {
    let mut freed = std::collections::BTreeSet::new();
    for handle in outputs.iter().filter_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }) {
        if same_grp2idx_handle(handle, prototype)
            || !freed.insert((handle.device_id, handle.buffer_id))
        {
            continue;
        }
        if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
            let _ = owner.free(handle);
        }
    }
}

#[runtime_builtin(
    name = "groupcounts",
    category = "array/grouping",
    summary = "Count the number of elements in each group.",
    keywords = "groupcounts,groups,count,table,categorical",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    extensions(crate::builtins::array::grouping::GROUPCOUNTS_EXTENSIONS),
    integer_capabilities(crate::builtins::array::grouping::GROUPCOUNTS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn groupcounts_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_groupcounts_extensions(&first, &rest)?;
    let first = gather_if_needed_async(&first).await?;
    let rest = gather_values(rest).await?;
    if let Value::Object(object) = first.clone() {
        if is_tabular_object(&object) {
            let (selector_args, option_args) = split_option_tail(rest)?;
            return groupcounts_table(object, selector_args, option_args);
        }
    }
    let (data_args, option_args) = split_option_tail(rest)?;
    if !data_args.is_empty() {
        return Err(grouping_error(
            "groupcounts: groupbins are documented but not yet implemented; unbinned grouping accepts no positional argument after A",
        ));
    }
    let options = GroupOptions::parse(&option_args, "groupcounts")?;
    let columns = if matches!(first, Value::Cell(_)) {
        columns_from_group_value("A", first, true)?
    } else {
        columns_from_group_args(vec![first])?
    };
    let grouping = build_grouping_with_options(&columns, options)?;
    let counts = grouping
        .row_groups
        .iter()
        .map(|rows| rows.len() as f64)
        .collect::<Vec<_>>();
    let b =
        Value::Tensor(Tensor::new(counts, vec![grouping.keys.len(), 1]).map_err(grouping_error)?);
    let bg_columns = group_label_outputs(&columns, &grouping)?;
    let bg = if bg_columns.len() == 1 {
        bg_columns.into_iter().next().expect("one grouping column")
    } else {
        Value::Cell(CellArray::new(bg_columns, 1, columns.len()).map_err(grouping_error)?)
    };
    let bp = Value::Tensor(
        Tensor::new(
            grouping
                .row_groups
                .iter()
                .map(|rows| {
                    if grouping.ids.is_empty() {
                        0.0
                    } else {
                        rows.len() as f64 * 100.0 / grouping.ids.len() as f64
                    }
                })
                .collect(),
            vec![grouping.keys.len(), 1],
        )
        .map_err(grouping_error)?,
    );
    multi_output(vec![b, bg, bp])
}

fn ensure_groupcounts_extensions(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::value_contains_gpu(first) || rest.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GROUPCOUNTS_RESIDENT_INPUT_EXTENSION,
            "groupcounts",
        )?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "splitapply",
    category = "array/grouping",
    summary = "Split data into groups and apply a function.",
    keywords = "splitapply,groups,apply,function,table",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn splitapply_builtin(
    func: Value,
    first_data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let func = gather_if_needed_async(&func).await?;
    let first_data = gather_if_needed_async(&first_data).await?;
    let rest = gather_values(rest).await?;
    if rest.is_empty() {
        return Err(grouping_error(
            "splitapply: expected data arguments followed by group numbers",
        ));
    }
    let (group_value, data_tail) = rest.split_last().expect("checked non-empty");
    let mut data_values = Vec::with_capacity(data_tail.len() + 1);
    data_values.push(first_data);
    data_values.extend_from_slice(data_tail);
    let group_columns = columns_from_group_value("G", group_value.clone(), true)?;
    if group_columns.len() != 1 {
        return Err(grouping_error("splitapply: G must be a grouping vector"));
    }
    let grouping = build_grouping(&group_columns)?;
    let expected_rows = grouping.ids.len();
    for value in &data_values {
        let rows = value_row_count(value)?;
        if rows != expected_rows {
            return Err(grouping_error(format!(
                "splitapply: data arguments must have {expected_rows} rows to match G, got {rows}"
            )));
        }
    }
    let requested_outputs = crate::output_count::current_output_count()
        .unwrap_or(1)
        .max(1);
    let mut collectors = (0..requested_outputs)
        .map(|_| Vec::<Value>::new())
        .collect::<Vec<_>>();
    for rows in &grouping.row_groups {
        let mut callback_args = Vec::with_capacity(data_values.len());
        for value in &data_values {
            callback_args.push(select_rows(value, rows)?);
        }
        let result = call_feval_async_with_outputs(func.clone(), &callback_args, requested_outputs)
            .await
            .map_err(|err| callback_error("splitapply: callback failed", Some(err)))?;
        let outputs = normalize_outputs(result, requested_outputs, "splitapply")?;
        for (collector, output) in collectors.iter_mut().zip(outputs) {
            collector.push(gather_if_needed_async(&output).await?);
        }
    }
    let outputs = collectors
        .into_iter()
        .map(|values| collect_group_results(values, grouping.keys.len(), "splitapply"))
        .collect::<BuiltinResult<Vec<_>>>()?;
    multi_output(outputs)
}

#[runtime_builtin(
    name = "accumarray",
    category = "array/grouping",
    summary = "Accumulate values into an array by subscript groups.",
    keywords = "accumarray,accumulate,groups,sum,sparse",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::ACCUMARRAY_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::grouping::ACCUMARRAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn accumarray_builtin(
    subs: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if matches!(&data, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        return Err(grouping_error(
            "accumarray: GPU input data must be logical, single, or double",
        ));
    }
    if matches!(
        rest.get(2),
        Some(Value::GpuTensor(handle))
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
    ) {
        return Err(grouping_error(
            "accumarray: GPU fill value must be logical, single, or double",
        ));
    }
    let subs = gather_if_needed_async(&subs).await?;
    let data = gather_if_needed_async(&data).await?;
    let rest = gather_values(rest).await?;
    accumarray_impl(subs, data, rest).await
}

#[runtime_builtin(
    name = "discretize",
    category = "array/grouping",
    summary = "Group numeric data into bins.",
    keywords = "discretize,bins,edges,categorical,grouping",
    accel = "cpu",
    integer_capabilities(crate::builtins::array::grouping::DISCRETIZE_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::array::grouping::DISCRETIZE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn discretize_builtin(
    x: Value,
    edges_or_n: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let x = gather_if_needed_async(&x).await?;
    let edges_or_n = gather_if_needed_async(&edges_or_n).await?;
    let rest = gather_values(rest).await?;
    let computed_edges = is_discretize_bin_count(&edges_or_n);
    let (output, edges) = discretize_impl(x, edges_or_n, rest)?;
    match crate::output_count::current_output_count() {
        None => Ok(output),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![output])),
        Some(2) if computed_edges => Ok(Value::OutputList(vec![output, edges])),
        Some(2) => Err(grouping_error(
            "discretize: the second edge output requires a scalar bin count",
        )),
        Some(_) => Err(grouping_error("discretize: too many output arguments")),
    }
}

#[runtime_builtin(
    name = "combinations",
    category = "array/grouping",
    summary = "Generate all element combinations of arrays.",
    keywords = "combinations,cartesian,table,combinatorics",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::COMBINATIONS_DESCRIPTOR),
    extensions(crate::builtins::array::grouping::COMBINATIONS_EXTENSIONS),
    integer_capabilities(crate::builtins::array::grouping::COMBINATIONS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn combinations_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(first, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMBINATIONS_RESIDENT_INPUT_EXTENSION,
            "combinations",
        )?;
    }
    let first = gather_if_needed_async(&first).await?;
    let rest = gather_values(rest).await?;
    combinations_impl(first, rest)
}

fn ensure_findgroups_extensions(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::value_contains_gpu(first) || rest.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FINDGROUPS_RESIDENT_INPUT_EXTENSION,
            "findgroups",
        )?;
    }
    if value_is_grouping_matrix(first) || rest.iter().any(value_is_grouping_matrix) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FINDGROUPS_MATRIX_COLUMNS_EXTENSION,
            "findgroups",
        )?;
    }
    if let Value::Object(object) = first {
        if object.is_class("timetable") {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FINDGROUPS_TIMETABLE_EXTENSION,
                "findgroups",
            )?;
        }
        if is_tabular_object(object) && !rest.is_empty() {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FINDGROUPS_TABLE_SELECTOR_EXTENSION,
                "findgroups",
            )?;
            if rest.len() != 1 {
                return Err(grouping_error(
                    "findgroups: table selector form accepts exactly one selector",
                ));
            }
        }
    }
    Ok(())
}

fn value_is_grouping_matrix(value: &Value) -> bool {
    let shape = match value {
        Value::Tensor(tensor) => &tensor.shape,
        Value::LogicalArray(array) => &array.shape,
        Value::StringArray(array) => &array.shape,
        Value::GpuTensor(handle) => &handle.shape,
        _ => return false,
    };
    shape.first().copied().unwrap_or(1) > 1 && shape.get(1).copied().unwrap_or(1) > 1
}

fn findgroups_columns(args: Vec<Value>) -> BuiltinResult<(Vec<GroupColumn>, bool, Vec<usize>)> {
    if args.is_empty() {
        return Err(grouping_error("findgroups: expected at least one input"));
    }
    if let Value::Object(object) = args[0].clone() {
        if is_tabular_object(&object) {
            let names = table_variable_names_from_object(&object)?;
            let selected = if let Some(selector) = args.get(1) {
                parse_name_selector(selector, &names, "findgroups")?
            } else {
                names
            };
            let variables = table_variables(&object)?;
            let mut columns = Vec::with_capacity(selected.len());
            for name in selected {
                let value = variables.fields.get(&name).cloned().ok_or_else(|| {
                    grouping_error(format!("findgroups: missing table variable '{name}'"))
                })?;
                validate_findgroups_grouping_value(&value)?;
                if value_is_grouping_matrix(&value) {
                    return Err(grouping_error(format!(
                        "findgroups: table variable '{name}' must be a vector"
                    )));
                }
                columns.push(GroupColumn {
                    rows: findgroups_value_row_count(&value)?,
                    name,
                    value,
                });
            }
            let height = columns.first().map(|column| column.rows).unwrap_or(0);
            return Ok((columns, true, vec![height, 1]));
        }
    }
    for value in &args {
        validate_findgroups_grouping_value(value)?;
    }
    let output_shape = findgroups_vector_shape(&args[0])?;
    for value in args.iter().skip(1) {
        if findgroups_vector_shape(value)? != output_shape {
            return Err(grouping_error(
                "findgroups: grouping vectors must have matching sizes",
            ));
        }
    }
    Ok((columns_from_group_args(args)?, false, output_shape))
}

fn findgroups_vector_shape(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) if tensor.rows() > 1 && tensor.cols() > 1 => {
            Ok(vec![tensor.rows(), 1])
        }
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::LogicalArray(array)
            if array.shape.first().copied().unwrap_or(1) > 1
                && array.shape.get(1).copied().unwrap_or(1) > 1 =>
        {
            Ok(vec![array.shape[0], 1])
        }
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::StringArray(array) if array.rows() > 1 && array.cols() > 1 => {
            Ok(vec![array.rows(), 1])
        }
        Value::StringArray(array) => Ok(array.shape.clone()),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Object(object) if object.is_class("categorical") => object
            .properties
            .get("Codes")
            .map(findgroups_vector_shape)
            .transpose()
            .map(|shape| shape.unwrap_or_else(|| vec![0, 1])),
        Value::Object(object) if object.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value).map(|tensor| tensor.shape)
        }
        Value::Object(object) if object.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map(|tensor| tensor.shape)
        }
        Value::Object(object) if object.is_class("calendarDuration") => object
            .properties
            .get("__months")
            .map(findgroups_vector_shape)
            .transpose()
            .map(|shape| shape.unwrap_or_else(|| vec![0, 1])),
        _ => Ok(vec![1, 1]),
    }
}

fn validate_findgroups_grouping_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::StringArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_) => Ok(()),
        Value::Cell(cell)
            if cell
                .data
                .iter()
                .all(|value| matches!(value, Value::CharArray(chars) if chars.rows <= 1)) =>
        {
            Ok(())
        }
        Value::Object(object)
            if object.is_class("categorical")
                || object.is_class("datetime")
                || object.is_class("duration")
                || object.is_class("calendarDuration") =>
        {
            Ok(())
        }
        Value::SparseTensor(_) => Err(grouping_error(
            "findgroups: sparse grouping variables are not supported",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(grouping_error(
            "findgroups: complex grouping variables are not supported",
        )),
        other => Err(grouping_error(format!(
            "findgroups: unsupported grouping variable {other:?}"
        ))),
    }
}

fn findgroups_value_row_count(value: &Value) -> BuiltinResult<usize> {
    if let Value::Object(object) = value {
        if object.is_class("calendarDuration") {
            return object
                .properties
                .get("__months")
                .map(value_row_count)
                .transpose()
                .map(|rows| rows.unwrap_or(0));
        }
    }
    value_row_count(value)
}

fn columns_from_group_args(args: Vec<Value>) -> BuiltinResult<Vec<GroupColumn>> {
    let mut columns = Vec::new();
    for (idx, value) in args.into_iter().enumerate() {
        columns.extend(columns_from_group_value(
            &format!("Var{}", idx + 1),
            value,
            true,
        )?);
    }
    Ok(columns)
}

fn columns_from_group_value(
    base_name: &str,
    value: Value,
    split_matrix: bool,
) -> BuiltinResult<Vec<GroupColumn>> {
    match value {
        Value::Tensor(tensor) => tensor_columns(base_name, tensor, split_matrix),
        Value::LogicalArray(array) => logical_columns(base_name, array, split_matrix),
        Value::StringArray(array) => string_columns(base_name, array, split_matrix),
        Value::Cell(cell) if cell_is_group_vector_list(&cell) => {
            let mut columns = Vec::with_capacity(cell.data.len());
            for (idx, value) in cell.data.into_iter().enumerate() {
                columns.extend(columns_from_group_value(
                    &format!("{base_name}{}", idx + 1),
                    value,
                    false,
                )?);
            }
            Ok(columns)
        }
        Value::Cell(cell) => Ok(vec![GroupColumn {
            rows: cell.rows.max(cell.cols).max(cell.data.len()),
            name: base_name.to_string(),
            value: Value::Cell(cell),
        }]),
        Value::Object(object) if object.is_class("categorical") => {
            let rows = value_row_count(&Value::Object(object.clone()))?;
            Ok(vec![GroupColumn {
                rows,
                name: base_name.to_string(),
                value: Value::Object(object),
            }])
        }
        Value::Object(object) if object.is_class("calendarDuration") => {
            let rows = object
                .properties
                .get("__months")
                .map(value_row_count)
                .transpose()?
                .unwrap_or(0);
            Ok(vec![GroupColumn {
                rows,
                name: base_name.to_string(),
                value: Value::Object(object),
            }])
        }
        other => {
            let rows = value_row_count(&other).unwrap_or(1);
            Ok(vec![GroupColumn {
                rows,
                name: base_name.to_string(),
                value: other,
            }])
        }
    }
}

fn tensor_columns(
    base_name: &str,
    tensor: Tensor,
    split_matrix: bool,
) -> BuiltinResult<Vec<GroupColumn>> {
    if !split_matrix || tensor.cols() <= 1 || tensor.rows() == 1 {
        let rows = tensor_utils::tensor_element_len(&tensor);
        let value = tensor.reshape(vec![rows, 1]).map_err(grouping_error)?;
        return Ok(vec![GroupColumn {
            name: base_name.to_string(),
            rows,
            value: Value::Tensor(value),
        }]);
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let storage = tensor.into_numeric_storage().map_err(grouping_error)?;
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let indices = (0..rows).map(|row| row + col * rows).collect::<Vec<_>>();
        let value = Tensor::from_numeric_storage(
            storage.gather(&indices).map_err(grouping_error)?,
            vec![rows, 1],
        )
        .map_err(grouping_error)?;
        out.push(GroupColumn {
            name: format!("{base_name}{col_plus}", col_plus = col + 1),
            rows,
            value: Value::Tensor(value),
        });
    }
    Ok(out)
}

fn logical_columns(
    base_name: &str,
    array: LogicalArray,
    split_matrix: bool,
) -> BuiltinResult<Vec<GroupColumn>> {
    let rows = array.shape.first().copied().unwrap_or(array.data.len());
    let cols = array.shape.get(1).copied().unwrap_or(1);
    if !split_matrix || cols <= 1 || rows == 1 {
        let len = array.data.len();
        return Ok(vec![GroupColumn {
            name: base_name.to_string(),
            rows: len,
            value: Value::LogicalArray(
                LogicalArray::new(array.data, vec![len, 1]).map_err(grouping_error)?,
            ),
        }]);
    }
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            data.push(*array.data.get(row + col * rows).ok_or_else(|| {
                grouping_error("grouping: logical array shape/data length mismatch")
            })?);
        }
        out.push(GroupColumn {
            name: format!("{base_name}{col_plus}", col_plus = col + 1),
            rows,
            value: Value::LogicalArray(
                LogicalArray::new(data, vec![rows, 1]).map_err(grouping_error)?,
            ),
        });
    }
    Ok(out)
}

fn string_columns(
    base_name: &str,
    array: StringArray,
    split_matrix: bool,
) -> BuiltinResult<Vec<GroupColumn>> {
    let rows = array.rows();
    let cols = array.cols();
    if !split_matrix || cols <= 1 || rows == 1 {
        let len = array.data.len();
        return Ok(vec![GroupColumn {
            name: base_name.to_string(),
            rows: len,
            value: Value::StringArray(
                StringArray::new(array.data, vec![len, 1]).map_err(grouping_error)?,
            ),
        }]);
    }
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            data.push(array.data[row + col * rows].clone());
        }
        out.push(GroupColumn {
            name: format!("{base_name}{col_plus}", col_plus = col + 1),
            rows,
            value: Value::StringArray(
                StringArray::new(data, vec![rows, 1]).map_err(grouping_error)?,
            ),
        });
    }
    Ok(out)
}

fn cell_is_group_vector_list(cell: &CellArray) -> bool {
    !cell.data.is_empty()
        && cell.data.iter().all(|value| {
            matches!(
                value,
                Value::Tensor(_)
                    | Value::LogicalArray(_)
                    | Value::StringArray(_)
                    | Value::Object(_)
                    | Value::Cell(_)
            )
        })
}

fn build_grouping(columns: &[GroupColumn]) -> BuiltinResult<Grouping> {
    build_grouping_with_options(
        columns,
        GroupOptions {
            include_missing: false,
        },
    )
}

fn build_grouping_with_options(
    columns: &[GroupColumn],
    options: GroupOptions,
) -> BuiltinResult<Grouping> {
    if columns.is_empty() {
        return Err(grouping_error(
            "grouping: expected at least one grouping variable",
        ));
    }
    let rows = columns[0].rows;
    for column in columns {
        if column.rows != rows {
            return Err(grouping_error(format!(
                "grouping: grouping variables must have matching row counts ({} vs {})",
                rows, column.rows
            )));
        }
    }
    let mut buckets = BTreeMap::<Vec<Atom>, Vec<usize>>::new();
    let mut row_keys = Vec::with_capacity(rows);
    for row in 0..rows {
        let key = columns
            .iter()
            .map(|column| atom_at(&column.value, row))
            .collect::<BuiltinResult<Vec<_>>>()?;
        if !options.include_missing && key.iter().any(|atom| matches!(atom, Atom::Missing)) {
            row_keys.push(None);
            continue;
        }
        buckets.entry(key.clone()).or_default().push(row);
        row_keys.push(Some(key));
    }
    let keys = buckets.keys().cloned().collect::<Vec<_>>();
    let mut key_to_index = BTreeMap::<Vec<Atom>, usize>::new();
    let mut first_rows = Vec::<usize>::with_capacity(keys.len());
    let mut row_groups = Vec::<Vec<usize>>::with_capacity(keys.len());
    for (idx, key) in keys.iter().enumerate() {
        key_to_index.insert(key.clone(), idx);
        let rows = buckets
            .get(key)
            .cloned()
            .expect("key collected from buckets must exist");
        first_rows.push(*rows.first().expect("nonmissing group has rows"));
        row_groups.push(rows);
    }
    let ids = row_keys
        .into_iter()
        .map(|key| {
            key.and_then(|key| key_to_index.get(&key).copied())
                .map(|idx| idx as f64 + 1.0)
                .unwrap_or(f64::NAN)
        })
        .collect();
    Ok(Grouping {
        ids,
        keys,
        first_rows,
        row_groups,
    })
}

fn atom_at(value: &Value, row: usize) -> BuiltinResult<Atom> {
    match value {
        Value::Tensor(tensor) => tensor
            .numeric_value_at(row)
            .map(numeric_scalar_atom)
            .ok_or_else(|| grouping_error("grouping: numeric row out of bounds")),
        Value::LogicalArray(array) => Ok(array
            .data
            .get(row)
            .map(|flag| Atom::Logical(*flag != 0))
            .unwrap_or(Atom::Missing)),
        Value::StringArray(array) => Ok(array
            .data
            .get(row)
            .map(|text| {
                if is_missing_text(text) {
                    Atom::Missing
                } else {
                    Atom::Text(text.clone())
                }
            })
            .unwrap_or(Atom::Missing)),
        Value::Cell(cell) => cell
            .data
            .get(row)
            .map(scalar_atom)
            .unwrap_or(Ok(Atom::Missing)),
        Value::Object(object) if object.is_class("categorical") => {
            let label = categorical_label_at(object, row);
            Ok(match label.as_deref() {
                None | Some("<undefined>") | Some("") => Atom::Missing,
                Some(text) => Atom::Text(text.to_string()),
            })
        }
        Value::Object(object) if object.is_class("datetime") => {
            let serials = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let value = if row < tensor_utils::tensor_element_len(&serials) {
                tensor_utils::tensor_value_f64(&serials, row)
            } else {
                f64::NAN
            };
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(value))
            }
        }
        Value::Object(object) if object.is_class("duration") => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let value = if row < tensor_utils::tensor_element_len(&tensor) {
                tensor_utils::tensor_value_f64(&tensor, row)
            } else {
                f64::NAN
            };
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(value))
            }
        }
        Value::Object(object) if object.is_class("calendarDuration") => {
            let months = calendar_duration_component(object, "__months", row)?;
            let days = calendar_duration_component(object, "__days", row)?;
            if months.is_nan() || days.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::CalendarDuration(months, days))
            }
        }
        other if row == 0 => scalar_atom(other),
        _ => Ok(Atom::Missing),
    }
}

fn numeric_scalar_atom(value: NumericScalar) -> Atom {
    match value {
        NumericScalar::F64(value) => {
            if value.is_nan() {
                Atom::Missing
            } else {
                Atom::Number(value)
            }
        }
        NumericScalar::F32(value) => {
            if value.is_nan() {
                Atom::Missing
            } else {
                Atom::Number(f64::from(value))
            }
        }
        NumericScalar::I8(value) => Atom::Integer(IntValue::I8(value)),
        NumericScalar::I16(value) => Atom::Integer(IntValue::I16(value)),
        NumericScalar::I32(value) => Atom::Integer(IntValue::I32(value)),
        NumericScalar::I64(value) => Atom::Integer(IntValue::I64(value)),
        NumericScalar::U8(value) => Atom::Integer(IntValue::U8(value)),
        NumericScalar::U16(value) => Atom::Integer(IntValue::U16(value)),
        NumericScalar::U32(value) => Atom::Integer(IntValue::U32(value)),
        NumericScalar::U64(value) => Atom::Integer(IntValue::U64(value)),
    }
}

fn scalar_atom(value: &Value) -> BuiltinResult<Atom> {
    match value {
        Value::Num(value) => {
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(*value))
            }
        }
        Value::Int(value) => Ok(Atom::Integer(value.clone())),
        Value::Bool(flag) => Ok(Atom::Logical(*flag)),
        Value::String(text) => {
            if is_missing_text(text) {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Text(text.clone()))
            }
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text: String = chars.data.iter().collect();
            if text.is_empty() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Text(text))
            }
        }
        other => Ok(Atom::Text(format!("{other}"))),
    }
}

fn calendar_duration_component(
    object: &ObjectInstance,
    field: &str,
    row: usize,
) -> BuiltinResult<f64> {
    match object.properties.get(field) {
        Some(Value::Tensor(tensor)) => tensor
            .numeric_value_at(row)
            .map(numeric_scalar_to_f64)
            .ok_or_else(|| grouping_error("findgroups: calendarDuration row out of bounds")),
        Some(Value::Num(value)) if row == 0 => Ok(*value),
        _ => Err(grouping_error(format!(
            "findgroups: invalid calendarDuration {field} storage"
        ))),
    }
}

fn numeric_scalar_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        NumericScalar::I8(value) => f64::from(value),
        NumericScalar::I16(value) => f64::from(value),
        NumericScalar::I32(value) => f64::from(value),
        NumericScalar::I64(value) => value as f64,
        NumericScalar::U8(value) => f64::from(value),
        NumericScalar::U16(value) => f64::from(value),
        NumericScalar::U32(value) => f64::from(value),
        NumericScalar::U64(value) => value as f64,
    }
}

fn findgroups_outputs(
    columns: &[GroupColumn],
    grouping: &Grouping,
    table_mode: bool,
    output_shape: Vec<usize>,
) -> BuiltinResult<Vec<Value>> {
    let g = Value::Tensor(Tensor::new(grouping.ids.clone(), output_shape).map_err(grouping_error)?);
    let mut outputs = vec![g];
    if table_mode {
        let mut names = Vec::with_capacity(columns.len());
        let mut values = Vec::with_capacity(columns.len());
        for column in columns {
            names.push(column.name.clone());
            values.push(select_group_rows(&column.value, &grouping.first_rows)?);
        }
        outputs.push(table_from_columns(names, values)?);
    } else {
        outputs.extend(group_label_outputs(columns, grouping)?);
    }
    Ok(outputs)
}

fn group_label_outputs(columns: &[GroupColumn], grouping: &Grouping) -> BuiltinResult<Vec<Value>> {
    columns
        .iter()
        .map(|column| select_group_rows(&column.value, &grouping.first_rows))
        .collect()
}

fn select_group_rows(value: &Value, rows: &[usize]) -> BuiltinResult<Value> {
    let Value::Object(object) = value else {
        return select_rows(value, rows);
    };
    if !object.is_class("calendarDuration") {
        return select_rows(value, rows);
    }
    let mut selected = object.clone();
    for field in ["__months", "__days"] {
        let component = object.properties.get(field).ok_or_else(|| {
            grouping_error(format!(
                "findgroups: missing calendarDuration {field} storage"
            ))
        })?;
        selected
            .properties
            .insert(field.to_string(), select_rows(component, rows)?);
    }
    Ok(Value::Object(selected))
}

fn split_option_tail(args: Vec<Value>) -> BuiltinResult<(Vec<Value>, Vec<Value>)> {
    let mut option_start = args.len();
    for (idx, value) in args.iter().enumerate() {
        if is_option_name(value) {
            option_start = idx;
            break;
        }
    }
    if option_start < args.len() && !(args.len() - option_start).is_multiple_of(2) {
        return Err(grouping_error(
            "groupcounts: name-value options must be provided in pairs",
        ));
    }
    Ok((args[..option_start].to_vec(), args[option_start..].to_vec()))
}

fn groupcounts_table(
    object: ObjectInstance,
    selector_args: Vec<Value>,
    option_args: Vec<Value>,
) -> BuiltinResult<Value> {
    let all_names = table_variable_names_from_object(&object)?;
    if selector_args.len() > 1 {
        return Err(grouping_error(
            "groupcounts: groupbins are documented but not yet implemented for table input",
        ));
    }
    let selector = selector_args.first().ok_or_else(|| {
        grouping_error("groupcounts: table input requires a grouping variable selector")
    })?;
    let options = GroupOptions::parse(&option_args, "groupcounts")?;
    let selected = parse_name_selector(selector, &all_names, "groupcounts")?;
    let variables = table_variables(&object)?;
    let height = table_height(&object)?;
    let mut columns = Vec::with_capacity(selected.len());
    for name in &selected {
        let value = variables
            .fields
            .get(name)
            .cloned()
            .ok_or_else(|| grouping_error(format!("groupcounts: missing variable '{name}'")))?;
        columns.push(GroupColumn {
            name: name.clone(),
            rows: value_row_count(&value)?,
            value,
        });
    }
    let grouping = build_grouping_with_options(&columns, options)?;
    let mut out_names = selected.clone();
    let mut out_columns = group_label_outputs(&columns, &grouping)?;
    out_names.push("GroupCount".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            grouping
                .row_groups
                .iter()
                .map(|rows| rows.len() as f64)
                .collect(),
            vec![grouping.keys.len(), 1],
        )
        .map_err(grouping_error)?,
    ));
    out_names.push("Percent".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            grouping
                .row_groups
                .iter()
                .map(|rows| {
                    if height == 0 {
                        0.0
                    } else {
                        rows.len() as f64 * 100.0 / height as f64
                    }
                })
                .collect(),
            vec![grouping.keys.len(), 1],
        )
        .map_err(grouping_error)?,
    ));
    table_from_columns(out_names, out_columns)
}

async fn accumarray_impl(subs: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let index_rows = accumarray_subscripts(subs)?;
    let rows = index_rows.len();
    let data = accumarray_data_column(data, rows)?;
    let output_shape = if let Some(size_value) = rest.first() {
        if is_empty_value(size_value) {
            infer_accumarray_shape(&index_rows)
        } else {
            parse_positive_size_vector(size_value, "accumarray")?
        }
    } else {
        infer_accumarray_shape(&index_rows)
    };
    let fun = rest.get(1).filter(|value| !is_empty_value(value)).cloned();
    let fill = rest.get(2).filter(|value| !is_empty_value(value)).cloned();
    let issparse = rest
        .get(3)
        .map(|value| binary_bool_scalar(value, "accumarray issparse"))
        .transpose()?
        .unwrap_or(false);
    let output_len = checked_element_count(&output_shape, "accumarray")?;
    if output_len > MAX_MATERIALIZED_ELEMENTS {
        return Err(too_large_error("accumarray: output is too large"));
    }
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, subs) in index_rows.iter().enumerate() {
        let lin = subscript_to_linear(subs, &output_shape)?;
        buckets.entry(lin).or_default().push(idx);
    }
    if let Some(fun) = fun {
        return accumarray_callback_output(
            data,
            buckets,
            output_shape,
            output_len,
            fun,
            fill,
            issparse,
        )
        .await;
    }
    if issparse && !is_double_numeric_value(&data) {
        return Err(grouping_error(
            "accumarray: sparse output requires double input data",
        ));
    }
    if issparse && fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: sparse output requires a double zero fill value",
        ));
    }
    if fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: fill value class must match the double default sum output",
        ));
    }
    let data_values = accumarray_data_values(data, rows)?;
    let fill_num = match fill.as_ref() {
        None => 0.0,
        Some(value) => numeric_scalar(value, "accumarray fill value")?,
    };
    if issparse && fill_num != 0.0 {
        return Err(grouping_error(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut data_out = vec![fill_num; output_len];
    for (lin, indices) in buckets {
        data_out[lin] = indices.iter().map(|index| data_values[*index]).sum();
    }
    accumarray_numeric_output(data_out, output_shape, issparse)
}

fn accumarray_data_column(data: Value, rows: usize) -> BuiltinResult<Value> {
    match data {
        Value::Num(value) => Tensor::new(vec![value; rows], vec![rows, 1])
            .map(Value::Tensor)
            .map_err(grouping_error),
        Value::Int(value) => {
            let prototype = IntegerStorage::from_scalar(value);
            Tensor::new_integer(repeat_integer_scalar(&prototype, rows)?, vec![rows, 1])
                .map(Value::Tensor)
                .map_err(grouping_error)
        }
        Value::Bool(value) => LogicalArray::new(vec![u8::from(value); rows], vec![rows, 1])
            .map(Value::LogicalArray)
            .map_err(grouping_error),
        Value::Tensor(tensor) => {
            let len = tensor_utils::tensor_element_len(&tensor);
            if len != 1 && len != rows {
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            if len == rows {
                return tensor
                    .reshape(vec![rows, 1])
                    .map(Value::Tensor)
                    .map_err(grouping_error);
            }
            if let Some(storage) = tensor.integer_storage() {
                return Tensor::new_integer(repeat_integer_scalar(storage, rows)?, vec![rows, 1])
                    .map(Value::Tensor)
                    .map_err(grouping_error);
            }
            Tensor::new_with_dtype(
                vec![tensor_utils::tensor_value_f64(&tensor, 0); rows],
                vec![rows, 1],
                tensor.numeric_dtype(),
            )
            .map(Value::Tensor)
            .map_err(grouping_error)
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 && array.data.len() != rows {
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            let values = if array.data.len() == 1 {
                vec![array.data[0]; rows]
            } else {
                array.data
            };
            LogicalArray::new(values, vec![rows, 1])
                .map(Value::LogicalArray)
                .map_err(grouping_error)
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn repeat_integer_scalar(storage: &IntegerStorage, len: usize) -> BuiltinResult<IntegerStorage> {
    let value = storage
        .value_at(0)
        .ok_or_else(|| grouping_error("accumarray: scalar integer data is empty"))?;
    storage
        .from_exact_values_like(vec![value; len])
        .map_err(grouping_error)
}

async fn accumarray_callback_output(
    data: Value,
    buckets: BTreeMap<usize, Vec<usize>>,
    output_shape: Vec<usize>,
    output_len: usize,
    fun: Value,
    fill: Option<Value>,
    issparse: bool,
) -> BuiltinResult<Value> {
    if issparse && !is_double_numeric_value(&data) {
        return Err(grouping_error(
            "accumarray: sparse output requires double input data",
        ));
    }
    if issparse && fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: sparse output requires a double zero fill value",
        ));
    }
    let mut computed = BTreeMap::new();
    for (linear, rows) in buckets {
        let group = select_rows(&data, &rows)?;
        let result = apply_accumarray_callback(fun.clone(), group).await?;
        computed.insert(linear, accumarray_scalar_result(result)?);
    }
    let prototype = computed.values().next();
    let fill = match fill {
        Some(value) => accumarray_scalar_result(value)?,
        None => prototype
            .map(accumarray_zero_like)
            .transpose()?
            .unwrap_or(Value::Num(0.0)),
    };
    if issparse && value_as_numeric_scalar(&fill) != Some(0.0) {
        return Err(grouping_error(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut values = vec![fill; output_len];
    for (linear, value) in computed {
        values[linear] = value;
    }
    accumarray_collect_results(values, output_shape, issparse)
}

fn accumarray_scalar_result(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(&tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)
                    .map(Value::Int)
                    .ok_or_else(|| grouping_error("accumarray: empty scalar result"));
            }
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(Value::Bool(array.data[0] != 0)),
        Value::Cell(cell) if cell.data.len() == 1 => Ok(cell.data[0].clone()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(value),
        other => Err(grouping_error(format!(
            "accumarray: group function must return a scalar, got {other:?}"
        ))),
    }
}

fn accumarray_zero_like(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Int(value) => IntegerStorage::from_scalar(value.clone())
            .zeros_like(1)
            .value_at(0)
            .map(Value::Int)
            .ok_or_else(|| grouping_error("accumarray: could not construct integer fill")),
        Value::Num(_) => Ok(Value::Num(0.0)),
        Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Tensor::new_with_dtype(vec![0.0], vec![1, 1], tensor.numeric_dtype())
                .map(Value::Tensor)
                .map_err(grouping_error)
        }
        other => Err(grouping_error(format!(
            "accumarray: explicit fill value required for group result {other:?}"
        ))),
    }
}

fn accumarray_collect_results(
    values: Vec<Value>,
    shape: Vec<usize>,
    issparse: bool,
) -> BuiltinResult<Value> {
    let integer_values = values
        .iter()
        .map(exact_integer_scalar)
        .collect::<Option<Vec<_>>>();
    if let Some(exact) = integer_values {
        if issparse {
            return Err(grouping_error(
                "accumarray: sparse output requires double group results",
            ));
        }
        let prototype = IntegerStorage::from_scalar(
            exact
                .first()
                .cloned()
                .ok_or_else(|| grouping_error("accumarray: empty integer result"))?,
        );
        let storage = prototype
            .from_exact_values_like(exact)
            .map_err(|_| grouping_error("accumarray: fill value class must match group output"))?;
        return Tensor::new_integer(storage, shape)
            .map(Value::Tensor)
            .map_err(grouping_error);
    }
    if values
        .iter()
        .any(|value| exact_integer_scalar(value).is_some())
    {
        return Err(grouping_error(
            "accumarray: fill value class must match group output",
        ));
    }
    if values.iter().all(|value| matches!(value, Value::Bool(_))) {
        if issparse {
            return Err(grouping_error(
                "accumarray: sparse output requires double scalar group results",
            ));
        }
        return LogicalArray::new(
            values
                .into_iter()
                .map(|value| match value {
                    Value::Bool(flag) => u8::from(flag),
                    _ => unreachable!("checked above"),
                })
                .collect(),
            shape,
        )
        .map(Value::LogicalArray)
        .map_err(grouping_error);
    }
    let floating_dtype = values
        .iter()
        .map(floating_scalar_dtype)
        .collect::<Option<Vec<_>>>()
        .and_then(|dtypes| {
            let first = dtypes.first().copied()?;
            dtypes.iter().all(|dtype| *dtype == first).then_some(first)
        });
    if floating_dtype.is_none()
        && values
            .iter()
            .any(|value| floating_scalar_dtype(value).is_some())
    {
        return Err(grouping_error(
            "accumarray: fill value class must match group output",
        ));
    }
    if let Some(dtype) = floating_dtype {
        if issparse && dtype != NumericDType::F64 {
            return Err(grouping_error(
                "accumarray: sparse output requires double scalar group results",
            ));
        }
        let data = values
            .iter()
            .map(|value| value_as_numeric_scalar(value).unwrap())
            .collect::<Vec<_>>();
        if issparse {
            return accumarray_numeric_output(data, shape, true);
        }
        return Tensor::new_with_dtype(data, shape, dtype)
            .map(Value::Tensor)
            .map_err(grouping_error);
    }
    if values
        .iter()
        .all(|value| value_as_numeric_scalar(value).is_some())
    {
        let data = values
            .iter()
            .map(|value| value_as_numeric_scalar(value).unwrap())
            .collect();
        return accumarray_numeric_output(data, shape, issparse);
    }
    if issparse {
        return Err(grouping_error(
            "accumarray: sparse output requires double scalar group results",
        ));
    }
    CellArray::new_with_shape(values, shape)
        .map(Value::Cell)
        .map_err(grouping_error)
}

fn floating_scalar_dtype(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Num(_) => Some(NumericDType::F64),
        Value::Tensor(tensor)
            if tensor_utils::is_scalar_tensor(tensor) && tensor.integer_storage().is_none() =>
        {
            Some(tensor.numeric_dtype())
        }
        _ => None,
    }
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor.integer_storage()?.value_at(0)
        }
        _ => None,
    }
}

fn is_integer_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_double_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
}

fn accumarray_subscripts(subs: Value) -> BuiltinResult<Vec<Vec<usize>>> {
    match subs {
        Value::Int(value) => Ok(vec![vec![positive_integer_value(
            &value,
            "accumarray subscript",
        )?]]),
        Value::Num(value) => Ok(vec![vec![positive_integer(value, "accumarray subscript")?]]),
        Value::Tensor(tensor) => {
            if tensor_utils::tensor_element_len(&tensor) == 0 {
                return Ok(Vec::new());
            }
            if let Some(storage) = tensor.integer_storage() {
                if tensor.cols() <= 1 || tensor.rows() == 1 {
                    return storage
                        .exact_values()
                        .into_iter()
                        .map(|value| {
                            Ok(vec![positive_integer_value(
                                &value,
                                "accumarray subscript",
                            )?])
                        })
                        .collect();
                }
                let mut out = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let mut subs = Vec::with_capacity(tensor.cols());
                    for col in 0..tensor.cols() {
                        let index = row + col * tensor.rows();
                        let value = storage.value_at(index).ok_or_else(|| {
                            grouping_error("accumarray: integer subscript index out of bounds")
                        })?;
                        subs.push(positive_integer_value(&value, "accumarray subscript")?);
                    }
                    out.push(subs);
                }
                return Ok(out);
            }
            if tensor.cols() <= 1 || tensor.rows() == 1 {
                tensor_utils::tensor_into_values_f64(tensor)
                    .into_iter()
                    .map(|value| Ok(vec![positive_integer(value, "accumarray subscript")?]))
                    .collect()
            } else {
                let mut out = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let mut subs = Vec::with_capacity(tensor.cols());
                    for col in 0..tensor.cols() {
                        subs.push(positive_integer(
                            tensor.get2(row, col).map_err(grouping_error)?,
                            "accumarray subscript",
                        )?);
                    }
                    out.push(subs);
                }
                Ok(out)
            }
        }
        Value::Cell(cell) => {
            let mut columns = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                let column = accumarray_subscripts(value)?;
                if column.iter().any(|row| row.len() != 1) {
                    return Err(grouping_error(
                        "accumarray: cell subscript entries must be vectors",
                    ));
                }
                columns.push(column.into_iter().map(|row| row[0]).collect::<Vec<_>>());
            }
            let rows = columns.first().map(Vec::len).unwrap_or(0);
            for column in &columns {
                if column.len() != rows {
                    return Err(grouping_error(
                        "accumarray: cell subscript vectors must have equal length",
                    ));
                }
            }
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                out.push(columns.iter().map(|column| column[row]).collect());
            }
            Ok(out)
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported subscript input {other:?}"
        ))),
    }
}

fn accumarray_data_values(data: Value, rows: usize) -> BuiltinResult<Vec<f64>> {
    match data {
        Value::Num(value) => Ok(vec![value; rows]),
        Value::Int(value) => Ok(vec![value.to_f64(); rows]),
        Value::Bool(value) => Ok(vec![if value { 1.0 } else { 0.0 }; rows]),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let values = integer_storage_to_f64_vec(storage);
                if values.len() == 1 {
                    return Ok(vec![values[0]; rows]);
                } else if values.len() == rows {
                    return Ok(values);
                }
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            let len = tensor_utils::tensor_element_len(&tensor);
            if len == 1 {
                Ok(vec![tensor_utils::tensor_value_f64(&tensor, 0); rows])
            } else if len == rows {
                Ok(tensor_utils::tensor_into_values_f64(tensor))
            } else {
                Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ))
            }
        }
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(vec![if array.data[0] != 0 { 1.0 } else { 0.0 }; rows])
            } else if array.data.len() == rows {
                Ok(array
                    .data
                    .into_iter()
                    .map(|flag| if flag != 0 { 1.0 } else { 0.0 })
                    .collect())
            } else {
                Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ))
            }
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn infer_accumarray_shape(index_rows: &[Vec<usize>]) -> Vec<usize> {
    let dims = index_rows.first().map(Vec::len).unwrap_or(1).max(1);
    let mut shape = vec![0usize; dims];
    for row in index_rows {
        for (dim, idx) in row.iter().enumerate() {
            shape[dim] = shape[dim].max(*idx);
        }
    }
    if dims == 1 {
        vec![shape[0], 1]
    } else {
        shape
    }
}

fn subscript_to_linear(subs: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if subs.len() > shape.len() {
        return Err(grouping_error("accumarray: too many subscript dimensions"));
    }
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (dim, &size) in shape.iter().enumerate() {
        let sub = subs.get(dim).copied().unwrap_or(1);
        if sub == 0 || sub > size {
            return Err(grouping_error("accumarray: subscript exceeds output size"));
        }
        linear = linear
            .checked_add(
                (sub - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| too_large_error("accumarray: output linear index overflow"))?,
            )
            .ok_or_else(|| too_large_error("accumarray: output linear index overflow"))?;
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| too_large_error("accumarray: output size overflow"))?;
    }
    Ok(linear)
}

async fn apply_accumarray_callback(func: Value, values: Value) -> BuiltinResult<Value> {
    let result = call_feval_async_with_outputs(func, &[values], 1)
        .await
        .map_err(|err| callback_error("accumarray: callback failed", Some(err)))?;
    match result {
        Value::OutputList(mut values) if values.len() == 1 => Ok(values.remove(0)),
        other => Ok(other),
    }
}

fn accumarray_numeric_output(
    data: Vec<f64>,
    shape: Vec<usize>,
    issparse: bool,
) -> BuiltinResult<Value> {
    if issparse {
        let (rows, cols) = shape_to_rows_cols(&shape)?;
        if shape.len() > 2 {
            return Err(grouping_error(
                "accumarray: sparse output is only supported for 2-D results",
            ));
        }
        let mut col_ptrs = Vec::with_capacity(cols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in 0..cols {
            for row in 0..rows {
                let value = data[row + col * rows];
                if value != 0.0 {
                    row_indices.push(row);
                    values.push(value);
                }
            }
            col_ptrs.push(values.len());
        }
        return SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
            .map(Value::SparseTensor)
            .map_err(grouping_error);
    }
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(grouping_error)
}

enum DiscretizeLabels {
    Text(Vec<String>),
    Numeric(NumericStorage),
}

impl DiscretizeLabels {
    fn len(&self) -> usize {
        match self {
            Self::Text(values) => values.len(),
            Self::Numeric(values) => values.len(),
        }
    }
}

fn discretize_impl(x: Value, edges_or_n: Value, rest: Vec<Value>) -> BuiltinResult<(Value, Value)> {
    let values = numeric_scalars(&x, "discretize X")?;
    let shape = value_shape(&x);
    let (edges, labels, included_right) = parse_discretize_args(&values, edges_or_n, rest)?;
    if edges.len() < 2 {
        return Err(grouping_error(
            "discretize: at least two bin edges are required",
        ));
    }
    let bins = values
        .iter()
        .map(|value| discretize_one_exact(*value, &edges, included_right))
        .collect::<Vec<_>>();
    let edge_output = Tensor::new(
        edges.iter().map(|value| value.materialize_f64()).collect(),
        vec![1, edges.len()],
    )
    .map(Value::Tensor)
    .map_err(grouping_error)?;
    let output = match labels {
        Some(DiscretizeLabels::Text(labels)) => {
            let data = bins
                .iter()
                .map(|bin| match bin {
                    Some(idx) => labels.get(*idx - 1).cloned().unwrap_or_default(),
                    None => String::new(),
                })
                .collect::<Vec<_>>();
            StringArray::new(data, shape)
                .map(Value::StringArray)
                .map_err(grouping_error)?
        }
        Some(DiscretizeLabels::Numeric(labels)) => {
            let mut output = missing_numeric_labels(&labels, bins.len());
            for (position, bin) in bins.iter().enumerate() {
                if let Some(index) = bin {
                    let label = labels.value_at(index - 1).ok_or_else(|| {
                        grouping_error("discretize: replacement value index is out of bounds")
                    })?;
                    output.set_value(position, label).map_err(grouping_error)?;
                }
            }
            Tensor::from_numeric_storage(output, shape)
                .map(Value::Tensor)
                .map_err(grouping_error)?
        }
        None => Tensor::new(
            bins.into_iter()
                .map(|bin| bin.map(|idx| idx as f64).unwrap_or(f64::NAN))
                .collect(),
            shape,
        )
        .map(Value::Tensor)
        .map_err(grouping_error)?,
    };
    Ok((output, edge_output))
}

fn is_discretize_bin_count(value: &Value) -> bool {
    match value {
        Value::Num(value) => is_positive_integer_f64(*value),
        Value::Int(value) => value.try_to_usize().is_some_and(|value| value > 0),
        _ => false,
    }
}

fn parse_discretize_args(
    values: &[NumericScalar],
    edges_or_n: Value,
    rest: Vec<Value>,
) -> BuiltinResult<(Vec<NumericScalar>, Option<DiscretizeLabels>, bool)> {
    let mut labels = None;
    let mut included_right = false;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if !is_option_name(first) {
            labels = Some(discretize_labels(first)?);
            idx = 1;
        }
    }
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(grouping_error(
                "discretize: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&rest[idx], "discretize option")?;
        if name.eq_ignore_ascii_case("IncludedEdge") {
            let edge = scalar_text(&rest[idx + 1], "IncludedEdge")?;
            included_right = match edge.to_ascii_lowercase().as_str() {
                "right" => true,
                "left" => false,
                other => {
                    return Err(grouping_error(format!(
                        "discretize: unsupported IncludedEdge '{other}'"
                    )))
                }
            };
        } else {
            return Err(grouping_error(format!(
                "discretize: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    let edges = match edges_or_n {
        Value::Num(n) if is_positive_integer_f64(n) => equal_width_edges(
            &values
                .iter()
                .map(|value| value.materialize_f64())
                .collect::<Vec<_>>(),
            n as usize,
        )?
        .into_iter()
        .map(NumericScalar::F64)
        .collect(),
        Value::Int(n) => match n.try_to_usize().filter(|bins| *bins > 0) {
            Some(bins) => equal_width_edges(
                &values
                    .iter()
                    .map(|value| value.materialize_f64())
                    .collect::<Vec<_>>(),
                bins,
            )?
            .into_iter()
            .map(NumericScalar::F64)
            .collect(),
            None => numeric_scalars(&Value::Int(n), "discretize edges")?,
        },
        other => numeric_scalars(&other, "discretize edges")?,
    };
    for pair in edges.windows(2) {
        let ordering = compare_numeric(pair[0], pair[1])
            .ok_or_else(|| grouping_error("discretize: bin edges must not contain NaN"))?;
        if ordering == Ordering::Greater {
            return Err(grouping_error(
                "discretize: bin edges must be monotonically increasing",
            ));
        }
    }
    if let Some(labels) = &labels {
        if labels.len() != edges.len().saturating_sub(1) {
            return Err(grouping_error(
                "discretize: number of labels must match number of bins",
            ));
        }
    }
    Ok((edges, labels, included_right))
}

fn discretize_one_exact(
    value: NumericScalar,
    edges: &[NumericScalar],
    included_right: bool,
) -> Option<usize> {
    if compare_numeric(value, value).is_none() {
        return None;
    }
    for bin in 0..edges.len() - 1 {
        let lower = edges[bin];
        let upper = edges[bin + 1];
        let lower_cmp = compare_numeric(value, lower)?;
        let upper_cmp = compare_numeric(value, upper)?;
        let hit = if included_right {
            (lower_cmp == Ordering::Greater || (bin == 0 && lower_cmp == Ordering::Equal))
                && upper_cmp != Ordering::Greater
        } else {
            lower_cmp != Ordering::Less
                && (upper_cmp == Ordering::Less
                    || (bin == edges.len() - 2 && upper_cmp == Ordering::Equal))
        };
        if hit {
            return Some(bin + 1);
        }
    }
    None
}

fn discretize_labels(value: &Value) -> BuiltinResult<DiscretizeLabels> {
    match value {
        Value::Num(value) => Ok(DiscretizeLabels::Numeric(NumericStorage::F64(vec![*value]))),
        Value::Int(value) => homogeneous_integer_values(&[Value::Int(value.clone())])
            .map(NumericStorage::from)
            .map(DiscretizeLabels::Numeric)
            .ok_or_else(|| grouping_error("discretize: invalid integer replacement values")),
        Value::Tensor(tensor) => tensor
            .clone()
            .into_numeric_storage()
            .map(DiscretizeLabels::Numeric)
            .map_err(grouping_error),
        _ => string_list(value).map(DiscretizeLabels::Text),
    }
}

fn missing_numeric_labels(labels: &NumericStorage, len: usize) -> NumericStorage {
    match labels {
        NumericStorage::F64(_) => NumericStorage::F64(vec![f64::NAN; len]),
        NumericStorage::F32(_) => NumericStorage::F32(vec![f32::NAN; len]),
        _ => labels.zeros_like(len),
    }
}

fn numeric_scalars(value: &Value, context: &str) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Bool(value) => Ok(vec![NumericScalar::F64(if *value { 1.0 } else { 0.0 })]),
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| grouping_error(format!("{context}: invalid numeric element")))
            })
            .collect(),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| NumericScalar::F64(if *flag != 0 { 1.0 } else { 0.0 }))
            .collect()),
        Value::SparseTensor(sparse) => numeric_scalars(
            &Value::Tensor(sparse.to_dense().map_err(grouping_error)?),
            context,
        ),
        other => Err(grouping_error(format!(
            "{context}: expected numeric input, got {other:?}"
        ))),
    }
}

fn compare_numeric(left: NumericScalar, right: NumericScalar) -> Option<Ordering> {
    match (numeric_integer(left), numeric_integer(right)) {
        (Some(left), Some(right)) => Some(left.cmp(&right)),
        (Some(left), None) => compare_integer_float(left, right.materialize_f64()),
        (None, Some(right)) => {
            compare_integer_float(right, left.materialize_f64()).map(Ordering::reverse)
        }
        (None, None) => left.materialize_f64().partial_cmp(&right.materialize_f64()),
    }
}

fn numeric_integer(value: NumericScalar) -> Option<i128> {
    value.into_int_value().map(|value| match value {
        IntValue::I8(value) => i128::from(value),
        IntValue::I16(value) => i128::from(value),
        IntValue::I32(value) => i128::from(value),
        IntValue::I64(value) => i128::from(value),
        IntValue::U8(value) => i128::from(value),
        IntValue::U16(value) => i128::from(value),
        IntValue::U32(value) => i128::from(value),
        IntValue::U64(value) => i128::from(value),
    })
}

fn compare_integer_float(integer: i128, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    let truncated = float.trunc() as i128;
    match integer.cmp(&truncated) {
        Ordering::Equal if float.fract() > 0.0 => Some(Ordering::Less),
        Ordering::Equal if float.fract() < 0.0 => Some(Ordering::Greater),
        ordering => Some(ordering),
    }
}

fn equal_width_edges(values: &[f64], bins: usize) -> BuiltinResult<Vec<f64>> {
    if bins == 0 {
        return Err(grouping_error(
            "discretize: number of bins must be positive",
        ));
    }
    if bins >= MAX_MATERIALIZED_ELEMENTS {
        return Err(too_large_error(
            "discretize: requested number of bins is too large",
        ));
    }
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(grouping_error(
            "discretize: cannot infer equal-width bins from all-missing data",
        ));
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        let half = 0.5;
        return Ok((0..=bins)
            .map(|idx| min - half + idx as f64 / bins as f64)
            .collect());
    }
    let step = (max - min) / bins as f64;
    Ok((0..=bins).map(|idx| min + idx as f64 * step).collect())
}

fn combinations_impl(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut values = vec![first];
    values.extend(rest);
    let names = (0..values.len())
        .map(|idx| format!("Var{}", idx + 1))
        .collect::<Vec<_>>();
    let empty_columns = values
        .iter()
        .map(empty_combination_column_like)
        .collect::<BuiltinResult<Vec<_>>>()?;
    let columns = values
        .into_iter()
        .map(vector_elements)
        .collect::<BuiltinResult<Vec<_>>>()?;
    let row_count = columns.iter().try_fold(1usize, |acc, column| {
        acc.checked_mul(column.len())
            .ok_or_else(|| too_large_error("combinations: output row count overflow"))
    })?;
    if row_count > MAX_MATERIALIZED_ELEMENTS {
        return Err(too_large_error("combinations: output is too large"));
    }
    if row_count == 0 {
        return table_from_columns(names, empty_columns);
    }
    let mut out_columns = Vec::with_capacity(columns.len());
    for col_idx in 0..columns.len() {
        let repeat_inner = columns[col_idx + 1..]
            .iter()
            .map(Vec::len)
            .product::<usize>()
            .max(1);
        let repeat_outer = columns[..col_idx]
            .iter()
            .map(Vec::len)
            .product::<usize>()
            .max(1);
        let mut values = Vec::with_capacity(row_count);
        for _ in 0..repeat_outer {
            for item in &columns[col_idx] {
                for _ in 0..repeat_inner {
                    values.push(item.clone());
                }
            }
        }
        out_columns.push(collect_column_values(values, row_count)?);
    }
    table_from_columns(names, out_columns)
}

fn empty_combination_column_like(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => Tensor::from_numeric_storage(
            tensor
                .clone()
                .into_numeric_storage()
                .map_err(grouping_error)?
                .zeros_like(0),
            vec![0, 1],
        )
        .map(Value::Tensor)
        .map_err(grouping_error),
        Value::Int(value) => {
            let sample = [Value::Int(value.clone())];
            let storage = homogeneous_integer_values(&sample)
                .expect("typed integer scalar storage")
                .zeros_like(0);
            Tensor::new_integer(storage, vec![0, 1])
                .map(Value::Tensor)
                .map_err(grouping_error)
        }
        Value::Num(_) => Tensor::new(Vec::new(), vec![0, 1])
            .map(Value::Tensor)
            .map_err(grouping_error),
        Value::Bool(_) | Value::LogicalArray(_) => LogicalArray::new(Vec::new(), vec![0, 1])
            .map(Value::LogicalArray)
            .map_err(grouping_error),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            StringArray::new(Vec::new(), vec![0, 1])
                .map(Value::StringArray)
                .map_err(grouping_error)
        }
        _ => CellArray::new(Vec::new(), 0, 1)
            .map(Value::Cell)
            .map_err(grouping_error),
    }
}

fn parse_name_selector(
    value: &Value,
    names: &[String],
    context: &str,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => {
            if names.contains(text) {
                Ok(vec![text.clone()])
            } else {
                Err(grouping_error(format!(
                    "{context}: unknown variable '{text}'"
                )))
            }
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            parse_name_selector(&Value::String(text), names, context)
        }
        Value::StringArray(array) => array
            .data
            .iter()
            .map(|name| {
                if names.contains(name) {
                    Ok(name.clone())
                } else {
                    Err(grouping_error(format!(
                        "{context}: unknown variable '{name}'"
                    )))
                }
            })
            .collect(),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| scalar_text(value, context))
            .map(|res| {
                let name = res?;
                if names.contains(&name) {
                    Ok(name)
                } else {
                    Err(grouping_error(format!(
                        "{context}: unknown variable '{name}'"
                    )))
                }
            })
            .collect(),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(|value| {
                        let idx = positive_integer_value(value, context)?;
                        names.get(idx - 1).cloned().ok_or_else(|| {
                            grouping_error(format!("{context}: variable index out of range"))
                        })
                    })
                    .collect();
            }
            tensor_utils::tensor_values_f64_cow(tensor)
                .iter()
                .map(|value| {
                    let idx = positive_integer(*value, context)?;
                    names.get(idx - 1).cloned().ok_or_else(|| {
                        grouping_error(format!("{context}: variable index out of range"))
                    })
                })
                .collect()
        }
        other => Err(grouping_error(format!(
            "{context}: unsupported variable selector {other:?}"
        ))),
    }
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed_async(&value).await?);
    }
    Ok(out)
}

fn normalize_outputs(value: Value, requested: usize, context: &str) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == requested => Ok(values),
        Value::OutputList(values) => Err(callback_error(
            format!(
                "{context}: callback returned {} outputs but {} were requested",
                values.len(),
                requested
            ),
            None,
        )),
        value if requested == 1 => Ok(vec![value]),
        _ => Err(callback_error(
            format!("{context}: callback did not return the requested number of outputs"),
            None,
        )),
    }
}

fn collect_group_results(values: Vec<Value>, rows: usize, context: &str) -> BuiltinResult<Value> {
    if values
        .iter()
        .all(|value| value_as_numeric_scalar(value).is_some())
    {
        return Tensor::new(
            values
                .iter()
                .map(|value| value_as_numeric_scalar(value).unwrap())
                .collect(),
            vec![rows, 1],
        )
        .map(Value::Tensor)
        .map_err(grouping_error);
    }
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(|err| callback_error(format!("{context}: {err}"), None))
}

fn collect_column_values(values: Vec<Value>, rows: usize) -> BuiltinResult<Value> {
    if let Some(storage) = homogeneous_integer_values(&values) {
        return Tensor::new_integer(storage, vec![rows, 1])
            .map(Value::Tensor)
            .map_err(grouping_error);
    }
    if values
        .iter()
        .all(|value| value_as_numeric_scalar(value).is_some())
    {
        return Tensor::new(
            values
                .iter()
                .map(|value| value_as_numeric_scalar(value).unwrap())
                .collect(),
            vec![rows, 1],
        )
        .map(Value::Tensor)
        .map_err(grouping_error);
    }
    if values.iter().all(|value| matches!(value, Value::String(_))) {
        return StringArray::new(
            values
                .into_iter()
                .map(|value| match value {
                    Value::String(text) => text,
                    _ => unreachable!("checked above"),
                })
                .collect(),
            vec![rows, 1],
        )
        .map(Value::StringArray)
        .map_err(grouping_error);
    }
    if values.iter().all(|value| matches!(value, Value::Bool(_))) {
        return LogicalArray::new(
            values
                .into_iter()
                .map(|value| match value {
                    Value::Bool(flag) => u8::from(flag),
                    _ => unreachable!("checked above"),
                })
                .collect(),
            vec![rows, 1],
        )
        .map(Value::LogicalArray)
        .map_err(grouping_error);
    }
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(grouping_error)
}

fn vector_elements(value: Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(storage.exact_values().into_iter().map(Value::Int).collect());
            }
            Ok(tensor_utils::tensor_into_values_f64(tensor)
                .into_iter()
                .map(Value::Num)
                .collect())
        }
        Value::LogicalArray(array) => Ok(array
            .data
            .into_iter()
            .map(|flag| Value::Bool(flag != 0))
            .collect()),
        Value::StringArray(array) => Ok(array.data.into_iter().map(Value::String).collect()),
        Value::Cell(cell) => Ok(cell.data),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars
            .data
            .into_iter()
            .map(|ch| Value::String(ch.to_string()))
            .collect()),
        Value::String(text) => Ok(vec![Value::String(text)]),
        other => Ok(vec![other]),
    }
}

fn homogeneous_integer_values(values: &[Value]) -> Option<IntegerStorage> {
    let first = match values.first()? {
        Value::Int(IntValue::I8(_)) => IntegerStorage::I8(Vec::new()),
        Value::Int(IntValue::I16(_)) => IntegerStorage::I16(Vec::new()),
        Value::Int(IntValue::I32(_)) => IntegerStorage::I32(Vec::new()),
        Value::Int(IntValue::I64(_)) => IntegerStorage::I64(Vec::new()),
        Value::Int(IntValue::U8(_)) => IntegerStorage::U8(Vec::new()),
        Value::Int(IntValue::U16(_)) => IntegerStorage::U16(Vec::new()),
        Value::Int(IntValue::U32(_)) => IntegerStorage::U32(Vec::new()),
        Value::Int(IntValue::U64(_)) => IntegerStorage::U64(Vec::new()),
        _ => return None,
    };
    let mut exact = Vec::with_capacity(values.len());
    for value in values {
        let Value::Int(value) = value else {
            return None;
        };
        exact.push(value.clone());
    }
    first.from_exact_values_like(exact).ok()
}

fn multi_output(outputs: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(outputs
        .into_iter()
        .next()
        .unwrap_or(Value::OutputList(Vec::new())))
}

fn numeric_values(value: &Value, context: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(tensor)),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| if *flag != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map(tensor_utils::tensor_into_values_f64)
            .map_err(grouping_error),
        other => Err(grouping_error(format!(
            "{context}: expected numeric input, got {other:?}"
        ))),
    }
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::LogicalArray(array) => array.shape.clone(),
        Value::StringArray(array) => array.shape.clone(),
        Value::SparseTensor(sparse) => sparse.shape(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => vec![1, 1],
        _ => vec![1, 1],
    }
}

fn parse_positive_size_vector(value: &Value, context: &str) -> BuiltinResult<Vec<usize>> {
    let dims = match value {
        Value::Int(value) => vec![positive_integer_value(value, context)?],
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| positive_integer_value(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            } else {
                numeric_values(value, context)?
                    .into_iter()
                    .map(|value| positive_integer(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            }
        }
        _ => numeric_values(value, context)?
            .into_iter()
            .map(|value| positive_integer(value, context))
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    if dims.is_empty() {
        return Err(grouping_error(format!(
            "{context}: size vector must not be empty"
        )));
    }
    Ok(if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims
    })
}

fn shape_to_rows_cols(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    let rows = shape.first().copied().unwrap_or(0);
    let cols = if shape.len() <= 1 {
        1
    } else if shape.len() == 2 {
        shape[1]
    } else {
        shape[1..].iter().product()
    };
    Ok((rows, cols))
}

fn checked_element_count(shape: &[usize], context: &str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| too_large_error(format!("{context}: output size overflow")))
    })
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(grouping_error(format!(
            "{context}: expected text scalar, got {other:?}"
        ))),
    }
}

fn string_list(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(chars) if chars.rows == 1 => Ok(vec![chars.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| scalar_text(value, "string list"))
            .collect(),
        other => Err(grouping_error(format!("expected text list, got {other:?}"))),
    }
}

fn binary_bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.to_f64() == 1.0 => Ok(true),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(grouping_error(format!(
            "{context}: expected logical or numeric 0 or 1, got {other:?}"
        ))),
    }
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    value_as_numeric_scalar(value)
        .ok_or_else(|| grouping_error(format!("{context}: expected numeric scalar")))
}

fn value_as_numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Some(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) == 0,
        Value::StringArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::CharArray(chars) => chars.data.is_empty(),
        _ => false,
    }
}

fn is_option_name(value: &Value) -> bool {
    scalar_text(value, "option")
        .map(|text| {
            matches!(
                text.to_ascii_lowercase().as_str(),
                "includededge" | "variablenames" | "includemissinggroups" | "includeemptygroups"
            )
        })
        .unwrap_or(false)
}

fn positive_integer(value: f64, context: &str) -> BuiltinResult<usize> {
    if let Some(value) = positive_platform_usize(value) {
        Ok(value)
    } else {
        Err(grouping_error(format!(
            "{context}: expected positive integer"
        )))
    }
}

fn positive_integer_value(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value > 0)
        .ok_or_else(|| grouping_error(format!("{context}: expected positive integer")))
}

#[cfg(test)]
fn nonnegative_integer(value: f64, context: &str) -> BuiltinResult<usize> {
    if let Some(value) = nonnegative_platform_usize(value) {
        Ok(value)
    } else {
        Err(grouping_error(format!(
            "{context}: expected nonnegative integer"
        )))
    }
}

fn integer_storage_to_f64_vec(storage: &IntegerStorage) -> Vec<f64> {
    storage
        .exact_values()
        .iter()
        .map(IntValue::to_f64)
        .collect()
}

fn is_positive_integer_f64(value: f64) -> bool {
    positive_platform_usize(value).is_some()
}

fn positive_platform_usize(value: f64) -> Option<usize> {
    nonnegative_platform_usize(value).filter(|value| *value > 0)
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn format_key_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        let mut text = format!("{value:.12}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

fn is_missing_text(text: &str) -> bool {
    text.eq_ignore_ascii_case("<missing>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, NumericStorage};

    #[test]
    fn discretize_descriptor_covers_public_forms_and_output_arity() {
        assert_eq!(
            DISCRETIZE_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(
            DISCRETIZE_SIGNATURES
                .iter()
                .map(|signature| signature.label)
                .collect::<Vec<_>>(),
            vec![
                "Y = discretize(X, edges)",
                "Y = discretize(X, N)",
                "Y = discretize(___, values)",
                "Y = discretize(___, \"IncludedEdge\", side)",
                "[Y, E] = discretize(X, N, ___)",
            ]
        );
        assert!(DISCRETIZE_SIGNATURES[..4]
            .iter()
            .all(|signature| signature.outputs.len() == 1));
        assert_eq!(DISCRETIZE_SIGNATURES[4].outputs.len(), 2);
        assert_eq!(DISCRETIZE_DESCRIPTOR.errors.len(), 2);
        assert!(DISCRETIZE_DESCRIPTOR
            .errors
            .iter()
            .all(|error| error.code != ERROR_CALLBACK.code));
    }

    #[test]
    fn discretize_typed_bin_count_preserves_exact_unsigned_values() {
        let values = [NumericScalar::F64(0.0), NumericScalar::F64(1.0)];
        let (edges, _, _) =
            parse_discretize_args(&values, Value::Int(IntValue::U16(2)), Vec::new()).unwrap();
        assert_eq!(
            edges,
            vec![
                NumericScalar::F64(0.0),
                NumericScalar::F64(0.5),
                NumericScalar::F64(1.0)
            ]
        );

        let (edges, _, _) =
            parse_discretize_args(&values, Value::Int(IntValue::U8(1)), Vec::new()).unwrap();
        assert_eq!(
            edges,
            vec![NumericScalar::F64(0.0), NumericScalar::F64(1.0)]
        );
    }

    #[test]
    fn discretize_explicit_integer_edges_do_not_use_f64_mirror() {
        let base = 9_007_199_254_740_992_u64;
        let out = block_on(discretize_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![base + 1]), vec![1, 1]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![base, base + 1, base + 2]),
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn discretize_integer_replacement_values_preserve_class_and_zero_missing() {
        let out = block_on(discretize_builtin(
            Value::Tensor(Tensor::new(vec![-1.0, 0.5, 1.5, 3.0], vec![1, 4]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            vec![Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 7]), vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![0, u64::MAX, 7, 0]))
            ),
            other => panic!("expected integer tensor, got {other:?}"),
        }
    }

    #[test]
    fn discretize_scalar_bin_count_returns_computed_edges_as_second_output() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let out = block_on(discretize_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Int(IntValue::U8(2)),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert!(
                    matches!(&values[0], Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 2.0])
                );
                assert!(
                    matches!(&values[1], Value::Tensor(tensor) if tensor.materialize_f64() == vec![0.0, 0.5, 1.0])
                );
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn discretize_bins_infinities_when_outer_edges_are_infinite() {
        let input = Value::Tensor(
            Tensor::new(
                vec![f64::NEG_INFINITY, -1.0, 1.0, f64::INFINITY, f64::NAN],
                vec![1, 5],
            )
            .unwrap(),
        );
        let edges = Value::Tensor(
            Tensor::new(vec![f64::NEG_INFINITY, 0.0, f64::INFINITY], vec![1, 3]).unwrap(),
        );

        for options in [
            Vec::new(),
            vec![Value::from("IncludedEdge"), Value::from("right")],
        ] {
            let output = block_on(discretize_builtin(input.clone(), edges.clone(), options))
                .expect("infinite values with infinite outer edges");
            let Value::Tensor(output) = output else {
                panic!("expected tensor")
            };
            let values = output.materialize_f64();
            assert_eq!(values[..4], [1.0, 1.0, 2.0, 2.0]);
            assert!(values[4].is_nan());
        }
    }

    #[test]
    fn grouping_dimension_parsers_reject_fractional_and_out_of_range_doubles() {
        assert!(positive_integer(1.5, "test").is_err());
        assert!(nonnegative_integer(1.5, "test").is_err());
        assert!(positive_integer(usize::MAX as f64 + 1.0, "test").is_err());
        assert!(nonnegative_integer(usize::MAX as f64 + 1.0, "test").is_err());
    }

    #[test]
    fn accumarray_sums_vector_and_matrix_subscripts() {
        let out = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 4.0, 1.0], vec![6, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![7.0, 4.0, 2.0, 8.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let out = block_on(accumarray_builtin(
            Value::Tensor(
                Tensor::new(
                    vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
                    vec![6, 2],
                )
                .unwrap(),
            ),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 2]);
                assert_eq!(
                    tensor.materialize_f64(),
                    vec![5.0, 0.0, 0.0, 6.0, 0.0, 7.0, 3.0, 0.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn accumarray_integer_metadata_covers_structural_default_callback_and_sparse_forms() {
        assert_eq!(ACCUMARRAY_DESCRIPTOR.signatures.len(), 5);
        assert_eq!(ACCUMARRAY_DESCRIPTOR.output_mode, BuiltinOutputMode::Fixed);
        assert_eq!(ACCUMARRAY_INTEGER_CAPABILITIES.len(), 4);
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES
            .iter()
            .any(|capability| capability.computation_domain
                == BuiltinIntegerComputationDomain::Structural));
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES
            .iter()
            .any(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES
            .iter()
            .any(|capability| capability
                .inputs
                .iter()
                .any(|input| input.availability == BuiltinIntegerInputAvailability::Rejected)));
    }

    #[test]
    fn accumarray_accepts_exact_integer_subscripts_data_and_size_vectors() {
        let subs_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 3, 4, 2]), vec![4, 1]).unwrap();
        let subs = Value::Tensor(subs_tensor);
        let data_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![10, 20, 30, 40]), vec![4, 1]).unwrap();
        let data = Value::Tensor(data_tensor);
        let size_tensor = Tensor::new_integer(IntegerStorage::U8(vec![4, 1]), vec![1, 2]).unwrap();
        let size = Value::Tensor(size_tensor);

        let out = block_on(accumarray_builtin(subs, data, vec![size])).unwrap();

        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 1]);
                assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0, 20.0, 30.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn grouping_numeric_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1]).unwrap();

        assert_eq!(
            value_as_numeric_scalar(&Value::Tensor(tensor)),
            Some(2026.0)
        );
    }

    #[test]
    fn grouping_name_selector_reads_typed_integer_storage_exactly() {
        let selector = Tensor::new_integer(IntegerStorage::U16(vec![2, 1]), vec![1, 2]).unwrap();

        let selected = parse_name_selector(
            &Value::Tensor(selector),
            &["alpha".into(), "beta".into()],
            "test selector",
        )
        .unwrap();

        assert_eq!(selected, vec!["beta", "alpha"]);
    }

    #[test]
    fn accumarray_empty_subscripts_read_typed_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();

        assert_eq!(
            accumarray_subscripts(Value::Tensor(tensor)).unwrap(),
            Vec::<Vec<usize>>::new()
        );
    }

    #[test]
    fn accumarray_data_values_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).unwrap();

        assert_eq!(
            accumarray_data_values(Value::Tensor(tensor), 3).unwrap(),
            vec![7.0; 3]
        );
    }

    #[test]
    fn accumarray_rejects_negative_exact_integer_subscripts_and_sizes() {
        let subs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, -1]), vec![2, 1]).unwrap(),
        );
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let err = block_on(accumarray_builtin(subs, data, Vec::new()))
            .expect_err("negative subscript should fail");
        assert!(err.message.contains("expected positive integer"));

        let subs =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap());
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let size = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).unwrap(),
        );
        let err = block_on(accumarray_builtin(subs, data, vec![size]))
            .expect_err("negative size should fail");
        assert!(err.message.contains("expected positive integer"));
    }

    fn all_integer_triplets() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![4, 2, 9]),
            IntegerStorage::I16(vec![4, 2, 9]),
            IntegerStorage::I32(vec![4, 2, 9]),
            IntegerStorage::I64(vec![4, 2, 9]),
            IntegerStorage::U8(vec![4, 2, 9]),
            IntegerStorage::U16(vec![4, 2, 9]),
            IntegerStorage::U32(vec![4, 2, 9]),
            IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1, 9]),
        ]
    }

    #[test]
    fn accumarray_default_sum_accepts_all_integer_data_classes_and_returns_double() {
        for storage in all_integer_triplets() {
            let data = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let out = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(data),
                Vec::new(),
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense numeric output");
            };
            assert!(out.integer_storage().is_none());
            assert_eq!(out.shape, vec![2, 1]);
        }
    }

    #[test]
    fn accumarray_default_integer_sum_rejects_typed_integer_fill() {
        let err = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![4, 9]), vec![2, 1]).unwrap(),
            ),
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                ),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Int(IntValue::I16(0)),
            ],
        ))
        .expect_err("default integer sum requires double fill");
        assert!(err.message.contains("double default sum output"));
    }

    #[test]
    fn accumarray_custom_callback_preserves_native_single_result_class() {
        let data =
            Tensor::new_with_dtype(vec![4.0, 2.0, 9.0], vec![3, 1], NumericDType::F32).unwrap();
        let out = block_on(accumarray_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
            ),
            Value::Tensor(data),
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                ),
                Value::FunctionHandle("min".into()),
            ],
        ))
        .unwrap();
        let Value::Tensor(out) = out else {
            panic!("expected native-single tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.materialize_f64(), vec![2.0, 9.0, 0.0]);
    }

    #[test]
    fn accumarray_custom_min_preserves_all_integer_classes_and_exact_fill() {
        for storage in all_integer_triplets() {
            let second = storage.value_at(1).unwrap();
            let third = storage.value_at(2).unwrap();
            let fill = storage.zeros_like(1).value_at(0).unwrap();
            let expected = storage
                .from_exact_values_like(vec![second, third, fill.clone()])
                .unwrap();
            let out = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).unwrap()),
                vec![
                    Value::Tensor(
                        Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                    ),
                    Value::FunctionHandle("min".into()),
                    Value::Int(fill),
                ],
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected typed integer output");
            };
            assert_eq!(out.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn accumarray_accepts_all_integer_subscript_and_size_classes_exactly() {
        let subs = vec![
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        let sizes = vec![
            IntegerStorage::I8(vec![3, 1]),
            IntegerStorage::I16(vec![3, 1]),
            IntegerStorage::I32(vec![3, 1]),
            IntegerStorage::I64(vec![3, 1]),
            IntegerStorage::U8(vec![3, 1]),
            IntegerStorage::U16(vec![3, 1]),
            IntegerStorage::U32(vec![3, 1]),
            IntegerStorage::U64(vec![3, 1]),
        ];
        for (subs, size) in subs.into_iter().zip(sizes) {
            let out = block_on(accumarray_builtin(
                Value::Tensor(Tensor::new_integer(subs, vec![2, 1]).unwrap()),
                Value::Num(1.0),
                vec![Value::Tensor(
                    Tensor::new_integer(size, vec![1, 2]).unwrap(),
                )],
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense output");
            };
            assert_eq!(out.shape, vec![3, 1]);
            assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn accumarray_accepts_all_integer_scalar_subscript_classes() {
        for storage in all_integer_triplets() {
            let subscript = storage
                .from_exact_values_like(vec![storage.cast_f64_assignment(1.0)])
                .unwrap()
                .value_at(0)
                .unwrap();
            let out = block_on(accumarray_builtin(
                Value::Int(subscript),
                Value::Int(storage.value_at(0).unwrap()),
                Vec::new(),
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense output");
            };
            assert_eq!(out.shape, vec![1, 1]);
            assert!(out.integer_storage().is_none());
        }
    }

    #[test]
    fn accumarray_sparse_rejects_all_integer_data_classes_and_nonbinary_controls() {
        let empty = || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        for storage in all_integer_triplets() {
            let err = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).unwrap()),
                vec![empty(), empty(), empty(), Value::Bool(true)],
            ))
            .expect_err("sparse integer data must reject");
            assert!(err.message.contains("requires double input data"));
        }

        for storage in all_integer_triplets() {
            let control = storage
                .from_exact_values_like(vec![storage.cast_f64_assignment(2.0)])
                .unwrap();
            let err = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap(),
                ),
                Value::Num(1.0),
                vec![
                    empty(),
                    empty(),
                    empty(),
                    Value::Int(control.value_at(0).unwrap()),
                ],
            ))
            .expect_err("issparse must be binary");
            assert!(err.message.contains("expected logical or numeric 0 or 1"));
        }

        let err = block_on(accumarray_builtin(
            Value::Num(1.0),
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32).unwrap(),
            ),
            vec![empty(), empty(), empty(), Value::Bool(true)],
        ))
        .expect_err("sparse single data must reject");
        assert!(err.message.contains("requires double input data"));
    }

    #[test]
    fn accumarray_rejects_resident_integer_data_before_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let error = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap(),
                ),
                Value::GpuTensor(handle.clone()),
                Vec::new(),
            ))
            .expect_err("resident integer data must reject");
            assert!(error.message.contains("GPU input data must be"));
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn discretize_assigns_bins_and_labels() {
        let out = block_on(discretize_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 0.2, 1.0, 2.5], vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.materialize_f64()[0], 1.0);
                assert_eq!(tensor.materialize_f64()[1], 1.0);
                assert_eq!(tensor.materialize_f64()[2], 2.0);
                assert!(tensor.materialize_f64()[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn findgroups_groupcounts_and_grp2idx_share_order() {
        let groups = Value::StringArray(
            StringArray::new(
                vec!["b".into(), "a".into(), "b".into(), "<missing>".into()],
                vec![4, 1],
            )
            .unwrap(),
        );
        let out = block_on(findgroups_builtin(groups.clone(), Vec::new())).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.materialize_f64()[0], 2.0);
                assert_eq!(tensor.materialize_f64()[1], 1.0);
                assert_eq!(tensor.materialize_f64()[2], 2.0);
                assert!(tensor.materialize_f64()[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let counted = block_on(groupcounts_builtin(groups.clone(), Vec::new())).unwrap();
        match counted {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let indexed = block_on(grp2idx_builtin(groups)).unwrap();
        match indexed {
            Value::Tensor(tensor) => assert!(tensor.materialize_f64()[3].is_nan()),
            other => panic!("expected tensor, got {other:?}"),
        }

        let empty_is_group = block_on(groupcounts_builtin(
            Value::StringArray(
                StringArray::new(vec![String::new(), "a".into(), String::new()], vec![3, 1])
                    .unwrap(),
            ),
            Vec::new(),
        ))
        .unwrap();
        match empty_is_group {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn grp2idx_preserves_exact_integer_levels_and_returns_cellstr_names() {
        let large = 9_007_199_254_740_992_u64;
        let groups = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![large + 1, large, large + 1]),
                vec![1, 3],
            )
            .unwrap(),
        );
        let _outputs = crate::output_count::push_output_count(Some(3));
        let Value::OutputList(outputs) = block_on(grp2idx_builtin(groups)).unwrap() else {
            panic!("expected three grp2idx outputs");
        };
        assert_eq!(outputs.len(), 3);
        let Value::Tensor(g) = &outputs[0] else {
            panic!("expected double group indices");
        };
        assert_eq!(g.materialize_f64(), vec![2.0, 1.0, 2.0]);
        let Value::Cell(names) = &outputs[1] else {
            panic!("expected cellstr group names");
        };
        assert_eq!(names.rows, 2);
        assert!(names
            .data
            .iter()
            .all(|value| matches!(value, Value::CharArray(chars) if chars.rows == 1)));
        let Value::Tensor(levels) = &outputs[2] else {
            panic!("expected typed integer levels");
        };
        assert_eq!(
            levels.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, large + 1]))
        );
        assert_eq!(levels.shape, vec![2, 1]);
    }

    #[test]
    fn grp2idx_resident_wide_integer_restores_numeric_outputs_without_aliasing() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let large = 9_007_199_254_740_992_u64;
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![large + 1, large, large + 1]),
                vec![3, 1],
            )
            .unwrap();
            let prototype = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
                .expect("upload exact grouping input");
            let _outputs = crate::output_count::push_output_count(Some(3));
            let Value::OutputList(outputs) =
                block_on(grp2idx_builtin(Value::GpuTensor(prototype.clone()))).unwrap()
            else {
                panic!("expected three grp2idx outputs");
            };
            let Value::GpuTensor(g) = &outputs[0] else {
                panic!("expected resident double indices");
            };
            let Value::Cell(names) = &outputs[1] else {
                panic!("expected host cellstr names");
            };
            let Value::GpuTensor(levels) = &outputs[2] else {
                panic!("expected resident integer levels");
            };
            assert!(!same_grp2idx_handle(g, &prototype));
            assert!(!same_grp2idx_handle(levels, &prototype));
            assert!(!same_grp2idx_handle(g, levels));
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(g).expect("g owner"),
                provider
            ));
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(levels).expect("gL owner"),
                provider
            ));
            assert_eq!(names.rows, 2);
            let gathered_g = crate::builtins::common::test_support::gather(outputs[0].clone())
                .expect("gather indices");
            assert_eq!(gathered_g.materialize_f64(), vec![2.0, 1.0, 2.0]);
            let gathered_levels = crate::builtins::common::test_support::gather(outputs[2].clone())
                .expect("gather levels");
            assert_eq!(
                gathered_levels.integer_storage(),
                Some(&IntegerStorage::U64(vec![large, large + 1]))
            );
        });
    }

    #[test]
    fn groupcounts_sorts_missing_last_and_requires_binary_controls() {
        let groups = Value::Tensor(Tensor::new(vec![f64::NAN, 1.0, f64::NAN], vec![3, 1]).unwrap());
        let counted = {
            let _outputs = crate::output_count::push_output_count(Some(3));
            block_on(groupcounts_builtin(groups.clone(), Vec::new())).unwrap()
        };
        let Value::OutputList(counted) = counted else {
            panic!("expected groupcounts outputs");
        };
        let Value::Tensor(counts) = &counted[0] else {
            panic!("expected counts");
        };
        assert_eq!(counts.materialize_f64(), vec![1.0, 2.0]);
        let Value::Tensor(labels) = &counted[1] else {
            panic!("expected labels");
        };
        assert_eq!(labels.materialize_f64()[0], 1.0);
        assert!(labels.materialize_f64()[1].is_nan());

        let excluded = block_on(groupcounts_builtin(
            groups,
            vec![
                Value::from("IncludeMissingGroups"),
                Value::Int(IntValue::U8(0)),
            ],
        ))
        .unwrap();
        let Value::Tensor(excluded) = excluded else {
            panic!("expected counts");
        };
        assert_eq!(excluded.materialize_f64(), vec![1.0]);

        let error = block_on(groupcounts_builtin(
            Value::Num(1.0),
            vec![
                Value::from("IncludeMissingGroups"),
                Value::Int(IntValue::U8(2)),
            ],
        ))
        .expect_err("nonbinary control must reject");
        assert!(error.message.contains("expected logical or numeric 0 or 1"));
    }

    #[test]
    fn groupcounts_strict_mode_gates_resident_but_not_documented_groupbins_syntax() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let resident_error = block_on(groupcounts_builtin(resident, Vec::new()))
            .expect_err("resident form must gate before provider access");
        assert_eq!(
            resident_error.identifier(),
            GROUPCOUNTS_RESIDENT_INPUT_EXTENSION.error_identifier
        );

        let groupbins_error = block_on(groupcounts_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap(),
            )],
        ))
        .expect_err("unimplemented groupbins must reject ordinarily");
        assert_eq!(groupbins_error.identifier(), ERROR_INVALID_INPUT.identifier);
        assert!(groupbins_error.message.contains("groupbins"));
    }

    #[test]
    fn groupcounts_multiple_grouping_vectors_return_one_bg_cell_and_bp_third() {
        let first = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9, 9, 7]), vec![3, 1]).unwrap(),
        );
        let second = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![1, 2, 1]), vec![3, 1]).unwrap(),
        );
        let groups = Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap());
        let _outputs = crate::output_count::push_output_count(Some(3));
        let Value::OutputList(outputs) = block_on(groupcounts_builtin(groups, Vec::new())).unwrap()
        else {
            panic!("expected three outputs");
        };
        assert_eq!(outputs.len(), 3);
        let Value::Cell(bg) = &outputs[1] else {
            panic!("BG must be one cell array");
        };
        assert_eq!((bg.rows, bg.cols), (1, 2));
        assert!(matches!(&bg.data[0], Value::Tensor(value) if value.integer_storage().is_some()));
        assert!(matches!(&bg.data[1], Value::Tensor(value) if value.integer_storage().is_some()));
        let Value::Tensor(bp) = &outputs[2] else {
            panic!("BP must be the third output");
        };
        assert_eq!(bp.len(), 3);
    }

    #[test]
    fn grouping_uses_exact_integer_keys_and_preserves_group_labels() {
        let large = 9_007_199_254_740_992_u64;
        let groups = Value::Tensor(
            Tensor::new_integer(
                runmat_builtins::IntegerStorage::U64(vec![large, large + 1, large]),
                vec![3, 1],
            )
            .unwrap(),
        );

        let grouped = block_on(findgroups_builtin(groups.clone(), Vec::new())).unwrap();
        match grouped {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let counted = block_on(groupcounts_builtin(groups, Vec::new())).unwrap();
        match counted {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn findgroups_preserves_row_vector_group_orientation() {
        let groups = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![7, 3, 7]), vec![1, 3]).unwrap(),
        );
        let Value::Tensor(g) = block_on(findgroups_builtin(groups, Vec::new())).unwrap() else {
            panic!("expected group tensor");
        };
        assert_eq!(g.shape, vec![1, 3]);
        assert_eq!(g.materialize_f64(), vec![2.0, 1.0, 2.0]);
    }

    #[test]
    fn findgroups_identifier_output_preserves_exact_integer_storage() {
        let large = 9_007_199_254_740_992_u64;
        let columns = columns_from_group_args(vec![Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![large + 1, large, large + 1]),
                vec![1, 3],
            )
            .unwrap(),
        )])
        .unwrap();
        let grouping = build_grouping(&columns).unwrap();
        let outputs = findgroups_outputs(&columns, &grouping, false, vec![1, 3]).unwrap();
        let Value::Tensor(ids) = &outputs[1] else {
            panic!("expected typed integer identifiers");
        };
        assert_eq!(
            ids.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, large + 1]))
        );
    }

    #[test]
    fn findgroups_empty_character_vectors_are_missing() {
        let groups = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(runmat_builtins::CharArray::new(Vec::new(), 1, 0).unwrap()),
                    Value::CharArray(runmat_builtins::CharArray::new(vec!['a'], 1, 1).unwrap()),
                ],
                2,
                1,
            )
            .unwrap(),
        );
        let Value::Tensor(g) = block_on(findgroups_builtin(groups, Vec::new())).unwrap() else {
            panic!("expected group tensor");
        };
        assert!(g.materialize_f64()[0].is_nan());
        assert_eq!(g.materialize_f64()[1], 1.0);
    }

    #[test]
    fn findgroups_strict_mode_gates_matrix_selector_and_timetable_forms() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let err =
            block_on(findgroups_builtin(matrix, Vec::new())).expect_err("matrix form must gate");
        assert_eq!(
            err.identifier(),
            FINDGROUPS_MATRIX_COLUMNS_EXTENSION.error_identifier
        );

        let table = Value::Object(ObjectInstance::new("table".to_string()));
        let err = ensure_findgroups_extensions(&table, &[Value::from("A")])
            .expect_err("table selector must gate");
        assert_eq!(
            err.identifier(),
            FINDGROUPS_TABLE_SELECTOR_EXTENSION.error_identifier
        );

        let timetable = Value::Object(ObjectInstance::new("timetable".to_string()));
        let err = ensure_findgroups_extensions(&timetable, &[]).expect_err("timetable must gate");
        assert_eq!(
            err.identifier(),
            FINDGROUPS_TIMETABLE_EXTENSION.error_identifier
        );
    }

    #[test]
    fn findgroups_strict_mode_gates_resident_input_before_provider_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = block_on(findgroups_builtin(resident, Vec::new()))
            .expect_err("resident form must gate before provider lookup");
        assert_eq!(
            err.identifier(),
            FINDGROUPS_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn findgroups_rejects_sparse_and_complex_grouping_values() {
        let sparse =
            runmat_builtins::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        block_on(findgroups_builtin(Value::SparseTensor(sparse), Vec::new()))
            .expect_err("sparse grouping must reject");
        block_on(findgroups_builtin(Value::Complex(1.0, 2.0), Vec::new()))
            .expect_err("complex grouping must reject");
    }

    #[test]
    fn findgroups_integer_metadata_covers_vector_multi_and_table_roles() {
        assert_eq!(FINDGROUPS_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(FINDGROUPS_EXTENSIONS.len(), 4);
        for capability in FINDGROUPS_INTEGER_CAPABILITIES {
            assert_eq!(
                capability.inputs[0].classes,
                &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
            );
            assert_eq!(
                capability.computation_domain,
                BuiltinIntegerComputationDomain::ExactInteger
            );
        }
    }

    #[test]
    fn grouping_columns_preserve_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 10.0, 20.0], vec![2, 2]).unwrap();
        let columns = tensor_columns("G", tensor, true).unwrap();
        assert_eq!(columns.len(), 2);
        for (column, expected) in columns
            .iter()
            .zip([vec![1.0_f32, 2.0], vec![10.0_f32, 20.0]])
        {
            let Value::Tensor(tensor) = &column.value else {
                panic!("expected numeric group column");
            };
            assert_eq!(
                tensor.clone().into_numeric_storage().unwrap(),
                NumericStorage::F32(expected)
            );
        }
    }

    #[test]
    fn combinations_returns_table_columns() {
        let out = block_on(combinations_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            vec![Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into()], vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();
        let Value::Object(object) = out else {
            panic!("expected table");
        };
        assert!(is_tabular_object(&object));
        assert_eq!(table_height(&object).unwrap(), 4);
    }

    #[test]
    fn combinations_preserves_typed_integer_columns_without_f64_mirror() {
        let first =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 9]), vec![1, 2]).unwrap();
        let out = block_on(combinations_builtin(
            Value::Tensor(first),
            vec![Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into()], vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();

        let Value::Object(object) = out else {
            panic!("expected table");
        };
        let variables = table_variables(&object).unwrap();
        let first_column = variables
            .fields
            .values()
            .next()
            .expect("first table variable");

        match first_column {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9, 9]))
            ),
            other => panic!("expected typed integer tensor column, got {other:?}"),
        }
    }

    #[test]
    fn combinations_preserves_all_integer_classes_and_linearizes_input_shape() {
        for storage in all_integer_triplets() {
            let expected = storage.clone();
            let input = Tensor::new_integer(storage, vec![1, 3]).unwrap();
            let out = block_on(combinations_builtin(Value::Tensor(input), Vec::new())).unwrap();
            let Value::Object(table) = out else {
                panic!("expected table");
            };
            assert_eq!(table_height(&table).unwrap(), 3);
            let variables = table_variables(&table).unwrap();
            let Value::Tensor(column) = variables.fields.values().next().unwrap() else {
                panic!("expected numeric table variable");
            };
            assert_eq!(column.shape, vec![3, 1]);
            assert_eq!(column.integer_storage(), Some(&expected));
        }
        assert_eq!(COMBINATIONS_INTEGER_INPUTS[0].classes.len(), 8);
    }

    #[test]
    fn combinations_preserves_empty_integer_class() {
        let empty = Tensor::new_integer(IntegerStorage::I32(Vec::new()), vec![0, 1]).unwrap();
        let out = block_on(combinations_builtin(Value::Tensor(empty), Vec::new())).unwrap();
        let Value::Object(table) = out else {
            panic!("expected table");
        };
        assert_eq!(table_height(&table).unwrap(), 0);
        let variables = table_variables(&table).unwrap();
        let Value::Tensor(column) = variables.fields.values().next().unwrap() else {
            panic!("expected numeric table variable");
        };
        assert_eq!(
            column.integer_storage(),
            Some(&IntegerStorage::I32(Vec::new()))
        );
    }

    #[test]
    fn combinations_empty_cartesian_product_keeps_every_integer_column_empty_and_typed() {
        for nonempty in all_integer_triplets() {
            let empty_storage = nonempty.zeros_like(0);
            let expected_empty = empty_storage.clone();
            let expected_other = nonempty.zeros_like(0);
            let out = block_on(combinations_builtin(
                Value::Tensor(Tensor::new_integer(empty_storage, vec![0, 1]).unwrap()),
                vec![Value::Tensor(
                    Tensor::new_integer(nonempty, vec![1, 3]).unwrap(),
                )],
            ))
            .unwrap();
            let Value::Object(table) = out else {
                panic!("expected table");
            };
            assert_eq!(table_height(&table).unwrap(), 0);
            let variables = table_variables(&table).unwrap();
            for (name, expected) in [("Var1", expected_empty), ("Var2", expected_other)] {
                let Value::Tensor(column) = variables.fields.get(name).unwrap() else {
                    panic!("expected typed numeric column");
                };
                assert_eq!(column.shape, vec![0, 1]);
                assert_eq!(column.integer_storage(), Some(&expected));
            }
        }
    }

    #[test]
    fn combinations_strict_mode_gates_resident_extension_before_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 99,
            buffer_id: 77,
        });
        let error = block_on(combinations_builtin(resident, Vec::new())).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CombinationsResidentInputExtension")
        );
    }

    #[test]
    fn combinations_treats_variable_names_text_as_data() {
        let out = block_on(combinations_builtin(
            Value::String("VariableNames".into()),
            vec![Value::StringArray(
                StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();
        let Value::Object(table) = out else {
            panic!("expected table");
        };
        assert_eq!(table_height(&table).unwrap(), 2);
    }

    #[test]
    fn accumarray_supports_callbacks_and_sparse_output() {
        let out = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4, 1]).unwrap()),
            vec![
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::FunctionHandle("mean".into()),
            ],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 6.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let sparse = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            Value::Num(1.0),
            vec![
                Value::Tensor(Tensor::new(vec![4.0, 1.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Bool(true),
            ],
        ))
        .unwrap();
        match sparse {
            Value::SparseTensor(st) => {
                assert_eq!(st.rows, 4);
                assert_eq!(st.cols, 1);
                assert_eq!(st.nnz(), 2);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn splitapply_invokes_callback_by_group() {
        let out = block_on(splitapply_builtin(
            Value::FunctionHandle("sum".into()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![2.0, 1.0, 2.0, 1.0], vec![4, 1]).unwrap(),
            )],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![6.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let err = block_on(splitapply_builtin(
            Value::FunctionHandle("sum".into()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap(),
            )],
        ))
        .unwrap_err();
        assert!(err.message().contains("must have 2 rows"));
    }

    #[test]
    fn groupcounts_table_returns_count_and_percent_columns() {
        let table = table_from_columns(
            vec!["G".into(), "X".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
        )
        .unwrap();
        let out = block_on(groupcounts_builtin(table, vec![Value::from("G")])).unwrap();
        let Value::Object(object) = out else {
            panic!("expected table");
        };
        let names = table_variable_names_from_object(&object).unwrap();
        assert_eq!(names, vec!["G", "GroupCount", "Percent"]);
        assert_eq!(table_height(&object).unwrap(), 2);

        let table = table_from_columns(
            vec!["G".into()],
            vec![Value::StringArray(
                StringArray::new(vec!["a".into(), "<missing>".into()], vec![2, 1]).unwrap(),
            )],
        )
        .unwrap();
        let out = block_on(groupcounts_builtin(
            table,
            vec![
                Value::from("G"),
                Value::from("IncludeMissingGroups"),
                Value::Bool(true),
            ],
        ))
        .unwrap();
        let Value::Object(object) = out else {
            panic!("expected table");
        };
        assert_eq!(table_height(&object).unwrap(), 2);
    }
}

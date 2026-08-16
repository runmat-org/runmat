use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, NumericDType,
};
use runmat_macros::runtime_builtin;

const ARRAY_DATASTORE_BUILTIN_NAME: &str = "arrayDatastore";

pub(crate) const TABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table with an explicit interactive GPU variable is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TableGpuInputExtension"),
    };
pub const TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [TABLE_GPU_INPUT_EXTENSION];
const TABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "var1,...,varN",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Any table variable may use one of the eight real integer classes and retains its own class and authoritative payload.",
    }];
const TABLE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented two-element numeric size vector is decoded exactly from native storage and its second element must match VariableTypes.",
    }];
pub const TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = table(integer_var1, ..., integer_varN, Name=Value...)",
        inputs: &TABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host and automatically resident variables become host table variables without an f64 mirror; unsupported explicit GPU intent is independently gated before gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = table(Size=integer_sz, VariableTypes=integer_type_names, Name=Value...)",
        inputs: &TABLE_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "int8 through uint64 VariableTypes allocate zero-filled authoritative native columns directly; floating VariableTypes allocate native f64 or f32 columns.",
    },
];

pub const ROWFILTER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "rowfilter is constructed from variable names, Parquet metadata/datastores, tables, or timetables; direct typed-integer arrays have no public data or control role and reject before gather. Integer columns nested in a tabular source remain opaque authoritative payloads while the filter captures row-selection metadata.",
    };

pub(crate) const ARRAY_DATASTORE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arraydatastore-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "arrayDatastore with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayDatastoreGpuInputExtension"),
    };

pub const ARRAY_DATASTORE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY_DATASTORE_GPU_INPUT_EXTENSION];

const PARQUET_DATASTORE_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ReadSize, BlockSize, location, or other constructor argument",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented ReadSize and BlockSize numeric datatype is double, while locations and other constructor options are text, logical, or objects. Typed-integer host or resident arguments reject before gather and cannot be silently ignored.",
    }];
pub const PARQUET_DATASTORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "pds = parquetDatastore(location, typed_integer_argument...)",
        inputs: &PARQUET_DATASTORE_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The constructor records supported host metadata only. Integer Parquet columns belong to parquetread/read datastore operations, not to this constructor input surface.",
    }];

const ARRAY_DATASTORE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented in-memory array input accepts matrices whose values use any of the eight real integer classes.",
    }];

const ARRAY_DATASTORE_READ_SIZE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ReadSize",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public property is explicitly double and typed-integer values are rejected.",
    }];

const ARRAY_DATASTORE_ITERATION_DIMENSION_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IterationDimension",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public property is explicitly double and typed-integer values are rejected.",
    }];

pub const ARRAY_DATASTORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(integer_A, Name,Value...)",
        inputs: &ARRAY_DATASTORE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The datastore object retains A in authoritative same-class storage. Interactive resident input is a mode-gated RunMat extension that gathers before host datastore construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(A, \"ReadSize\", typed_integer)",
        inputs: &ARRAY_DATASTORE_READ_SIZE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "ReadSize is documented as a double-only value; all typed-integer scalar and tensor classes reject.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(A, \"IterationDimension\", typed_integer)",
        inputs: &ARRAY_DATASTORE_ITERATION_DIMENSION_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "IterationDimension is documented as a double-only value; all typed-integer scalar and tensor classes reject.",
    },
];

pub(crate) const CATEGORICAL_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "categorical-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "categorical with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CategoricalGpuInputExtension"),
    };

pub const CATEGORICAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CATEGORICAL_GPU_INPUT_EXTENSION];

const ORDINAL_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ordinal-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ordinal host construction for explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OrdinalExplicitGpuInputExtension"),
};
const ORDINAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ORDINAL_GPU_INPUT_EXTENSION];

const CATEGORICAL_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts every real integer class and maps exact source values to categorical codes without a binary64 intermediary.",
    }];

const CATEGORICAL_INTEGER_VALUESET_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts every real integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "valueset",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A documented numeric valueset must use the same integer class as A, contain unique values, and is matched through exact native storage.",
    },
];

const CATEGORICAL_INTEGER_FLAG_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Ordinal or Protected",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented flags accept a scalar numeric zero or one; other integer values reject rather than being coerced by generic truthiness.",
    }];

const CATEGORICAL_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident argument",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive resident numeric input is not a documented gpuArray form; RunMat mode gates it before exact gather into the host categorical constructor.",
    }];

pub const CATEGORICAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(integer_A, ...)",
        inputs: &CATEGORICAL_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer values determine sorted categories and one-based categorical codes; the result is a host categorical metadata object rather than an integer array.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(integer_A, integer_valueset, ...)",
        inputs: &CATEGORICAL_INTEGER_VALUESET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A and valueset retain their exact common integer class for uniqueness, ordering, and membership; category names are object metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(A, ..., \"Ordinal\" or \"Protected\", integer_flag)",
        inputs: &CATEGORICAL_INTEGER_FLAG_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Every integer class is accepted only at exact scalar values zero and one, matching the documented numeric flag domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(gpuArray(integer_argument), ...)",
        inputs: &CATEGORICAL_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode gathers resident integer arguments exactly after the compatibility gate and returns a host categorical metadata object.",
    },
];

const ORDINAL_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts all eight integer classes; exact native values determine sorted ordinal levels and labels.",
    }];
const ORDINAL_INTEGER_LEVEL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer source values retain exact identity while levels are assigned.",
    },
    BuiltinIntegerInputCapability {
        name: "levels",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Explicit numeric levels accept all eight integer classes and are matched through the same exact categorical construction path.",
    },
];
const ORDINAL_INTEGER_EDGE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric source accepts all eight integer classes and is compared directly with each bin edge.",
    },
    BuiltinIntegerInputCapability {
        name: "edges",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric edge vector accepts all eight integer classes; strict ordering and half-open bin membership use exact mixed-class numeric comparison.",
    },
];
pub const ORDINAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X)",
        inputs: &ORDINAL_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer identity determines levels and codes before default labels cross the documented display-only five-significant-digit boundary; colliding display labels reject. Automatic residency gathers transparently, while explicit gpuArray input is gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X, labels, integer_levels)",
        inputs: &ORDINAL_INTEGER_LEVEL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Source and level vectors preserve exact signedness, width, and value identity until the opaque ordinal metadata object is assembled.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X, labels, [], integer_edges)",
        inputs: &ORDINAL_INTEGER_EDGE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Strictly increasing edges are compared without a binary64 mirror; bins are left-closed and right-open except that the last bin includes the final edge, and duplicate textual labels merge bins.",
    },
];

pub(crate) const DICTIONARY_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dictionary-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "dictionary with an interactive resident argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DictionaryGpuInputExtension"),
    };
pub const DICTIONARY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DICTIONARY_GPU_INPUT_EXTENSION];
const DICTIONARY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "keys",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dictionary keys accept every numeric integer class and retain exact configured-class identity, including wide uint64 values.",
    },
    BuiltinIntegerInputCapability {
        name: "values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dictionary values accept every numeric integer class and retain exact configured-class storage.",
    },
];
const DICTIONARY_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident keys or values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive resident input is not a documented gpuArray overload and is gated before exact gather into the host dictionary object.",
    }];
pub const DICTIONARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "d = dictionary(integer_keys, integer_values)",
        inputs: &DICTIONARY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Construction, duplicate resolution, lookup, assignment, and removal compare authoritative configured-class integer values without a binary64 mirror; the result is a host dictionary object.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "d = dictionary(gpuArray(integer_keys_or_values), ...)",
        inputs: &DICTIONARY_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode gathers admitted resident inputs through the owning provider and preserves exact typed values in the host object.",
    },
];

#[runtime_builtin(
    name = "table",
    category = "table",
    summary = "Create a table from named column variables.",
    keywords = "table,VariableNames,RowNames,Properties",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::TABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::constructors::TABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::TABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE_GPU_INPUT_EXTENSION,
            "table",
        )?;
    }
    let gathered = gather_values(&args).await?;
    let (variables, options) = split_table_constructor_args(gathered)?;
    if options.size.is_some() || options.variable_types.is_some() {
        return preallocated_table(variables, options);
    }
    let names = if let Some(names) = options.variable_names {
        names
    } else {
        generated_variable_names(variables.len())
    };
    table_from_columns_with_properties(names, variables, options.row_names)
}

fn preallocated_table(
    variables: Vec<Value>,
    options: TableConstructorOptions,
) -> BuiltinResult<Value> {
    if !variables.is_empty() {
        return Err(invalid_argument(
            "table: Size/VariableTypes preallocation cannot be combined with data variables",
        ));
    }
    let size = options
        .size
        .as_ref()
        .ok_or_else(|| invalid_argument("table: preallocation requires Size"))?;
    let variable_types = options
        .variable_types
        .as_ref()
        .ok_or_else(|| invalid_argument("table: preallocation requires VariableTypes"))?;
    let [rows, width] = table_preallocation_size(size)?;
    if variable_types.len() != width {
        return Err(invalid_argument(format!(
            "table: VariableTypes has {} entries but Size requests {width} variables",
            variable_types.len()
        )));
    }
    let mut columns = Vec::with_capacity(width);
    for type_name in variable_types {
        let dtype = table_preallocation_numeric_dtype(type_name).ok_or_else(|| {
            invalid_argument(format!(
                "table: preallocation type '{type_name}' is not yet supported; use double, single, or a built-in integer type"
            ))
        })?;
        columns.push(Value::Tensor(
            crate::builtins::common::tensor::zeros_with_dtype(&[rows, 1], dtype)
                .map_err(invalid_variable)?,
        ));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(width));
    table_from_columns_with_properties(names, columns, options.row_names)
}

fn table_preallocation_size(value: &Value) -> BuiltinResult<[usize; 2]> {
    let Value::Tensor(tensor) = value else {
        return Err(invalid_argument(
            "table: Size must be a two-element numeric vector",
        ));
    };
    if tensor.len() != 2 || (tensor.rows() != 1 && tensor.cols() != 1) {
        return Err(invalid_argument(
            "table: Size must be a two-element numeric vector",
        ));
    }
    let mut size = [0usize; 2];
    for (index, output) in size.iter_mut().enumerate() {
        let scalar = tensor
            .numeric_value_at(index)
            .ok_or_else(|| invalid_argument("table: Size contains a missing value"))?;
        *output = match scalar {
            runmat_builtins::NumericScalar::F64(value) => table_size_float(value)?,
            runmat_builtins::NumericScalar::F32(value) => table_size_float(f64::from(value))?,
            integer => integer
                .into_int_value()
                .and_then(|value| value.try_to_usize())
                .ok_or_else(|| {
                    invalid_argument("table: Size values must be nonnegative platform integers")
                })?,
        };
    }
    Ok(size)
}

fn table_size_float(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return Err(invalid_argument(
            "table: Size values must be nonnegative platform integers",
        ));
    }
    Ok(value as usize)
}

fn table_preallocation_numeric_dtype(type_name: &str) -> Option<NumericDType> {
    match type_name.trim().to_ascii_lowercase().as_str() {
        "double" => Some(NumericDType::F64),
        "single" => Some(NumericDType::F32),
        "int8" => Some(NumericDType::I8),
        "int16" => Some(NumericDType::I16),
        "int32" => Some(NumericDType::I32),
        "int64" => Some(NumericDType::I64),
        "uint8" => Some(NumericDType::U8),
        "uint16" => Some(NumericDType::U16),
        "uint32" => Some(NumericDType::U32),
        "uint64" => Some(NumericDType::U64),
        _ => None,
    }
}

#[runtime_builtin(
    name = "categorical",
    category = "table",
    summary = "Create a categorical array.",
    keywords = "categorical,categories,ordinal,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::constructors::CATEGORICAL_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::CATEGORICAL_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn categorical_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CATEGORICAL_GPU_INPUT_EXTENSION,
            "categorical",
        )?;
    }
    let args = gather_values(&args).await?;
    categorical_from_args(args)
}

#[runtime_builtin(
    name = "ordinal",
    category = "table",
    summary = "Create an ordinal categorical array.",
    keywords = "ordinal,categorical,categories,statistics",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::constructors::ORDINAL_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::ORDINAL_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn ordinal_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ORDINAL_GPU_INPUT_EXTENSION,
            "ordinal",
        )?;
    }
    let args = gather_values(&args).await?;
    ordinal_from_args(args)
}

#[runtime_builtin(
    name = "dictionary",
    category = "table",
    summary = "Create a key-value dictionary object.",
    keywords = "dictionary,containers.Map,key,value,map",
    accel = "cpu",
    descriptor(crate::builtins::table::DICTIONARY_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::constructors::DICTIONARY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::DICTIONARY_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DICTIONARY_GPU_INPUT_EXTENSION,
            "dictionary",
        )?;
    }
    let args = gather_values(&args).await?;
    dictionary_from_args(args)
}

#[runtime_builtin(
    name = "timerange",
    category = "table",
    summary = "Create a timetable row-time range selector.",
    keywords = "timerange,timetable,row times",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timerange_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.len() > 3 {
        return Err(invalid_argument(
            "timerange: expected start, end, and optional inclusivity",
        ));
    }
    let gathered = gather_values(&args).await?;
    let mut object = ObjectInstance::new(TIMERANGE_CLASS.to_string());
    object.properties.insert(
        "Start".to_string(),
        gathered
            .first()
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "End".to_string(),
        gathered
            .get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "Inclusivity".to_string(),
        gathered
            .get(2)
            .cloned()
            .unwrap_or_else(|| Value::from("closed")),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "vartype",
    category = "table",
    summary = "Create a table variable type selector.",
    keywords = "vartype,table,selector,variable type",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn vartype_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let mut object = ObjectInstance::new(VARTYPE_CLASS.to_string());
    object.properties.insert("Type".to_string(), value);
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "rowfilter",
    category = "table",
    summary = "Create a table row filter descriptor.",
    keywords = "rowfilter,table,rows,filter",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::table::builtins::constructors::ROWFILTER_INTEGER_AUDIT),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn rowfilter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_has_native_integer_class(value)
            || matches!(value, Value::GpuTensor(_))
    }) {
        return Err(invalid_argument(
            "rowfilter: typed numeric arrays are not valid rowfilter constructor inputs",
        ));
    }
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(ROWFILTER_CLASS.to_string());
    object.properties.insert(
        "Variables".to_string(),
        args.first()
            .cloned()
            .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
    );
    object.properties.insert(
        "Predicate".to_string(),
        args.get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "arrayDatastore",
    category = "io/tabular",
    summary = "Create a datastore for an in-memory array.",
    keywords = "arrayDatastore,datastore,array,ReadSize,IterationDimension,OutputType",
    accel = "gather",
    descriptor(crate::builtins::table::ARRAY_DATASTORE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::constructors::ARRAY_DATASTORE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::ARRAY_DATASTORE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY_DATASTORE_GPU_INPUT_EXTENSION,
            ARRAY_DATASTORE_BUILTIN_NAME,
        )?;
    }
    let args = gather_values(&args).await?;
    let Some(data) = args.first().cloned() else {
        return Err(invalid_argument(
            "arrayDatastore: input array A is required",
        ));
    };
    if (args.len() - 1) % 2 != 0 {
        return Err(invalid_argument(
            "arrayDatastore: name-value options must be provided in pairs",
        ));
    }
    let mut read_size = 1usize;
    let mut iteration_dimension = 1usize;
    let mut output_type = "cell".to_string();
    let mut index = 1usize;
    while index < args.len() {
        let name = scalar_text(&args[index], "arrayDatastore option")?;
        let value = &args[index + 1];
        if name.eq_ignore_ascii_case("ReadSize") {
            read_size = array_datastore_positive_double_integer(value, "ReadSize")?;
        } else if name.eq_ignore_ascii_case("IterationDimension") {
            iteration_dimension =
                array_datastore_positive_double_integer(value, "IterationDimension")?;
        } else if name.eq_ignore_ascii_case("OutputType") {
            output_type = scalar_text(value, "arrayDatastore OutputType")?.to_ascii_lowercase();
            if output_type != "cell" && output_type != "same" {
                return Err(invalid_argument(
                    "arrayDatastore: OutputType must be 'cell' or 'same'",
                ));
            }
        } else {
            return Err(invalid_argument(format!(
                "arrayDatastore: unsupported option '{name}'"
            )));
        }
        index += 2;
    }
    if output_type == "same" && iteration_dimension != 1 {
        return Err(invalid_argument(
            "arrayDatastore: IterationDimension must be 1 when OutputType is 'same'",
        ));
    }
    let mut object = ObjectInstance::new(ARRAY_DATASTORE_CLASS.to_string());
    object.properties.insert("Data".to_string(), data);
    object
        .properties
        .insert("ReadSize".to_string(), Value::Num(read_size as f64));
    object.properties.insert(
        "IterationDimension".to_string(),
        Value::Num(iteration_dimension as f64),
    );
    object
        .properties
        .insert("OutputType".to_string(), Value::String(output_type));
    Ok(Value::Object(object))
}

fn array_datastore_positive_double_integer(value: &Value, property: &str) -> BuiltinResult<usize> {
    let value = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 =>
        {
            crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(invalid_argument(format!(
                "arrayDatastore: {property} must be a positive double integer"
            )))
        }
    };
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return Err(invalid_argument(format!(
            "arrayDatastore: {property} must be a positive double integer"
        )));
    }
    Ok(value as usize)
}

#[runtime_builtin(
    name = "fileDatastore",
    category = "io/tabular",
    summary = "Create a file datastore descriptor.",
    keywords = "fileDatastore,datastore,file,readfcn",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn file_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let Some(files) = args.first().cloned() else {
        return Err(invalid_argument(
            "fileDatastore: files location is required",
        ));
    };
    if args.len() > 1 && (args.len() - 1) % 2 != 0 {
        return Err(invalid_argument(
            "fileDatastore: name-value options must be provided in pairs",
        ));
    }

    let mut read_fcn = Value::String(String::new());
    let mut file_extensions = Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap());
    let mut include_subfolders = Value::Bool(false);
    let mut read_mode = Value::from("file");
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "fileDatastore option")?;
        let value = args[idx + 1].clone();
        if name.eq_ignore_ascii_case("ReadFcn") {
            read_fcn = value;
        } else if name.eq_ignore_ascii_case("FileExtensions") {
            string_list(&value).map_err(|_| {
                invalid_argument(
                    "fileDatastore: FileExtensions must be text, a string array, or cellstr",
                )
            })?;
            file_extensions = value;
        } else if name.eq_ignore_ascii_case("IncludeSubfolders") {
            include_subfolders = Value::Bool(bool_scalar(&value, "IncludeSubfolders")?);
        } else if name.eq_ignore_ascii_case("ReadMode") {
            let mode = scalar_text(&value, "ReadMode")?;
            let lower = mode.to_ascii_lowercase();
            if lower != "file" && lower != "partialfile" {
                return Err(invalid_argument(
                    "fileDatastore: ReadMode must be 'file' or 'partialfile'",
                ));
            }
            read_mode = Value::String(mode);
        } else {
            return Err(invalid_argument(format!(
                "fileDatastore: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }

    let mut object = ObjectInstance::new(FILE_DATASTORE_CLASS.to_string());
    object.properties.insert("Files".to_string(), files);
    object.properties.insert("ReadFcn".to_string(), read_fcn);
    object
        .properties
        .insert("FileExtensions".to_string(), file_extensions);
    object
        .properties
        .insert("IncludeSubfolders".to_string(), include_subfolders);
    object.properties.insert("ReadMode".to_string(), read_mode);
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "parquetDatastore",
    category = "io/tabular",
    summary = "Create a parquet datastore descriptor.",
    keywords = "parquetDatastore,datastore,parquet,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::table::builtins::constructors::PARQUET_DATASTORE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquet_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.iter().any(parquet_value_contains_typed_integer) {
        return Err(invalid_argument(
            "parquetDatastore: typed-integer constructor arguments are not supported",
        ));
    }
    let args = gather_values(&args).await?;
    let Some(files) = args.first().cloned() else {
        return Err(invalid_argument(
            "parquetDatastore: files location is required",
        ));
    };
    if (args.len() - 1) % 2 != 0 {
        return Err(invalid_argument(
            "parquetDatastore: name-value options must be provided in pairs",
        ));
    }
    let mut read_size = Value::from("rowgroup");
    let mut block_size = Value::Num(128_000_000.0);
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "parquetDatastore option")?;
        let value = args[idx + 1].clone();
        if name.eq_ignore_ascii_case("ReadSize") {
            read_size = match &value {
                Value::String(text)
                    if text.eq_ignore_ascii_case("file")
                        || text.eq_ignore_ascii_case("rowgroup") =>
                {
                    value
                }
                Value::CharArray(_) | Value::StringArray(_) => {
                    let text = scalar_text(&value, "ReadSize")?;
                    if !text.eq_ignore_ascii_case("file") && !text.eq_ignore_ascii_case("rowgroup")
                    {
                        return Err(invalid_argument("parquetDatastore: ReadSize must be 'file', 'rowgroup', or a positive double integer"));
                    }
                    Value::String(text)
                }
                _ => {
                    Value::Num(array_datastore_positive_double_integer(&value, "ReadSize")? as f64)
                }
            };
        } else if name.eq_ignore_ascii_case("BlockSize") {
            block_size =
                Value::Num(array_datastore_positive_double_integer(&value, "BlockSize")? as f64);
        } else {
            return Err(invalid_argument(format!(
                "parquetDatastore: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    let mut object = ObjectInstance::new(PARQUET_DATASTORE_CLASS.to_string());
    object.properties.insert("Files".to_string(), files);
    object.properties.insert("ReadSize".to_string(), read_size);
    object
        .properties
        .insert("BlockSize".to_string(), block_size);
    Ok(Value::Object(object))
}

fn parquet_value_contains_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Cell(cell) if cell.data.iter().any(parquet_value_contains_typed_integer))
        || matches!(value, Value::Struct(structure) if structure.fields.values().any(parquet_value_contains_typed_integer))
        || matches!(value, Value::Object(object) if object.properties.values().any(parquet_value_contains_typed_integer))
}

#[runtime_builtin(
    name = "uitable",
    category = "table",
    summary = "Create a table UI component descriptor.",
    keywords = "uitable,ui,table,Data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn uitable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(UITABLE_CLASS.to_string());
    let data = parse_named_option(&args, "Data")
        .cloned()
        .or_else(|| args.first().cloned())
        .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()));
    object.properties.insert("Data".to_string(), data);
    object.properties.insert(
        "ColumnName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    object.properties.insert(
        "RowName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    Ok(Value::Object(object))
}

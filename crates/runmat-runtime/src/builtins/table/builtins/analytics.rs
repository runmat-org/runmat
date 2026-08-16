use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    IntValue, NumericScalar,
};
use runmat_macros::runtime_builtin;

const GROUPSUMMARY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupsummary-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "groupsummary on interactive resident GPU data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupsummaryResidentInputExtension"),
    };
const GROUPSUMMARY_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupsummary-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "groupsummary with a native-class integer name-value control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupsummaryIntegerControlExtension"),
    };
pub const GROUPSUMMARY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    GROUPSUMMARY_RESIDENT_INPUT_EXTENSION,
    GROUPSUMMARY_INTEGER_CONTROL_EXTENSION,
];

const GRPSTATS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "grpstats-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "grpstats on interactive resident GPU data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GrpstatsResidentInputExtension"),
};
const GRPSTATS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "grpstats-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "grpstats with a native-class integer matrix X is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GrpstatsIntegerDataExtension"),
};
const GRPSTATS_INTEGER_ALPHA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "grpstats-integer-alpha",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "grpstats with a native-class integer Alpha value is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GrpstatsIntegerAlphaExtension"),
};
const GRPSTATS_INTEGER_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "grpstats-integer-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "grpstats with a native-class integer table selector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GrpstatsIntegerSelectorExtension"),
    };
pub const GRPSTATS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    GRPSTATS_RESIDENT_INPUT_EXTENSION,
    GRPSTATS_INTEGER_DATA_EXTENSION,
    GRPSTATS_INTEGER_ALPHA_EXTENSION,
    GRPSTATS_INTEGER_SELECTOR_EXTENSION,
];

const PIVOT_INTEGER_GROUP_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer grouping or DataVariable table columns",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target permits numeric grouping and data variables. Group keys are compared exactly; summary results follow the selected Method contract.",
    }];
const PIVOT_INTEGER_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Columns, Rows, or DataVariable indices",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public variable-index selectors are exact one-based structural values; logical selectors remain a distinct documented form.",
    }];
const PIVOT_INTEGER_BIN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ColumnsBinMethod or RowsBinMethod count/edges",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents positive-integer bin counts and numeric edge vectors; counts are structural and edges require exact ordering before summary computation.",
    }];
const PIVOT_INTEGER_BOOLEAN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IncludeTotals, IncludeMissingGroups, or IncludeEmptyGroups",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public controls admit numeric zero or one alongside logical values and are structural rather than aggregation data.",
    }];
pub const PIVOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(table_with_integer_group_or_data_variables,___)",
        inputs: &PIVOT_INTEGER_GROUP_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Implemented grouping keys retain exact native values above flintmax; aggregation crosses only the selected method's explicit result domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(T,Columns=integer_indices,Rows=integer_indices,DataVariable=integer_index,___)",
        inputs: &PIVOT_INTEGER_SELECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Implemented selector parsing reads native integer storage directly; unsupported portions of the newer name-value grammar remain general pivot surface gaps.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(T,ColumnsBinMethod=integer_bins,RowsBinMethod=integer_bins,___)",
        inputs: &PIVOT_INTEGER_BIN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented contract is recorded without treating unimplemented general binning syntax as an implicit integer-to-floating boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "P = pivot(T,IncludeTotals=integer_flag,IncludeMissingGroups=integer_flag,IncludeEmptyGroups=integer_flag,___)",
        inputs: &PIVOT_INTEGER_BOOLEAN_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact zero/one values are compatible; unsupported general options reject before aggregation.",
    },
];

const GROUPSUMMARY_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer grouping variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table grouping variables are compared exactly and preserve class in the output table.",
    }];
const GROUPSUMMARY_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer data variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer table data variables are numeric and are valid data-variable selections; exact-class extrema and explicit floating statistics preserve the documented method contract.",
    }];
const GROUPSUMMARY_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "native-class integer bin count or boolean name-value control",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer scalars in these structurally ambiguous control positions are accepted only in RunMat extension mode.",
    }];
pub const GROUPSUMMARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(T_or_A,integer_groupvars,method,___)",
        inputs: &GROUPSUMMARY_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Table and array grouping is exact. Table grouping columns and array BG preserve class, while GroupCount/BC and summaries are function-specific.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(integer_A,groupvars,method) or table integer data variables",
        inputs: &GROUPSUMMARY_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer array and table data uses exact native-class min and max plus explicit floating computation for mean, sum, median, and counts. One-group numeric-edge and scalar-count forms compare integers exactly; multi-group bin specifications, time bins, additional named methods, and function handles currently reject explicitly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "groupsummary(T,groupvars,integer_numbins,method,___) or integer boolean controls",
        inputs: &GROUPSUMMARY_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This is an explicitly gated input-form extension; equivalent scalar-double and logical controls remain available in strict compatibility mode.",
    },
];

const GRPSTATS_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "group",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Statistics grouping variables explicitly admit signed and unsigned integer classes and are compared exactly.",
    }];
const GRPSTATS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix X datatype list excludes signed and unsigned integer data; RunMat admission is independently mode-gated.",
    }];
const GRPSTATS_INTEGER_TABLE_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer table data variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Table variables may be numeric, so selected native integer variables use the documented table form.",
    }];
pub const GRPSTATS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "stats = grpstats(X,integer_group,___)",
        inputs: &GRPSTATS_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer grouping keys remain exact, including above flintmax; summary output is governed by X and whichstats.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "stats = grpstats(integer_X,group,___)",
        inputs: &GRPSTATS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer matrix data crosses a checked binary64 boundary after its compatibility gate; values that cannot be represented exactly reject rather than silently rounding.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tblstats = grpstats(tbl,groupvars,___) with integer DataVars",
        inputs: &GRPSTATS_INTEGER_TABLE_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Selected integer table variables are documented numeric data; extrema retain exact integer values while floating statistics enter their explicit computation domain.",
    },
];

#[runtime_builtin(
    name = "grpstats",
    category = "stats/summary",
    summary = "Compute summary statistics organized by group.",
    keywords = "grpstats,group,statistics,mean,std,confidence interval,table",
    accel = "cpu",
    descriptor(crate::builtins::table::GRPSTATS_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::analytics::GRPSTATS_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::analytics::GRPSTATS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn grpstats_builtin(
    value: Value,
    group: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_grpstats_extensions(&value, &group, &rest)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let group = gather_if_needed_async(&group)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    grpstats_impl(value, group, rest)
}

#[runtime_builtin(
    name = "pivot",
    category = "table",
    summary = "Pivot or summarize table data by grouping variables.",
    keywords = "pivot,table,reshape,groupsummary",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    integer_capabilities(crate::builtins::table::builtins::analytics::PIVOT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn pivot_builtin(
    table: Value,
    rowvars: Value,
    colvars: Value,
    datavar: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let rowvars = gather_if_needed_async(&rowvars)
        .await
        .map_err(map_control_flow)?;
    let colvars = gather_if_needed_async(&colvars)
        .await
        .map_err(map_control_flow)?;
    let datavar = gather_if_needed_async(&datavar)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let method = parse_named_text_option(&rest, "Method", "sum", "pivot")?;
    pivot_impl(table, rowvars, colvars, datavar, &method)
}

#[runtime_builtin(
    name = "groupsummary",
    category = "table",
    summary = "Group table rows and compute summary statistics for data variables.",
    keywords = "groupsummary,group,table,mean,sum,count,median,min,max",
    accel = "cpu",
    descriptor(crate::builtins::table::GROUPSUMMARY_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::analytics::GROUPSUMMARY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::analytics::GROUPSUMMARY_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn groupsummary_builtin(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_groupsummary_extensions(&table, &groupvars, &method, &rest)?;
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let groupvars = gather_if_needed_async(&groupvars)
        .await
        .map_err(map_control_flow)?;
    let method = gather_if_needed_async(&method)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    if matches!(&table, Value::Object(object) if is_tabular_object(object)) {
        groupsummary_with_numeric_bins(table, groupvars, method, rest)
    } else {
        groupsummary_array(table, groupvars, method, rest)
    }
}

fn groupsummary_array(
    data: Value,
    groupvars: Value,
    method_or_bins: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let data = groupsummary_array_columns(data, "input array")?;
    let groups = groupsummary_group_columns(groupvars)?;
    let rows = data.first().map(value_row_count).transpose()?.unwrap_or(0);
    for value in &groups {
        if value_row_count(value)? != rows {
            return Err(invalid_argument(
                "groupsummary: grouping vectors must have the same number of rows as A",
            ));
        }
    }
    let group_names = (1..=groups.len())
        .map(|index| format!("Group{index}"))
        .collect::<Vec<_>>();
    let data_names = (1..=data.len())
        .map(|index| format!("Data{index}"))
        .collect::<Vec<_>>();
    let mut names = group_names.clone();
    names.extend(data_names.iter().cloned());
    let mut columns = groups;
    columns.extend(data);
    let table = table_from_columns(names, columns)?;
    let group_selector = if group_names.len() == 1 {
        Value::from(group_names[0].as_str())
    } else {
        Value::StringArray(
            StringArray::new(group_names.clone(), vec![1, group_names.len()])
                .map_err(invalid_variable)?,
        )
    };
    let result = groupsummary_with_numeric_bins(table, group_selector, method_or_bins, rest)?;
    groupsummary_array_outputs(result, &group_names)
}

fn groupsummary_array_columns(value: Value, role: &str) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.cols())
            .map(|column| groupsummary_tensor_column(&tensor, column).map(Value::Tensor))
            .collect(),
        Value::Cell(cell) => {
            let mut columns = Vec::new();
            for value in cell.data {
                let mut nested = groupsummary_array_columns(value, role)?;
                columns.append(&mut nested);
            }
            if columns.is_empty() {
                return Err(invalid_argument(format!(
                    "groupsummary: {role} must contain at least one numeric column"
                )));
            }
            Ok(columns)
        }
        _ => Err(invalid_argument(format!(
            "groupsummary: {role} must be a numeric vector, matrix, or cell array of numeric arrays"
        ))),
    }
}

fn groupsummary_group_columns(value: Value) -> BuiltinResult<Vec<Value>> {
    groupsummary_array_columns(value, "groupvars")
}

fn groupsummary_tensor_column(tensor: &Tensor, column: usize) -> BuiltinResult<Tensor> {
    let rows = tensor.rows();
    let indices = (0..rows).map(|row| row + column * rows).collect::<Vec<_>>();
    if let Some(storage) = tensor.integer_storage() {
        let exact = storage.exact_values();
        let values = indices
            .iter()
            .map(|index| {
                exact
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| invalid_index("groupsummary: array column index out of bounds"))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        return Tensor::new_integer(
            storage
                .from_exact_values_like(values)
                .map_err(invalid_variable)?,
            vec![rows, 1],
        )
        .map_err(invalid_variable);
    }
    if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        return Tensor::from_f32(
            indices
                .iter()
                .map(|index| {
                    tensor
                        .numeric_value_at(*index)
                        .map(|value| value.materialize_f64() as f32)
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid_index("groupsummary: array column index out of bounds"))?,
            vec![rows, 1],
        )
        .map_err(invalid_variable);
    }
    Tensor::new(
        indices
            .iter()
            .map(|index| {
                tensor
                    .numeric_value_at(*index)
                    .map(NumericScalar::materialize_f64)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid_index("groupsummary: array column index out of bounds"))?,
        vec![rows, 1],
    )
    .map_err(invalid_variable)
}

fn groupsummary_array_outputs(result: Value, group_names: &[String]) -> BuiltinResult<Value> {
    let Value::Object(object) = result else {
        return Err(invalid_argument(
            "groupsummary: internal array summary did not produce a table",
        ));
    };
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let group_count = variables
        .fields
        .get("GroupCount")
        .cloned()
        .ok_or_else(|| invalid_variable("groupsummary: missing GroupCount output"))?;
    let summaries = names
        .iter()
        .filter(|name| !group_names.contains(name) && name.as_str() != "GroupCount")
        .map(|name| {
            variables
                .fields
                .get(name)
                .cloned()
                .ok_or_else(|| invalid_variable("groupsummary: missing summary output"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let summary = groupsummary_concat_columns(summaries)?;
    let groups = group_names
        .iter()
        .map(|name| {
            variables
                .fields
                .get(name)
                .cloned()
                .ok_or_else(|| invalid_variable("groupsummary: missing group output"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let groups = if groups.len() == 1 {
        groups.into_iter().next().expect("one grouping output")
    } else {
        Value::Cell(CellArray::new(groups, 1, group_names.len()).map_err(invalid_variable)?)
    };
    groupsummary_multi_output(vec![summary, groups, group_count])
}

fn groupsummary_concat_columns(columns: Vec<Value>) -> BuiltinResult<Value> {
    let tensors = columns
        .into_iter()
        .map(|value| match value {
            Value::Tensor(tensor) => Ok(tensor),
            _ => Err(invalid_variable(
                "groupsummary: array summary methods must return numeric columns",
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let rows = tensors.first().map(Tensor::rows).unwrap_or(0);
    if tensors
        .iter()
        .any(|tensor| tensor.rows() != rows || tensor.cols() != 1)
    {
        return Err(invalid_variable(
            "groupsummary: array summary columns have incompatible sizes",
        ));
    }
    if let Some(first) = tensors.first().and_then(Tensor::integer_storage) {
        if tensors.iter().all(|tensor| {
            tensor
                .integer_storage()
                .is_some_and(|storage| storage.numeric_dtype() == first.numeric_dtype())
        }) {
            let mut values = Vec::with_capacity(rows.saturating_mul(tensors.len()));
            for tensor in &tensors {
                values.extend(
                    tensor
                        .integer_storage()
                        .expect("checked integer summary")
                        .exact_values(),
                );
            }
            return first
                .from_exact_values_like(values)
                .and_then(|storage| Tensor::new_integer(storage, vec![rows, tensors.len()]))
                .map(Value::Tensor)
                .map_err(invalid_variable);
        }
    }
    if tensors
        .iter()
        .all(|tensor| tensor.numeric_dtype() == runmat_builtins::NumericDType::F32)
    {
        let values = tensors
            .iter()
            .flat_map(|tensor| {
                tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|value| value as f32)
            })
            .collect();
        return Tensor::from_f32(values, vec![rows, tensors.len()])
            .map(Value::Tensor)
            .map_err(invalid_variable);
    }
    let values = tensors.iter().flat_map(Tensor::materialize_f64).collect();
    Tensor::new(values, vec![rows, tensors.len()])
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

fn groupsummary_multi_output(outputs: Vec<Value>) -> BuiltinResult<Value> {
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

fn value_is_native_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::CharArray(value) if value.rows == 1 => Some(value.data.iter().collect()),
        Value::StringArray(value) if value.data.len() == 1 => Some(value.data[0].clone()),
        _ => None,
    }
}

fn ensure_groupsummary_extensions(
    table: &Value,
    groupvars: &Value,
    method: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    if groupsummary_contains_explicit_gpu(table)
        || groupsummary_contains_explicit_gpu(groupvars)
        || groupsummary_contains_explicit_gpu(method)
        || rest.iter().any(groupsummary_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GROUPSUMMARY_RESIDENT_INPUT_EXTENSION,
            "groupsummary",
        )?;
    }
    if rest.windows(2).any(|pair| {
        option_name(&pair[0]).is_some_and(|name| {
            name.eq_ignore_ascii_case("IncludeMissingGroups")
                || name.eq_ignore_ascii_case("IncludeEmptyGroups")
        }) && value_is_native_integer(&pair[1])
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GROUPSUMMARY_INTEGER_CONTROL_EXTENSION,
            "groupsummary",
        )?;
    }
    if groupsummary_typed_integer_scalar(method) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GROUPSUMMARY_INTEGER_CONTROL_EXTENSION,
            "groupsummary",
        )?;
    }
    Ok(())
}

fn groupsummary_typed_integer_scalar(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if handle.shape.iter().product::<usize>() == 1 && runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Cell(cell) if cell.data.len() == 1 && groupsummary_typed_integer_scalar(&cell.data[0]))
}

fn groupsummary_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(groupsummary_contains_explicit_gpu),
        Value::Struct(value) => value
            .fields
            .values()
            .any(groupsummary_contains_explicit_gpu),
        Value::Object(value) => value
            .properties
            .values()
            .any(groupsummary_contains_explicit_gpu),
        Value::Closure(value) => value
            .captures
            .iter()
            .any(groupsummary_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(groupsummary_contains_explicit_gpu),
        _ => false,
    }
}

fn groupsummary_with_numeric_bins(
    table: Value,
    groupvars: Value,
    method_or_bins: Value,
    mut rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !groupsummary_is_numeric_bins(&method_or_bins) {
        return groupsummary_impl(table, groupvars, method_or_bins, rest);
    }
    if rest.is_empty() {
        return Err(invalid_argument(
            "groupsummary: a computation method must follow groupbins",
        ));
    }
    let method = rest.remove(0);
    let bin_spec = match &method_or_bins {
        Value::Cell(cell) if cell.data.len() == 1 => &cell.data[0],
        Value::Cell(_) => {
            return Err(invalid_argument(
                "groupsummary: a cell groupbins argument must contain exactly one specification for one grouping variable",
            ))
        }
        value => value,
    };
    let object = into_table_object(table, "groupsummary")?;
    let names = table_variable_names_from_object(&object)?;
    let selected = parse_variable_selector_for_object(Some(&groupvars), &object, &names)?;
    if selected.len() != 1 {
        return Err(invalid_argument(
            "groupsummary: numeric groupbins currently require one numeric grouping variable; multiple and time grouping variables are not implemented",
        ));
    }
    let variables = table_variables(&object)?;
    let group_value = variables
        .fields
        .get(&selected[0])
        .ok_or_else(|| invalid_variable("groupsummary: missing grouping variable"))?;
    let values = groupsummary_numeric_scalars(group_value)?;
    if scalar_text(bin_spec, "groupbins").is_ok_and(|text| text.eq_ignore_ascii_case("none")) {
        let (_, _, forwarded) = groupsummary_bin_options(rest)?;
        return groupsummary_impl(Value::Object(object), groupvars, method, forwarded);
    }
    let (included_right, include_empty, forwarded) = groupsummary_bin_options(rest)?;
    let (assignments, labels) = groupsummary_bin_assignments(&values, bin_spec, included_right)?;
    let internal = assignments
        .into_iter()
        .map(|bin| {
            bin.map(|index| format!("{index:08}|{}", labels[index]))
                .unwrap_or_else(|| "<missing>".to_string())
        })
        .collect::<Vec<_>>();
    let mut replaced = Vec::with_capacity(names.len());
    for name in &names {
        if name == &selected[0] {
            replaced.push(Value::StringArray(
                StringArray::new(internal.clone(), vec![internal.len(), 1])
                    .map_err(invalid_variable)?,
            ));
        } else {
            replaced.push(
                variables
                    .fields
                    .get(name)
                    .cloned()
                    .ok_or_else(|| invalid_variable("groupsummary: missing table variable"))?,
            );
        }
    }
    let binned_table = table_from_columns(names, replaced)?;
    let result = groupsummary_impl(binned_table, groupvars, method, forwarded)?;
    groupsummary_finalize_bins(result, &selected[0], &labels, include_empty)
}

fn groupsummary_is_numeric_bins(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
        || matches!(value, Value::Cell(cell) if cell.data.len() == 1 && groupsummary_is_numeric_bins(&cell.data[0]))
        || matches!(value, Value::String(text) if text.eq_ignore_ascii_case("none"))
        || matches!(value, Value::CharArray(chars) if chars.data.iter().collect::<String>().eq_ignore_ascii_case("none"))
}

fn groupsummary_bin_options(rest: Vec<Value>) -> BuiltinResult<(bool, bool, Vec<Value>)> {
    let mut included_right = false;
    let mut include_empty = false;
    let mut forwarded = Vec::with_capacity(rest.len());
    let mut index = 0;
    if rest.first().is_some_and(|value| {
        !option_name(value).is_some_and(|name| {
            name.eq_ignore_ascii_case("IncludedEdge")
                || name.eq_ignore_ascii_case("IncludeEmptyGroups")
                || name.eq_ignore_ascii_case("IncludeMissingGroups")
        })
    }) {
        forwarded.push(rest[0].clone());
        index = 1;
    }
    while index < rest.len() {
        if index + 1 >= rest.len() {
            return Err(invalid_argument(
                "groupsummary: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&rest[index], "groupsummary option name")?;
        if name.eq_ignore_ascii_case("IncludedEdge") {
            included_right = match scalar_text(&rest[index + 1], "IncludedEdge")?
                .to_ascii_lowercase()
                .as_str()
            {
                "left" => false,
                "right" => true,
                _ => {
                    return Err(invalid_argument(
                        "groupsummary: IncludedEdge must be 'left' or 'right'",
                    ))
                }
            };
        } else if name.eq_ignore_ascii_case("IncludeEmptyGroups") {
            include_empty = zero_one_bool_scalar(&rest[index + 1], "IncludeEmptyGroups")?;
        } else {
            forwarded.push(rest[index].clone());
            forwarded.push(rest[index + 1].clone());
        }
        index += 2;
    }
    Ok((included_right, include_empty, forwarded))
}

fn groupsummary_numeric_scalars(value: &Value) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid_argument("groupsummary: invalid numeric grouping value"))
            })
            .collect(),
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        _ => Err(invalid_argument(
            "groupsummary: numeric groupbins require a numeric grouping variable",
        )),
    }
}

fn groupsummary_bin_assignments(
    values: &[NumericScalar],
    spec: &Value,
    included_right: bool,
) -> BuiltinResult<(Vec<Option<usize>>, Vec<String>)> {
    if scalar_text(spec, "groupbins").is_ok_and(|text| text.eq_ignore_ascii_case("none")) {
        return Err(invalid_argument(
            "groupsummary: 'none' groupbins does not transform numeric groups; use the ordinary method form",
        ));
    }
    if groupsummary_is_bin_count(spec) {
        let count = groupsummary_bin_count(spec)?;
        return groupsummary_equal_count_bins(values, count, included_right);
    }
    let edges = groupsummary_numeric_scalars(spec)?;
    if edges.len() < 2 {
        return Err(invalid_argument(
            "groupsummary: numeric bin edges must contain at least two values",
        ));
    }
    for pair in edges.windows(2) {
        if groupsummary_compare_numeric(pair[0], pair[1]) != Some(Ordering::Less) {
            return Err(invalid_argument(
                "groupsummary: numeric bin edges must be finite and strictly increasing",
            ));
        }
    }
    let labels = (0..edges.len() - 1)
        .map(|index| {
            groupsummary_edge_label(
                edges[index],
                edges[index + 1],
                index,
                edges.len() - 1,
                included_right,
            )
        })
        .collect::<Vec<_>>();
    let assignments = values
        .iter()
        .map(|value| groupsummary_find_edge_bin(*value, &edges, included_right))
        .collect();
    Ok((assignments, labels))
}

fn groupsummary_is_bin_count(value: &Value) -> bool {
    matches!(value, Value::Num(value) if value.is_finite() && *value > 0.0 && value.fract() == 0.0)
        || matches!(value, Value::Int(value) if value.try_to_usize().is_some_and(|value| value > 0))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
}

fn groupsummary_bin_count(value: &Value) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Num(value) if value.is_finite() && *value > 0.0 && value.fract() == 0.0 => {
            Some(*value as usize)
        }
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .and_then(|value| value.try_to_usize())
            .or_else(|| {
                let value = tensor.materialize_f64()[0];
                (value.is_finite() && value > 0.0 && value.fract() == 0.0).then_some(value as usize)
            }),
        _ => None,
    };
    parsed
        .filter(|value| *value > 0 && *value <= 1_000_000)
        .ok_or_else(|| {
            invalid_argument(
                "groupsummary: number of bins must be a positive integer no greater than 1000000",
            )
        })
}

fn groupsummary_numeric_integer(value: NumericScalar) -> Option<i128> {
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

fn groupsummary_compare_numeric(left: NumericScalar, right: NumericScalar) -> Option<Ordering> {
    match (
        groupsummary_numeric_integer(left),
        groupsummary_numeric_integer(right),
    ) {
        (Some(left), Some(right)) => Some(left.cmp(&right)),
        (Some(left), None) => groupsummary_compare_integer_float(left, right.materialize_f64()),
        (None, Some(right)) => {
            groupsummary_compare_integer_float(right, left.materialize_f64()).map(Ordering::reverse)
        }
        (None, None) => left.materialize_f64().partial_cmp(&right.materialize_f64()),
    }
}

fn groupsummary_compare_integer_float(integer: i128, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    let truncated = float.trunc();
    if truncated < i128::MIN as f64 {
        return Some(Ordering::Greater);
    }
    if truncated > i128::MAX as f64 {
        return Some(Ordering::Less);
    }
    let ordering = integer.cmp(&(truncated as i128));
    if ordering == Ordering::Equal && float.fract() != 0.0 {
        Some(if float.is_sign_positive() {
            Ordering::Less
        } else {
            Ordering::Greater
        })
    } else {
        Some(ordering)
    }
}

fn groupsummary_find_edge_bin(
    value: NumericScalar,
    edges: &[NumericScalar],
    included_right: bool,
) -> Option<usize> {
    if groupsummary_compare_numeric(value, value).is_none() {
        return None;
    }
    for index in 0..edges.len() - 1 {
        let lower = groupsummary_compare_numeric(value, edges[index])?;
        let upper = groupsummary_compare_numeric(value, edges[index + 1])?;
        let hit = if included_right {
            (lower == Ordering::Greater || (index == 0 && lower == Ordering::Equal))
                && upper != Ordering::Greater
        } else {
            lower != Ordering::Less
                && (upper == Ordering::Less
                    || (index + 2 == edges.len() && upper == Ordering::Equal))
        };
        if hit {
            return Some(index);
        }
    }
    None
}

fn groupsummary_equal_count_bins(
    values: &[NumericScalar],
    count: usize,
    included_right: bool,
) -> BuiltinResult<(Vec<Option<usize>>, Vec<String>)> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| groupsummary_compare_numeric(*value, *value).is_some())
        .collect::<Vec<_>>();
    finite.sort_by(|left, right| {
        groupsummary_compare_numeric(*left, *right).unwrap_or(Ordering::Equal)
    });
    let Some(minimum) = finite.first().copied() else {
        return Ok((
            vec![None; values.len()],
            (1..=count).map(|index| format!("bin{index}")).collect(),
        ));
    };
    let maximum = *finite.last().expect("nonempty finite grouping values");
    let assignments = if let (Some(minimum), Some(maximum)) = (
        groupsummary_numeric_integer(minimum),
        groupsummary_numeric_integer(maximum),
    ) {
        let range = (maximum - minimum) as u128;
        values
            .iter()
            .map(|value| {
                let value = groupsummary_numeric_integer(*value)?;
                if range == 0 {
                    return Some(0);
                }
                let delta = (value - minimum) as u128;
                let scaled = delta * count as u128;
                let index = if included_right && delta > 0 {
                    scaled.div_ceil(range).saturating_sub(1)
                } else {
                    scaled / range
                };
                Some((index as usize).min(count - 1))
            })
            .collect()
    } else {
        let minimum = minimum.materialize_f64();
        let maximum = maximum.materialize_f64();
        values
            .iter()
            .map(|value| {
                let value = value.materialize_f64();
                if !value.is_finite() {
                    None
                } else if minimum == maximum {
                    Some(0)
                } else {
                    let scaled = (value - minimum) * count as f64 / (maximum - minimum);
                    let index = if included_right && value > minimum {
                        scaled.ceil() - 1.0
                    } else {
                        scaled.floor()
                    };
                    Some((index.max(0.0) as usize).min(count - 1))
                }
            })
            .collect()
    };
    Ok((
        assignments,
        (1..=count).map(|index| format!("bin{index}")).collect(),
    ))
}

fn groupsummary_scalar_label(value: NumericScalar) -> String {
    value
        .into_int_value()
        .map(|value| match value {
            IntValue::I8(value) => value.to_string(),
            IntValue::I16(value) => value.to_string(),
            IntValue::I32(value) => value.to_string(),
            IntValue::I64(value) => value.to_string(),
            IntValue::U8(value) => value.to_string(),
            IntValue::U16(value) => value.to_string(),
            IntValue::U32(value) => value.to_string(),
            IntValue::U64(value) => value.to_string(),
        })
        .unwrap_or_else(|| format_key_number(value.materialize_f64()))
}

fn groupsummary_edge_label(
    lower: NumericScalar,
    upper: NumericScalar,
    index: usize,
    count: usize,
    included_right: bool,
) -> String {
    let lower = groupsummary_scalar_label(lower);
    let upper = groupsummary_scalar_label(upper);
    if included_right {
        format!("{}{lower}, {upper}]", if index == 0 { "[" } else { "(" })
    } else {
        format!(
            "[{lower}, {upper}{}",
            if index + 1 == count { "]" } else { ")" }
        )
    }
}

fn groupsummary_finalize_bins(
    result: Value,
    group_name: &str,
    labels: &[String],
    include_empty: bool,
) -> BuiltinResult<Value> {
    let Value::Object(object) = result else {
        return Err(invalid_argument("groupsummary: expected table result"));
    };
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let Value::StringArray(groups) = variables
        .fields
        .get(group_name)
        .ok_or_else(|| invalid_variable("groupsummary: missing output grouping variable"))?
    else {
        return Err(invalid_variable(
            "groupsummary: invalid binned grouping output",
        ));
    };
    let existing = groups
        .data
        .iter()
        .enumerate()
        .map(|(row, label)| (label.clone(), row))
        .collect::<BTreeMap<_, _>>();
    let mut desired = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (format!("{index:08}|{label}"), label.clone()))
        .filter(|(internal, _)| include_empty || existing.contains_key(internal))
        .collect::<Vec<_>>();
    if existing.contains_key("<missing>") {
        desired.push(("<missing>".to_string(), "<missing>".to_string()));
    }
    let mut output = Vec::with_capacity(names.len());
    for name in &names {
        if name == group_name {
            output.push(Value::StringArray(
                StringArray::new(
                    desired.iter().map(|(_, label)| label.clone()).collect(),
                    vec![desired.len(), 1],
                )
                .map_err(invalid_variable)?,
            ));
            continue;
        }
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable("groupsummary: missing output variable"))?;
        let Value::Tensor(tensor) = value else {
            return Err(invalid_variable(
                "groupsummary: binned output supports numeric summary variables",
            ));
        };
        if tensor.integer_storage().is_some()
            && desired
                .iter()
                .any(|(internal, _)| !existing.contains_key(internal))
        {
            return Err(invalid_argument(
                "groupsummary: empty binned groups for integer-class min/max are not implemented because the public fill class is unresolved",
            ));
        }
        if tensor.integer_storage().is_some() {
            let rows = desired
                .iter()
                .filter_map(|(internal, _)| existing.get(internal).copied())
                .collect::<Vec<_>>();
            output.push(select_rows(value, &rows)?);
        } else {
            let values = desired
                .iter()
                .map(|(internal, _)| {
                    existing
                        .get(internal)
                        .map(|row| tensor.materialize_f64()[*row])
                        .unwrap_or_else(|| if name == "GroupCount" { 0.0 } else { f64::NAN })
                })
                .collect::<Vec<_>>();
            output.push(Value::Tensor(
                Tensor::new(values, vec![desired.len(), 1]).map_err(invalid_variable)?,
            ));
        }
    }
    table_from_columns(names, output)
}

fn ensure_grpstats_extensions(value: &Value, group: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::value_contains_gpu(value)
        || crate::value_contains_gpu(group)
        || rest.iter().any(crate::value_contains_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRPSTATS_RESIDENT_INPUT_EXTENSION,
            "grpstats",
        )?;
    }
    let table_input = matches!(value, Value::Object(object) if is_tabular_object(object));
    if !table_input && value_is_native_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRPSTATS_INTEGER_DATA_EXTENSION,
            "grpstats",
        )?;
    }
    if rest.first().is_some_and(value_is_native_integer)
        || rest.windows(2).any(|pair| {
            option_name(&pair[0]).is_some_and(|name| name.eq_ignore_ascii_case("Alpha"))
                && value_is_native_integer(&pair[1])
        })
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRPSTATS_INTEGER_ALPHA_EXTENSION,
            "grpstats",
        )?;
    }
    if table_input
        && (value_is_native_integer(group)
            || rest.windows(2).any(|pair| {
                option_name(&pair[0]).is_some_and(|name| name.eq_ignore_ascii_case("DataVars"))
                    && value_is_native_integer(&pair[1])
            }))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRPSTATS_INTEGER_SELECTOR_EXTENSION,
            "grpstats",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod integer_compatibility_tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn grouping_metadata_distinguishes_documented_groups_from_runmat_data() {
        assert_eq!(GROUPSUMMARY_EXTENSIONS.len(), 2);
        assert_eq!(GRPSTATS_EXTENSIONS.len(), 4);
        assert_eq!(GROUPSUMMARY_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(GRPSTATS_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            GROUPSUMMARY_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            GROUPSUMMARY_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            GRPSTATS_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            GRPSTATS_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            GRPSTATS_INTEGER_CAPABILITIES[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
    }

    #[test]
    fn grouping_strict_mode_gates_resident_inputs_before_provider_access() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        };
        runmat_accelerate_api::mark_handle_explicit(&handle);
        let resident = Value::GpuTensor(handle.clone());
        let group_summary =
            ensure_groupsummary_extensions(&resident, &Value::from("G"), &Value::from("mean"), &[])
                .expect_err("resident groupsummary must gate");
        assert_eq!(
            group_summary.identifier(),
            GROUPSUMMARY_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        let grpstats = ensure_grpstats_extensions(&resident, &Value::Num(1.0), &[])
            .expect_err("resident grpstats must gate");
        assert_eq!(
            grpstats.identifier(),
            GRPSTATS_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }

    #[test]
    fn grpstats_strict_mode_separates_integer_group_data_selector_and_alpha_roles() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_data = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap(),
        );
        let data_error = ensure_grpstats_extensions(
            &integer_data,
            &Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            &[],
        )
        .expect_err("integer nongrouping data must gate");
        assert_eq!(
            data_error.identifier(),
            GRPSTATS_INTEGER_DATA_EXTENSION.error_identifier
        );

        let table = table_from_columns(
            vec!["G".into(), "X".into(), "I".into()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1]), vec![2, 1])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let selector_error = ensure_grpstats_extensions(
            &table,
            &Value::Int(IntValue::U8(1)),
            &[
                Value::from("DataVars"),
                Value::StringArray(
                    runmat_builtins::StringArray::new(vec!["X".into()], vec![1, 1]).unwrap(),
                ),
            ],
        )
        .expect_err("native integer table selector must gate independently");
        assert_eq!(
            selector_error.identifier(),
            GRPSTATS_INTEGER_SELECTOR_EXTENSION.error_identifier
        );
        ensure_grpstats_extensions(
            &table,
            &Value::Num(1.0),
            &[
                Value::from("DataVars"),
                Value::StringArray(
                    runmat_builtins::StringArray::new(vec!["I".into()], vec![1, 1]).unwrap(),
                ),
            ],
        )
        .expect("double selector selecting an integer data column must remain documented");
        let alpha_error = ensure_grpstats_extensions(
            &Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            &Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            &[Value::from("Alpha"), Value::Int(IntValue::U8(1))],
        )
        .expect_err("integer Alpha must gate");
        assert_eq!(
            alpha_error.identifier(),
            GRPSTATS_INTEGER_ALPHA_EXTENSION.error_identifier
        );
    }

    #[test]
    fn groupsummary_numeric_bins_preserve_wide_integer_edges_and_empty_bins() {
        let base = 1_u64 << 53;
        let table = table_from_columns(
            vec!["G".into(), "X".into()],
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![base, base + 2]), vec![2, 1])
                        .unwrap(),
                ),
                Value::Tensor(Tensor::new(vec![10.0, 30.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let edges = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1, base + 2, base + 3]),
                vec![4, 1],
            )
            .unwrap(),
        );
        let result = futures::executor::block_on(groupsummary_builtin(
            table,
            Value::from("G"),
            edges,
            vec![
                Value::from("mean"),
                Value::from("X"),
                Value::from("IncludeEmptyGroups"),
                Value::Bool(true),
            ],
        ))
        .expect("numeric groupbins");
        let Value::Object(result) = result else {
            panic!("expected table");
        };
        let variables = table_variables(&result).unwrap();
        let Value::Tensor(counts) = variables.fields.get("GroupCount").unwrap() else {
            panic!("expected GroupCount");
        };
        assert_eq!(counts.materialize_f64(), vec![1.0, 0.0, 1.0]);
        let Value::Tensor(means) = variables.fields.get("mean_X").unwrap() else {
            panic!("expected means");
        };
        let means = means.materialize_f64();
        assert_eq!(means[0], 10.0);
        assert!(means[1].is_nan());
        assert_eq!(means[2], 30.0);
    }

    #[test]
    fn groupsummary_array_form_preserves_wide_integer_groups_and_extrema() {
        let base = 9_007_199_254_740_992_u64;
        let data = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base + 3, base + 1, base + 4]),
                vec![3, 1],
            )
            .unwrap(),
        );
        let groups = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![base, base + 1, base]), vec![3, 1])
                .unwrap(),
        );
        let _outputs = crate::output_count::push_output_count(Some(3));
        let result = futures::executor::block_on(groupsummary_builtin(
            data,
            groups,
            Value::from("max"),
            Vec::new(),
        ))
        .expect("array groupsummary");
        let Value::OutputList(values) = result else {
            panic!("expected three array outputs");
        };
        assert!(matches!(
            &values[0],
            Value::Tensor(tensor)
                if tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![base + 4, base + 1]))
        ));
        assert!(matches!(
            &values[1],
            Value::Tensor(tensor)
                if tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![base, base + 1]))
        ));
        assert!(matches!(
            &values[2],
            Value::Tensor(tensor) if tensor.materialize_f64() == vec![2.0, 1.0]
        ));
    }

    #[test]
    fn groupsummary_array_numeric_bins_compare_wide_integer_edges_exactly() {
        let base = 9_007_199_254_740_992_u64;
        let data = Value::Tensor(Tensor::new(vec![10.0, 30.0], vec![2, 1]).unwrap());
        let groups = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![base, base + 2]), vec![2, 1]).unwrap(),
        );
        let edges = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1, base + 2, base + 3]),
                vec![4, 1],
            )
            .unwrap(),
        );
        let _outputs = crate::output_count::push_output_count(Some(3));
        let result = futures::executor::block_on(groupsummary_builtin(
            data,
            groups,
            edges,
            vec![
                Value::from("mean"),
                Value::from("IncludeEmptyGroups"),
                Value::Bool(true),
            ],
        ))
        .expect("array numeric groupbins");
        let Value::OutputList(values) = result else {
            panic!("expected three array outputs");
        };
        assert!(matches!(
            &values[0],
            Value::Tensor(tensor)
                if tensor.materialize_f64().len() == 3
                    && tensor.materialize_f64()[0] == 10.0
                    && tensor.materialize_f64()[1].is_nan()
                    && tensor.materialize_f64()[2] == 30.0
        ));
        assert!(matches!(
            &values[2],
            Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn groupsummary_residency_gate_only_applies_to_explicit_handles() {
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 426,
        };
        let value = Value::GpuTensor(handle.clone());
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        runmat_accelerate_api::mark_handle_automatic(&handle);
        ensure_groupsummary_extensions(&value, &Value::from("G"), &Value::from("mean"), &[])
            .expect("automatic residency remains transparent");
        runmat_accelerate_api::mark_handle_explicit(&handle);
        let error =
            ensure_groupsummary_extensions(&value, &Value::from("G"), &Value::from("mean"), &[])
                .expect_err("explicit residency must gate");
        assert_eq!(
            error.identifier(),
            GROUPSUMMARY_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }
}

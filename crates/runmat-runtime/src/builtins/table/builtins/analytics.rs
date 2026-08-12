use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
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
pub const GROUPSUMMARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = groupsummary(T,groupvars,method,___) with integer grouping variables",
        inputs: &GROUPSUMMARY_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Grouping is exact; output grouping columns preserve class while GroupCount and summaries are function-specific.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = groupsummary(T,groupvars,method,integer_datavars) for implemented methods",
        inputs: &GROUPSUMMARY_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented integer table data uses exact native-class min/max and explicit floating computation for mean, sum, median, and counts. [integer-audit-open] Documented integer groupbins remain unresolved and keep this name in the quantitative audit queue; empty-group expansion, additional named methods, and function handles are general coverage gaps.",
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
    groupsummary_impl(table, groupvars, method, rest)
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
    if crate::value_contains_gpu(table)
        || crate::value_contains_gpu(groupvars)
        || crate::value_contains_gpu(method)
        || rest.iter().any(crate::value_contains_gpu)
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
    Ok(())
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
        assert_eq!(GROUPSUMMARY_INTEGER_CAPABILITIES.len(), 2);
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
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
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
}

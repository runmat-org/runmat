//! MATLAB-compatible array grouping, binning, and grouped-apply builtins.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, IntValue, IntegerStorage, LogicalArray, ObjectInstance, SparseTensor, StringArray,
    Tensor, Value,
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

pub const GROUPING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Debug)]
enum Atom {
    Missing,
    Logical(bool),
    Number(f64),
    Integer(IntValue),
    Text(String),
}

impl Atom {
    fn rank(&self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Logical(_) => 1,
            Self::Number(_) => 2,
            Self::Integer(_) => 3,
            Self::Text(_) => 4,
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
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Integer(a), Self::Integer(b)) => compare_integer_values(a, b),
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
        let mut include_missing = false;
        let mut idx = 0usize;
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(grouping_error(format!(
                    "{context}: name-value options must be provided in pairs"
                )));
            }
            let name = scalar_text(&args[idx], context)?;
            if name.eq_ignore_ascii_case("IncludeMissingGroups") {
                include_missing = bool_scalar(&args[idx + 1], "IncludeMissingGroups")?;
            } else if name.eq_ignore_ascii_case("IncludeEmptyGroups") {
                let include_empty = bool_scalar(&args[idx + 1], "IncludeEmptyGroups")?;
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
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn findgroups_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut args = Vec::with_capacity(rest.len() + 1);
    args.push(gather_if_needed_async(&first).await?);
    for value in rest {
        args.push(gather_if_needed_async(&value).await?);
    }
    let (columns, table_mode) = findgroups_columns(args)?;
    let grouping = build_grouping(&columns)?;
    let outputs = findgroups_outputs(&columns, &grouping, table_mode)?;
    multi_output(outputs)
}

#[runtime_builtin(
    name = "grp2idx",
    category = "array/grouping",
    summary = "Create an index vector from a grouping variable.",
    keywords = "grp2idx,groups,index,categorical,statistics",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn grp2idx_builtin(value: Value) -> BuiltinResult<Value> {
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
    let gn = Value::StringArray(
        StringArray::new(names.clone(), vec![names.len(), 1]).map_err(grouping_error)?,
    );
    let gl = Value::StringArray(
        StringArray::new(names, vec![grouping.keys.len(), 1]).map_err(grouping_error)?,
    );
    multi_output(vec![g, gn, gl])
}

#[runtime_builtin(
    name = "groupcounts",
    category = "array/grouping",
    summary = "Count the number of elements in each group.",
    keywords = "groupcounts,groups,count,table,categorical",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn groupcounts_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let first = gather_if_needed_async(&first).await?;
    let rest = gather_values(rest).await?;
    if let Value::Object(object) = first.clone() {
        if is_tabular_object(&object) {
            let (selector_args, option_args) = split_option_tail(rest)?;
            return groupcounts_table(object, selector_args, option_args);
        }
    }
    let (data_args, option_args) = split_option_tail(rest)?;
    let options = GroupOptions::parse(&option_args, "groupcounts")?;
    let columns = if matches!(first, Value::Cell(_)) {
        columns_from_group_value("A", first, true)?
    } else {
        let mut args = vec![first];
        args.extend(data_args);
        columns_from_group_args(args)?
    };
    let grouping = build_grouping_with_options(&columns, options)?;
    let counts = grouping
        .row_groups
        .iter()
        .map(|rows| rows.len() as f64)
        .collect::<Vec<_>>();
    let b =
        Value::Tensor(Tensor::new(counts, vec![grouping.keys.len(), 1]).map_err(grouping_error)?);
    let bg = group_label_outputs(&columns, &grouping)?;
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
    let mut outputs = vec![b];
    outputs.extend(bg);
    outputs.push(bp);
    multi_output(outputs)
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
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn accumarray_builtin(
    subs: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
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
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
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
    discretize_impl(x, edges_or_n, rest)
}

#[runtime_builtin(
    name = "combinations",
    category = "array/grouping",
    summary = "Generate all element combinations of arrays.",
    keywords = "combinations,cartesian,table,combinatorics",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::GROUPING_DESCRIPTOR),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn combinations_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let first = gather_if_needed_async(&first).await?;
    let rest = gather_values(rest).await?;
    combinations_impl(first, rest)
}

fn findgroups_columns(args: Vec<Value>) -> BuiltinResult<(Vec<GroupColumn>, bool)> {
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
                columns.push(GroupColumn {
                    rows: value_row_count(&value)?,
                    name,
                    value,
                });
            }
            return Ok((columns, true));
        }
    }
    Ok((columns_from_group_args(args)?, false))
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
        let rows = tensor.data.len();
        let value = if let Some(storage) = tensor.integer_storage() {
            Tensor::new_integer(storage.clone(), vec![rows, 1]).map_err(grouping_error)?
        } else {
            Tensor::new_with_dtype(tensor.data, vec![rows, 1], tensor.dtype)
                .map_err(grouping_error)?
        };
        return Ok(vec![GroupColumn {
            name: base_name.to_string(),
            rows,
            value: Value::Tensor(value),
        }]);
    }
    let mut out = Vec::with_capacity(tensor.cols());
    for col in 0..tensor.cols() {
        let value = if let Some(storage) = tensor.integer_storage() {
            let values = (0..tensor.rows())
                .map(|row| {
                    storage
                        .value_at(row + col * tensor.rows())
                        .expect("integer tensor storage matches tensor shape")
                })
                .collect();
            Tensor::new_integer(
                storage
                    .from_exact_values_like(values)
                    .map_err(grouping_error)?,
                vec![tensor.rows(), 1],
            )
            .map_err(grouping_error)?
        } else {
            let mut data = Vec::with_capacity(tensor.rows());
            for row in 0..tensor.rows() {
                data.push(tensor.get2(row, col).map_err(grouping_error)?);
            }
            Tensor::new_with_dtype(data, vec![tensor.rows(), 1], tensor.dtype)
                .map_err(grouping_error)?
        };
        out.push(GroupColumn {
            name: format!("{base_name}{col_plus}", col_plus = col + 1),
            rows: tensor.rows(),
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
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(row)
                    .map(Atom::Integer)
                    .ok_or_else(|| grouping_error("grouping: numeric row out of bounds"));
            }
            let value = *tensor
                .data
                .get(row)
                .ok_or_else(|| grouping_error("grouping: numeric row out of bounds"))?;
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(value))
            }
        }
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
            let value = serials.data.get(row).copied().unwrap_or(f64::NAN);
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(value))
            }
        }
        Value::Object(object) if object.is_class("duration") => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let value = tensor.data.get(row).copied().unwrap_or(f64::NAN);
            if value.is_nan() {
                Ok(Atom::Missing)
            } else {
                Ok(Atom::Number(value))
            }
        }
        other if row == 0 => scalar_atom(other),
        _ => Ok(Atom::Missing),
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
        Value::CharArray(chars) if chars.rows == 1 => Ok(Atom::Text(chars.data.iter().collect())),
        other => Ok(Atom::Text(format!("{other}"))),
    }
}

fn findgroups_outputs(
    columns: &[GroupColumn],
    grouping: &Grouping,
    table_mode: bool,
) -> BuiltinResult<Vec<Value>> {
    let g = Value::Tensor(
        Tensor::new(grouping.ids.clone(), vec![grouping.ids.len(), 1]).map_err(grouping_error)?,
    );
    let mut outputs = vec![g];
    if table_mode {
        let mut names = Vec::with_capacity(columns.len());
        let mut values = Vec::with_capacity(columns.len());
        for column in columns {
            names.push(column.name.clone());
            values.push(select_rows(&column.value, &grouping.first_rows)?);
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
        .map(|column| select_rows(&column.value, &grouping.first_rows))
        .collect()
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
            "groupcounts: table input accepts one grouping variable selector before options",
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
    let data_values = accumarray_data_values(data, rows)?;
    let output_shape = if let Some(size_value) = rest.first() {
        if is_empty_value(size_value) {
            infer_accumarray_shape(&index_rows)
        } else {
            parse_size_vector(size_value, "accumarray")?
        }
    } else {
        infer_accumarray_shape(&index_rows)
    };
    let fun = rest.get(1).filter(|value| !is_empty_value(value)).cloned();
    let fill = rest.get(2).filter(|value| !is_empty_value(value)).cloned();
    let issparse = rest
        .get(3)
        .map(|value| bool_scalar(value, "accumarray issparse"))
        .transpose()?
        .unwrap_or(false);
    let output_len = checked_element_count(&output_shape, "accumarray")?;
    if output_len > MAX_MATERIALIZED_ELEMENTS {
        return Err(too_large_error("accumarray: output is too large"));
    }
    let mut buckets: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (idx, subs) in index_rows.iter().enumerate() {
        let lin = subscript_to_linear(subs, &output_shape)?;
        buckets.entry(lin).or_default().push(data_values[idx]);
    }
    let fill_num = match fill.as_ref() {
        None => 0.0,
        Some(value) => numeric_scalar(value, "accumarray fill value")?,
    };
    let mut data_out = vec![fill_num; output_len];
    if let Some(fun) = fun {
        let mut cell_out = vec![Value::Num(fill_num); data_out.len()];
        let mut all_numeric = true;
        for (lin, values) in buckets {
            let result = apply_accumarray_callback(fun.clone(), values).await?;
            if let Some(num) = value_as_numeric_scalar(&result) {
                data_out[lin] = num;
                cell_out[lin] = Value::Num(num);
            } else {
                all_numeric = false;
                cell_out[lin] = result;
            }
        }
        if all_numeric {
            return accumarray_numeric_output(data_out, output_shape, issparse);
        }
        if issparse {
            return Err(grouping_error(
                "accumarray: sparse output requires numeric scalar group results",
            ));
        }
        let (rows, cols) = shape_to_rows_cols(&output_shape)?;
        return CellArray::new(cell_out, rows, cols)
            .map(Value::Cell)
            .map_err(grouping_error);
    }
    for (lin, values) in buckets {
        data_out[lin] = values.iter().sum();
    }
    accumarray_numeric_output(data_out, output_shape, issparse)
}

fn accumarray_subscripts(subs: Value) -> BuiltinResult<Vec<Vec<usize>>> {
    match subs {
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
                tensor
                    .data
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
                Ok(tensor.data)
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

async fn apply_accumarray_callback(func: Value, values: Vec<f64>) -> BuiltinResult<Value> {
    if let Some(name) = function_name(&func) {
        match name.to_ascii_lowercase().as_str() {
            "sum" => return Ok(Value::Num(values.iter().sum())),
            "mean" => {
                return Ok(Value::Num(if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }))
            }
            "numel" | "length" => return Ok(Value::Num(values.len() as f64)),
            "min" => return Ok(Value::Num(values.into_iter().fold(f64::INFINITY, f64::min))),
            "max" => {
                return Ok(Value::Num(
                    values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                ))
            }
            _ => {}
        }
    }
    let arg =
        Value::Tensor(Tensor::new(values.clone(), vec![values.len(), 1]).map_err(grouping_error)?);
    call_feval_async_with_outputs(func, &[arg], 1)
        .await
        .map_err(|err| callback_error("accumarray: callback failed", Some(err)))
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

fn discretize_impl(x: Value, edges_or_n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let values = numeric_values(&x, "discretize X")?;
    let shape = value_shape(&x);
    let (edges, labels, included_right) = parse_discretize_args(&values, edges_or_n, rest)?;
    if edges.len() < 2 {
        return Err(grouping_error(
            "discretize: at least two bin edges are required",
        ));
    }
    let bins = values
        .iter()
        .map(|value| discretize_one(*value, &edges, included_right))
        .collect::<Vec<_>>();
    if let Some(labels) = labels {
        let data = bins
            .iter()
            .map(|bin| match bin {
                Some(idx) => labels.get(*idx - 1).cloned().unwrap_or_default(),
                None => String::new(),
            })
            .collect::<Vec<_>>();
        return StringArray::new(data, shape)
            .map(Value::StringArray)
            .map_err(grouping_error);
    }
    Tensor::new(
        bins.into_iter()
            .map(|bin| bin.map(|idx| idx as f64).unwrap_or(f64::NAN))
            .collect(),
        shape,
    )
    .map(Value::Tensor)
    .map_err(grouping_error)
}

fn parse_discretize_args(
    values: &[f64],
    edges_or_n: Value,
    rest: Vec<Value>,
) -> BuiltinResult<(Vec<f64>, Option<Vec<String>>, bool)> {
    let mut labels = None;
    let mut included_right = false;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if !is_option_name(first) {
            labels = Some(string_list(first)?);
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
        Value::Num(n) if is_positive_integer_f64(n) => equal_width_edges(values, n as usize)?,
        Value::Int(n) => match n.try_to_usize().filter(|bins| *bins > 0) {
            Some(bins) => equal_width_edges(values, bins)?,
            None => numeric_values(&Value::Int(n), "discretize edges")?,
        },
        other => numeric_values(&other, "discretize edges")?,
    };
    if let Some(labels) = &labels {
        if labels.len() != edges.len().saturating_sub(1) {
            return Err(grouping_error(
                "discretize: number of labels must match number of bins",
            ));
        }
    }
    Ok((edges, labels, included_right))
}

fn discretize_one(value: f64, edges: &[f64], included_right: bool) -> Option<usize> {
    if value.is_nan() {
        return None;
    }
    for bin in 0..edges.len() - 1 {
        let lower = edges[bin];
        let upper = edges[bin + 1];
        let hit = if included_right {
            (value > lower || (bin == 0 && value == lower)) && value <= upper
        } else {
            value >= lower && (value < upper || (bin == edges.len() - 2 && value == upper))
        };
        if hit {
            return Some(bin + 1);
        }
    }
    None
}

fn equal_width_edges(values: &[f64], bins: usize) -> BuiltinResult<Vec<f64>> {
    if bins == 0 {
        return Err(grouping_error(
            "discretize: number of bins must be positive",
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
    let mut options_start = rest.len();
    let mut idx = 0usize;
    while idx < rest.len() {
        if is_option_name(&rest[idx]) {
            options_start = idx;
            break;
        }
        values.push(rest[idx].clone());
        idx += 1;
    }
    let mut names = (0..values.len())
        .map(|idx| format!("Var{}", idx + 1))
        .collect::<Vec<_>>();
    if options_start < rest.len() {
        let mut opt = options_start;
        while opt < rest.len() {
            if opt + 1 >= rest.len() {
                return Err(grouping_error(
                    "combinations: name-value options must be provided in pairs",
                ));
            }
            let name = scalar_text(&rest[opt], "combinations option")?;
            if name.eq_ignore_ascii_case("VariableNames") {
                names = string_list(&rest[opt + 1])?;
                if names.len() != values.len() {
                    return Err(grouping_error(
                        "combinations: VariableNames length must match input count",
                    ));
                }
            } else {
                return Err(grouping_error(format!(
                    "combinations: unsupported option '{name}'"
                )));
            }
            opt += 2;
        }
    }
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
            tensor
                .data
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
            Ok(tensor.data.into_iter().map(Value::Num).collect())
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
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(integer_storage_to_f64_vec(storage));
            }
            Ok(tensor.data.clone())
        }
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| if *flag != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map(|tensor| tensor.data)
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

fn parse_size_vector(value: &Value, context: &str) -> BuiltinResult<Vec<usize>> {
    let mut dims = match value {
        Value::Int(value) => vec![nonnegative_integer_value(value, context)?],
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| nonnegative_integer_value(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            } else {
                numeric_values(value, context)?
                    .into_iter()
                    .map(|value| nonnegative_integer(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            }
        }
        _ => numeric_values(value, context)?
            .into_iter()
            .map(|value| nonnegative_integer(value, context))
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    if dims.is_empty() {
        return Err(grouping_error(format!(
            "{context}: size vector must not be empty"
        )));
    }
    if dims.len() == 1 {
        dims.push(1);
    }
    Ok(dims)
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

fn bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Int(value) => Ok(!value.is_zero()),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(grouping_error(format!(
            "{context}: expected logical scalar, got {other:?}"
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

fn function_name(value: &Value) -> Option<&str> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name)
        | Value::BoundFunctionHandle { name, .. }
        | Value::String(name) => Some(name.as_str()),
        _ => None,
    }
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

fn nonnegative_integer(value: f64, context: &str) -> BuiltinResult<usize> {
    if let Some(value) = nonnegative_platform_usize(value) {
        Ok(value)
    } else {
        Err(grouping_error(format!(
            "{context}: expected nonnegative integer"
        )))
    }
}

fn nonnegative_integer_value(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| grouping_error(format!("{context}: expected nonnegative integer")))
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
    use runmat_builtins::{IntValue, IntegerStorage};

    #[test]
    fn discretize_typed_bin_count_preserves_exact_unsigned_values() {
        let values = [0.0, 1.0];
        let (edges, _, _) =
            parse_discretize_args(&values, Value::Int(IntValue::U16(2)), Vec::new()).unwrap();
        assert_eq!(edges, vec![0.0, 0.5, 1.0]);

        let (edges, _, _) =
            parse_discretize_args(&values, Value::Int(IntValue::U8(1)), Vec::new()).unwrap();
        assert_eq!(edges, vec![0.0, 1.0]);
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
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![7.0, 4.0, 2.0, 8.0]),
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
                assert_eq!(tensor.data, vec![5.0, 0.0, 0.0, 6.0, 0.0, 7.0, 3.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn accumarray_accepts_exact_integer_subscripts_data_and_size_vectors() {
        let mut subs_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 3, 4, 2]), vec![4, 1]).unwrap();
        subs_tensor.data = vec![0.0; 4];
        let subs = Value::Tensor(subs_tensor);
        let mut data_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![10, 20, 30, 40]), vec![4, 1]).unwrap();
        data_tensor.data = vec![0.0; 4];
        let data = Value::Tensor(data_tensor);
        let mut size_tensor =
            Tensor::new_integer(IntegerStorage::U8(vec![4, 1]), vec![1, 2]).unwrap();
        size_tensor.data = vec![0.0; 2];
        let size = Value::Tensor(size_tensor);

        let out = block_on(accumarray_builtin(subs, data, vec![size])).unwrap();

        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 1]);
                assert_eq!(tensor.data, vec![10.0, 40.0, 20.0, 30.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn grouping_numeric_scalar_reads_typed_integer_storage_exactly() {
        let mut tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1]).unwrap();
        tensor.data.clear();

        assert_eq!(
            value_as_numeric_scalar(&Value::Tensor(tensor)),
            Some(2026.0)
        );
    }

    #[test]
    fn grouping_name_selector_reads_typed_integer_storage_exactly() {
        let mut selector =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 1]), vec![1, 2]).unwrap();
        selector.data = vec![0.0, 0.0];

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
        let mut tensor = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
        tensor.data = vec![1.0];

        assert_eq!(
            accumarray_subscripts(Value::Tensor(tensor)).unwrap(),
            Vec::<Vec<usize>>::new()
        );
    }

    #[test]
    fn accumarray_data_values_reads_typed_integer_storage_exactly() {
        let mut tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).unwrap();
        tensor.data.clear();

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
        assert!(err.message.contains("expected nonnegative integer"));
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
                assert_eq!(tensor.data[0], 1.0);
                assert_eq!(tensor.data[1], 1.0);
                assert_eq!(tensor.data[2], 2.0);
                assert!(tensor.data[3].is_nan());
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
                assert_eq!(tensor.data[0], 2.0);
                assert_eq!(tensor.data[1], 1.0);
                assert_eq!(tensor.data[2], 2.0);
                assert!(tensor.data[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        let counted = block_on(groupcounts_builtin(groups.clone(), Vec::new())).unwrap();
        match counted {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        let indexed = block_on(grp2idx_builtin(groups)).unwrap();
        match indexed {
            Value::Tensor(tensor) => assert!(tensor.data[3].is_nan()),
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
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
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
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let counted = block_on(groupcounts_builtin(groups, Vec::new())).unwrap();
        match counted {
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
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
        let mut first =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 9]), vec![1, 2]).unwrap();
        first.data = Vec::new();
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
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 6.0]),
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
            Value::Tensor(tensor) => assert_eq!(tensor.data, vec![6.0, 4.0]),
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

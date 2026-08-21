use super::*;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::stats::summary::distribution_math::student_t_inv;
use runmat_value::NumericScalar;

pub(in crate::builtins::table) fn grpstats_impl(
    value: Value,
    group: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if table_object(&value).is_some() {
        return grpstats_table_impl(value, group, rest);
    }
    grpstats_matrix_impl(value, group, rest)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GrpStat {
    Mean,
    Sem,
    Std,
    Var,
    Min,
    Max,
    Range,
    MeanCi,
    PredCi,
    Numel,
    GName,
}

impl GrpStat {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "sem" => Ok(Self::Sem),
            "std" => Ok(Self::Std),
            "var" => Ok(Self::Var),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "range" => Ok(Self::Range),
            "meanci" => Ok(Self::MeanCi),
            "predci" => Ok(Self::PredCi),
            "numel" | "count" | "groupcount" => Ok(Self::Numel),
            "gname" => Ok(Self::GName),
            other => Err(invalid_argument(format!(
                "grpstats: unsupported summary statistic '{other}'"
            ))),
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Sem => "sem",
            Self::Std => "std",
            Self::Var => "var",
            Self::Min => "min",
            Self::Max => "max",
            Self::Range => "range",
            Self::MeanCi => "meanci",
            Self::PredCi => "predci",
            Self::Numel => "numel",
            Self::GName => "gname",
        }
    }

    fn is_table_data_stat(self) -> bool {
        !matches!(self, Self::GName | Self::Numel)
    }
}

#[derive(Clone, Debug)]
struct GrpStatsOptions {
    stats: Vec<GrpStat>,
    alpha: f64,
    data_vars: Option<Vec<String>>,
    var_names: Option<Vec<String>>,
}

impl Default for GrpStatsOptions {
    fn default() -> Self {
        Self {
            stats: vec![GrpStat::Mean],
            alpha: 0.05,
            data_vars: None,
            var_names: None,
        }
    }
}

fn parse_grpstats_options(
    rest: Vec<Value>,
    table_context: Option<(&ObjectInstance, &[String])>,
) -> BuiltinResult<GrpStatsOptions> {
    let mut options = GrpStatsOptions::default();
    let mut idx = 0;
    if let Some(first) = rest.first() {
        if let Some(alpha) = scalar_number(first) {
            options.alpha = validate_alpha(alpha)?;
            idx = 1;
        } else if !is_option_name(first, "Alpha")
            && !is_option_name(first, "DataVars")
            && !is_option_name(first, "VarNames")
        {
            options.stats = parse_grpstats_stat_list(first)?;
            idx = 1;
        }
    }
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "grpstats option name")?;
        idx += 1;
        if idx >= rest.len() {
            return Err(invalid_argument(format!(
                "grpstats: option '{name}' requires a value"
            )));
        }
        let option_value = &rest[idx];
        idx += 1;
        match name.to_ascii_lowercase().as_str() {
            "alpha" => {
                let alpha = scalar_number(option_value)
                    .ok_or_else(|| invalid_argument("grpstats: Alpha must be numeric"))?;
                options.alpha = validate_alpha(alpha)?;
            }
            "datavars" => {
                let Some((object, names)) = table_context else {
                    return Err(invalid_argument(
                        "grpstats: DataVars is only valid for table input",
                    ));
                };
                options.data_vars = Some(parse_variable_selector_for_object(
                    Some(option_value),
                    object,
                    names,
                )?);
            }
            "varnames" => {
                options.var_names = Some(string_list(option_value)?);
            }
            other => {
                return Err(invalid_argument(format!(
                    "grpstats: unsupported option '{other}'"
                )))
            }
        }
    }
    if options.stats.is_empty() {
        return Err(invalid_argument(
            "grpstats: summary statistic list must not be empty",
        ));
    }
    Ok(options)
}

fn parse_grpstats_stat_list(value: &Value) -> BuiltinResult<Vec<GrpStat>> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name)
        | Value::BoundFunctionHandle { name, .. } => Ok(vec![GrpStat::parse(name)?]),
        Value::Closure(_) => Err(invalid_argument(
            "grpstats: custom function-handle summary statistics are not supported yet",
        )),
        Value::Cell(cell) => {
            let mut stats = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                stats.extend(parse_grpstats_stat_list(value)?);
            }
            Ok(stats)
        }
        _ => string_list(value)?
            .iter()
            .map(|name| GrpStat::parse(name))
            .collect(),
    }
}

fn validate_alpha(alpha: f64) -> BuiltinResult<f64> {
    if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        Ok(alpha)
    } else {
        Err(invalid_argument(
            "grpstats: Alpha must be a finite scalar in the open interval (0,1)",
        ))
    }
}

fn is_option_name(value: &Value, expected: &str) -> bool {
    scalar_text(value, "grpstats option")
        .map(|text| text.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn grpstats_table_impl(table: Value, groupvars: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let object = into_table_object(table, "grpstats")?;
    let names = table_variable_names_from_object(&object)?;
    let group_names = if option_value_is_empty(&groupvars) {
        Vec::new()
    } else {
        parse_variable_selector_for_object(Some(&groupvars), &object, &names)?
    };
    let options = parse_grpstats_options(rest, Some((&object, &names)))?;
    let variables = table_variables(&object)?;
    let height = table_height(&object)?;
    let data_names = match options.data_vars.clone() {
        Some(data_vars) => data_vars,
        None => names
            .iter()
            .filter(|name| !group_names.contains(name))
            .filter(|name| {
                variables
                    .fields
                    .get(*name)
                    .map(is_numeric_or_logical_column)
                    .unwrap_or(false)
            })
            .cloned()
            .collect(),
    };
    for name in &data_names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("grpstats: missing data variable '{name}'")))?;
        if !is_numeric_or_logical_column(value) {
            return Err(invalid_variable(format!(
                "grpstats: data variable '{name}' must be numeric or logical"
            )));
        }
    }
    let groups = table_grpstats_groups(&variables, &group_names, height);
    let group_rows = groups
        .rows
        .iter()
        .filter_map(|rows| rows.first().copied())
        .collect::<Vec<_>>();
    let mut out_names = Vec::new();
    let mut out_columns = Vec::new();
    for name in &group_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            invalid_variable(format!("grpstats: missing group variable '{name}'"))
        })?;
        out_names.push(name.clone());
        out_columns.push(select_rows(value, &group_rows)?);
    }
    out_names.push("GroupCount".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            groups.rows.iter().map(|rows| rows.len() as f64).collect(),
            vec![groups.len(), 1],
        )
        .map_err(invalid_variable)?,
    ));
    let table_stats = options
        .stats
        .iter()
        .copied()
        .filter(|stat| stat.is_table_data_stat())
        .collect::<Vec<_>>();
    for name in &data_names {
        for stat in table_stats.iter().copied() {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("grpstats: missing data variable '{name}'"))
            })?;
            if let Value::Tensor(tensor) = value {
                if tensor.integer_storage().is_some() {
                    out_names.push(format!("{}_{}", stat.prefix(), name));
                    out_columns.push(summarize_integer_table_groups(
                        tensor,
                        groups.rows.iter(),
                        stat,
                        options.alpha,
                    )?);
                    continue;
                }
            }
            let matrix = matrix_from_table_column(value)?;
            let summary =
                summarize_matrix_groups(&matrix, groups.rows.iter(), stat, options.alpha)?;
            let shape = if summary.depth == 1 {
                vec![groups.len(), 1]
            } else {
                vec![groups.len(), summary.depth]
            };
            out_names.push(format!("{}_{}", stat.prefix(), name));
            out_columns.push(Value::Tensor(
                Tensor::new(summary.data, shape).map_err(invalid_variable)?,
            ));
        }
    }
    if let Some(var_names) = options.var_names {
        if var_names.len() != out_names.len() {
            return Err(invalid_argument(format!(
                "grpstats: VarNames must contain {} names",
                out_names.len()
            )));
        }
        out_names = var_names;
    }
    table_from_columns(out_names, out_columns)
}

fn summarize_integer_table_groups<'a>(
    tensor: &Tensor,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    stat: GrpStat,
    alpha: f64,
) -> BuiltinResult<Value> {
    let storage = tensor
        .integer_storage()
        .ok_or_else(|| invalid_variable("grpstats: expected integer table storage"))?;
    let groups = groups.collect::<Vec<_>>();
    if matches!(stat, GrpStat::Min | GrpStat::Max) {
        let mut extrema = Vec::with_capacity(groups.len());
        for rows in &groups {
            let mut values = rows.iter().map(|row| {
                storage
                    .value_at(*row)
                    .ok_or_else(|| invalid_index("grpstats: integer row out of bounds"))
            });
            let mut selected = values.next().transpose()?.ok_or_else(|| {
                invalid_argument("grpstats: observed integer groups cannot be empty")
            })?;
            for value in values {
                let value = value?;
                let ordering = compare_integer_values(&value, &selected);
                if (stat == GrpStat::Min && ordering == Ordering::Less)
                    || (stat == GrpStat::Max && ordering == Ordering::Greater)
                {
                    selected = value;
                }
            }
            extrema.push(selected);
        }
        let output = storage
            .from_exact_values_like(extrema)
            .map_err(invalid_variable)?;
        return Tensor::new_integer(output, vec![groups.len(), 1])
            .map(Value::Tensor)
            .map_err(invalid_variable);
    }

    let mut data = vec![f64::NAN; storage.len()];
    for row in groups.iter().flat_map(|rows| rows.iter().copied()) {
        let value = storage
            .value_at(row)
            .ok_or_else(|| invalid_index("grpstats: integer row out of bounds"))?;
        if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&value) {
            return Err(invalid_argument(
                "grpstats: integer table data must be exactly representable as double for floating summary statistics",
            ));
        }
        data[row] = value.to_f64();
    }
    let matrix = NumericMatrix {
        data,
        rows: tensor.rows(),
        cols: 1,
    };
    let summary = summarize_matrix_groups(&matrix, groups.into_iter(), stat, alpha)?;
    let shape = if summary.depth == 1 {
        vec![summary.data.len(), 1]
    } else {
        vec![summary.data.len() / summary.depth, summary.depth]
    };
    Tensor::new(summary.data, shape)
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

fn grpstats_matrix_impl(value: Value, group: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let options = parse_grpstats_options(rest, None)?;
    let matrix = numeric_matrix(value, "grpstats")?;
    let groups = matrix_groups(&group, matrix.rows)?;
    let mut outputs = Vec::with_capacity(options.stats.len());
    for stat in &options.stats {
        outputs.push(match stat {
            GrpStat::GName => group_names_value(groups.keys.iter()),
            _ => {
                let summary =
                    summarize_matrix_groups(&matrix, groups.rows.iter(), *stat, options.alpha)?;
                Value::Tensor(summary.into_tensor(groups.len())?)
            }
        });
    }
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count != outputs.len() {
            return Err(invalid_argument(
                "grpstats: number of outputs must match requested statistics",
            ));
        }
        return Ok(Value::OutputList(outputs));
    }
    Ok(outputs
        .into_iter()
        .next()
        .unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0]))))
}

#[derive(Clone, Debug)]
struct NumericMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

fn numeric_matrix(value: Value, context: &str) -> BuiltinResult<NumericMatrix> {
    match value {
        Value::Tensor(tensor) => {
            // Typed tensors keep their exact values and authoritative element
            // count outside the floating compatibility mirror.
            let len = tensor_utils::tensor_element_len(&tensor);
            let mut rows = tensor.rows();
            let mut cols = tensor.cols();
            if rows == 1 && len > 1 {
                rows = len;
                cols = 1;
            }
            let data = checked_tensor_values_f64(&tensor, context)?;
            Ok(NumericMatrix { data, rows, cols })
        }
        Value::LogicalArray(array) => {
            let mut rows = array.shape.first().copied().unwrap_or(array.data.len());
            let mut cols = array.shape.get(1).copied().unwrap_or(1);
            if rows == 1 && array.data.len() > 1 {
                rows = array.data.len();
                cols = 1;
            }
            Ok(NumericMatrix {
                data: array
                    .data
                    .iter()
                    .map(|value| (*value != 0) as u8 as f64)
                    .collect(),
                rows,
                cols,
            })
        }
        other => Err(invalid_argument(format!(
            "{context}: X must be a numeric or logical matrix, got {other:?}"
        ))),
    }
}

fn checked_tensor_values_f64(tensor: &Tensor, context: &str) -> BuiltinResult<Vec<f64>> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .exact_values()
            .into_iter()
            .map(|value| {
                if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&value) {
                    Ok(value.to_f64())
                } else {
                    Err(invalid_argument(format!(
                        "{context}: integer data must be exactly representable as double for floating summary statistics"
                    )))
                }
            })
            .collect();
    }
    Ok(tensor_utils::tensor_values_f64(tensor))
}

fn matrix_from_table_column(value: &Value) -> BuiltinResult<NumericMatrix> {
    match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => Ok(NumericMatrix {
            data: tensor_utils::tensor_values_f64(tensor),
            rows: tensor.rows(),
            cols: 1,
        }),
        Value::LogicalArray(array) if array.shape.get(1).copied().unwrap_or(1) == 1 => {
            Ok(NumericMatrix {
                data: array
                    .data
                    .iter()
                    .map(|value| (*value != 0) as u8 as f64)
                    .collect(),
                rows: array.shape.first().copied().unwrap_or(array.data.len()),
                cols: 1,
            })
        }
        _ => Err(invalid_variable(
            "grpstats: table data variables must be numeric or logical column vectors",
        )),
    }
}

fn is_numeric_or_logical_column(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.cols() == 1,
        Value::LogicalArray(array) => array.shape.get(1).copied().unwrap_or(1) == 1,
        _ => false,
    }
}

#[derive(Clone, Debug)]
struct GroupedRows {
    keys: Vec<Vec<GroupAtom>>,
    rows: Vec<Vec<usize>>,
}

impl GroupedRows {
    fn len(&self) -> usize {
        self.rows.len()
    }
}

fn table_grpstats_groups(
    variables: &StructValue,
    group_names: &[String],
    height: usize,
) -> GroupedRows {
    if group_names.is_empty() {
        return GroupedRows {
            keys: vec![Vec::new()],
            rows: vec![(0..height).collect()],
        };
    }
    let mut row_keys = Vec::new();
    for row in 0..height {
        let mut key = Vec::with_capacity(group_names.len());
        let mut missing = false;
        for name in group_names {
            let atom = variables
                .fields
                .get(name)
                .map(|value| cell_group_atom(value, row))
                .unwrap_or(GroupAtom::Missing);
            if group_atom_is_missing(&atom) {
                missing = true;
                break;
            }
            key.push(atom);
        }
        if !missing {
            row_keys.push((row, key));
        }
    }
    grouped_rows_from_row_keys(row_keys)
}

fn matrix_groups(group: &Value, rows: usize) -> BuiltinResult<GroupedRows> {
    let key_columns = grouping_columns(group, rows)?;
    if key_columns.is_empty() {
        return Ok(GroupedRows {
            keys: vec![Vec::new()],
            rows: vec![(0..rows).collect()],
        });
    }
    let mut row_keys = Vec::new();
    for row in 0..rows {
        let mut key = Vec::with_capacity(key_columns.len());
        let mut missing = false;
        for column in &key_columns {
            let atom = column[row].clone();
            if group_atom_is_missing(&atom) {
                missing = true;
                break;
            }
            key.push(atom);
        }
        if !missing {
            row_keys.push((row, key));
        }
    }
    Ok(grouped_rows_from_row_keys(row_keys))
}

fn grouped_rows_from_row_keys(row_keys: Vec<(usize, Vec<GroupAtom>)>) -> GroupedRows {
    let preserve_first_seen = row_keys
        .iter()
        .any(|(_, key)| key.iter().any(|atom| matches!(atom, GroupAtom::Text(_))));
    let mut first_seen = Vec::<Vec<GroupAtom>>::new();
    let mut buckets = BTreeMap::<Vec<GroupAtom>, Vec<usize>>::new();
    for (row, key) in row_keys {
        if !buckets.contains_key(&key) {
            first_seen.push(key.clone());
        }
        buckets.entry(key).or_default().push(row);
    }
    let keys = if preserve_first_seen {
        first_seen
    } else {
        buckets.keys().cloned().collect()
    };
    let rows = keys
        .iter()
        .map(|key| buckets.get(key).cloned().unwrap_or_default())
        .collect();
    GroupedRows { keys, rows }
}

fn grouping_columns(group: &Value, rows: usize) -> BuiltinResult<Vec<Vec<GroupAtom>>> {
    if option_value_is_empty(group) {
        return Ok(Vec::new());
    }
    if let Value::Cell(cell) = group {
        let treats_as_grouping_list = cell.data.len() > 1
            && cell
                .data
                .iter()
                .all(|value| grouping_vector_len(value).is_some_and(|len| len == rows));
        if treats_as_grouping_list {
            return cell
                .data
                .iter()
                .map(|value| grouping_atoms(value, rows))
                .collect();
        }
    }
    Ok(vec![grouping_atoms(group, rows)?])
}

fn grouping_vector_len(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor_utils::tensor_element_len(tensor)),
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::StringArray(array) => Some(array.data.len()),
        Value::CharArray(array) => Some(array.rows),
        Value::Cell(cell) => Some(cell.data.len()),
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => value_row_count(value).ok(),
        _ => None,
    }
}

fn grouping_atoms(value: &Value, rows: usize) -> BuiltinResult<Vec<GroupAtom>> {
    let len = grouping_vector_len(value)
        .ok_or_else(|| invalid_argument("grpstats: unsupported grouping variable"))?;
    if len != rows {
        return Err(invalid_argument(format!(
            "grpstats: grouping variable length {len} does not match data rows {rows}"
        )));
    }
    match value {
        Value::Tensor(tensor) => Ok((0..tensor.len())
            .map(|index| {
                match tensor
                    .numeric_value_at(index)
                    .expect("validated grouping tensor storage")
                {
                    NumericScalar::F64(value) => number_group_atom(value),
                    NumericScalar::F32(value) => number_group_atom(f64::from(value)),
                    value => GroupAtom::Integer(
                        value
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                    ),
                }
            })
            .collect()),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| GroupAtom::Logical(*value != 0))
            .collect()),
        Value::StringArray(array) => Ok(array.data.iter().cloned().map(text_group_atom).collect()),
        Value::CharArray(array) => Ok((0..array.rows)
            .map(|row| {
                let start = row * array.cols;
                GroupAtom::Text(
                    array.data[start..start + array.cols]
                        .iter()
                        .collect::<String>()
                        .trim()
                        .to_string(),
                )
            })
            .collect()),
        Value::Cell(cell) => Ok(cell.data.iter().map(cell_atom).collect()),
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => Ok((0..rows)
            .map(|row| {
                categorical_label_at(obj, row)
                    .map(GroupAtom::Text)
                    .unwrap_or(GroupAtom::Missing)
            })
            .collect()),
        _ => Err(invalid_argument("grpstats: unsupported grouping variable")),
    }
}

fn cell_atom(value: &Value) -> GroupAtom {
    match value {
        Value::Num(value) => number_group_atom(*value),
        Value::Int(value) => GroupAtom::Integer(value.clone()),
        Value::Bool(value) => GroupAtom::Logical(*value),
        Value::String(value) => text_group_atom(value.clone()),
        Value::CharArray(array) if array.rows == 1 => {
            text_group_atom(array.data.iter().collect::<String>().trim().to_string())
        }
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .map(GroupAtom::Integer)
            .unwrap_or_else(|| number_group_atom(tensor_utils::tensor_value_f64(tensor, 0))),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            GroupAtom::Logical(array.data[0] != 0)
        }
        other => GroupAtom::Text(cell_key_string(other, 0)),
    }
}

#[derive(Clone, Debug)]
struct GroupSummary {
    data: Vec<f64>,
    cols: usize,
    depth: usize,
}

impl GroupSummary {
    fn into_tensor(self, groups: usize) -> BuiltinResult<Tensor> {
        let shape = if self.depth == 1 {
            vec![groups, self.cols]
        } else {
            vec![groups, self.cols, self.depth]
        };
        Tensor::new(self.data, shape).map_err(invalid_variable)
    }
}

fn summarize_matrix_groups<'a>(
    matrix: &NumericMatrix,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    stat: GrpStat,
    alpha: f64,
) -> BuiltinResult<GroupSummary> {
    let groups = groups.collect::<Vec<_>>();
    let depth = match stat {
        GrpStat::MeanCi | GrpStat::PredCi => 2,
        _ => 1,
    };
    let mut data = vec![f64::NAN; groups.len() * matrix.cols * depth];
    for (group_idx, rows) in groups.iter().enumerate() {
        for col in 0..matrix.cols {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows.iter().copied() {
                let value = matrix
                    .data
                    .get(row + col * matrix.rows)
                    .copied()
                    .ok_or_else(|| invalid_index("grpstats: matrix index out of range"))?;
                if !value.is_nan() {
                    values.push(value);
                }
            }
            let stats = evaluate_group_stat(&values, stat, alpha);
            for (depth_idx, value) in stats.iter().copied().enumerate() {
                data[group_idx + col * groups.len() + depth_idx * groups.len() * matrix.cols] =
                    value;
            }
        }
    }
    Ok(GroupSummary {
        data,
        cols: matrix.cols,
        depth,
    })
}

fn evaluate_group_stat(values: &[f64], stat: GrpStat, alpha: f64) -> Vec<f64> {
    let n = values.len();
    match stat {
        GrpStat::Mean => vec![mean(values)],
        GrpStat::Sem => vec![sem(values)],
        GrpStat::Std => vec![stddev(values)],
        GrpStat::Var => vec![variance(values)],
        GrpStat::Min => vec![values.iter().copied().fold(f64::INFINITY, f64::min)],
        GrpStat::Max => vec![values.iter().copied().fold(f64::NEG_INFINITY, f64::max)],
        GrpStat::Range => vec![
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min),
        ],
        GrpStat::Numel => vec![n as f64],
        GrpStat::MeanCi => interval(values, alpha, false),
        GrpStat::PredCi => interval(values, alpha, true),
        GrpStat::GName => vec![f64::NAN],
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = mean(values);
    values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn stddev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

fn sem(values: &[f64]) -> f64 {
    if values.len() < 2 {
        f64::NAN
    } else {
        stddev(values) / (values.len() as f64).sqrt()
    }
}

fn interval(values: &[f64], alpha: f64, prediction: bool) -> Vec<f64> {
    if values.len() < 2 {
        return vec![f64::NAN, f64::NAN];
    }
    let center = mean(values);
    let sd = stddev(values);
    let scale = if prediction {
        sd * (1.0 + 1.0 / values.len() as f64).sqrt()
    } else {
        sd / (values.len() as f64).sqrt()
    };
    let crit = student_t_inv(1.0 - alpha / 2.0, (values.len() - 1) as f64);
    vec![center - crit * scale, center + crit * scale]
}

fn group_names_value<'a>(keys: impl Iterator<Item = &'a Vec<GroupAtom>>) -> Value {
    let keys = keys.collect::<Vec<_>>();
    let cols = keys.iter().map(|key| key.len()).max().unwrap_or(0).max(1);
    let mut data = Vec::with_capacity(keys.len() * cols);
    for col in 0..cols {
        for key in &keys {
            let text = if key.is_empty() {
                "All".to_string()
            } else {
                key.get(col).map(group_atom_label).unwrap_or_default()
            };
            data.push(Value::from(text));
        }
    }
    Value::Cell(CellArray::new(data, keys.len(), cols).expect("group names cell should build"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn scalar_number_reads_typed_integer_storage_exactly() {
        let alpha = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();

        assert_eq!(scalar_number(&Value::Tensor(alpha)), Some(1.0));
    }

    #[test]
    fn numeric_matrix_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]).unwrap();

        let matrix = numeric_matrix(Value::Tensor(tensor), "grpstats").unwrap();

        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 1);
    }

    #[test]
    fn grpstats_uses_typed_storage_for_row_and_group_lengths() {
        let wide = (1_u64 << 53) + 1;
        let data =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2]).unwrap();
        let group = Tensor::new_integer(IntegerStorage::I64(vec![-3, -3]), vec![1, 2]).unwrap();

        let error = numeric_matrix(Value::Tensor(data), "grpstats")
            .expect_err("wide matrix data must reject at the floating boundary");
        assert!(error.message.contains("exactly representable as double"));
        assert_eq!(grouping_vector_len(&Value::Tensor(group)), Some(2));
    }

    #[test]
    fn matrix_from_table_column_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![4, 5]), vec![2, 1]).unwrap();

        let matrix = matrix_from_table_column(&Value::Tensor(tensor)).unwrap();

        assert_eq!(matrix.data, vec![4.0, 5.0]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 1);
    }
}

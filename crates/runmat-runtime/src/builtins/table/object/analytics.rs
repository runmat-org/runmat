use super::*;
use crate::builtins::stats::summary::distribution_math::student_t_inv;

pub fn sortrows_table(value: Value, rest: &[Value]) -> BuiltinResult<(Value, Tensor)> {
    let object = into_table_object(value, "sortrows")?;
    let names = table_variable_names_from_object(&object)?;
    let sort_spec = SortSpec::parse(rest, &names)?;
    let height = table_height(&object)?;
    let variables = table_variables(&object)?;
    let mut indices: Vec<usize> = (0..height).collect();
    indices.sort_by(|&a, &b| {
        for key in &sort_spec.keys {
            let Some(value) = variables.fields.get(&key.name) else {
                continue;
            };
            let ord = compare_table_cells(value, a, b).unwrap_or(Ordering::Equal);
            let ord = if key.descending { ord.reverse() } else { ord };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.cmp(&b)
    });
    let mut sorted_columns = Vec::with_capacity(names.len());
    for name in &names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{name}'")))?;
        sorted_columns.push(select_rows(value, &indices)?);
    }
    let row_names = selected_row_names(&object, &indices)?;
    let sorted = table_from_columns_with_properties(names, sorted_columns, row_names)?;
    let indices_tensor = Tensor::new(
        indices.iter().map(|idx| *idx as f64 + 1.0).collect(),
        vec![indices.len(), 1],
    )
    .map_err(invalid_variable)?;
    Ok((sorted, indices_tensor))
}

pub(in crate::builtins::table) struct SortSpec {
    keys: Vec<SortKey>,
}

pub(in crate::builtins::table) struct SortKey {
    name: String,
    descending: bool,
}

impl SortSpec {
    fn parse(rest: &[Value], names: &[String]) -> BuiltinResult<Self> {
        let mut keys = if rest.is_empty() {
            names
                .iter()
                .map(|name| SortKey {
                    name: name.clone(),
                    descending: false,
                })
                .collect::<Vec<_>>()
        } else {
            parse_variable_selector(rest.first(), names)?
                .into_iter()
                .map(|name| SortKey {
                    name,
                    descending: false,
                })
                .collect()
        };
        if let Some(direction) = rest.get(1) {
            let directions = string_list(direction)?;
            if directions.len() == 1 {
                let descending = directions[0].eq_ignore_ascii_case("descend")
                    || directions[0].eq_ignore_ascii_case("desc");
                for key in &mut keys {
                    key.descending = descending;
                }
            } else {
                for (key, direction) in keys.iter_mut().zip(directions.iter()) {
                    key.descending = direction.eq_ignore_ascii_case("descend")
                        || direction.eq_ignore_ascii_case("desc");
                }
            }
        }
        Ok(Self { keys })
    }
}

pub(in crate::builtins::table) fn compare_table_cells(
    value: &Value,
    a: usize,
    b: usize,
) -> BuiltinResult<Ordering> {
    match value {
        Value::Tensor(tensor) => Ok(tensor
            .get2(a, 0)
            .map_err(invalid_index)?
            .partial_cmp(&tensor.get2(b, 0).map_err(invalid_index)?)
            .unwrap_or(Ordering::Greater)),
        Value::StringArray(array) => {
            let av = array.data.get(a).cloned().unwrap_or_default();
            let bv = array.data.get(b).cloned().unwrap_or_default();
            Ok(av.cmp(&bv))
        }
        Value::LogicalArray(array) => {
            let av = *array.data.get(a).unwrap_or(&0);
            let bv = *array.data.get(b).unwrap_or(&0);
            Ok(av.cmp(&bv))
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            Ok(tensor
                .data
                .get(a)
                .copied()
                .unwrap_or(f64::NAN)
                .partial_cmp(&tensor.data.get(b).copied().unwrap_or(f64::NAN))
                .unwrap_or(Ordering::Greater))
        }
        other => Ok(cell_key_string(other, a).cmp(&cell_key_string(other, b))),
    }
}

#[derive(Clone, Debug)]
pub(in crate::builtins::table) enum GroupAtom {
    Number(f64),
    Text(String),
    Logical(bool),
    Missing,
}

impl GroupAtom {
    fn rank(&self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Logical(_) => 1,
            Self::Number(_) => 2,
            Self::Text(_) => 3,
        }
    }
}

impl PartialEq for GroupAtom {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for GroupAtom {}

impl PartialOrd for GroupAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GroupAtom {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = self.rank().cmp(&other.rank());
        if rank != Ordering::Equal {
            return rank;
        }
        match (self, other) {
            (Self::Missing, Self::Missing) => Ordering::Equal,
            (Self::Logical(a), Self::Logical(b)) => a.cmp(b),
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

pub(in crate::builtins::table) fn cell_group_atom(value: &Value, row: usize) -> GroupAtom {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(GroupAtom::Number)
            .unwrap_or(GroupAtom::Missing),
        Value::StringArray(array) => array
            .data
            .get(row)
            .cloned()
            .map(GroupAtom::Text)
            .unwrap_or(GroupAtom::Missing),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| GroupAtom::Logical(*value != 0))
            .unwrap_or(GroupAtom::Missing),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(GroupAtom::Number)
                .unwrap_or(GroupAtom::Missing)
        }
        other => GroupAtom::Text(cell_key_string(other, row)),
    }
}

pub(in crate::builtins::table) fn pivot_impl(
    table: Value,
    rowvars: Value,
    colvars: Value,
    datavar: Value,
    method: &str,
) -> BuiltinResult<Value> {
    let object = into_table_object(table, "pivot")?;
    let names = table_variable_names_from_object(&object)?;
    let row_names = parse_variable_selector_for_object(Some(&rowvars), &object, &names)?;
    let col_names = parse_variable_selector_for_object(Some(&colvars), &object, &names)?;
    let data_names = parse_variable_selector_for_object(Some(&datavar), &object, &names)?;
    if row_names.is_empty() || col_names.is_empty() || data_names.is_empty() {
        return Err(invalid_argument(
            "pivot: rowvars, colvars, and datavar must select at least one variable",
        ));
    }
    if data_names.len() != 1 {
        return Err(invalid_argument(
            "pivot: exactly one data variable is currently supported",
        ));
    }
    let data_name = &data_names[0];
    let variables = table_variables(&object)?;
    let data_value = variables
        .fields
        .get(data_name)
        .ok_or_else(|| invalid_variable(format!("pivot: missing data variable '{data_name}'")))?;
    if !matches!(data_value, Value::Tensor(tensor) if tensor.cols() == 1) {
        return Err(invalid_variable(
            "pivot: data variable must be a numeric column vector",
        ));
    }

    let height = table_height(&object)?;
    let mut row_order = Vec::<Vec<GroupAtom>>::new();
    let mut row_first_index = BTreeMap::<Vec<GroupAtom>, usize>::new();
    let mut col_order = Vec::<Vec<GroupAtom>>::new();
    let mut col_seen = BTreeMap::<Vec<GroupAtom>, ()>::new();
    let mut buckets = BTreeMap::<(Vec<GroupAtom>, Vec<GroupAtom>), Vec<usize>>::new();
    for row in 0..height {
        let row_key = group_key_for_row(&variables, &row_names, row);
        let col_key = group_key_for_row(&variables, &col_names, row);
        if !row_first_index.contains_key(&row_key) {
            row_first_index.insert(row_key.clone(), row);
            row_order.push(row_key.clone());
        }
        if !col_seen.contains_key(&col_key) {
            col_seen.insert(col_key.clone(), ());
            col_order.push(col_key.clone());
        }
        buckets.entry((row_key, col_key)).or_default().push(row);
    }

    let mut out_names = row_names.clone();
    let mut out_columns = Vec::with_capacity(row_names.len() + col_order.len());
    for name in &row_names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("pivot: missing row variable '{name}'")))?;
        let rows = row_order
            .iter()
            .filter_map(|key| row_first_index.get(key).copied())
            .collect::<Vec<_>>();
        out_columns.push(select_rows(value, &rows)?);
    }
    for col_key in &col_order {
        let mut values = Vec::with_capacity(row_order.len());
        for row_key in &row_order {
            let summary_rows = buckets
                .get(&(row_key.clone(), col_key.clone()))
                .cloned()
                .unwrap_or_default();
            if summary_rows.is_empty() {
                values.push(f64::NAN);
            } else {
                values.push(
                    summarize_groups(data_value, std::iter::once(&summary_rows), method)?
                        .into_iter()
                        .next()
                        .unwrap_or(f64::NAN),
                );
            }
        }
        out_names.push(format!(
            "{}_{}",
            make_valid_variable_name(&group_key_label(col_key), out_names.len() + 1),
            data_name
        ));
        out_columns.push(Value::Tensor(
            Tensor::new(values, vec![row_order.len(), 1]).map_err(invalid_variable)?,
        ));
    }
    let out_names = make_unique_variable_names(out_names);
    table_from_columns(out_names, out_columns)
}

pub(in crate::builtins::table) fn group_key_for_row(
    variables: &StructValue,
    names: &[String],
    row: usize,
) -> Vec<GroupAtom> {
    names
        .iter()
        .map(|name| {
            variables
                .fields
                .get(name)
                .map(|value| cell_group_atom(value, row))
                .unwrap_or(GroupAtom::Missing)
        })
        .collect()
}

pub(in crate::builtins::table) fn group_key_label(key: &[GroupAtom]) -> String {
    if key.is_empty() {
        return "missing".to_string();
    }
    key.iter()
        .map(group_atom_label)
        .collect::<Vec<_>>()
        .join("_")
}

pub(in crate::builtins::table) fn group_atom_label(atom: &GroupAtom) -> String {
    match atom {
        GroupAtom::Number(value) => format_key_number(*value),
        GroupAtom::Text(text) => text.clone(),
        GroupAtom::Logical(flag) => flag.to_string(),
        GroupAtom::Missing => "missing".to_string(),
    }
}

pub(in crate::builtins::table) fn groupsummary_impl(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let object = into_table_object(table, "groupsummary")?;
    let names = table_variable_names_from_object(&object)?;
    let group_names = parse_variable_selector_for_object(Some(&groupvars), &object, &names)?;
    let methods = string_list(&method)?;
    if methods.is_empty() {
        return Err(invalid_argument(
            "groupsummary: method list must not be empty",
        ));
    }
    let data_names = if let Some(value) = rest.first() {
        parse_variable_selector_for_object(Some(value), &object, &names)?
    } else {
        names
            .iter()
            .filter(|name| !group_names.contains(name))
            .filter(|name| {
                table_variables(&object)
                    .ok()
                    .and_then(|vars| vars.fields.get(*name).cloned())
                    .map(|value| matches!(value, Value::Tensor(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    };
    let variables = table_variables(&object)?;
    let height = table_height(&object)?;
    let mut groups: BTreeMap<Vec<GroupAtom>, Vec<usize>> = BTreeMap::new();
    for row in 0..height {
        let key = group_names
            .iter()
            .map(|name| {
                variables
                    .fields
                    .get(name)
                    .map(|value| cell_group_atom(value, row))
                    .unwrap_or(GroupAtom::Missing)
            })
            .collect::<Vec<_>>();
        groups.entry(key).or_default().push(row);
    }
    let group_rows = groups
        .values()
        .filter_map(|rows| rows.first().copied())
        .collect::<Vec<_>>();
    let mut out_names = Vec::new();
    let mut out_columns = Vec::new();
    for name in &group_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            invalid_variable(format!("groupsummary: missing group variable '{name}'"))
        })?;
        out_names.push(name.clone());
        out_columns.push(select_rows(value, &group_rows)?);
    }
    out_names.push("GroupCount".to_string());
    out_columns.push(Value::Tensor(
        Tensor::new(
            groups.values().map(|rows| rows.len() as f64).collect(),
            vec![groups.len(), 1],
        )
        .map_err(invalid_variable)?,
    ));
    for method in &methods {
        for name in &data_names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("groupsummary: missing data variable '{name}'"))
            })?;
            let values = summarize_groups(value, groups.values(), method)?;
            out_names.push(format!("{}_{}", method.to_ascii_lowercase(), name));
            out_columns.push(Value::Tensor(
                Tensor::new(values, vec![groups.len(), 1]).map_err(invalid_variable)?,
            ));
        }
    }
    table_from_columns(out_names, out_columns)
}

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
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data.first().copied(),
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
            let mut rows = tensor.rows();
            let mut cols = tensor.cols();
            if rows == 1 && tensor.data.len() > 1 {
                rows = tensor.data.len();
                cols = 1;
            }
            Ok(NumericMatrix {
                data: tensor.data,
                rows,
                cols,
            })
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

fn matrix_from_table_column(value: &Value) -> BuiltinResult<NumericMatrix> {
    match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => Ok(NumericMatrix {
            data: tensor.data.clone(),
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

fn group_atom_is_missing(atom: &GroupAtom) -> bool {
    match atom {
        GroupAtom::Missing => true,
        GroupAtom::Number(value) => value.is_nan(),
        GroupAtom::Text(value) => value.is_empty(),
        GroupAtom::Logical(_) => false,
    }
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
        Value::Tensor(tensor) => Some(tensor.data.len()),
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
        Value::Tensor(tensor) => Ok(tensor.data.iter().copied().map(GroupAtom::Number).collect()),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| GroupAtom::Logical(*value != 0))
            .collect()),
        Value::StringArray(array) => Ok(array.data.iter().cloned().map(GroupAtom::Text).collect()),
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
        Value::Num(value) => GroupAtom::Number(*value),
        Value::Int(value) => GroupAtom::Number(value.to_f64()),
        Value::Bool(value) => GroupAtom::Logical(*value),
        Value::String(value) => GroupAtom::Text(value.clone()),
        Value::CharArray(array) if array.rows == 1 => {
            GroupAtom::Text(array.data.iter().collect::<String>().trim().to_string())
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => GroupAtom::Number(tensor.data[0]),
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

pub(in crate::builtins::table) fn summarize_groups<'a>(
    value: &Value,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    method: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => tensor,
        _ => {
            return Err(invalid_variable(
                "groupsummary: summary data variables must be numeric column vectors",
            ))
        }
    };
    groups
        .map(|rows| {
            let mut values = rows
                .iter()
                .map(|row| tensor.get2(*row, 0).map_err(invalid_index))
                .collect::<BuiltinResult<Vec<_>>>()?;
            values.retain(|value| !value.is_nan());
            let result = match method.to_ascii_lowercase().as_str() {
                "mean" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                }
                "sum" => values.iter().sum(),
                "min" => values.into_iter().fold(f64::INFINITY, f64::min),
                "max" => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                "median" => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = values.len() / 2;
                        if values.len() % 2 == 0 {
                            (values[mid - 1] + values[mid]) / 2.0
                        } else {
                            values[mid]
                        }
                    }
                }
                "count" | "numel" => values.len() as f64,
                other => {
                    return Err(invalid_argument(format!(
                        "groupsummary: unsupported method '{other}'"
                    )))
                }
            };
            Ok(result)
        })
        .collect()
}

pub(in crate::builtins::table) fn cell_key_string(value: &Value, row: usize) -> String {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(format_key_number)
            .unwrap_or_default(),
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .ok()
                .and_then(|tensor| tensor.data.get(row).copied())
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => {
            categorical_label_at(obj, row).unwrap_or_default()
        }
        Value::Cell(cell) => cell
            .get(row, 0)
            .map(|item| cell_to_text(&item))
            .unwrap_or_default(),
        other => format!("{other}"),
    }
}

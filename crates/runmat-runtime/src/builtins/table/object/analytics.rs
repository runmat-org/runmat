use super::selectors::parse_variable_selector;
use super::*;
use runmat_builtins::IntValue;

mod grpstats;

pub(in crate::builtins::table) use grpstats::grpstats_impl;

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
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let left = storage
                    .value_at(a)
                    .ok_or_else(|| invalid_index("table: numeric row index out of bounds"))?;
                let right = storage
                    .value_at(b)
                    .ok_or_else(|| invalid_index("table: numeric row index out of bounds"))?;
                return Ok(compare_integer_values(&left, &right));
            }
            Ok(tensor
                .get2(a, 0)
                .map_err(invalid_index)?
                .partial_cmp(&tensor.get2(b, 0).map_err(invalid_index)?)
                .unwrap_or(Ordering::Greater))
        }
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
            Ok(double_value_at(&tensor, a)
                .unwrap_or(f64::NAN)
                .partial_cmp(&double_value_at(&tensor, b).unwrap_or(f64::NAN))
                .unwrap_or(Ordering::Greater))
        }
        other => Ok(cell_key_string(other, a).cmp(&cell_key_string(other, b))),
    }
}

#[derive(Clone, Debug)]
pub(in crate::builtins::table) enum GroupAtom {
    Number(f64),
    Integer(IntValue),
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
            Self::Integer(_) => 3,
            Self::Text(_) => 4,
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

pub(in crate::builtins::table) fn cell_group_atom(value: &Value, row: usize) -> GroupAtom {
    match value {
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(row)
                    .map(GroupAtom::Integer)
                    .unwrap_or(GroupAtom::Missing);
            }
            tensor
                .get2(row, 0)
                .map(GroupAtom::Number)
                .unwrap_or(GroupAtom::Missing)
        }
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
                .and_then(|tensor| double_value_at(&tensor, row))
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
        GroupAtom::Integer(value) => format_integer_key(value),
        GroupAtom::Text(text) => text.clone(),
        GroupAtom::Logical(flag) => flag.to_string(),
        GroupAtom::Missing => "missing".to_string(),
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
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(row)
                    .map(|value| format_integer_key(&value))
                    .unwrap_or_default();
            }
            tensor
                .get2(row, 0)
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .ok()
                .and_then(|tensor| double_value_at(&tensor, row))
                .map(format_key_number)
                .unwrap_or_default()
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .ok()
                .and_then(|tensor| double_value_at(&tensor, row))
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

fn double_value_at(tensor: &Tensor, index: usize) -> Option<f64> {
    tensor.as_f64_slice()?.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn typed_integer_group_atoms_and_table_ordering_remain_exact() {
        let large = 9_007_199_254_740_992_u64;
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large, large + 1]), vec![2, 1]).unwrap(),
        );

        let first = cell_group_atom(&value, 0);
        let second = cell_group_atom(&value, 1);
        assert_ne!(first, second);
        assert_eq!(compare_table_cells(&value, 0, 1).unwrap(), Ordering::Less);
        assert_eq!(group_atom_label(&second), (large + 1).to_string());
    }

    #[test]
    fn cell_key_string_reads_typed_integer_storage_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap();

        assert_eq!(
            cell_key_string(&Value::Tensor(tensor), 0),
            "9007199254740993"
        );
    }
}

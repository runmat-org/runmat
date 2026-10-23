//! Table-owned projection for heatmap's already-aggregated plotting surface.
//! Group identities and variable selection remain the same ones used by pivot.

use std::collections::{BTreeMap, BTreeSet};

use runmat_value::{NumericStorage, ObjectInstance, Tensor, Value};

use super::*;
use crate::RuntimeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeatmapAggregation {
    Count,
    Mean,
    Median,
    Sum,
    Max,
    Min,
    None,
}

impl HeatmapAggregation {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "count" => Some(Self::Count),
            "mean" => Some(Self::Mean),
            "median" => Some(Self::Median),
            "sum" => Some(Self::Sum),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    fn summary_method(self) -> Option<SummaryMethod> {
        match self {
            Self::Mean => Some(SummaryMethod::Mean),
            Self::Median => Some(SummaryMethod::Median),
            Self::Sum => Some(SummaryMethod::Sum),
            Self::Max => Some(SummaryMethod::Max),
            Self::Min => Some(SummaryMethod::Min),
            Self::Count | Self::None => None,
        }
    }

    fn empty_cell(self) -> f64 {
        match self {
            Self::Count | Self::Sum => 0.0,
            _ => f64::NAN,
        }
    }
}

pub(crate) struct AggregatedHeatmap {
    pub(crate) x_labels: Vec<String>,
    pub(crate) y_labels: Vec<String>,
    pub(crate) color_data: Tensor,
}

fn heatmap_table_invalid_message(detail: &str) -> String {
    let error = &runmat_builtins::HEATMAP_ERROR_INVALID_ARGUMENT;
    let detail = detail
        .trim()
        .strip_prefix("heatmap:")
        .map(str::trim)
        .unwrap_or(detail.trim());
    if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    }
}

fn heatmap_table_invalid(message: impl Into<String>) -> RuntimeError {
    let error = &runmat_builtins::HEATMAP_ERROR_INVALID_ARGUMENT;
    let message = message.into();
    let mut builder =
        crate::build_runtime_error(heatmap_table_invalid_message(&message)).with_builtin("heatmap");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

/// Retag failures from the synchronous, non-effecting table validators used
/// below. Provider, callback, cancellation, and allocation-admission errors
/// must not cross this boundary; those retain their typed terminal errors at
/// their operation owner instead.
fn map_pure_heatmap_table_validation(source: RuntimeError) -> RuntimeError {
    let error = &runmat_builtins::HEATMAP_ERROR_INVALID_ARGUMENT;
    let message = heatmap_table_invalid_message(source.message());
    let mut builder = crate::build_runtime_error(message)
        .with_builtin("heatmap")
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn selected_name(
    selector: &Value,
    table: &ObjectInstance,
    names: &[String],
) -> BuiltinResult<String> {
    let selected = parse_variable_selector_for_object(Some(selector), table, names)
        .map_err(map_pure_heatmap_table_validation)?;
    match selected.as_slice() {
        [name] => Ok(name.clone()),
        _ => Err(heatmap_table_invalid(
            "heatmap: each axis or color variable must select exactly one table variable",
        )),
    }
}

fn category_order(value: &Value, keys: BTreeSet<GroupAtom>) -> BuiltinResult<Vec<GroupAtom>> {
    let mut ordered = keys.into_iter().collect::<Vec<_>>();
    if let Value::Object(object) = value {
        if object.is_class(CATEGORICAL_CLASS) {
            let categories =
                categorical_categories(object).map_err(map_pure_heatmap_table_validation)?;
            ordered.sort_by_key(|key| match key {
                GroupAtom::Text(label) => categories
                    .iter()
                    .position(|category| category == label)
                    .unwrap_or(categories.len()),
                _ => categories.len(),
            });
        }
    }
    Ok(ordered)
}

fn numeric_color_column(value: &Value, height: usize) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor)
            if tensor.cols() == 1 && tensor.rows() == height && tensor.len() == height =>
        {
            Ok(value.clone())
        }
        Value::LogicalArray(array) if array.data.len() == height => {
            let values = array
                .data
                .iter()
                .map(|&value| if value == 0 { 0.0 } else { 1.0 })
                .collect();
            Tensor::new(values, vec![height, 1])
                .map(Value::Tensor)
                .map_err(heatmap_table_invalid)
        }
        _ => Err(heatmap_table_invalid(
            "heatmap: ColorVariable must be a numeric or logical column",
        )),
    }
}

fn admit_axis_column(value: &Value, height: usize) -> BuiltinResult<()> {
    let vector = match value {
        Value::Tensor(tensor) => {
            tensor.cols() == 1 && tensor.rows() == height && tensor.len() == height
        }
        Value::StringArray(array) => {
            array.cols() == 1 && array.rows() == height && array.data().len() == height
        }
        Value::LogicalArray(array) => {
            array.shape.first().copied() == Some(height)
                && array.shape.iter().skip(1).all(|&axis| axis == 1)
        }
        Value::Cell(array) => {
            array.cols == 1
                && array.rows == height
                && array.data().len() == height
                && array.data().iter().all(|item| {
                    matches!(item, Value::String(_))
                        || matches!(item, Value::CharArray(chars) if chars.rows == 1)
                })
        }
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            categorical_label_view(object)
                .map_err(map_pure_heatmap_table_validation)?
                .len()
                == height
        }
        _ => false,
    };
    if vector {
        Ok(())
    } else {
        Err(heatmap_table_invalid(
            "heatmap: axis variables must be discrete single-column values",
        ))
    }
}

/// Admit all table selectors and column extents before preparing a chart or
/// publishing a plot handle. A table value is borrowed; only the result grid
/// and grouping indices are materialized.
pub(crate) fn aggregate_heatmap_table(
    table: &ObjectInstance,
    x_selector: &Value,
    y_selector: &Value,
    color_selector: Option<&Value>,
    method: HeatmapAggregation,
) -> BuiltinResult<AggregatedHeatmap> {
    if !is_tabular_object(table) {
        return Err(heatmap_table_invalid(
            "heatmap: expected a table or timetable",
        ));
    }
    let variables = table_variables_ref(table)
        .map_err(map_pure_heatmap_table_validation)?
        .ok_or_else(|| heatmap_table_invalid("heatmap: table has no variable storage"))?;
    let names =
        table_variable_names_from_object(table).map_err(map_pure_heatmap_table_validation)?;
    let x_name = selected_name(x_selector, table, &names)?;
    let y_name = selected_name(y_selector, table, &names)?;
    let color_name = color_selector
        .map(|selector| selected_name(selector, table, &names))
        .transpose()?;
    if method != HeatmapAggregation::Count && color_name.is_none() {
        return Err(heatmap_table_invalid(
            "heatmap: ColorMethod other than count requires ColorVariable",
        ));
    }
    let x = variables
        .fields
        .get(&x_name)
        .ok_or_else(|| heatmap_table_invalid("heatmap: missing XVariable"))?;
    let y = variables
        .fields
        .get(&y_name)
        .ok_or_else(|| heatmap_table_invalid("heatmap: missing YVariable"))?;
    let height = value_row_count(x).map_err(map_pure_heatmap_table_validation)?;
    if value_row_count(y).map_err(map_pure_heatmap_table_validation)? != height {
        return Err(heatmap_table_invalid(
            "heatmap: axis column heights disagree",
        ));
    }
    if matches!(x, Value::GpuTensor(_)) || matches!(y, Value::GpuTensor(_)) {
        return Err(heatmap_table_invalid(
            "heatmap: resident table axis columns require a supported provider path",
        ));
    }
    admit_axis_column(x, height)?;
    admit_axis_column(y, height)?;
    let color = color_name
        .as_ref()
        .map(|name| {
            variables
                .fields
                .get(name)
                .ok_or_else(|| heatmap_table_invalid("heatmap: missing ColorVariable"))
        })
        .transpose()?;
    if let Some(color) = color.filter(|_| method != HeatmapAggregation::Count) {
        if value_row_count(color).map_err(map_pure_heatmap_table_validation)? != height {
            return Err(heatmap_table_invalid(
                "heatmap: color column height disagrees",
            ));
        }
        if matches!(color, Value::GpuTensor(_)) {
            return Err(heatmap_table_invalid(
                "heatmap: resident table color columns require a supported provider path",
            ));
        }
    }
    let color = if method == HeatmapAggregation::Count {
        None
    } else {
        Some(numeric_color_column(
            color.expect("non-count method admitted a color variable"),
            height,
        )?)
    };

    let mut x_keys = BTreeSet::new();
    let mut y_keys = BTreeSet::new();
    let mut groups = BTreeMap::<(GroupAtom, GroupAtom), Vec<usize>>::new();
    for row in 0..height {
        let x_key = cell_group_atom(x, row)?;
        let y_key = cell_group_atom(y, row)?;
        // RunMat's scoped table-heatmap policy excludes rows without a
        // discrete axis category. A literal text category "missing" remains
        // distinct and never collides with an invented display sentinel.
        if group_atom_is_missing(&x_key) || group_atom_is_missing(&y_key) {
            continue;
        }
        x_keys.insert(x_key.clone());
        y_keys.insert(y_key.clone());
        groups.entry((x_key, y_key)).or_default().push(row);
    }
    let x_keys = category_order(x, x_keys)?;
    let y_keys = category_order(y, y_keys)?;
    if x_keys.is_empty() || y_keys.is_empty() {
        return Err(heatmap_table_invalid(
            "heatmap: table has no observed axis values",
        ));
    }
    let cell_count = x_keys
        .len()
        .checked_mul(y_keys.len())
        .ok_or_else(|| heatmap_table_invalid("heatmap: aggregated grid is too large"))?;
    if method == HeatmapAggregation::None {
        let Value::Tensor(source) = color.as_ref().expect("none requires a color variable") else {
            unreachable!("numeric color normalized to Tensor")
        };
        let color_data =
            exact_unaggregated_color_data(source, &x_keys, &y_keys, &groups, cell_count)?;
        return Ok(AggregatedHeatmap {
            x_labels: x_keys.iter().map(group_atom_label).collect(),
            y_labels: y_keys.iter().map(group_atom_label).collect(),
            color_data,
        });
    }
    let mut values = Vec::with_capacity(cell_count);
    for x_key in &x_keys {
        for y_key in &y_keys {
            let rows = groups.get(&(x_key.clone(), y_key.clone()));
            let value = match (method, rows) {
                (_, None) => method.empty_cell(),
                (HeatmapAggregation::Count, Some(rows)) => rows.len() as f64,
                (HeatmapAggregation::None, Some(_)) => {
                    unreachable!("unaggregated colors use their native-class path")
                }
                (summary, Some(rows)) => {
                    let method = summary.summary_method().expect("summary method admitted");
                    let reduced = summarize_groups(
                        color.as_ref().expect("color admitted"),
                        std::iter::once(rows),
                        method,
                    )
                    .map_err(map_pure_heatmap_table_validation)?
                    .into_iter()
                    .next()
                    .expect("one admitted group");
                    if summary != HeatmapAggregation::Sum && rows.iter().all(|&row| {
                        matches!(color.as_ref(), Some(Value::Tensor(t)) if t.get2(row, 0).is_ok_and(f64::is_nan))
                    }) {
                        f64::NAN
                    } else {
                        reduced
                    }
                }
            };
            values.push(value);
        }
    }
    let color_data =
        Tensor::new(values, vec![y_keys.len(), x_keys.len()]).map_err(heatmap_table_invalid)?;
    Ok(AggregatedHeatmap {
        x_labels: x_keys.iter().map(|key| group_atom_label(key)).collect(),
        y_labels: y_keys.iter().map(|key| group_atom_label(key)).collect(),
        color_data,
    })
}

fn exact_unaggregated_color_data(
    source: &Tensor,
    x_keys: &[GroupAtom],
    y_keys: &[GroupAtom],
    groups: &BTreeMap<(GroupAtom, GroupAtom), Vec<usize>>,
    cell_count: usize,
) -> BuiltinResult<Tensor> {
    if groups.values().any(|rows| rows.len() != 1) {
        return Err(heatmap_table_invalid(
            "heatmap: ColorMethod none requires unique axis pairs",
        ));
    }
    let has_holes = groups.len() != cell_count;
    if has_holes {
        if let Some(integer) = source.integer_storage() {
            for rows in groups.values() {
                let value = integer
                    .value_at(rows[0])
                    .ok_or_else(|| heatmap_table_invalid("heatmap: color row out of bounds"))?;
                if !crate::builtins::common::validation::integer_is_exact_f64(&value) {
                    let error = &runmat_builtins::HEATMAP_ERROR_UNSUPPORTED_COLOR_DATA;
                    let mut builder =
                        crate::build_runtime_error(error.message).with_builtin("heatmap");
                    if let Some(identifier) = error.identifier {
                        builder = builder.with_identifier(identifier);
                    }
                    return Err(builder.build());
                }
            }
        }
    }
    let mut storage = if has_holes {
        if source.integer_storage().is_some() {
            NumericStorage::F64(vec![f64::NAN; cell_count])
        } else {
            match source.numeric_dtype() {
                runmat_value::NumericDType::F32 => NumericStorage::F32(vec![f32::NAN; cell_count]),
                _ => NumericStorage::F64(vec![f64::NAN; cell_count]),
            }
        }
    } else {
        NumericStorage::zeros(source.numeric_dtype(), cell_count)
    };
    for (x_index, x_key) in x_keys.iter().enumerate() {
        for (y_index, y_key) in y_keys.iter().enumerate() {
            let Some(rows) = groups.get(&(x_key.clone(), y_key.clone())) else {
                continue;
            };
            let original = source
                .numeric_value_at(rows[0])
                .ok_or_else(|| heatmap_table_invalid("heatmap: color row out of bounds"))?;
            let value = if has_holes && source.integer_storage().is_some() {
                runmat_value::NumericScalar::F64(original.materialize_f64())
            } else {
                original
            };
            storage
                .set_value(x_index * y_keys.len() + y_index, value)
                .map_err(heatmap_table_invalid)?;
        }
    }
    Tensor::from_numeric_storage(storage, vec![y_keys.len(), x_keys.len()])
        .map_err(heatmap_table_invalid)
}

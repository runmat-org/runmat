use std::collections::BTreeSet;

use crate::builtins::array::grouping::groupcounts::bins::BinLevels;
use crate::builtins::array::grouping::keys::{GroupIndex, KeyAtom, KeyOrder};
use crate::builtins::array::grouping::variables::{build_index, GroupColumn};
use crate::builtins::table::{categorical_categories, categorical_observation_labels};
use crate::BuiltinResult;

use super::error;

const MAX_GROUPS: usize = 50_000_000;

pub(super) fn build(
    columns: &[GroupColumn],
    include_missing: bool,
    include_empty: bool,
    bins: Option<&BinLevels>,
) -> BuiltinResult<GroupIndex> {
    let explicit = explicit_levels(columns, include_missing, include_empty, bins)?;
    let order = explicit.map_or(KeyOrder::Sorted, KeyOrder::Explicit);
    let mut index = build_index(columns, include_missing, order).map_err(error::invalid)?;
    if !include_empty {
        retain_observed(&mut index);
    }
    Ok(index)
}

fn explicit_levels(
    columns: &[GroupColumn],
    include_missing: bool,
    include_empty: bool,
    bins: Option<&BinLevels>,
) -> BuiltinResult<Option<Vec<Vec<KeyAtom>>>> {
    let categorical = columns.iter().any(|column| {
        matches!(&column.value, runmat_value::Value::Object(value) if value.is_class(runmat_types::standard::CATEGORICAL))
    });
    if !include_empty && !categorical && bins.is_none() {
        return Ok(None);
    }
    let levels = columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            if index == 0 {
                if let Some(bins) = bins {
                    return bin_levels(column, bins, include_missing);
                }
            }
            column_levels(column, include_missing, include_empty)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    cartesian(&levels).map(Some)
}

fn column_levels(
    column: &GroupColumn,
    include_missing: bool,
    include_empty: bool,
) -> BuiltinResult<Vec<KeyAtom>> {
    if let runmat_value::Value::Object(value) = &column.value {
        if value.is_class(runmat_types::standard::CATEGORICAL) {
            let mut levels = categorical_categories(value)?
                .into_iter()
                .map(KeyAtom::Text)
                .collect::<Vec<_>>();
            let missing = categorical_observation_labels(value)?
                .iter()
                .any(Option::is_none);
            if include_missing && missing {
                levels.push(KeyAtom::Missing);
            }
            return Ok(levels);
        }
    }
    if include_empty
        && matches!(
            column.value,
            runmat_value::Value::LogicalArray(_) | runmat_value::Value::Bool(_)
        )
    {
        let mut levels = vec![KeyAtom::Logical(false), KeyAtom::Logical(true)];
        append_observed_missing(column, include_missing, &mut levels)?;
        return Ok(levels);
    }
    observed_levels(column, include_missing)
}

fn bin_levels(
    column: &GroupColumn,
    bins: &BinLevels,
    include_missing: bool,
) -> BuiltinResult<Vec<KeyAtom>> {
    let mut levels = bins
        .labels
        .iter()
        .cloned()
        .map(KeyAtom::Text)
        .collect::<Vec<_>>();
    append_observed_missing(column, include_missing, &mut levels)?;
    Ok(levels)
}

fn observed_levels(column: &GroupColumn, include_missing: bool) -> BuiltinResult<Vec<KeyAtom>> {
    let mut levels = column
        .atoms()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|atom| include_missing || *atom != KeyAtom::Missing)
        .collect::<Vec<_>>();
    levels.sort();
    Ok(levels)
}

fn append_observed_missing(
    column: &GroupColumn,
    include_missing: bool,
    levels: &mut Vec<KeyAtom>,
) -> BuiltinResult<()> {
    if include_missing && column.atoms().contains(&KeyAtom::Missing) {
        levels.push(KeyAtom::Missing);
    }
    Ok(())
}

fn cartesian(levels: &[Vec<KeyAtom>]) -> BuiltinResult<Vec<Vec<KeyAtom>>> {
    let count = levels.iter().try_fold(1usize, |count, values| {
        count
            .checked_mul(values.len())
            .filter(|count| *count <= MAX_GROUPS)
            .ok_or_else(|| {
                error::too_large("groupcounts: possible empty-group combinations are too large")
            })
    })?;
    let mut keys = Vec::with_capacity(count);
    expand(levels, 0, &mut Vec::with_capacity(levels.len()), &mut keys);
    Ok(keys)
}

fn expand(
    levels: &[Vec<KeyAtom>],
    index: usize,
    prefix: &mut Vec<KeyAtom>,
    keys: &mut Vec<Vec<KeyAtom>>,
) {
    if index == levels.len() {
        keys.push(prefix.clone());
        return;
    }
    for value in &levels[index] {
        prefix.push(value.clone());
        expand(levels, index + 1, prefix, keys);
        prefix.pop();
    }
}

fn retain_observed(index: &mut GroupIndex) {
    let retained = index
        .row_groups
        .iter()
        .enumerate()
        .filter_map(|(position, rows)| (!rows.is_empty()).then_some(position))
        .collect::<Vec<_>>();
    index.keys = retained
        .iter()
        .map(|position| index.keys[*position].clone())
        .collect();
    index.first_rows = retained
        .iter()
        .map(|position| index.first_rows[*position])
        .collect();
    index.row_groups = retained
        .iter()
        .map(|position| index.row_groups[*position].clone())
        .collect();
}

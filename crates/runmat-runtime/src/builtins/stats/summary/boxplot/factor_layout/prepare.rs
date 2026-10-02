//! Allocate category slots and finite geometry before selecting graphics axes.
use super::*;
use crate::builtins::stats::summary::boxplot::groups::labels::group_label_text;
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::builtins::stats::summary::boxplot) fn prepare_layout(
    mut boxes: Vec<BoxStats>,
    options: &BoxplotOptions,
) -> BuiltinResult<BoxLayout> {
    let keys = boxes
        .iter()
        .map(|stats| stats.group_key.clone())
        .collect::<Vec<_>>();
    let factor_count = keys.first().map_or(0, Vec::len);
    if keys.iter().any(|key| key.len() != factor_count) {
        return Err(invalid("boxplot: grouping factors have inconsistent arity"));
    }
    let levels = collect_levels(&keys, factor_count);
    validate_factor_controls(options, factor_count)?;
    let directions = resolved_directions(&options.factor_direction, &levels, options.orientation)?;
    let slot_count = if options.full_factors && factor_count > 0 {
        cartesian_slot_count(&levels)?
    } else {
        keys.len()
    };
    if slot_count == 0 {
        return Err(invalid("boxplot: no groups remain for display"));
    }
    if let Some(labels) = &options.labels {
        if labels.len() != slot_count {
            return Err(invalid(
                "boxplot: Labels length must match the number of plotted slots",
            ));
        }
    }
    if let Some(positions) = &options.positions {
        if positions.len() != slot_count {
            return Err(invalid(
                "boxplot: Positions length must match the number of plotted slots",
            ));
        }
    }
    let mut slots = if options.full_factors && factor_count > 0 {
        cartesian_slots(&levels, &directions, &keys, slot_count)?
    } else {
        observed_slots(&keys, &directions, options.group_order.is_some())
    };
    if options.full_factors && options.group_order.is_some() {
        // GroupOrder already resolved public labels to exact typed keys. Keep
        // those observed keys in the requested order; append unobserved
        // Cartesian combinations in their deterministic factor order.
        let observed_rank = keys
            .iter()
            .cloned()
            .enumerate()
            .map(|(rank, key)| (key, rank))
            .collect::<BTreeMap<_, _>>();
        slots.sort_by_key(|slot| observed_rank.get(&slot.key).copied().unwrap_or(usize::MAX));
    }
    let positions = slot_positions(&slots, options)?;
    let widths = super::super::plot::widths_for(slots.len(), options);
    for (index, slot) in slots.iter_mut().enumerate() {
        slot.position = positions[index];
        slot.label = options.labels.as_ref().map_or_else(
            || group_label_text(&slot.key),
            |labels| labels[index].clone(),
        );
        if let Some(box_index) = slot.box_index {
            let stats = &mut boxes[box_index];
            stats.position = slot.position;
            stats.width = widths[index];
            stats.label = slot.label.clone();
        }
    }
    boxes.sort_by(|left, right| left.position.total_cmp(&right.position));
    let box_index_by_key = boxes
        .iter()
        .enumerate()
        .map(|(index, stats)| (stats.group_key.clone(), index))
        .collect::<BTreeMap<_, _>>();
    for slot in &mut slots {
        slot.box_index = box_index_by_key.get(&slot.key).copied();
    }
    let separators = separator_positions(&slots, options);
    Ok(BoxLayout {
        boxes,
        slots,
        separators,
    })
}

fn collect_levels(keys: &[Vec<KeyAtom>], factor_count: usize) -> Vec<Vec<KeyAtom>> {
    let mut levels = vec![Vec::new(); factor_count];
    let mut seen = vec![BTreeSet::new(); factor_count];
    for key in keys {
        for (factor, value) in key.iter().enumerate() {
            if seen[factor].insert(value.clone()) {
                levels[factor].push(value.clone());
            }
        }
    }
    levels
}

fn resolved_directions(
    requested: &[FactorDirection],
    levels: &[Vec<KeyAtom>],
    orientation: Orientation,
) -> BuiltinResult<Vec<bool>> {
    validate_direction_count(requested, levels.len())?;
    Ok(levels
        .iter()
        .enumerate()
        .map(|(index, values)| {
            let direction = requested[index.min(requested.len() - 1)].resolved(values);
            orientation == Orientation::Horizontal && direction == FactorDirection::List
        })
        .collect())
}

fn observed_slots(
    keys: &[Vec<KeyAtom>],
    reverse: &[bool],
    group_order_explicit: bool,
) -> Vec<FactorSlot> {
    let mut indices = (0..keys.len()).collect::<Vec<_>>();
    if !group_order_explicit && reverse.iter().any(|value| *value) {
        let levels = collect_levels(keys, reverse.len());
        let ranks = levels
            .iter()
            .map(|factor_levels| {
                factor_levels
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(rank, key)| (key, rank))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            for factor in 0..reverse.len() {
                let left_rank = ranks[factor][&keys[left][factor]];
                let right_rank = ranks[factor][&keys[right][factor]];
                let order = if reverse[factor] {
                    right_rank.cmp(&left_rank)
                } else {
                    left_rank.cmp(&right_rank)
                };
                if !order.is_eq() {
                    return order;
                }
            }
            left.cmp(&right)
        });
    }
    indices
        .into_iter()
        .map(|index| FactorSlot {
            key: keys[index].clone(),
            label: String::new(),
            position: 0.0,
            box_index: Some(index),
        })
        .collect()
}

fn cartesian_slots(
    levels: &[Vec<KeyAtom>],
    reverse: &[bool],
    observed: &[Vec<KeyAtom>],
    count: usize,
) -> BuiltinResult<Vec<FactorSlot>> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(count)
        .map_err(|_| invalid("boxplot: FullFactors slots exceed available memory"))?;
    let observed_by_key = observed
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let mut digits = vec![0usize; levels.len()];
    for _ in 0..count {
        let key = digits
            .iter()
            .enumerate()
            .map(|(factor, &digit)| {
                let level = if reverse[factor] {
                    levels[factor].len() - 1 - digit
                } else {
                    digit
                };
                levels[factor][level].clone()
            })
            .collect::<Vec<_>>();
        let box_index = observed_by_key.get(&key).copied();
        slots.push(FactorSlot {
            key,
            label: String::new(),
            position: 0.0,
            box_index,
        });
        for digit in (0..digits.len()).rev() {
            digits[digit] += 1;
            if digits[digit] < levels[digit].len() {
                break;
            }
            digits[digit] = 0;
        }
    }
    Ok(slots)
}

fn cartesian_slot_count(levels: &[Vec<KeyAtom>]) -> BuiltinResult<usize> {
    levels
        .iter()
        .try_fold(1usize, |count, level| count.checked_mul(level.len()))
        .ok_or_else(|| invalid("boxplot: FullFactors combination count overflows"))
}

fn validate_factor_controls(options: &BoxplotOptions, factor_count: usize) -> BuiltinResult<()> {
    validate_gap_count(&options.factor_gap, factor_count)?;
    validate_separator_count(&options.factor_separator, factor_count)?;
    Ok(())
}

fn slot_positions(slots: &[FactorSlot], options: &BoxplotOptions) -> BuiltinResult<Vec<f64>> {
    if let Some(positions) = &options.positions {
        if positions.len() != slots.len() {
            return Err(invalid(
                "boxplot: Positions length must match the number of plotted slots",
            ));
        }
        return Ok(positions.clone());
    }
    let percentages = slots
        .windows(2)
        .map(|pair| {
            pair[0]
                .key
                .iter()
                .zip(&pair[1].key)
                .position(|(prior, current)| prior != current)
                .map_or(0.0, |factor| gap_percent(&options.factor_gap, factor))
        })
        .collect::<Vec<_>>();
    // A percentage is measured against the final category-axis width,
    // including one unit per slot. Solve that width before placing any slot.
    let total_percent = percentages.iter().sum::<f64>();
    if !total_percent.is_finite() || total_percent >= 100.0 {
        return Err(invalid(
            "boxplot: FactorGap exceeds the representable plot width",
        ));
    }
    let span = slots.len() as f64 / (1.0 - total_percent / 100.0);
    if !span.is_finite() {
        return Err(invalid(
            "boxplot: FactorGap plot width is not representable",
        ));
    }
    let mut positions = Vec::with_capacity(slots.len());
    let mut next = 1.0;
    for index in 0..slots.len() {
        if index > 0 {
            let extra = span * (percentages[index - 1] / 100.0);
            next += 1.0 + extra;
        }
        if !next.is_finite() {
            return Err(invalid("boxplot: factor positions are not representable"));
        }
        positions.push(next);
    }
    Ok(positions)
}

fn gap_percent(gap: &FactorGap, factor: usize) -> f64 {
    match gap {
        FactorGap::None => 0.0,
        FactorGap::Auto => {
            if factor == 0 {
                3.0
            } else {
                1.0
            }
        }
        FactorGap::Percentages(gaps) => gaps.get(factor).copied().unwrap_or(0.0),
    }
}

fn separator_positions(slots: &[FactorSlot], options: &BoxplotOptions) -> Vec<f64> {
    let selected = match &options.factor_separator {
        FactorSeparator::None => return Vec::new(),
        FactorSeparator::Auto => (1..=slots[0].key.len()).collect::<Vec<_>>(),
        FactorSeparator::Factors(indices) => indices.clone(),
    };
    slots
        .windows(2)
        .filter_map(|pair| {
            selected
                .iter()
                .any(|&factor| pair[0].key[factor - 1] != pair[1].key[factor - 1])
                .then_some(pair[0].position * 0.5 + pair[1].position * 0.5)
        })
        .collect()
}

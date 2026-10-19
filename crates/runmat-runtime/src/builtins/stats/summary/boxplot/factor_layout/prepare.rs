//! Allocate category slots and finite geometry before selecting graphics axes.
use super::*;
use crate::builtins::stats::summary::boxplot::groups::labels::group_label_text;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
pub(in crate::builtins::stats::summary::boxplot) fn prepare_layout(
    boxes: Vec<BoxStats>,
    options: &BoxplotOptions,
) -> BuiltinResult<BoxLayout> {
    futures::executor::block_on(prepare_layout_cooperative(
        boxes,
        options,
        &mut super::super::cooperation::BoxplotCooperation::new(None),
    ))
}

pub(in crate::builtins::stats::summary::boxplot) async fn prepare_layout_cooperative(
    mut boxes: Vec<BoxStats>,
    options: &BoxplotOptions,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<BoxLayout> {
    let mut keys = Vec::new();
    keys.try_reserve_exact(boxes.len()).map_err(|error| {
        allocation_failure("boxplot: grouping keys exceed available memory", error)
    })?;
    for stats in &boxes {
        let mut key = Vec::new();
        key.try_reserve_exact(stats.group_key.len())
            .map_err(|error| {
                allocation_failure("boxplot: grouping key exceeds available memory", error)
            })?;
        for atom in &stats.group_key {
            cooperation.completed(key_clone_work(atom)).await?;
            key.push(atom.clone());
        }
        keys.push(key);
    }
    let factor_count = keys.first().map_or(0, Vec::len);
    if keys.iter().any(|key| key.len() != factor_count) {
        return Err(invalid("boxplot: grouping factors have inconsistent arity"));
    }
    let levels = collect_levels_cooperative(&keys, factor_count, cooperation).await?;
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
        cartesian_slots(
            &levels,
            &directions,
            &keys,
            slot_count,
            options.group_order.is_some(),
            cooperation,
        )
        .await?
    } else {
        observed_slots(
            &keys,
            &directions,
            options.group_order.is_some(),
            cooperation,
        )
        .await?
    };
    let positions = slot_positions(&slots, options, cooperation).await?;
    for (index, slot) in slots.iter_mut().enumerate() {
        slot.position = positions[index];
        slot.label = options.labels.as_ref().map_or_else(
            || group_label_text(&slot.key),
            |labels| labels[index].clone(),
        );
        if let Some(box_index) = slot.box_index {
            let stats = &mut boxes[box_index];
            stats.position = slot.position;
            stats.width = options.widths[index % options.widths.len()];
            stats.label = slot.label.clone();
        }
        cooperation.completed(1).await?;
    }
    let boxes = if options.positions.is_some() {
        // Explicit Positions may be nonmonotone, so preserve the established
        // stable position ordering for that form.
        boxes.sort_by(|left, right| left.position.total_cmp(&right.position));
        cooperation.completed(0).await?;
        boxes
    } else {
        // Generated slot positions are strictly increasing. Move observed
        // statistics into slot order without sorting a large factor product.
        let mut original = Vec::new();
        original.try_reserve_exact(boxes.len()).map_err(|error| {
            allocation_failure("boxplot: observed boxes exceed available memory", error)
        })?;
        for stats in boxes {
            original.push(Some(stats));
            cooperation.completed(1).await?;
        }
        let mut ordered = Vec::new();
        ordered.try_reserve_exact(original.len()).map_err(|error| {
            allocation_failure("boxplot: observed boxes exceed available memory", error)
        })?;
        for slot in &slots {
            if let Some(index) = slot.box_index {
                ordered.push(
                    original[index]
                        .take()
                        .ok_or_else(|| internal("boxplot: repeated observed factor slot"))?,
                );
            }
            cooperation.completed(1).await?;
        }
        ordered
    };
    if options.positions.is_none() {
        let mut next_box = 0;
        for slot in &mut slots {
            if slot.box_index.is_some() {
                slot.box_index = Some(next_box);
                next_box += 1;
            }
            cooperation.completed(1).await?;
        }
    } else {
        let mut box_index_by_key = BTreeMap::new();
        for (index, stats) in boxes.iter().enumerate() {
            box_index_by_key.insert(stats.group_key.clone(), index);
            cooperation.completed(1).await?;
        }
        for slot in &mut slots {
            slot.box_index = box_index_by_key.get(&slot.key).copied();
            cooperation.completed(1).await?;
        }
    }
    let separators = separator_positions(&slots, options, cooperation).await?;
    Ok(BoxLayout {
        boxes,
        slots,
        separators,
    })
}

async fn collect_levels_cooperative(
    keys: &[Vec<KeyAtom>],
    factor_count: usize,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<Vec<KeyAtom>>> {
    let mut levels = vec![Vec::new(); factor_count];
    let mut seen = vec![BTreeSet::new(); factor_count];
    for key in keys {
        for (factor, value) in key.iter().enumerate() {
            cooperation.completed(key_clone_work(value)).await?;
            if seen[factor].insert(value.clone()) {
                cooperation.completed(key_clone_work(value)).await?;
                levels[factor].push(value.clone());
            }
        }
    }
    Ok(levels)
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

async fn observed_slots(
    keys: &[Vec<KeyAtom>],
    reverse: &[bool],
    group_order_explicit: bool,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<FactorSlot>> {
    let mut indices = (0..keys.len()).collect::<Vec<_>>();
    if !group_order_explicit && reverse.iter().any(|value| *value) {
        let levels = collect_levels_cooperative(keys, reverse.len(), cooperation).await?;
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
    let mut slots = Vec::new();
    slots.try_reserve_exact(indices.len()).map_err(|error| {
        allocation_failure("boxplot: observed slots exceed available memory", error)
    })?;
    for index in indices {
        slots.push(FactorSlot {
            key: keys[index].clone(),
            label: String::new(),
            position: 0.0,
            box_index: Some(index),
        });
        cooperation.completed(1).await?;
    }
    Ok(slots)
}

async fn cartesian_slots(
    levels: &[Vec<KeyAtom>],
    reverse: &[bool],
    observed: &[Vec<KeyAtom>],
    count: usize,
    observed_first: bool,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<FactorSlot>> {
    cooperation.completed(0).await?;
    let mut slots = Vec::new();
    slots.try_reserve_exact(count).map_err(|error| {
        allocation_failure("boxplot: FullFactors slots exceed available memory", error)
    })?;
    let observed_by_key = observed
        .iter()
        .enumerate()
        .map(|(index, key)| (key.as_slice(), index))
        .collect::<BTreeMap<_, _>>();
    if observed_first {
        // GroupOrder has already put the observed keys in public order. Emit
        // them first, then append absent Cartesian combinations; no large
        // synchronous post-expansion sort can monopolize a browser turn.
        for (index, key) in observed.iter().enumerate() {
            let mut copied = Vec::new();
            copied.try_reserve_exact(key.len()).map_err(|error| {
                allocation_failure("boxplot: grouping key exceeds available memory", error)
            })?;
            for atom in key {
                cooperation.completed(key_clone_work(atom)).await?;
                copied.push(atom.clone());
            }
            slots.push(FactorSlot {
                key: copied,
                label: String::new(),
                position: 0.0,
                box_index: Some(index),
            });
            cooperation.completed(1).await?;
        }
    }
    let mut digits = vec![0usize; levels.len()];
    for _ in 0..count {
        let mut key = Vec::new();
        key.try_reserve_exact(levels.len()).map_err(|error| {
            allocation_failure("boxplot: grouping key exceeds available memory", error)
        })?;
        for (factor, &digit) in digits.iter().enumerate() {
            let level = if reverse[factor] {
                levels[factor].len() - 1 - digit
            } else {
                digit
            };
            let atom = &levels[factor][level];
            cooperation.completed(key_clone_work(atom)).await?;
            key.push(atom.clone());
        }
        let box_index = observed_by_key.get(key.as_slice()).copied();
        if !observed_first || box_index.is_none() {
            slots.push(FactorSlot {
                key,
                label: String::new(),
                position: 0.0,
                box_index,
            });
        }
        cooperation.completed(1).await?;
        for digit in (0..digits.len()).rev() {
            digits[digit] += 1;
            if digits[digit] < levels[digit].len() {
                break;
            }
            digits[digit] = 0;
        }
    }
    cooperation.completed(0).await?;
    Ok(slots)
}

fn cartesian_slot_count(levels: &[Vec<KeyAtom>]) -> BuiltinResult<usize> {
    levels
        .iter()
        .try_fold(1usize, |count, level| count.checked_mul(level.len()))
        .ok_or_else(|| invalid("boxplot: FullFactors combination count overflows"))
}

fn key_clone_work(atom: &KeyAtom) -> u64 {
    match atom {
        KeyAtom::Text(text) => u64::try_from(text.len().max(1)).unwrap_or(u64::MAX),
        _ => 1,
    }
}

fn validate_factor_controls(options: &BoxplotOptions, factor_count: usize) -> BuiltinResult<()> {
    validate_gap_count(&options.factor_gap, factor_count)?;
    validate_separator_count(&options.factor_separator, factor_count)?;
    Ok(())
}

async fn slot_positions(
    slots: &[FactorSlot],
    options: &BoxplotOptions,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<f64>> {
    if let Some(positions) = &options.positions {
        if positions.len() != slots.len() {
            return Err(invalid(
                "boxplot: Positions length must match the number of plotted slots",
            ));
        }
        let mut copied = Vec::new();
        copied.try_reserve_exact(positions.len()).map_err(|error| {
            allocation_failure("boxplot: Positions exceed available memory", error)
        })?;
        for &position in positions {
            copied.push(position);
            cooperation.completed(1).await?;
        }
        return Ok(copied);
    }
    let mut percentages = Vec::new();
    percentages
        .try_reserve_exact(slots.len().saturating_sub(1))
        .map_err(|error| {
            allocation_failure(
                "boxplot: FactorGap boundaries exceed available memory",
                error,
            )
        })?;
    let mut total_percent = 0.0;
    for pair in slots.windows(2) {
        let mut changed = None;
        for (factor, (prior, current)) in pair[0].key.iter().zip(&pair[1].key).enumerate() {
            cooperation.completed(1).await?;
            if prior != current {
                changed = Some(factor);
                break;
            }
        }
        let percent = changed.map_or(0.0, |factor| gap_percent(&options.factor_gap, factor));
        percentages.push(percent);
        total_percent += percent;
        cooperation.completed(1).await?;
    }
    // A percentage is measured against the final category-axis width,
    // including one unit per slot. Solve that width before placing any slot.
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
    let mut positions = Vec::new();
    positions.try_reserve_exact(slots.len()).map_err(|error| {
        allocation_failure("boxplot: factor positions exceed available memory", error)
    })?;
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
        cooperation.completed(1).await?;
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

async fn separator_positions(
    slots: &[FactorSlot],
    options: &BoxplotOptions,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<f64>> {
    let selected = match &options.factor_separator {
        FactorSeparator::None => return Ok(Vec::new()),
        FactorSeparator::Auto => (1..=slots[0].key.len()).collect::<Vec<_>>(),
        FactorSeparator::Factors(indices) => indices.clone(),
    };
    let mut separators = Vec::new();
    for pair in slots.windows(2) {
        let mut boundary = false;
        for &factor in &selected {
            cooperation.completed(1).await?;
            if pair[0].key[factor - 1] != pair[1].key[factor - 1] {
                boundary = true;
                break;
            }
        }
        if boundary {
            separators.push(pair[0].position * 0.5 + pair[1].position * 0.5);
        }
        cooperation.completed(1).await?;
    }
    Ok(separators)
}

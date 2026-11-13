use super::super::numeric::is_missing_text;
use super::super::text_output::TextSelection;
use super::lines::FillLines;
use super::Neighbor;
use runmat_value::{StringArray, StringElement};

pub(super) fn fill_neighbor_text(data: &mut [StringElement], lines: &FillLines, dir: Neighbor) {
    for line in lines.iter() {
        fill_line_text(data, line.start, line.length, line.stride, dir);
    }
}

pub(super) fn fill_line_text(
    data: &mut [StringElement],
    start: usize,
    len: usize,
    step: usize,
    dir: Neighbor,
) {
    match dir {
        Neighbor::Previous => {
            let mut last: Option<StringElement> = None;
            for i in 0..len {
                let idx = start + i * step;
                if is_missing_text(&data[idx]) {
                    if let Some(value) = &last {
                        data[idx] = value.clone();
                    }
                } else {
                    last = Some(data[idx].clone());
                }
            }
        }
        Neighbor::Next => {
            let mut next: Option<StringElement> = None;
            for i in (0..len).rev() {
                let idx = start + i * step;
                if is_missing_text(&data[idx]) {
                    if let Some(value) = &next {
                        data[idx] = value.clone();
                    }
                } else {
                    next = Some(data[idx].clone());
                }
            }
        }
    }
}

pub(super) fn fill_nearest_text(data: &mut [StringElement], lines: &FillLines) {
    for line in lines.iter() {
        fill_nearest_line_text(data, line.start, line.length, line.stride);
    }
}

pub(super) fn fill_nearest_line_text(
    data: &mut [StringElement],
    start: usize,
    len: usize,
    step: usize,
) {
    let mut runs = super::runs::MissingRuns::new(len);
    while let Some(run) = runs.next(|position| is_missing_text(&data[start + position * step])) {
        for position in run.start..run.end {
            data[start + position * step] = run.nearest_source(position).map_or_else(
                || StringElement::Missing,
                |source| data[start + source * step].clone(),
            );
        }
    }
}

/// Source indices are computed from the original input, never from a
/// partially replaced line. `usize::MAX` denotes the one retained literal.
pub(super) fn fill_selection_map(
    input: &StringArray,
    lines: &FillLines,
    method: NeighborSelection,
    destinations: &mut [usize],
    mut checkpoint: impl FnMut(usize) -> crate::BuiltinResult<()>,
) -> crate::BuiltinResult<()> {
    for (index, destination) in destinations.iter_mut().enumerate() {
        if index % 4096 == 0 {
            checkpoint(index)?;
        }
        *destination = index;
    }
    for line in lines.iter() {
        match method {
            NeighborSelection::Constant => {
                for position in 0..line.length {
                    let index = line.start + position * line.stride;
                    if position % 4096 == 0 {
                        checkpoint(index)?;
                    }
                    if is_missing_text(&input.data()[index]) {
                        destinations[index] = usize::MAX;
                    }
                }
            }
            NeighborSelection::Previous | NeighborSelection::Next => {
                let reverse = matches!(method, NeighborSelection::Next);
                let mut source = None;
                for ordinal in 0..line.length {
                    let position = if reverse {
                        line.length - 1 - ordinal
                    } else {
                        ordinal
                    };
                    let index = line.start + position * line.stride;
                    if ordinal % 4096 == 0 {
                        checkpoint(index)?;
                    }
                    if is_missing_text(&input.data()[index]) {
                        if let Some(source) = source {
                            destinations[index] = source;
                        }
                    } else {
                        source = Some(index);
                    }
                }
            }
            NeighborSelection::Nearest => {
                let mut previous = None;
                for position in 0..line.length {
                    let index = line.start + position * line.stride;
                    if position % 4096 == 0 {
                        checkpoint(index)?;
                    }
                    if is_missing_text(&input.data()[index]) {
                        destinations[index] = previous.unwrap_or(usize::MAX);
                    } else {
                        previous = Some(index);
                    }
                }
                let mut next = None;
                for ordinal in 0..line.length {
                    let position = line.length - 1 - ordinal;
                    let index = line.start + position * line.stride;
                    if ordinal % 4096 == 0 {
                        checkpoint(index)?;
                    }
                    if is_missing_text(&input.data()[index]) {
                        if let Some(next_index) = next {
                            let previous_distance = if destinations[index] == usize::MAX {
                                usize::MAX
                            } else {
                                (index - destinations[index]) / line.stride
                            };
                            let next_distance = (next_index - index) / line.stride;
                            if next_distance < previous_distance {
                                destinations[index] = next_index;
                            }
                        }
                    } else {
                        next = Some(index);
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum NeighborSelection {
    Constant,
    Previous,
    Next,
    Nearest,
}

pub(super) fn selected(index: usize) -> TextSelection {
    if index == usize::MAX {
        TextSelection::Literal
    } else {
        TextSelection::Input(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_matches_original_neighbor_oracle_for_all_eight_entry_patterns() {
        for pattern in 0..256u16 {
            for step in [1, 2] {
                let original: Vec<StringElement> = (0..8)
                    .map(|index| {
                        if pattern & (1 << index) == 0 {
                            format!("value-{index}").into()
                        } else {
                            StringElement::Missing
                        }
                    })
                    .collect();
                let mut data = vec![StringElement::text("other-line"); 8 * step + 1];
                for (index, value) in original.iter().enumerate() {
                    data[1 + index * step] = value.clone();
                }
                fill_nearest_line_text(&mut data, 1, 8, step);
                for (index, value) in original.iter().enumerate() {
                    let expected = if !is_missing_text(value) {
                        value.clone()
                    } else {
                        original
                            .iter()
                            .enumerate()
                            .filter(|(_, value)| !is_missing_text(value))
                            .min_by_key(|(source, _)| (index.abs_diff(*source), *source))
                            .map_or_else(|| StringElement::Missing, |(_, value)| value.clone())
                    };
                    assert_eq!(
                        data[1 + index * step],
                        expected,
                        "pattern {pattern}, step {step}, index {index}"
                    );
                }
                assert!(data
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index == 0 || (index - 1) % step != 0)
                    .all(|(_, value)| value == "other-line"));
            }
        }
    }
}

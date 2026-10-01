use super::super::metadata::MISSING_TEXT;
use super::super::numeric::is_missing_text;
use super::lines::FillLines;
use super::Neighbor;

pub(super) fn fill_neighbor_text(data: &mut [String], lines: &FillLines, dir: Neighbor) {
    for line in lines.iter() {
        fill_line_text(data, line.start, line.length, line.stride, dir);
    }
}

pub(super) fn fill_line_text(
    data: &mut [String],
    start: usize,
    len: usize,
    step: usize,
    dir: Neighbor,
) {
    match dir {
        Neighbor::Previous => {
            let mut last: Option<String> = None;
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
            let mut next: Option<String> = None;
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

pub(super) fn fill_nearest_text(data: &mut [String], lines: &FillLines) {
    for line in lines.iter() {
        fill_nearest_line_text(data, line.start, line.length, line.stride);
    }
}

pub(super) fn fill_nearest_line_text(data: &mut [String], start: usize, len: usize, step: usize) {
    let mut runs = super::runs::MissingRuns::new(len);
    while let Some(run) = runs.next(|position| is_missing_text(&data[start + position * step])) {
        for position in run.start..run.end {
            data[start + position * step] = run.nearest_source(position).map_or_else(
                || MISSING_TEXT.to_owned(),
                |source| data[start + source * step].clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_matches_original_neighbor_oracle_for_all_eight_entry_patterns() {
        for pattern in 0..256u16 {
            for step in [1, 2] {
                let original: Vec<String> = (0..8)
                    .map(|index| {
                        if pattern & (1 << index) == 0 {
                            format!("value-{index}")
                        } else {
                            MISSING_TEXT.to_ascii_uppercase()
                        }
                    })
                    .collect();
                let mut data = vec!["other-line".to_owned(); 8 * step + 1];
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
                            .map_or_else(|| MISSING_TEXT.to_owned(), |(_, value)| value.clone())
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

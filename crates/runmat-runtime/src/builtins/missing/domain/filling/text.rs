use super::super::metadata::MISSING_TEXT;
use super::super::numeric::is_missing_text;
use super::Neighbor;

pub(super) fn fill_neighbor_text(
    data: &mut [String],
    rows: usize,
    cols: usize,
    dim: usize,
    dir: Neighbor,
) {
    if dim == 1 {
        for col in 0..cols {
            fill_line_text(data, col * rows, rows, 1, dir);
        }
    } else {
        for row in 0..rows {
            fill_line_text(data, row, cols, rows, dir);
        }
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

pub(super) fn fill_nearest_text(data: &mut [String], rows: usize, cols: usize, dim: usize) {
    let original = data.to_vec();
    if dim == 1 {
        for col in 0..cols {
            fill_nearest_line_text(&original, data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_nearest_line_text(&original, data, row, cols, rows);
        }
    }
}

pub(super) fn fill_nearest_line_text(
    original: &[String],
    data: &mut [String],
    start: usize,
    len: usize,
    step: usize,
) {
    for i in 0..len {
        let idx = start + i * step;
        if !is_missing_text(&original[idx]) {
            continue;
        }
        let prev = (0..i).rev().find_map(|j| {
            let value = &original[start + j * step];
            (!is_missing_text(value)).then_some((i - j, value.clone()))
        });
        let next = ((i + 1)..len).find_map(|j| {
            let value = &original[start + j * step];
            (!is_missing_text(value)).then_some((j - i, value.clone()))
        });
        data[idx] = match (prev, next) {
            (Some((pd, _)), Some((nd, nv))) if nd < pd => nv,
            (Some((_, pv)), _) => pv,
            (_, Some((_, nv))) => nv,
            _ => MISSING_TEXT.to_string(),
        };
    }
}

pub(in crate::builtins::missing::domain) enum Summary {
    Mean,
    Median,
}

pub(super) fn fill_summary_numeric(
    data: &mut [f64],
    rows: usize,
    cols: usize,
    dim: usize,
    summary: Summary,
) {
    if dim == 1 {
        for col in 0..cols {
            let vals = finite_slice(data, rows, col, true);
            let replacement = summary_value(vals, &summary);
            for row in 0..rows {
                let idx = row + col * rows;
                if data[idx].is_nan() {
                    data[idx] = replacement;
                }
            }
        }
    } else {
        for row in 0..rows {
            let vals = finite_slice(data, rows, row, false);
            let replacement = summary_value(vals, &summary);
            for col in 0..cols {
                let idx = row + col * rows;
                if data[idx].is_nan() {
                    data[idx] = replacement;
                }
            }
        }
    }
}

pub(super) fn finite_slice(data: &[f64], rows: usize, fixed: usize, along_rows: bool) -> Vec<f64> {
    let mut out = Vec::new();
    if along_rows {
        let cols = data.len() / rows;
        for row in 0..rows {
            let value = data[row + fixed * rows];
            if !value.is_nan() {
                out.push(value);
            }
        }
        debug_assert!(fixed < cols);
    } else {
        let cols = data.len() / rows;
        for col in 0..cols {
            let value = data[fixed + col * rows];
            if !value.is_nan() {
                out.push(value);
            }
        }
    }
    out
}

pub(in crate::builtins::missing::domain) fn summary_value(
    mut vals: Vec<f64>,
    summary: &Summary,
) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    match summary {
        Summary::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
        Summary::Median => {
            vals.sort_by(|a, b| a.total_cmp(b));
            let mid = vals.len() / 2;
            if vals.len().is_multiple_of(2) {
                (vals[mid - 1] + vals[mid]) / 2.0
            } else {
                vals[mid]
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Neighbor {
    Previous,
    Next,
}

pub(super) fn fill_neighbor_numeric(
    data: &mut [f64],
    rows: usize,
    cols: usize,
    dim: usize,
    dir: Neighbor,
) {
    if dim == 1 {
        for col in 0..cols {
            fill_line_numeric(data, rows, col, rows, 1, dir);
        }
    } else {
        for row in 0..rows {
            fill_line_numeric(data, rows, row, cols, rows, dir);
        }
    }
}

pub(super) fn fill_line_numeric(
    data: &mut [f64],
    _rows: usize,
    start: usize,
    len: usize,
    step: usize,
    dir: Neighbor,
) {
    match dir {
        Neighbor::Previous => {
            let mut last = None;
            for i in 0..len {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = last {
                        data[idx] = value;
                    }
                } else {
                    last = Some(data[idx]);
                }
            }
        }
        Neighbor::Next => {
            let mut next = None;
            for i in (0..len).rev() {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = next {
                        data[idx] = value;
                    }
                } else {
                    next = Some(data[idx]);
                }
            }
        }
    }
}

pub(super) fn fill_nearest_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize) {
    let original = data.to_vec();
    if dim == 1 {
        for col in 0..cols {
            fill_nearest_line_numeric(&original, data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_nearest_line_numeric(&original, data, row, cols, rows);
        }
    }
}

pub(super) fn fill_nearest_line_numeric(
    original: &[f64],
    data: &mut [f64],
    start: usize,
    len: usize,
    step: usize,
) {
    for i in 0..len {
        let idx = start + i * step;
        if !original[idx].is_nan() {
            continue;
        }
        let prev = (0..i).rev().find_map(|j| {
            let value = original[start + j * step];
            (!value.is_nan()).then_some((i - j, value))
        });
        let next = ((i + 1)..len).find_map(|j| {
            let value = original[start + j * step];
            (!value.is_nan()).then_some((j - i, value))
        });
        data[idx] = match (prev, next) {
            (Some((pd, _)), Some((nd, nv))) if nd < pd => nv,
            (Some((_, pv)), _) => pv,
            (_, Some((_, nv))) => nv,
            _ => f64::NAN,
        };
    }
}

pub(super) fn fill_linear_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize) {
    if dim == 1 {
        for col in 0..cols {
            fill_linear_line(data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_linear_line(data, row, cols, rows);
        }
    }
}

pub(super) fn fill_linear_line(data: &mut [f64], start: usize, len: usize, step: usize) {
    let mut i = 0;
    while i < len {
        let idx = start + i * step;
        if !data[idx].is_nan() {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < len && data[start + i * step].is_nan() {
            i += 1;
        }
        let run_end = i;
        let prev = (run_start > 0).then(|| data[start + (run_start - 1) * step]);
        let next = (run_end < len).then(|| data[start + run_end * step]);
        match (prev, next) {
            (Some(a), Some(b)) if !a.is_nan() && !b.is_nan() => {
                let span = (run_end - run_start + 1) as f64;
                for (offset, pos) in (run_start..run_end).enumerate() {
                    data[start + pos * step] = a + (b - a) * ((offset + 1) as f64 / span);
                }
            }
            (Some(a), _) if !a.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = a;
                }
            }
            (_, Some(b)) if !b.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = b;
                }
            }
            _ => {}
        }
    }
}

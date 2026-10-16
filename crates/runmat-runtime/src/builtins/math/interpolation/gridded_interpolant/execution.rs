use super::*;

pub(super) fn interpolate_point(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    if point.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if point_outside_grid(spec, point) {
        return extrapolate_point(spec, values, point, series_offset);
    }
    match spec.method {
        InterpMethod::Linear => interpolate_linear(spec, values, point, series_offset),
        InterpMethod::Nearest => interpolate_nearest(spec, values, point, series_offset),
        InterpMethod::Next => interpolate_step(spec, values, point[0], series_offset, true),
        InterpMethod::Previous => interpolate_step(spec, values, point[0], series_offset, false),
        InterpMethod::Pchip | InterpMethod::Spline | InterpMethod::Makima => {
            unreachable!("higher-order methods use evaluate_piecewise_interpolant")
        }
    }
}

pub(super) fn point_outside_grid(spec: &InterpolantSpec, point: &[f64]) -> bool {
    point.iter().enumerate().any(|(dim, coord)| {
        *coord < spec.grid_vectors[dim][0]
            || *coord > spec.grid_vectors[dim][spec.grid_vectors[dim].len() - 1]
    })
}

pub(super) fn extrapolate_point(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    match spec.extrapolation {
        ExtrapolationMethod::None => f64::NAN,
        ExtrapolationMethod::Linear => interpolate_linear(spec, values, point, series_offset),
        ExtrapolationMethod::Nearest => interpolate_nearest(spec, values, point, series_offset),
        ExtrapolationMethod::Next => interpolate_step(spec, values, point[0], series_offset, true),
        ExtrapolationMethod::Previous => {
            interpolate_step(spec, values, point[0], series_offset, false)
        }
        ExtrapolationMethod::Pchip | ExtrapolationMethod::Spline | ExtrapolationMethod::Makima => {
            unreachable!("piecewise extrapolation uses evaluate_piecewise_interpolant")
        }
    }
}

pub(super) fn is_piecewise_method(method: InterpMethod) -> bool {
    matches!(
        method,
        InterpMethod::Pchip | InterpMethod::Spline | InterpMethod::Makima
    )
}

pub(super) fn is_step_method(method: InterpMethod) -> bool {
    matches!(method, InterpMethod::Next | InterpMethod::Previous)
}

pub(super) fn is_step_extrapolation(method: ExtrapolationMethod) -> bool {
    matches!(
        method,
        ExtrapolationMethod::Next | ExtrapolationMethod::Previous
    )
}

pub(super) fn evaluate_piecewise_scalar(
    spec: &InterpolantSpec,
    values: &[f64],
    pp: &PiecewisePolynomial,
    coord: f64,
    series_offset: usize,
) -> f64 {
    if coord.is_nan() {
        return f64::NAN;
    }
    let grid = &spec.grid_vectors[0];
    let in_range = coord >= grid[0] && coord <= grid[grid.len() - 1];
    if !in_range {
        match spec.extrapolation {
            ExtrapolationMethod::None => return f64::NAN,
            ExtrapolationMethod::Nearest => {
                let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
                return values[series_offset + idx];
            }
            ExtrapolationMethod::Next => {
                return interpolate_step(spec, values, coord, series_offset, true);
            }
            ExtrapolationMethod::Previous => {
                return interpolate_step(spec, values, coord, series_offset, false);
            }
            ExtrapolationMethod::Linear => {
                return interpolate_linear(spec, values, &[coord], series_offset);
            }
            ExtrapolationMethod::Pchip
            | ExtrapolationMethod::Spline
            | ExtrapolationMethod::Makima => {}
        }
    }
    eval_pp_scalar(pp, 0, coord, &Extrapolation::Extrapolate)
}

pub(super) fn interpolate_linear(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    let mut brackets = Vec::with_capacity(point.len());
    for (dim, &coord) in point.iter().enumerate() {
        let Some(bracket) = bracket_for_linear(&spec.grid_vectors[dim], coord, spec.extrapolation)
        else {
            return f64::NAN;
        };
        brackets.push(bracket);
    }
    if brackets.len() >= usize::BITS as usize {
        return f64::NAN;
    }
    let corners = 1usize << brackets.len();
    let mut acc = 0.0;
    for corner in 0..corners {
        let mut weight = 1.0;
        let mut subs = Vec::with_capacity(brackets.len());
        for (dim, bracket) in brackets.iter().enumerate() {
            if (corner >> dim) & 1 == 0 {
                weight *= 1.0 - bracket.t;
                subs.push(bracket.lo);
            } else {
                weight *= bracket.t;
                subs.push(bracket.hi);
            }
        }
        acc += weight * values[series_offset + column_major_offset(&subs, spec.values.shape())];
    }
    acc
}

pub(super) fn bracket_for_linear(
    grid: &[f64],
    coord: f64,
    extrapolation: ExtrapolationMethod,
) -> Option<Bracket> {
    if coord < grid[0] || coord > grid[grid.len() - 1] {
        match extrapolation {
            ExtrapolationMethod::None => return None,
            ExtrapolationMethod::Nearest => {
                let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
                return Some(Bracket {
                    lo: idx,
                    hi: idx,
                    t: 0.0,
                });
            }
            _ => {}
        }
    }
    let lo = if coord <= grid[0] {
        0
    } else if coord >= grid[grid.len() - 1] {
        grid.len() - 2
    } else {
        match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
            Ok(idx) => {
                return Some(Bracket {
                    lo: idx,
                    hi: idx,
                    t: 0.0,
                })
            }
            Err(idx) => idx - 1,
        }
    };
    let hi = lo + 1;
    let t = (coord - grid[lo]) / (grid[hi] - grid[lo]);
    Some(Bracket { lo, hi, t })
}

pub(super) fn interpolate_nearest(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    let mut subs = Vec::with_capacity(point.len());
    for (dim, &coord) in point.iter().enumerate() {
        let Some(idx) = nearest_index(&spec.grid_vectors[dim], coord, spec.extrapolation) else {
            return f64::NAN;
        };
        subs.push(idx);
    }
    values[series_offset + column_major_offset(&subs, spec.values.shape())]
}

pub(super) fn nearest_index(
    grid: &[f64],
    coord: f64,
    extrapolation: ExtrapolationMethod,
) -> Option<usize> {
    if coord < grid[0] {
        return (extrapolation != ExtrapolationMethod::None).then_some(0);
    }
    if coord > grid[grid.len() - 1] {
        return (extrapolation != ExtrapolationMethod::None).then_some(grid.len() - 1);
    }
    match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
        Ok(idx) => Some(idx),
        Err(idx) => {
            let lo = idx - 1;
            let hi = idx;
            if coord - grid[lo] < grid[hi] - coord {
                Some(lo)
            } else {
                Some(hi)
            }
        }
    }
}

pub(super) fn interpolate_step(
    spec: &InterpolantSpec,
    values: &[f64],
    coord: f64,
    series_offset: usize,
    next: bool,
) -> f64 {
    let grid = &spec.grid_vectors[0];
    if coord < grid[0] || coord > grid[grid.len() - 1] {
        if spec.extrapolation == ExtrapolationMethod::None {
            return f64::NAN;
        }
        let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
        return values[series_offset + idx];
    }
    let idx = match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
        Ok(idx) => idx,
        Err(idx) if next => idx,
        Err(idx) => idx.saturating_sub(1),
    };
    values[series_offset + idx.min(grid.len() - 1)]
}

pub(super) fn column_major_offset(subs: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (dim, &sub) in subs.iter().enumerate() {
        offset += sub * stride;
        stride *= shape.get(dim).copied().unwrap_or(1);
    }
    offset
}

pub(super) fn unravel_column_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .map(|&dim| {
            let sub = if dim == 0 { 0 } else { linear % dim };
            if dim > 0 {
                linear /= dim;
            }
            sub
        })
        .collect()
}

pub(super) fn interp_method_name(method: InterpMethod) -> &'static str {
    match method {
        InterpMethod::Linear => "linear",
        InterpMethod::Nearest => "nearest",
        InterpMethod::Next => "next",
        InterpMethod::Previous => "previous",
        InterpMethod::Pchip => "pchip",
        InterpMethod::Spline => "spline",
        InterpMethod::Makima => "makima",
    }
}

pub(super) fn extrap_method_name(method: ExtrapolationMethod) -> &'static str {
    match method {
        ExtrapolationMethod::Linear => "linear",
        ExtrapolationMethod::Nearest => "nearest",
        ExtrapolationMethod::Next => "next",
        ExtrapolationMethod::Previous => "previous",
        ExtrapolationMethod::Pchip => "pchip",
        ExtrapolationMethod::Spline => "spline",
        ExtrapolationMethod::Makima => "makima",
        ExtrapolationMethod::None => "none",
    }
}

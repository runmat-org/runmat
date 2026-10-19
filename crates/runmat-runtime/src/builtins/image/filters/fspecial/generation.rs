use std::f64::consts::PI;

use runmat_value::Tensor;

use super::{FspecialKernelError, FspecialKernelResult};

pub(super) fn generate_average(rows: usize, cols: usize) -> FspecialKernelResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: LENGTHS are too large"))?;
    if total == 0 {
        return Err(FspecialKernelError::invalid(
            "fspecial: LENGTHS must be positive integers",
        ));
    }
    let fill = 1.0 / total as f64;
    let data = vec![fill; total];
    Tensor::new(data, vec![rows, cols])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_disk(radius: f64, size: usize) -> FspecialKernelResult<Tensor> {
    if radius == 0.0 {
        return Tensor::new(vec![1.0], vec![1, 1])
            .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")));
    }

    let total = size
        .checked_mul(size)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: RADIUS is too large"))?;
    let mut data = vec![0.0f64; total];
    let center = (size as isize / 2) as f64;

    for row in 0..size {
        let y1 = row as f64 - center - 0.5;
        let y2 = y1 + 1.0;
        for col in 0..size {
            let x1 = col as f64 - center - 0.5;
            let x2 = x1 + 1.0;
            let area = circle_rect_area(radius, x1, x2, y1, y2);
            data[col * size + row] = area;
        }
    }

    let normaliser = PI * radius * radius;
    if normaliser <= f64::EPSILON {
        return Err(FspecialKernelError::invalid(
            "fspecial: radius is too small",
        ));
    }
    let mut sum = 0.0;
    for value in &mut data {
        *value /= normaliser;
        sum += *value;
    }
    if sum <= 0.0 {
        return Err(FspecialKernelError::invalid(
            "fspecial: failed to generate disk filter",
        ));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![size, size])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_gaussian(
    rows: usize,
    cols: usize,
    sigma: f64,
) -> FspecialKernelResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: LENGTHS are too large"))?;
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let denom = 2.0 * sigma * sigma;
    let mut data = Vec::with_capacity(total);
    let mut sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / denom)).exp();
            data.push(value);
            sum += value;
        }
    }
    if sum == 0.0 {
        return Err(FspecialKernelError::invalid(
            "fspecial: gaussian generation failed (degenerate sigma)",
        ));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![rows, cols])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_laplacian(alpha: f64) -> FspecialKernelResult<Tensor> {
    let scale = 4.0 / (alpha + 1.0);
    let a = alpha / 4.0;
    let b = (1.0 - alpha) / 4.0;
    let mut data = vec![
        a, b, a, //
        b, -1.0, b, //
        a, b, a,
    ];
    for value in &mut data {
        *value *= scale;
    }
    Tensor::new(data, vec![3, 3])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_log(rows: usize, cols: usize, sigma: f64) -> FspecialKernelResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: LENGTHS are too large"))?;
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let mut gauss = Vec::with_capacity(total);
    let mut gauss_sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / (2.0 * sigma * sigma))).exp();
            gauss_sum += value;
            gauss.push((x, y, value));
        }
    }
    if gauss_sum == 0.0 {
        return Err(FspecialKernelError::invalid(
            "fspecial: failed to normalise Laplacian of Gaussian",
        ));
    }
    let mut data = Vec::with_capacity(rows * cols);
    let sigma2 = sigma * sigma;
    let normaliser = 2.0 * PI * sigma6(sigma);
    for (x, y, g) in gauss {
        let radial = x * x + y * y;
        let value = ((radial - 2.0 * sigma2) * g) / normaliser;
        data.push(value / gauss_sum);
    }
    let sum: f64 = data.iter().sum();
    if sum != 0.0 {
        let correction = sum / data.len() as f64;
        for value in &mut data {
            *value -= correction;
        }
    }
    Tensor::new(data, vec![rows, cols])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

fn sigma6(sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    sigma2 * sigma2 * sigma2
}

pub(super) fn generate_motion(
    length: usize,
    kernel_size: usize,
    angle_degrees: f64,
    oversample: usize,
) -> FspecialKernelResult<Tensor> {
    let total = kernel_size
        .checked_mul(kernel_size)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: LENGTH is too large"))?;
    let total_samples = length
        .checked_mul(oversample)
        .ok_or_else(|| FspecialKernelError::invalid("fspecial: LENGTH is too large"))?;
    let mut data = vec![0.0f64; total];
    let center = (kernel_size as f64 - 1.0) / 2.0;
    let theta = angle_degrees.to_radians();
    let dir_x = theta.cos();
    let dir_y = theta.sin();
    let step = 1.0 / oversample as f64;
    let half = (length as f64 - 1.0) / 2.0;

    for idx in 0..total_samples {
        let t = -half + (idx as f64 + 0.5) * step;
        let x = center + t * dir_x;
        let y = center + t * dir_y;
        deposit_bilinear(&mut data, kernel_size, x, y, 1.0);
    }

    let mut sum = 0.0;
    for value in &data {
        sum += *value;
    }
    if sum == 0.0 {
        return Err(FspecialKernelError::invalid(
            "fspecial: failed to build motion kernel",
        ));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![kernel_size, kernel_size])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

fn deposit_bilinear(data: &mut [f64], size: usize, x: f64, y: f64, contribution: f64) {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let xi = xi as isize;
    let yi = yi as isize;
    for dy in 0..=1 {
        for dx in 0..=1 {
            let px = xi + dx;
            let py = yi + dy;
            if px < 0 || py < 0 || px >= size as isize || py >= size as isize {
                continue;
            }
            let wx = if dx == 0 { 1.0 - xf } else { xf };
            let wy = if dy == 0 { 1.0 - yf } else { yf };
            let weight = wx * wy;
            let idx = (px as usize) * size + py as usize;
            data[idx] += contribution * weight;
        }
    }
}

pub(super) fn generate_prewitt() -> FspecialKernelResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_sobel() -> FspecialKernelResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            2.0, 0.0, -2.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

pub(super) fn generate_unsharp(alpha: f64) -> FspecialKernelResult<Tensor> {
    let denom = alpha + 1.0;
    let mut data = vec![
        -alpha,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        alpha + 5.0,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        -alpha,
    ];
    for value in &mut data {
        *value /= denom;
    }
    Tensor::new(data, vec![3, 3])
        .map_err(|e| FspecialKernelError::internal(format!("fspecial: {e}")))
}

fn circle_rect_area(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if (x1 >= r || y1 >= r || x2 <= -r || y2 <= -r) && min_distance_to_circle(x1, y1, x2, y2) >= r {
        return 0.0;
    }

    if x1 < 0.0 && x2 > 0.0 {
        let left = circle_rect_area(r, x1, 0.0, y1, y2);
        let right = circle_rect_area(r, 0.0, x2, y1, y2);
        return left + right;
    }
    if y1 < 0.0 && y2 > 0.0 {
        let bottom = circle_rect_area(r, x1, x2, y1, 0.0);
        let top = circle_rect_area(r, x1, x2, 0.0, y2);
        return bottom + top;
    }
    if x2 <= 0.0 {
        return circle_rect_area(r, -x2, -x1, y1, y2);
    }
    if y2 <= 0.0 {
        return circle_rect_area(r, x1, x2, -y2, -y1);
    }
    circle_rect_area_first_quadrant(r, x1.max(0.0), x2.min(r), y1.max(0.0), y2.min(r))
}

fn min_distance_to_circle(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let cx = if x1 > 0.0 {
        x1
    } else if x2 < 0.0 {
        x2
    } else {
        0.0
    };
    let cy = if y1 > 0.0 {
        y1
    } else if y2 < 0.0 {
        y2
    } else {
        0.0
    };
    (cx * cx + cy * cy).sqrt()
}

fn circle_rect_area_first_quadrant(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if x1 >= r || y1 >= r {
        return 0.0;
    }
    let xa = x1.max(0.0);
    let xb = x2.min(r);
    if xb <= xa {
        return 0.0;
    }
    let ya = y1.max(0.0);
    let yb = y2.min(r);
    if yb <= ya {
        return 0.0;
    }
    let rsq = r * r;
    let x_for_y = |y: f64| -> f64 {
        if y >= r {
            0.0
        } else {
            (rsq - y * y).sqrt()
        }
    };
    let x_full = xb.min(x_for_y(yb));
    let mut area = 0.0;
    if x_full > xa {
        area += (yb - ya) * (x_full - xa);
    }
    let x_partial_start = x_full.max(xa);
    let x_partial_end = xb.min(x_for_y(ya));
    if x_partial_end > x_partial_start {
        let arc = arc_integral(r, x_partial_start, x_partial_end);
        area += arc - ya * (x_partial_end - x_partial_start);
    }
    area
}

fn arc_integral(radius: f64, a: f64, b: f64) -> f64 {
    primitive_arc(radius, b) - primitive_arc(radius, a)
}

fn primitive_arc(radius: f64, x: f64) -> f64 {
    let r = radius;
    if x <= -r {
        return -0.5 * PI * r * r;
    }
    if x >= r {
        return 0.5 * PI * r * r;
    }
    let term = (r * r - x * x).max(0.0).sqrt();
    0.5 * (x * term + r * r * clamp_asin(x / r))
}

fn clamp_asin(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).asin()
}

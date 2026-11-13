//! Eigenvectors of a retained real generalized Schur form.
//!
//! Back-substitution acts on its one- and two-dimensional diagonal blocks.
//! The two real columns of a conjugate pair encode the real and imaginary
//! parts of one complex vector, as required by the real-QZ output contract.

use faer::{c64, Mat};

use super::algorithm::RealQz;
use super::eigvectors::{
    component_scale, conj, divide, homogeneous_row_residual, magnitude, normalize,
    same_homogeneous_eigenvalue, scaled,
};

fn blocks(form: &RealQz) -> Result<Vec<(usize, usize)>, &'static str> {
    let n = form.aa.nrows();
    let mut result = Vec::new();
    let mut i = 0;
    while i < n {
        let size = if i + 1 < n && form.aa[(i + 1, i)] != 0.0 {
            if i + 2 < n && form.aa[(i + 2, i + 1)] != 0.0 {
                return Err("overlapping generalized Schur blocks");
            }
            2
        } else {
            1
        };
        result.push((i, size));
        i += size;
    }
    Ok(result)
}

fn coefficient(form: &RealQz, row: usize, col: usize, alpha: c64, beta: c64, scale: f64) -> c64 {
    beta * c64::new(form.aa[(row, col)] / scale, 0.0)
        - alpha * c64::new(form.bb[(row, col)] / scale, 0.0)
}

fn entry(
    form: &RealQz,
    row: usize,
    col: usize,
    alpha: c64,
    beta: c64,
    scale: f64,
    left: bool,
) -> c64 {
    if left {
        conj(coefficient(form, col, row, alpha, beta, scale))
    } else {
        coefficient(form, row, col, alpha, beta, scale)
    }
}

fn residual_terms(
    form: &RealQz,
    position: (usize, usize),
    eigenvalue: (c64, c64),
    scale: f64,
    left: bool,
    component: c64,
) -> (c64, c64) {
    let (row, col) = position;
    let (alpha, beta) = eigenvalue;
    let (a, b) = if left {
        (form.aa[(col, row)], form.bb[(col, row)])
    } else {
        (form.aa[(row, col)], form.bb[(row, col)])
    };
    let a = beta * c64::new(a / scale, 0.0);
    let b = alpha * c64::new(b / scale, 0.0);
    if left {
        (conj(a) * component, conj(b) * component)
    } else {
        (a * component, b * component)
    }
}

fn solve_two(
    coefficients: [[c64; 2]; 2],
    rhs: [c64; 2],
    tolerance: f64,
) -> Result<[c64; 2], &'static str> {
    let local_scale = coefficients
        .iter()
        .flatten()
        .copied()
        .map(magnitude)
        .fold(0.0_f64, f64::max);
    if local_scale == 0.0 {
        return if rhs.iter().all(|&value| magnitude(value) == 0.0) {
            Ok([c64::new(0.0, 0.0); 2])
        } else {
            Err("generalized eigenvector block is singular")
        };
    }
    let a = scaled(coefficients[0][0], local_scale);
    let b = scaled(coefficients[0][1], local_scale);
    let c = scaled(coefficients[1][0], local_scale);
    let d = scaled(coefficients[1][1], local_scale);
    let det = a * d - b * c;
    if magnitude(det) <= tolerance {
        return if rhs
            .iter()
            .all(|&value| magnitude(value) <= tolerance * local_scale)
        {
            Ok([c64::new(0.0, 0.0); 2])
        } else {
            Err("generalized eigenvector block solve is ill-conditioned")
        };
    }
    let r0 = scaled(rhs[0], local_scale);
    let r1 = scaled(rhs[1], local_scale);
    let answer = [divide(d * r0 - b * r1, det), divide(a * r1 - c * r0, det)];
    if answer.iter().any(|&value| !magnitude(value).is_finite()) {
        return Err("generalized eigenvector block solve overflowed");
    }
    Ok(answer)
}

fn seed_two(coefficients: [[c64; 2]; 2]) -> Result<[c64; 2], &'static str> {
    let row0 = magnitude(coefficients[0][0]).max(magnitude(coefficients[0][1]));
    let row1 = magnitude(coefficients[1][0]).max(magnitude(coefficients[1][1]));
    let local_scale = row0.max(row1);
    if local_scale == 0.0 {
        return Err("indeterminate generalized eigenvalue has no unique eigenvector");
    }
    let mut seed = if row0 >= row1 {
        [
            -scaled(coefficients[0][1], local_scale),
            scaled(coefficients[0][0], local_scale),
        ]
    } else {
        [
            -scaled(coefficients[1][1], local_scale),
            scaled(coefficients[1][0], local_scale),
        ]
    };
    normalize(&mut seed)?;
    // The full homogeneous row check below certifies this seed. Testing only
    // the already-subtracted 2x2 coefficients rejects rounded eigenpairs.
    Ok(seed)
}

fn solve_column(
    form: &RealQz,
    blocks: &[(usize, usize)],
    selected: usize,
    left: bool,
    scale: f64,
) -> Result<Vec<c64>, &'static str> {
    let n = form.aa.nrows();
    let alpha = c64::new(form.alpha_re[selected], form.alpha_im[selected]);
    let beta = c64::new(form.beta[selected], 0.0);
    let eig_scale = component_scale(alpha).max(component_scale(beta));
    if eig_scale == 0.0 {
        return Err("indeterminate generalized eigenvalue has no unique eigenvector");
    }
    let alpha = scaled(alpha, eig_scale);
    let beta = scaled(beta, eig_scale);
    let threshold = 128.0 * f64::EPSILON * (n as f64).max(1.0);
    let selected_block = blocks
        .iter()
        .position(|&(start, size)| selected >= start && selected < start + size)
        .ok_or("generalized Schur eigenvalue is outside the retained block order")?;
    let (start, size) = blocks[selected_block];
    let mut vector = vec![c64::new(0.0, 0.0); n];
    if size == 1 {
        vector[start] = c64::new(1.0, 0.0);
    } else {
        let matrix = [
            [
                entry(form, start, start, alpha, beta, scale, left),
                entry(form, start, start + 1, alpha, beta, scale, left),
            ],
            [
                entry(form, start + 1, start, alpha, beta, scale, left),
                entry(form, start + 1, start + 1, alpha, beta, scale, left),
            ],
        ];
        let seed = seed_two(matrix)?;
        vector[start] = seed[0];
        vector[start + 1] = seed[1];
    }

    let indices: Box<dyn Iterator<Item = usize>> = if left {
        Box::new((selected_block + 1)..blocks.len())
    } else {
        Box::new((0..selected_block).rev())
    };
    for index in indices {
        let (block_start, block_size) = blocks[index];
        let mut rhs = [c64::new(0.0, 0.0); 2];
        for offset in 0..block_size {
            let row = block_start + offset;
            for col in 0..n {
                if (left && col < block_start) || (!left && col >= block_start + block_size) {
                    rhs[offset] -= entry(form, row, col, alpha, beta, scale, left) * vector[col];
                }
            }
        }
        if block_size == 1 {
            let diagonal = entry(form, block_start, block_start, alpha, beta, scale, left);
            let local_scale = magnitude(diagonal).max(magnitude(rhs[0]));
            if magnitude(diagonal) <= threshold * local_scale {
                if magnitude(rhs[0]) > threshold * local_scale {
                    return Err("generalized eigenvector solve is singular or defective");
                }
                vector[block_start] = c64::new(0.0, 0.0);
            } else {
                vector[block_start] = divide(rhs[0], diagonal);
            }
        } else {
            let matrix = [
                [
                    entry(form, block_start, block_start, alpha, beta, scale, left),
                    entry(form, block_start, block_start + 1, alpha, beta, scale, left),
                ],
                [
                    entry(form, block_start + 1, block_start, alpha, beta, scale, left),
                    entry(
                        form,
                        block_start + 1,
                        block_start + 1,
                        alpha,
                        beta,
                        scale,
                        left,
                    ),
                ],
            ];
            let answer = solve_two(matrix, rhs, threshold)?;
            vector[block_start] = answer[0];
            vector[block_start + 1] = answer[1];
        }
    }
    normalize(&mut vector)?;
    for row in 0..n {
        if !homogeneous_row_residual(
            (0..n).map(|col| {
                residual_terms(form, (row, col), (alpha, beta), scale, left, vector[col])
            }),
            threshold * 16.0,
        ) {
            return Err("generalized eigenvector residual is too large");
        }
    }
    Ok(vector)
}

fn backtransform(unitary: &Mat<f64>, basis: &[c64]) -> Result<Vec<c64>, &'static str> {
    let n = unitary.nrows();
    let mut result = vec![c64::new(0.0, 0.0); n];
    for row in 0..n {
        for col in 0..n {
            result[row] += c64::new(unitary[(row, col)], 0.0) * basis[col];
        }
    }
    normalize(&mut result)?;
    Ok(result)
}

fn solve_equivalent_column(
    form: &RealQz,
    blocks: &[(usize, usize)],
    selected: usize,
    left: bool,
    scale: f64,
) -> Result<Vec<c64>, &'static str> {
    let alpha = c64::new(form.alpha_re[selected], form.alpha_im[selected]);
    let beta = c64::new(form.beta[selected], 0.0);
    let eig_scale = component_scale(alpha).max(component_scale(beta));
    if eig_scale == 0.0 {
        return Err("indeterminate generalized eigenvalue has no unique eigenvector");
    }
    let desired_alpha = scaled(alpha, eig_scale);
    let desired_beta = scaled(beta, eig_scale);
    let tolerance = 128.0 * f64::EPSILON * (form.aa.nrows() as f64).max(1.0) * 16.0;
    let mut failure = "generalized eigenvector solve is ill-conditioned";
    for candidate in
        std::iter::once(selected).chain((0..form.aa.nrows()).filter(|&i| i != selected))
    {
        let candidate_alpha = c64::new(form.alpha_re[candidate], form.alpha_im[candidate]);
        let candidate_beta = c64::new(form.beta[candidate], 0.0);
        if !same_homogeneous_eigenvalue(alpha, beta, candidate_alpha, candidate_beta) {
            continue;
        }
        match solve_column(form, blocks, candidate, left, scale) {
            Ok(vector) => {
                let satisfies_selected = (0..form.aa.nrows()).all(|row| {
                    homogeneous_row_residual(
                        (0..form.aa.nrows()).map(|col| {
                            residual_terms(
                                form,
                                (row, col),
                                (desired_alpha, desired_beta),
                                scale,
                                left,
                                vector[col],
                            )
                        }),
                        tolerance,
                    )
                });
                if satisfies_selected {
                    return Ok(vector);
                }
                failure = "generalized eigenvector residual is too large";
            }
            Err(error) => failure = error,
        }
    }
    Err(failure)
}

pub(super) struct RealEigenvectors {
    pub right: Mat<f64>,
    pub left: Option<Mat<f64>>,
}

pub(super) fn real_eigenvectors(
    form: &RealQz,
    need_left: bool,
) -> Result<RealEigenvectors, &'static str> {
    let n = form.aa.nrows();
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max(form.aa[(i, j)].abs()).max(form.bb[(i, j)].abs())
        });
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let diagonal_blocks = blocks(form)?;
    let mut right = Mat::<f64>::zeros(n, n);
    let mut left = need_left.then(|| Mat::<f64>::zeros(n, n));
    for &(start, size) in &diagonal_blocks {
        if size == 2
            && (form.alpha_im[start] == 0.0
                || form.alpha_im[start + 1] == 0.0
                || form.beta[start] == 0.0
                || form.beta[start + 1] == 0.0
                || (form.alpha_im[start].is_sign_positive() == form.beta[start].is_sign_positive())
                    == (form.alpha_im[start + 1].is_sign_positive()
                        == form.beta[start + 1].is_sign_positive()))
        {
            return Err("real generalized Schur pair lacks conjugate eigenvalue metadata");
        }
        let chosen = if size == 2
            && form.alpha_im[start].is_sign_positive() != form.beta[start].is_sign_positive()
        {
            start + 1
        } else {
            start
        };
        let right_basis = solve_equivalent_column(form, &diagonal_blocks, chosen, false, scale)?;
        let right_vector = backtransform(&form.right, &right_basis)?;
        let left_vector = if need_left {
            let basis = solve_equivalent_column(form, &diagonal_blocks, chosen, true, scale)?;
            Some(backtransform(&form.left, &basis)?)
        } else {
            None
        };
        for row in 0..n {
            right[(row, start)] = right_vector[row].re;
            if size == 2 {
                let sign = if chosen == start { 1.0 } else { -1.0 };
                right[(row, start + 1)] = sign * right_vector[row].im;
            }
            if let (Some(matrix), Some(vector)) = (left.as_mut(), left_vector.as_ref()) {
                matrix[(row, start)] = vector[row].re;
                if size == 2 {
                    let sign = if chosen == start { 1.0 } else { -1.0 };
                    matrix[(row, start + 1)] = sign * vector[row].im;
                }
            }
        }
    }
    Ok(RealEigenvectors { right, left })
}

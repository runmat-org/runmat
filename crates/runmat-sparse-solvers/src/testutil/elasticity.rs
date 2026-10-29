//! Structured-grid trilinear (Q1 hexahedron) linear elasticity, assembled row-parallel.

use crate::csr::CsrMatrix;
use crate::par;

/// Clamped-face elasticity problem on an `ex × ey × ez` element grid of unit cubes.
pub struct ElasticityProblem {
    pub matrix: CsrMatrix<f64>,
    /// Coordinates of the free nodes (3 per node), aligned with the matrix dofs / 3.
    pub coordinates: Vec<f64>,
    /// Body-force load (gravity-like in −z).
    pub load: Vec<f64>,
}

/// 24×24 element stiffness of a unit cube (E, ν), 2×2×2 Gauss, dof order (node, component).
fn element_stiffness(young: f64, poisson: f64) -> [[f64; 24]; 24] {
    let lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = young / (2.0 * (1.0 + poisson));
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let g = 0.5 / 3f64.sqrt();
    let points = [0.5 - g, 0.5 + g];
    let mut k = [[0.0; 24]; 24];
    for &x in &points {
        for &y in &points {
            for &z in &points {
                // Shape function gradients at (x,y,z) for the unit cube.
                let mut grad = [[0.0; 3]; 8];
                for (a, c) in corners.iter().enumerate() {
                    let fx = if c[0] == 1.0 { x } else { 1.0 - x };
                    let fy = if c[1] == 1.0 { y } else { 1.0 - y };
                    let fz = if c[2] == 1.0 { z } else { 1.0 - z };
                    let dx = if c[0] == 1.0 { 1.0 } else { -1.0 };
                    let dy = if c[1] == 1.0 { 1.0 } else { -1.0 };
                    let dz = if c[2] == 1.0 { 1.0 } else { -1.0 };
                    grad[a] = [dx * fy * fz, fx * dy * fz, fx * fy * dz];
                }
                let weight = 0.125;
                for a in 0..8 {
                    for b in 0..8 {
                        for i in 0..3 {
                            for j in 0..3 {
                                // λ ∂_i N_a ∂_j N_b + μ (δ_ij ∇N_a·∇N_b + ∂_j N_a ∂_i N_b)
                                let mut value =
                                    lambda * grad[a][i] * grad[b][j] + mu * grad[a][j] * grad[b][i];
                                if i == j {
                                    value += mu
                                        * (grad[a][0] * grad[b][0]
                                            + grad[a][1] * grad[b][1]
                                            + grad[a][2] * grad[b][2]);
                                }
                                k[3 * a + i][3 * b + j] += weight * value;
                            }
                        }
                    }
                }
            }
        }
    }
    k
}

/// Builds the clamped (`z = 0` face fixed) problem with free dofs only.
pub fn elasticity3d(ex: usize, ey: usize, ez: usize) -> ElasticityProblem {
    let (nx, ny, nz) = (ex + 1, ey + 1, ez + 1);
    let ke = element_stiffness(1.0, 0.3);
    // Free nodes are those with k ≥ 1; renumber.
    let free_index = |i: usize, j: usize, k: usize| -> Option<usize> {
        (k >= 1).then(|| ((k - 1) * ny + j) * nx + i)
    };
    let free_nodes = nx * ny * (nz - 1);
    let n = 3 * free_nodes;
    let rows: Vec<Vec<(Vec<usize>, Vec<f64>)>> = par::map_range(free_nodes, 64, |f| {
        let i = f % nx;
        let j = (f / nx) % ny;
        let k = f / (nx * ny) + 1;
        // Neighbor free nodes (sorted) and their 3×3 blocks.
        let mut neighbors: Vec<(usize, [[f64; 3]; 3])> = Vec::with_capacity(27);
        for dk in [-1i64, 0, 1] {
            for dj in [-1i64, 0, 1] {
                for di in [-1i64, 0, 1] {
                    let (ii, jj, kk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                    if ii < 0
                        || jj < 0
                        || kk < 1
                        || ii >= nx as i64
                        || jj >= ny as i64
                        || kk >= nz as i64
                    {
                        continue;
                    }
                    neighbors.push((
                        free_index(ii as usize, jj as usize, kk as usize).unwrap(),
                        [[0.0; 3]; 3],
                    ));
                }
            }
        }
        neighbors.sort_by_key(|e| e.0);
        // Elements touching this node.
        let element_range = |c: usize, count: usize| c.saturating_sub(1)..(c + 1).min(count);
        for ek in element_range(k, ez) {
            for ej in element_range(j, ey) {
                for ei in element_range(i, ex) {
                    let local_nodes = [
                        (ei, ej, ek),
                        (ei + 1, ej, ek),
                        (ei + 1, ej + 1, ek),
                        (ei, ej + 1, ek),
                        (ei, ej, ek + 1),
                        (ei + 1, ej, ek + 1),
                        (ei + 1, ej + 1, ek + 1),
                        (ei, ej + 1, ek + 1),
                    ];
                    let Some(a) = local_nodes
                        .iter()
                        .position(|&(x, y, z)| (x, y, z) == (i, j, k))
                    else {
                        continue;
                    };
                    for (b, &(x, y, z)) in local_nodes.iter().enumerate() {
                        let Some(column) = free_index(x, y, z) else {
                            continue;
                        };
                        let position = neighbors.binary_search_by_key(&column, |e| e.0).unwrap();
                        for r in 0..3 {
                            for c in 0..3 {
                                neighbors[position].1[r][c] += ke[3 * a + r][3 * b + c];
                            }
                        }
                    }
                }
            }
        }
        (0..3)
            .map(|r| {
                let mut cols = Vec::with_capacity(3 * neighbors.len());
                let mut vals = Vec::with_capacity(3 * neighbors.len());
                for (column, block) in &neighbors {
                    for c in 0..3 {
                        cols.push(3 * column + c);
                        vals.push(block[r][c]);
                    }
                }
                (cols, vals)
            })
            .collect()
    });
    let matrix = CsrMatrix::from_rows(n, n, rows.into_iter().flatten().collect());
    let mut coordinates = Vec::with_capacity(n);
    let mut load = vec![0.0; n];
    for f in 0..free_nodes {
        let i = f % nx;
        let j = (f / nx) % ny;
        let k = f / (nx * ny) + 1;
        coordinates.extend_from_slice(&[i as f64, j as f64, k as f64]);
        load[3 * f + 2] = -1.0e-3;
    }
    ElasticityProblem {
        matrix,
        coordinates,
        load,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_rows_sum_to_zero_and_matrix_is_symmetric() {
        let ke = element_stiffness(1.0, 0.3);
        for row in ke.iter() {
            // Rigid translation in each component has zero energy.
            for c in 0..3 {
                let s: f64 = (0..8).map(|b| row[3 * b + c]).sum();
                assert!(s.abs() < 1e-12);
            }
        }
        let p = elasticity3d(3, 2, 2);
        assert!(p.matrix.as_ref().is_symmetric(1e-12, false));
    }
}

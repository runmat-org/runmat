//! Assembled Jacobian of the coupled momentum block.
//!
//! The momentum equations' explicit terms (non-orthogonal diffusion, the transpose viscous
//! stress, linear-upwind advection, the symmetry-plane traction and the trial outlet trace)
//! are linear in the velocity through the cell gradient reconstruction, whose stencil reaches
//! a cell's face neighbours. A face term therefore reaches two rings of cells. The sparsity of
//! that two-ring operator and the linear part of the reconstruction depend only on the mesh
//! and the boundary laws: they are derived once per prepared problem. Each Picard step fills
//! the values, with the flux, the viscosity and the time history frozen, so the momentum
//! block J = A − ∂b/∂u is one assembled operator with exactly the matrix-free action.
use super::{
    boundary::FaceBoundary, geometry_ops::*, momentum::MomentumMode, CanonicalFluidError as Error,
    CanonicalFluidProblem,
};
use crate::{algebra::RealSparseMatrix, topology::FiniteVolumeTopology};
use runmat_sparse_solvers::csr::CsrRef;

/// T[k][j][d] = ∂(∂u_k/∂x_d)/∂u_j(m).
type GradientTerm = [[V; 3]; 3];

/// Mesh- and law-fixed structure of the momentum Jacobian.
pub(super) struct MomentumJacobianPattern {
    gradient: Vec<Vec<(usize, GradientTerm)>>,
    /// Sorted block columns (cells) of each block row (cell).
    rows: Vec<Vec<usize>>,
    /// Scalar row offsets of the component-major 3n × 3n operator.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl MomentumJacobianPattern {
    /// Scalar entries of the assembled 3n × 3n block.
    pub(super) fn entries(&self) -> usize {
        self.col_idx.len()
    }

    pub(super) fn derive(
        t: &FiniteVolumeTopology,
        boundaries: &[FaceBoundary],
        stencils: &[super::velocity_gradient::VelocityGradientStencil],
    ) -> Result<Self, Error> {
        let n = t.cells.len();
        let gradient = super::velocity_gradient::linear_terms(t, stencils);
        let mut rows: Vec<Vec<usize>> = (0..n).map(|c| vec![c]).collect();
        for (i, f) in t.faces.iter().enumerate() {
            if matches!(boundaries[i], FaceBoundary::Symmetry { .. }) {
                continue;
            }
            let mut columns = vec![f.owner];
            columns.extend(gradient[f.owner].iter().map(|(m, _)| *m));
            if let Some(other) = f.neighbor {
                columns.push(other);
                columns.extend(gradient[other].iter().map(|(m, _)| *m));
                rows[other].extend(columns.iter().copied());
            }
            rows[f.owner].extend(columns);
        }
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
            row.shrink_to_fit();
        }
        let mut row_ptr = Vec::with_capacity(3 * n + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for _k in 0..3 {
            for row in &rows {
                for j in 0..3 {
                    col_idx.extend(row.iter().map(|m| j * n + m));
                }
                row_ptr.push(col_idx.len());
            }
        }
        Ok(Self {
            gradient,
            rows,
            row_ptr,
            col_idx,
        })
    }

    /// Sum over cells of their block-row stencil sizes: the scalar operator holds nine
    /// entries per block column.
    pub(super) fn block_column_count(&self) -> u64 {
        self.rows.iter().map(|row| row.len() as u64).sum()
    }

    fn position(&self, k: usize, r: usize, j: usize, m: usize) -> Result<usize, Error> {
        let n = self.rows.len();
        let row = &self.rows[r];
        let at = row
            .binary_search(&m)
            .map_err(|_| Error::Invalid("momentum Jacobian entry outside its pattern"))?;
        Ok(self.row_ptr[k * n + r] + j * row.len() + at)
    }
}

/// Values of the assembled 3n × 3n momentum block on the problem's pattern,
/// component-major (row k·n + c).
pub(super) struct MomentumJacobian {
    values: Vec<f64>,
}

/// One velocity component's preconditioning operator as an owned n × n CSR matrix.
pub(super) struct ComponentBlock {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl ComponentBlock {
    pub(super) fn view(&self) -> CsrRef<'_, f64> {
        CsrRef::new_unchecked(self.n, self.n, &self.row_ptr, &self.col_idx, &self.values)
    }
}

/// Per-component preconditioning operators: the assembled implicit operator A, whose
/// symmetry-plane coefficient c sits on every component, keeps on component k only the
/// normal share c n_k² that J carries. A stays an M-matrix, so its ILU is robust, unlike the
/// ILU of J's indefinite explicit blocks.
pub(super) fn component_operators(
    p: &CanonicalFluidProblem<'_>,
    matrix: &RealSparseMatrix,
    viscosity_gradients: &[[V; 3]],
) -> Result<[ComponentBlock; 3], Error> {
    let n = p.topology.cells.len();
    let offsets = matrix.pattern().row_offsets();
    let columns = matrix.pattern().column_indices();
    let row_ptr = offsets.iter().map(|v| *v as usize).collect::<Vec<_>>();
    let col_idx = columns.iter().map(|v| *v as usize).collect::<Vec<_>>();
    let diagonal = (0..n)
        .map(|r| {
            (row_ptr[r]..row_ptr[r + 1])
                .find(|&at| col_idx[at] == r)
                .ok_or(Error::Invalid("momentum operator lacks a diagonal entry"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut blocks = [
        matrix.values().to_vec(),
        matrix.values().to_vec(),
        matrix.values().to_vec(),
    ];
    for (i, f) in p.topology.faces.iter().enumerate() {
        if let FaceBoundary::Symmetry { .. } = p.boundaries[i] {
            let (coefficient, _) =
                symmetry_traction(p, i, [0.0; 3], &viscosity_gradients[f.owner])?;
            let normal = scale(f.owner_area_vector_m2, 1.0 / f.area_m2.canonical_value());
            for (k, values) in blocks.iter_mut().enumerate() {
                values[diagonal[f.owner]] -= coefficient * (1.0 - normal[k] * normal[k]);
            }
        }
    }
    Ok(blocks.map(|values| ComponentBlock {
        n,
        row_ptr: row_ptr.clone(),
        col_idx: col_idx.clone(),
        values,
    }))
}

impl MomentumJacobian {
    pub(super) fn view<'a>(&'a self, pattern: &'a MomentumJacobianPattern) -> CsrRef<'a, f64> {
        let size = 3 * pattern.rows.len();
        CsrRef::new_unchecked(size, size, &pattern.row_ptr, &pattern.col_idx, &self.values)
    }
}

/// Fill J = A − ∂b/∂u at the frozen flux `q`, time history and face viscosity.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble(
    p: &CanonicalFluidProblem<'_>,
    matrix: &RealSparseMatrix,
    q: &[f64],
    mode: MomentumMode,
    time: Option<super::transient::TimeDerivative<'_>>,
    viscosity_gradients: &[[V; 3]],
) -> Result<MomentumJacobian, Error> {
    let pattern = &p.momentum_jacobian;
    let mut values = vec![0.0; pattern.col_idx.len()];
    // Assembled implicit operator, the same on every component.
    let offsets = matrix.pattern().row_offsets();
    let columns = matrix.pattern().column_indices();
    for r in 0..p.topology.cells.len() {
        for at in offsets[r] as usize..offsets[r + 1] as usize {
            let m = columns[at] as usize;
            for k in 0..3 {
                values[pattern.position(k, r, k, m)?] += matrix.values()[at];
            }
        }
    }
    // − ∂b/∂u: subtract a block contribution of the right-hand side.
    let mut subtract = |r: usize, m: usize, block: &[[f64; 3]; 3]| -> Result<(), Error> {
        for k in 0..3 {
            for j in 0..3 {
                if block[k][j] != 0.0 {
                    values[pattern.position(k, r, j, m)?] -= block[k][j];
                }
            }
        }
        Ok(())
    };
    for (i, f) in p.topology.faces.iter().enumerate() {
        if let FaceBoundary::Symmetry { .. } = p.boundaries[i] {
            // b gains c (u − (u·n̂)n̂).
            let (coefficient, _) =
                symmetry_traction(p, i, [0.0; 3], &viscosity_gradients[f.owner])?;
            let normal = scale(f.owner_area_vector_m2, 1.0 / f.area_m2.canonical_value());
            let block = std::array::from_fn(|k| {
                std::array::from_fn(|j| {
                    coefficient * (f64::from(u8::from(k == j)) - normal[k] * normal[j])
                })
            });
            subtract(f.owner, f.owner, &block)?;
            continue;
        }
        let (_, tangent, w) = face_geometry(p, f)?;
        let normal_diffusion = !matches!(p.boundaries[i], FaceBoundary::PressureOutlet { .. });
        let viscosity = {
            let g = if let Some(other) = f.neighbor {
                std::array::from_fn(|k| {
                    interpolate(
                        viscosity_gradients[f.owner][k],
                        viscosity_gradients[other][k],
                        w,
                    )
                })
            } else {
                viscosity_gradients[f.owner]
            };
            face_viscosity(p, &g)
        };
        let area = f.owner_area_vector_m2;
        // Face gradient = Σ λ · (cell gradient), λ = w and 1 − w on an interior face.
        let sources: Vec<(usize, f64)> = match f.neighbor {
            Some(other) => vec![(f.owner, w), (other, 1.0 - w)],
            None => vec![(f.owner, 1.0)],
        };
        // Correction blocks per column cell.
        let mut blocks: Vec<(usize, [[f64; 3]; 3])> = Vec::new();
        let mut add_block =
            |m: usize, block: [[f64; 3]; 3]| match blocks.iter_mut().find(|(cell, _)| *cell == m) {
                Some((_, existing)) => {
                    for k in 0..3 {
                        for j in 0..3 {
                            existing[k][j] += block[k][j];
                        }
                    }
                }
                None => blocks.push((m, block)),
            };
        // Viscous remainder μ [ n_d (∂u_k/∂x · t) + Σ_i (∂u_i/∂x_k) A_i ].
        for &(cell, lambda) in &sources {
            for (m, term) in &p.momentum_jacobian.gradient[cell] {
                let block = std::array::from_fn(|k| {
                    std::array::from_fn(|j| {
                        let normal_part = if normal_diffusion {
                            dot(term[k][j], tangent)
                        } else {
                            0.0
                        };
                        let transpose = (0..3).map(|i| term[i][j][k] * area[i]).sum::<f64>();
                        viscosity * lambda * (normal_part + transpose)
                    })
                });
                add_block(*m, block);
            }
        }
        // Advection: the correction subtracts the linear-upwind remainder (or, for a trial
        // outlet backflow, the full owner trace times the mass flux).
        let transported = q[i] - time.map_or(0.0, |t| t.mesh_flux(i));
        let mass_flux = p.density * transported;
        let backflow =
            matches!(p.boundaries[i], FaceBoundary::PressureOutlet { .. }) && transported < 0.0;
        let trace = |cell: usize, factor: f64, add_block: &mut dyn FnMut(usize, [[f64; 3]; 3])| {
            let d = sub(f.centroid_m, p.topology.cells[cell].centroid_m);
            for (m, term) in &p.momentum_jacobian.gradient[cell] {
                add_block(
                    *m,
                    std::array::from_fn(|k| std::array::from_fn(|j| -factor * dot(term[k][j], d))),
                );
            }
        };
        if backflow {
            if mode == MomentumMode::Original {
                return Err(Error::Unsupported("pressure outlet backflow"));
            }
            add_block(
                f.owner,
                std::array::from_fn(|k| {
                    std::array::from_fn(|j| -mass_flux * f64::from(u8::from(k == j)))
                }),
            );
            trace(f.owner, mass_flux, &mut add_block);
        } else {
            let velocity_face =
                f.neighbor.is_none() && matches!(p.boundaries[i], FaceBoundary::Velocity { .. });
            if velocity_face {
                // Upwind is the boundary value; the remainder ρq (value − u_owner) for q ≥ 0.
                if transported >= 0.0 {
                    add_block(
                        f.owner,
                        std::array::from_fn(|k| {
                            std::array::from_fn(|j| mass_flux * f64::from(u8::from(k == j)))
                        }),
                    );
                }
            } else if transported >= 0.0 {
                trace(f.owner, mass_flux, &mut add_block);
            } else if let Some(other) = f.neighbor {
                trace(other, mass_flux, &mut add_block);
            }
        }
        for (m, block) in &blocks {
            subtract(f.owner, *m, block)?;
            if let Some(other) = f.neighbor {
                let negated = block.map(|row| row.map(|v| -v));
                subtract(other, *m, &negated)?;
            }
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite momentum Jacobian"));
    }
    Ok(MomentumJacobian { values })
}

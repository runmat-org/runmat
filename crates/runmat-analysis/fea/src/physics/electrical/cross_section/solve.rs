//! The two electrostatic solves of a cross-section and its Maxwell capacitance matrices.
//!
//! On linear triangles the dielectric potential-gradient operator has the element matrix
//! K_e = ε0 εr / (4 A) (b bᵀ + c cᵀ), with b and c the coordinate differences of the corners.
//! Conductor nodes are prescribed (1 V on the excited signal conductor, 0 V on the others,
//! the coplanar ground and the reference planes); the side walls and an open lid are
//! insulating (zero normal flux). One Cholesky factor of the free block serves every signal
//! conductor as a right-hand side. The Maxwell capacitance matrix is the energy form
//! C_ij = u_iᵀ K u_j of the full potential vectors, symmetric by construction.
use super::{geometry::Geometry, mesh::CrossSectionMesh, ConductorRole, CrossSectionError};
use crate::physics::electrical::VACUUM_PERMITTIVITY_F_PER_M;
use runmat_sparse_solvers::direct::{DirectOptions, DirectSolver, FactorKind, SymbolicFactorization};
use runmat_sparse_solvers::TripletBuilder;

/// Square matrix, row-major.
pub(super) type Matrix = Vec<Vec<f64>>;

pub(super) struct CapacitanceMatrices {
    /// Maxwell capacitance per length with the real dielectrics, F/m.
    pub dielectric: Matrix,
    /// The same with every dielectric replaced by vacuum, F/m.
    pub vacuum: Matrix,
    /// Largest relative residual ||K x − b|| / ||b|| over the solves.
    pub relative_residual: f64,
}

/// Which conductor holds a node: `None` free, `Some(None)` a zero-potential conductor,
/// `Some(Some(k))` signal conductor k.
pub(super) type Holder = Option<Option<usize>>;

pub(super) fn holders(geometry: &Geometry, mesh: &CrossSectionMesh) -> Vec<Holder> {
    let tolerance = 1e-9 * (geometry.y_high - geometry.y_low);
    let mut signal_index = Vec::with_capacity(geometry.traces.len());
    let mut next = 0;
    for trace in &geometry.traces {
        if trace.role == ConductorRole::Signal {
            signal_index.push(Some(next));
            next += 1;
        } else {
            signal_index.push(None);
        }
    }
    let mut holders = vec![None; mesh.x.len()];
    for (node, holder) in holders.iter_mut().enumerate() {
        let row = node / mesh.columns;
        if (row == 0 && geometry.plane_low) || (row + 1 == mesh.rows && geometry.plane_high) {
            *holder = Some(None);
            continue;
        }
        let (x, y) = (mesh.x[node], mesh.y[node]);
        if let Some(t) = geometry
            .traces
            .iter()
            .position(|trace| trace.contains(x, y, tolerance))
        {
            *holder = Some(signal_index[t]);
        }
    }
    holders
}

/// Element gradient coefficients (b, c) and area of one triangle.
fn shape(mesh: &CrossSectionMesh, corners: [usize; 3]) -> ([f64; 3], [f64; 3], f64) {
    let [p, q, r] = corners;
    let (x, y) = (&mesh.x, &mesh.y);
    let b = [y[q] - y[r], y[r] - y[p], y[p] - y[q]];
    let c = [x[r] - x[q], x[p] - x[r], x[q] - x[p]];
    let area = 0.5 * (b[0] * c[1] - b[1] * c[0]);
    (b, c, area)
}

pub(super) fn capacitance_matrices(
    geometry: &Geometry,
    mesh: &CrossSectionMesh,
) -> Result<CapacitanceMatrices, CrossSectionError> {
    let holders = holders(geometry, mesh);
    let signals = geometry.signal_names().len();
    let mut free_index = vec![usize::MAX; holders.len()];
    let mut free = 0;
    for (node, holder) in holders.iter().enumerate() {
        if holder.is_none() {
            free_index[node] = free;
            free += 1;
        }
    }
    for k in 0..signals {
        if !holders.contains(&Some(Some(k))) {
            return Err(CrossSectionError::Invalid(format!(
                "signal conductor `{}` holds no mesh node; refine the mesh",
                geometry.signal_names()[k]
            )));
        }
    }
    // Element relative permittivity at the centroid, and the element geometry.
    let mut elements = Vec::with_capacity(mesh.triangles.len());
    for corners in &mesh.triangles {
        let (b, c, area) = shape(mesh, *corners);
        if !(area > 0.0) {
            return Err(CrossSectionError::Operator(
                "a mesh triangle is inverted or degenerate".into(),
            ));
        }
        let cx = corners.iter().map(|&n| mesh.x[n]).sum::<f64>() / 3.0;
        let cy = corners.iter().map(|&n| mesh.y[n]).sum::<f64>() / 3.0;
        elements.push((b, c, area, geometry.relative_permittivity(cx, cy)));
    }
    let mut residual = 0.0_f64;
    let mut solve = |vacuum: bool| -> Result<Matrix, CrossSectionError> {
        let local = |e: usize, i: usize, j: usize| {
            let (b, c, area, er) = &elements[e];
            let er = if vacuum { 1.0 } else { *er };
            VACUUM_PERMITTIVITY_F_PER_M.canonical_value() * er / (4.0 * area)
                * (b[i] * b[j] + c[i] * c[j])
        };
        let usage = runmat_simulation_usage::UsageContext::capture();
        // Nine entries of six operations each per element, plus the right-hand sides.
        let operations = (mesh.triangles.len() as u64).saturating_mul(54 + 9 * signals as u64);
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    runmat_simulation_usage::UsageCategory::Assembly,
                    runmat_simulation_usage::OperationCount::from_operations(operations),
                )
            })
            .transpose()
            .map_err(CrossSectionError::Accounting)?;
        let mut triplets = TripletBuilder::<f64>::with_capacity(free, free, 9 * mesh.triangles.len());
        // Column-major right-hand sides, one column per signal conductor.
        let mut rhs = vec![0.0; free * signals];
        for (e, corners) in mesh.triangles.iter().enumerate() {
            for i in 0..3 {
                let Some(row) = holders[corners[i]].is_none().then(|| free_index[corners[i]]) else {
                    continue;
                };
                for j in 0..3 {
                    let value = local(e, i, j);
                    match holders[corners[j]] {
                        None => triplets.push(row, free_index[corners[j]], value),
                        Some(Some(k)) => rhs[k * free + row] -= value,
                        Some(None) => {}
                    }
                }
            }
        }
        if let Some(batch) = &mut batch {
            batch
                .complete(runmat_simulation_usage::OperationCount::from_operations(
                    operations,
                ))
                .map_err(CrossSectionError::Accounting)?;
        }
        let matrix = triplets.build();
        let mut solution = rhs.clone();
        if free > 0 {
            let options = DirectOptions::new(FactorKind::Cholesky);
            let plan = crate::solve::shared_sparse::ambient_plan();
            crate::solve::shared_sparse::with_plan(plan, || {
                let control = crate::solve::shared_sparse::control();
                let symbolic =
                    SymbolicFactorization::analyze::<f64>(matrix.as_ref().into(), &options, control)?;
                let required = symbolic.estimate().peak_bytes();
                if let Some(budget) = crate::solve::shared_sparse::default_memory_budget() {
                    if required > budget {
                        return Ok(Err(CrossSectionError::Budget {
                            required_bytes: required,
                            budget_bytes: budget,
                        }));
                    }
                }
                let factor = DirectSolver::<f64>::factorize(
                    symbolic,
                    matrix.as_ref().into(),
                    options,
                    None,
                    control,
                )?;
                factor.solve_in_place(&mut solution, signals)?;
                Ok(Ok(()))
            })
            .map_err(|error| match error {
                runmat_sparse_solvers::SolverError::Cancelled => CrossSectionError::Cancelled,
                error => CrossSectionError::Operator(format!(
                    "the cross-section operator is not positive definite: {error}"
                )),
            })??;
            let mut product = vec![0.0; free];
            for k in 0..signals {
                let column = &solution[k * free..(k + 1) * free];
                if column.iter().any(|v| !v.is_finite()) {
                    return Err(CrossSectionError::Operator(
                        "the cross-section solve produced non-finite potentials".into(),
                    ));
                }
                matrix.as_ref().spmv(column, &mut product);
                let b = &rhs[k * free..(k + 1) * free];
                let norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                let miss = product
                    .iter()
                    .zip(b)
                    .map(|(p, b)| (p - b) * (p - b))
                    .sum::<f64>()
                    .sqrt();
                if norm > 0.0 {
                    residual = residual.max(miss / norm);
                }
            }
        }
        // Energy form over the full potential vectors.
        let potential = |node: usize, k: usize| match holders[node] {
            None => solution[k * free + free_index[node]],
            Some(Some(owner)) => f64::from(u8::from(owner == k)),
            Some(None) => 0.0,
        };
        let mut capacitance = vec![vec![0.0; signals]; signals];
        let mut local_u = vec![[0.0; 3]; signals];
        for (e, corners) in mesh.triangles.iter().enumerate() {
            for (k, u) in local_u.iter_mut().enumerate() {
                for i in 0..3 {
                    u[i] = potential(corners[i], k);
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    let value = local(e, i, j);
                    if value == 0.0 {
                        continue;
                    }
                    for (p, row) in capacitance.iter_mut().enumerate() {
                        for (q, entry) in row.iter_mut().enumerate() {
                            *entry += local_u[p][i] * value * local_u[q][j];
                        }
                    }
                }
            }
        }
        // Symmetrize the round-off.
        for p in 0..signals {
            for q in 0..p {
                let mean = 0.5 * (capacitance[p][q] + capacitance[q][p]);
                capacitance[p][q] = mean;
                capacitance[q][p] = mean;
            }
        }
        Ok(capacitance)
    };
    let dielectric = solve(false)?;
    let vacuum = solve(true)?;
    Ok(CapacitanceMatrices {
        dielectric,
        vacuum,
        relative_residual: residual,
    })
}

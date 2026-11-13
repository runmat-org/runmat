//! Acoustic cavity modes: K phi = omega² M phi on the canonical Tet4/Tet10 pressure operators.
//!
//! Rigid walls are the natural boundary; pressure-release faces (prescribed zero pressure)
//! remove their nodes. Lossy boundaries (impedance, radiation) have complex modes and are
//! solved with a harmonic sweep instead, so a modal step rejects them. A closed cavity keeps
//! its uniform-pressure mode at 0 Hz.
//!
//! The eigensolver is a restarted block shift-invert Krylov method in the M inner product:
//! each block applies (K + s M)⁻¹ M, the basis is M-orthonormalized twice, and Rayleigh-Ritz
//! with K gives the Ritz pairs. Blocks resolve repeated (degenerate) cavity modes. Every
//! returned pair passes its own physical residual |K phi - lambda M phi|.
use super::{
    assembly::AcousticAssembly,
    real_solve::{dot, SpdOperator},
    AcousticHarmonicError,
};
use crate::{
    algebra::{spmv, MatrixProperties, OperatorRole, RealSparseMatrix},
    topology::AnalysisProjection,
};
use runmat_analysis_core::{
    AcousticConstraintKind, AnalysisConstraint, AnalysisModel, AnalysisProcedure, BoundaryOperator,
    IndependentDomain, PhysicsFamily, PrescribedValue, ProcedureSettings, RestartPolicy,
    SolutionSpace, StepObjectRef,
};
use runmat_meshing_core::SolverMeshArtifact;

/// Numerical controls for cavity-mode extraction. The defaults are the authored-study policy.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticModalPolicy {
    /// Relative eigenpair residual |K phi - lambda M phi| / ((|lambda| + s) |M phi|).
    pub relative_tolerance: f64,
    /// Relative residual of each inner (K + s M) solve.
    pub linear_relative_tolerance: f64,
    /// Ceiling on shift-invert operator applications across all restarts.
    pub maximum_operator_applications: usize,
}

impl Default for AcousticModalPolicy {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-8,
            linear_relative_tolerance: 1.0e-12,
            maximum_operator_applications: 4_000,
        }
    }
}

impl AcousticModalPolicy {
    pub fn validate(self) -> Result<(), AcousticHarmonicError> {
        let unit = |value: f64| value.is_finite() && value > 0.0 && value < 1.0;
        if !unit(self.relative_tolerance)
            || !unit(self.linear_relative_tolerance)
            || self.maximum_operator_applications == 0
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "cavity-mode tolerance or iteration policy".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalAcousticModes {
    pub projection: AnalysisProjection,
    /// Ascending natural frequencies in Hz.
    pub frequencies_hz: Vec<f64>,
    /// Mass-normalized pressure mode shapes (phi' M phi = 1) in projection node order.
    /// The sign makes the largest-magnitude nodal value positive.
    pub mode_pressures: Vec<Vec<f64>>,
    pub relative_residuals: Vec<f64>,
    /// max |phi_i' M phi_j - delta_ij| over the returned modes.
    pub mass_orthogonality_error: f64,
    pub operator_applications: usize,
    pub linear_iterations: usize,
}

/// Extract the lowest `requested_count` cavity modes of an authored acoustic modal step.
pub fn solve_canonical_modal(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    policy: AcousticModalPolicy,
) -> Result<CanonicalAcousticModes, AcousticHarmonicError> {
    policy.validate()?;
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| AcousticHarmonicError::InvalidInput("step not found".into()))?;
    if step.family != PhysicsFamily::Acoustic
        || step.procedure != AnalysisProcedure::ModalExtraction
        || step.solution_space != SolutionSpace::Direct
        || !step.state_inputs.is_empty()
        || step.settings != ProcedureSettings::None
        || step.restart != RestartPolicy::Fresh
    {
        return Err(AcousticHarmonicError::Unsupported(
            "a cavity-mode step must be a fresh direct acoustic modal step".into(),
        ));
    }
    let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain else {
        return Err(AcousticHarmonicError::InvalidInput(
            "cavity modes require a requested mode count".into(),
        ));
    };
    if requested_count == 0 {
        return Err(AcousticHarmonicError::InvalidInput(
            "cavity modes require at least one mode".into(),
        ));
    }
    if model.structural.is_some() || !model.connectors.is_empty() || !model.interfaces.is_empty() {
        return Err(AcousticHarmonicError::Unsupported(
            "explicit structural or coupled interfaces".into(),
        ));
    }
    if crate::excitation::active_actions(model, Some(step))
        .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?
        .next()
        .is_some()
    {
        return Err(AcousticHarmonicError::Unsupported(
            "cavity modes are source-free; sources belong to harmonic or transient steps".into(),
        ));
    }
    for boundary in &model.boundaries {
        if !step.object_is_active(&StepObjectRef::Boundary(boundary.boundary_id.clone())) {
            continue;
        }
        match boundary.operator {
            BoundaryOperator::AcousticRigidWall => {}
            BoundaryOperator::AcousticImpedance { .. }
            | BoundaryOperator::AcousticImpedanceLaw { .. }
            | BoundaryOperator::AcousticRadiation
            | BoundaryOperator::AcousticSphericalRadiation { .. } => {
                return Err(AcousticHarmonicError::Unsupported(
                    "impedance and radiation boundaries absorb energy and have complex modes; \
                     cavity modes admit rigid walls and pressure release (use a harmonic sweep \
                     for lossy enclosures)"
                        .into(),
                ))
            }
            _ => {
                return Err(AcousticHarmonicError::Unsupported(
                    "non-acoustic boundary in acoustic model".into(),
                ))
            }
        }
    }
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| AcousticHarmonicError::Material(error.to_string()))?;
    let assembly = AcousticAssembly::assemble(
        projection,
        &properties,
        runmat_types::physical::AngularFrequency::new(0.0),
    )?;
    if assembly.lossy {
        return Err(AcousticHarmonicError::Unsupported(
            "material loss in cavity modes (damped modes are complex)".into(),
        ));
    }
    let mut released = vec![false; assembly.projection.nodes.len()];
    for constraint in &model.constraints {
        if !step.object_is_active(&StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        )) {
            continue;
        }
        let AnalysisConstraint::Acoustic(constraint) = constraint else {
            return Err(AcousticHarmonicError::Unsupported(
                "non-acoustic constraint in acoustic model".into(),
            ));
        };
        match &constraint.kind {
            AcousticConstraintKind::PrescribedPressure {
                target,
                value: PrescribedValue::Constant(value),
            } if value.canonical_value() == 0.0 => {
                let faces = assembly
                    .projection
                    .boundary_faces_for_target(target)
                    .collect::<Vec<_>>();
                if faces.is_empty() {
                    return Err(AcousticHarmonicError::Boundary(format!(
                        "{} must resolve exterior faces",
                        target.region_id
                    )));
                }
                for face in faces {
                    for &node in &face.node_indices {
                        released[node] = true;
                    }
                }
            }
            AcousticConstraintKind::PrescribedPressure { .. } => {
                return Err(AcousticHarmonicError::Unsupported(
                    "cavity modes admit only zero (pressure-release) prescribed pressure".into(),
                ))
            }
            _ => {
                return Err(AcousticHarmonicError::Unsupported(
                    "periodic acoustic constraints in a cavity-mode step".into(),
                ))
            }
        }
    }
    let free = released
        .iter()
        .enumerate()
        .filter_map(|(index, released)| (!released).then_some(index))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return Err(AcousticHarmonicError::Boundary(
            "every pressure node is released; the cavity has no modes".into(),
        ));
    }
    let count = requested_count.min(free.len());
    let stiffness = assembly
        .stiffness
        .with_canonical_storage(|matrix| matrix.principal_submatrix(&free))
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let mass = assembly
        .mass
        .with_canonical_storage(|matrix| matrix.principal_submatrix(&free))
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let pairs = lowest_eigenpairs(&stiffness, &mass, count, policy)?;
    let node_count = assembly.projection.nodes.len();
    let mut frequencies_hz = Vec::with_capacity(count);
    let mut mode_pressures = Vec::with_capacity(count);
    for (eigenvalue, vector) in pairs.values.iter().zip(&pairs.vectors) {
        // Roundoff can leave the rigid mode at a tiny negative eigenvalue.
        frequencies_hz.push(eigenvalue.max(0.0).sqrt() / std::f64::consts::TAU);
        let mut full = vec![0.0; node_count];
        for (reduced, &node) in free.iter().enumerate() {
            full[node] = vector[reduced];
        }
        mode_pressures.push(full);
    }
    Ok(CanonicalAcousticModes {
        projection: assembly.projection,
        frequencies_hz,
        mode_pressures,
        relative_residuals: pairs.residuals,
        mass_orthogonality_error: pairs.orthogonality_error,
        operator_applications: pairs.applications,
        linear_iterations: pairs.linear_iterations,
    })
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    residuals: Vec<f64>,
    orthogonality_error: f64,
    applications: usize,
    linear_iterations: usize,
}

/// Lowest `count` eigenpairs of the symmetric definite pencil (K, M), K semidefinite.
fn lowest_eigenpairs(
    stiffness: &RealSparseMatrix,
    mass: &RealSparseMatrix,
    count: usize,
    policy: AcousticModalPolicy,
) -> Result<Eigenpairs, AcousticHarmonicError> {
    let n = stiffness.rows();
    let block = count.clamp(1, 4).min(n);
    // A small positive shift makes K + s M definite for the 0 Hz cavity mode while keeping
    // the low end of the spectrum well separated under shift-invert.
    let trace_k: f64 = stiffness.diagonal_values().iter().sum();
    let trace_m: f64 = mass.diagonal_values().iter().sum();
    if !(trace_m.is_finite() && trace_m > 0.0 && trace_k.is_finite() && trace_k >= 0.0) {
        return Err(AcousticHarmonicError::Operator(
            "cavity-mode operators are not positive".into(),
        ));
    }
    let shift = (1.0e-4 * trace_k / trace_m).max(f64::MIN_POSITIVE.sqrt());
    let shifted = RealSparseMatrix::scaled_sum(
        &[(1.0, stiffness), (shift, mass)],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let operator = SpdOperator::factor(&shifted, policy.linear_relative_tolerance)?;
    let apply_m =
        |v: &[f64]| spmv(mass, v).map_err(|error| AcousticHarmonicError::Operator(error.into()));
    let apply_k = |v: &[f64]| {
        spmv(stiffness, v).map_err(|error| AcousticHarmonicError::Operator(error.into()))
    };
    // Basis capacity before a thick restart keeps the lowest Ritz vectors.
    let capacity = (2 * count + 6 * block + 20).min(n);
    let retained_on_restart = (count + block)
        .min(capacity.saturating_sub(block))
        .max(count);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut mass_basis: Vec<Vec<f64>> = Vec::new();
    let mut stiffness_basis: Vec<Vec<f64>> = Vec::new();
    let mut applications = 0usize;
    let mut linear_iterations = 0usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next_block = (0..block)
        .map(|_| {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cancellation = crate::progress::cancellation_token();
    loop {
        if cancellation.is_cancelled() {
            return Err(AcousticHarmonicError::Cancelled);
        }
        // Shift-invert the pending block, then M-orthonormalize it against the basis.
        let mut appended = 0usize;
        for vector in next_block.drain(..) {
            if basis.len() >= capacity {
                break;
            }
            let rhs = apply_m(&vector)?;
            let solved = operator.solve(&rhs)?;
            applications += 1;
            linear_iterations += solved.iterations;
            let mut candidate = solved.values;
            let original = mass_norm(&candidate, &apply_m(&candidate)?);
            if !(original.is_finite() && original > 0.0) {
                continue;
            }
            for _ in 0..2 {
                let image = apply_m(&candidate)?;
                for existing in &basis {
                    let projection = dot(existing, &image);
                    for (value, basis_value) in candidate.iter_mut().zip(existing) {
                        *value -= projection * basis_value;
                    }
                }
            }
            let image = apply_m(&candidate)?;
            let length = mass_norm(&candidate, &image);
            if !(length.is_finite() && length > 1.0e-10 * original) {
                continue;
            }
            candidate.iter_mut().for_each(|value| *value /= length);
            let image = image.into_iter().map(|value| value / length).collect();
            stiffness_basis.push(apply_k(&candidate)?);
            mass_basis.push(image);
            basis.push(candidate);
            appended += 1;
        }
        if basis.len() < count && appended == 0 {
            return Err(AcousticHarmonicError::Operator(
                "cavity-mode Krylov basis cannot be extended".into(),
            ));
        }
        // Rayleigh-Ritz with K on the M-orthonormal basis.
        let m = basis.len();
        let mut projected = vec![0.0; m * m];
        for row in 0..m {
            for column in row..m {
                let value = 0.5
                    * (dot(&basis[row], &stiffness_basis[column])
                        + dot(&basis[column], &stiffness_basis[row]));
                projected[row * m + column] = value;
                projected[column * m + row] = value;
            }
        }
        // Ascending Ritz values; column-major eigenvectors (`rotations[ritz * m + index]`).
        let (values, rotations) = runmat_sparse_solvers::dense::symmetric_eigen(&projected, m)
            .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        let combine = |source: &[Vec<f64>], ritz: usize| {
            let mut result = vec![0.0; n];
            for (index, vector) in source.iter().enumerate() {
                let weight = rotations[ritz * m + index];
                if weight != 0.0 {
                    for (value, basis_value) in result.iter_mut().zip(vector) {
                        *value += weight * basis_value;
                    }
                }
            }
            result
        };
        let wanted = count.min(m);
        let mut residuals = Vec::with_capacity(wanted);
        for ritz in 0..wanted {
            let lambda = values[ritz];
            let k_phi = combine(&stiffness_basis, ritz);
            let m_phi = combine(&mass_basis, ritz);
            let residual = k_phi
                .iter()
                .zip(&m_phi)
                .map(|(k, m): (&f64, &f64)| (k - lambda * m).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale = (lambda.abs() + shift) * m_phi.iter().map(|v| v * v).sum::<f64>().sqrt();
            residuals.push(residual / scale);
        }
        let converged = wanted == count
            && residuals
                .iter()
                .all(|value| value.is_finite() && *value <= policy.relative_tolerance);
        if converged {
            let mut vectors = (0..count)
                .map(|ritz| combine(&basis, ritz))
                .collect::<Vec<_>>();
            for vector in &mut vectors {
                let largest = vector.iter().copied().fold(0.0_f64, |best, value| {
                    if value.abs() > best.abs() {
                        value
                    } else {
                        best
                    }
                });
                if largest < 0.0 {
                    vector.iter_mut().for_each(|value| *value = -*value);
                }
            }
            let mass_images = vectors
                .iter()
                .map(|vector| apply_m(vector))
                .collect::<Result<Vec<_>, _>>()?;
            let mut orthogonality_error = 0.0_f64;
            for (i, vector) in vectors.iter().enumerate() {
                for (j, image) in mass_images.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    orthogonality_error =
                        orthogonality_error.max((dot(vector, image) - expected).abs());
                }
            }
            return Ok(Eigenpairs {
                values: values[..count].to_vec(),
                vectors,
                residuals,
                orthogonality_error,
                applications,
                linear_iterations,
            });
        }
        if applications >= policy.maximum_operator_applications || (m == n && wanted == count) {
            return Err(AcousticHarmonicError::NotConverged);
        }
        // Expand from the unconverged Ritz vectors; restart thick when the basis is full.
        let unconverged = (0..wanted)
            .filter(|&ritz| residuals[ritz] > policy.relative_tolerance)
            .chain(wanted..m)
            .take(block)
            .collect::<Vec<_>>();
        next_block = unconverged
            .iter()
            .map(|&ritz| combine(&basis, ritz))
            .collect();
        if next_block.is_empty() {
            next_block = (0..block).map(|ritz| combine(&basis, ritz)).collect();
        }
        if m + next_block.len() > capacity {
            let keep = retained_on_restart.min(m);
            let rotate = |source: &[Vec<f64>]| (0..keep).map(|r| combine(source, r)).collect();
            let new_basis: Vec<Vec<f64>> = rotate(&basis);
            let new_mass: Vec<Vec<f64>> = rotate(&mass_basis);
            let new_stiffness: Vec<Vec<f64>> = rotate(&stiffness_basis);
            basis = new_basis;
            mass_basis = new_mass;
            stiffness_basis = new_stiffness;
        }
    }
}

fn mass_norm(vector: &[f64], image: &[f64]) -> f64 {
    dot(vector, image).max(0.0).sqrt()
}

#[cfg(test)]
mod tests;

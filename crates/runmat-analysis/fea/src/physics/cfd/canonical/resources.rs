use super::{CanonicalFluidError as Error, CanonicalFluidPolicy};
use crate::{
    solve::krylov::{GmresPolicy, GmresWorkEnvelope},
    topology::{AnalysisProjection, FiniteVolumeReservation, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    AnalysisBoundary, ExcitationAction, ExcitationTarget, FluidConstraint, FluidConstraintKind,
    PhysicalSignal, RegionTarget,
};
use runmat_meshing_core::PersistentEntityId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidReservation {
    pub cells: u64,
    pub faces: u64,
    pub sparse_entries: u64,
    pub resident_bytes_upper_bound: u64,
    /// Resident bound for verifying a retained result: the full bound minus the
    /// solver-only workspace (GMRES, ILU, coupled momentum block with its component
    /// AMG hierarchies, fixed-point acceleration). Verification re-assembles the
    /// operators and re-evaluates the residual but runs no linear or outer solve.
    pub verification_resident_bytes_upper_bound: u64,
    pub operator_work_upper_bound: u64,
    pub linear_iterations_upper_bound: u64,
}

/// The admitted reservation once the momentum pattern exists: its exact block-column count
/// replaces the tetrahedral per-row bound.
pub(super) fn reserve_exact(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    p: CanonicalFluidPolicy,
    momentum_block_columns: u64,
) -> Result<CanonicalFluidReservation, Error> {
    let geometry = FiniteVolumeTopology::reserve(source, t.policy, &t.selection)
        .map_err(|_| Error::Resource("source geometry admission"))?;
    if geometry.cell_count != t.cells.len() {
        return Err(Error::Invalid("finite-volume source count"));
    }
    CanonicalFluidReservation::for_counts_with_momentum(
        &geometry,
        t.faces.len(),
        p,
        Some(momentum_block_columns),
    )
}

/// Borrowed source/authority work admission. The public count-only reservation
/// covers numerical work; this adds the actual metadata/comparison envelope
/// before boundary/source binding allocates or traverses exact selections.
pub(super) fn admit_authority_profile(
    source: &AnalysisProjection,
    constraints: &[FluidConstraint],
    operators: &[AnalysisBoundary],
    signals: &[PhysicalSignal],
    actions: &[ExcitationAction],
    p: CanonicalFluidPolicy,
    mut reservation: CanonicalFluidReservation,
) -> Result<CanonicalFluidReservation, Error> {
    let mut profile = AuthorityProfile {
        maximum_units: p.operator_work_budget(),
        maximum_entities: p.maximum_authority_items as u64,
        ..AuthorityProfile::default()
    };
    for constraint in constraints {
        profile.string(&constraint.constraint_id)?;
        let target = match &constraint.kind {
            FluidConstraintKind::PrescribedVelocity { target, .. }
            | FluidConstraintKind::MovingWall { target, .. }
            | FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::SlipWall { target }
            | FluidConstraintKind::Symmetry { target }
            | FluidConstraintKind::PressureReference { target, .. } => target,
            FluidConstraintKind::Periodic(_) => {
                return Err(Error::Unsupported(crate::admission::FLUID_PERIODIC.message))
            }
        };
        profile.target(target)?;
    }
    for operator in operators {
        profile.string(&operator.boundary_id)?;
        profile.target(&operator.target)?;
    }
    for action in actions {
        let excitation = action.excitation();
        profile.string(&excitation.excitation_id.0)?;
        if let ExcitationTarget::Region(target) = &excitation.target {
            profile.target(target)?;
        }
    }
    let records = add(
        add(constraints.len() as u64, operators.len() as u64)?,
        add(signals.len() as u64, actions.len() as u64)?,
    )?;
    if add(records, profile.resolved_entities)? > p.maximum_authority_items as u64 {
        return Err(Error::Resource("authority records and resolved entities"));
    }
    let signal_profile =
        runmat_canonical_codec::profile_serialized(&signals, super::INPUT_IDENTITY_LIMITS)
            .map_err(|error| match &error.kind {
                runmat_canonical_codec::CanonicalCodecErrorKind::Invalid => {
                    Error::InputRealization(error.to_string())
                }
                runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit => {
                    Error::Resource("fluid original signal profile")
                }
            })?;
    reservation.resident_bytes_upper_bound = add(
        reservation.resident_bytes_upper_bound,
        signal_profile.materialization_resident_bytes_estimate,
    )?;
    reservation.operator_work_upper_bound = add(
        reservation.operator_work_upper_bound,
        mul(signal_profile.encoded_bytes_upper_bound, 64)?,
    )?;
    let mut max_signal_bytes = 1;
    for signal in signals {
        let bytes = add(signal.signal_id().0.len() as u64, 1)?;
        max_signal_bytes = max_signal_bytes.max(bytes);
        profile.string(&signal.signal_id().0)?;
    }
    let mut max_aliases = 0;
    for face in &source.boundary_faces {
        if face.provenance.len() > 64 {
            return Err(Error::Resource("boundary provenance count"));
        }
        max_aliases = max_aliases.max(face.provenance.len() as u64);
        for entity in &face.provenance {
            profile.entity(entity)?;
        }
    }
    for cell in &source.elements {
        profile.entity(&cell.region_id)?;
    }
    // Each comparison visits at most both original identities' actual strings
    // and path components. Binary searches use at most word-width comparisons.
    let compare = mul(
        mul(profile.maximum_entity_units.max(1), 2)?,
        u64::from(usize::BITS),
    )?;
    // Every face-bound action (mass flow, inlet velocity, wall motion) selects faces.
    let mass_flow_actions = actions
        .iter()
        .filter(|a| {
            matches!(
                a,
                ExcitationAction::MassFlow(_)
                    | ExcitationAction::InletVelocity(_)
                    | ExcitationAction::WallMotion(_)
            )
        })
        .count() as u64;
    let boundary_work = mul(
        mul(
            add(
                add(constraints.len() as u64, operators.len() as u64)?,
                mass_flow_actions,
            )?,
            reservation.faces,
        )?,
        mul(max_aliases, mul(compare, 2)?)?,
    )?;
    let forcing_work = mul(mul(actions.len() as u64, reservation.cells)?, compare)?;
    let signal_work = mul(
        mul(
            add(
                add(constraints.len() as u64, operators.len() as u64)?,
                actions.len() as u64,
            )?,
            signals.len() as u64,
        )?,
        mul(max_signal_bytes, 2)?,
    )?;
    let work = add(
        add(
            add(boundary_work, forcing_work)?,
            mul(mul(mass_flow_actions, reservation.faces)?, 64)?,
        )?,
        add(signal_work, mul(profile.metadata_units, 128)?)?,
    )?;
    reservation.operator_work_upper_bound = add(reservation.operator_work_upper_bound, work)?;
    // Covered-selection flags and traversal bookkeeping coexist with the base
    // numerical reservation. No metadata strings are cloned by this preflight.
    reservation.resident_bytes_upper_bound = add(
        reservation.resident_bytes_upper_bound,
        mul(profile.resolved_entities, 16)?,
    )?;
    if reservation.operator_work_upper_bound > p.operator_work_budget() {
        return Err(Error::Resource("source authority operator work"));
    }
    if reservation.resident_bytes_upper_bound > p.resident_byte_budget()
        || reservation.resident_bytes_upper_bound > isize::MAX as u64
    {
        return Err(Error::Resource("source authority resident bytes"));
    }
    Ok(reservation)
}

#[derive(Default)]
struct AuthorityProfile {
    resolved_entities: u64,
    metadata_units: u64,
    maximum_entity_units: u64,
    maximum_units: u64,
    maximum_entities: u64,
}
impl AuthorityProfile {
    fn string(&mut self, value: &str) -> Result<(), Error> {
        self.metadata_units = add(self.metadata_units, add(value.len() as u64, 1)?)?;
        if self.metadata_units > self.maximum_units {
            return Err(Error::Resource("authority metadata traversal"));
        }
        Ok(())
    }
    fn entity(&mut self, entity: &PersistentEntityId) -> Result<(), Error> {
        let mut units = add(entity.source_topology_id.len() as u64, 1)?;
        for component in &entity.assembly_path {
            units = add(units, add(component.len() as u64, 1)?)?;
            if units > self.maximum_units {
                return Err(Error::Resource("entity metadata traversal"));
            }
        }
        self.metadata_units = add(self.metadata_units, units)?;
        if self.metadata_units > self.maximum_units {
            return Err(Error::Resource("authority metadata traversal"));
        }
        self.maximum_entity_units = self.maximum_entity_units.max(units);
        Ok(())
    }
    fn target(&mut self, target: &RegionTarget) -> Result<(), Error> {
        self.string(&target.region_id)?;
        self.resolved_entities = add(
            self.resolved_entities,
            target.resolved_entities().len() as u64,
        )?;
        if self.resolved_entities > self.maximum_entities {
            return Err(Error::Resource("resolved authority entities"));
        }
        for entity in target.resolved_entities() {
            self.entity(entity)?;
        }
        Ok(())
    }
}

impl CanonicalFluidReservation {
    /// Count-only numerical admission before projection/topology allocations. The
    /// caller retains shared geometry admission and may conservatively pass all
    /// face incidences; exact topology admission rechecks the same numerical law.
    pub fn for_counts(
        geometry: &FiniteVolumeReservation,
        face_count: usize,
        p: CanonicalFluidPolicy,
    ) -> Result<Self, Error> {
        Self::for_counts_with_momentum(geometry, face_count, p, None)
    }

    /// As `for_counts`, with the coupled momentum block's block-column count (the sum over
    /// cells of their two-ring stencil sizes) when the derived pattern exists; without it the
    /// tetrahedral bound of 41 cells per row applies.
    pub fn for_counts_with_momentum(
        geometry: &FiniteVolumeReservation,
        face_count: usize,
        p: CanonicalFluidPolicy,
        momentum_block_columns: Option<u64>,
    ) -> Result<Self, Error> {
        if let Some(acceleration) = p.fixed_point_acceleration {
            acceleration
                .validate()
                .map_err(|_| Error::Invalid("fixed-point acceleration policy"))?;
        }
        if geometry.cell_count == 0
            || face_count == 0
            || face_count > geometry.face_incidence_count
            || p.maximum_outer_iterations == 0
            || p.maximum_nonorthogonal_iterations == 0
            || !p.momentum_relaxation.is_finite()
            || !(0.0..=1.0).contains(&p.momentum_relaxation)
            || p.momentum_relaxation == 0.0
            || !p.pressure_relaxation.is_finite()
            || !(0.0..=1.0).contains(&p.pressure_relaxation)
            || p.pressure_relaxation == 0.0
            || !p.physical_relative_tolerance.is_finite()
            || p.physical_relative_tolerance <= 0.0
        {
            return Err(Error::Invalid("iteration or relaxation policy"));
        }
        let gmres = GmresWorkEnvelope::derive(
            geometry.cell_count,
            GmresPolicy {
                relative_tolerance: p.linear_relative_tolerance,
                restart: p.gmres_restart,
                maximum_iterations: p.maximum_linear_iterations,
            },
        )
        .map_err(|_| Error::Resource("GMRES envelope"))?;
        let n = u64::try_from(geometry.cell_count).map_err(|_| Error::Resource("cell count"))?;
        let f = u64::try_from(face_count).map_err(|_| Error::Resource("face count"))?;
        // A tetrahedral cell has at most four distinct face-neighbours plus its diagonal.
        let nnz = mul(n, 5)?;
        let ilu = super::linear::PreconditionerEnvelope::for_counts(n, nnz)
            .ok_or(Error::Resource("shared preconditioner envelope"))?;
        // Three coexisting CSR operators/pattern clones plus assembly/output temporaries,
        // immutable boundary/bodyforce, all cell vectors/gradients/3x3 LS stencils, and
        // shared GMRES peak. Includes doubled Vec payload/header growth allowances.
        // Per cell: velocity (24 B), pressure triple (24), two gradient fields (96), retained
        // residual and vorticity (100), momentum right-hand sides and diagonal (32) and the
        // 3x3 least-squares stencil (72): 348 B, one evaluation's coexisting copy of each,
        // doubled for Vec growth: 1392 B.
        const CELL_STATE_BYTES: u64 = (24 + 24 + 96 + 100 + 32 + 72) * 2 * 2;
        let baseline_bytes = add(
            add(
                add(mul(nnz, 192)?, mul(n, CELL_STATE_BYTES)?)?,
                add(mul(f, 1024)?, gmres.resident_bytes_upper_bound)?,
            )?,
            // Finite-volume derivation scratch belongs to source preparation, which the
            // caller admits separately; it is released before any numerical storage exists.
            128 * 1024 * 1024,
        )?;
        // Current component/reference, original action, defect and correction
        // coexist with the existing shared Krylov peak in a correction solve.
        // Original governing fluctuation coexists with its absolute-Pa output projection.
        // Charge the extra retained vector with the same doubled payload allowance.
        let mut bytes = add(add(baseline_bytes, mul(n, 4 * 8)?)?, add(mul(n, 32)?, 96)?)?;
        // Solver-only workspace, which verification of a retained result never allocates:
        // the shared ILU envelope, then the coupled momentum block: the two-ring pattern (at
        // most 41 cells per row for tetrahedra), its 3 × 3-block values and the linear
        // reconstruction terms, plus one AMG hierarchy per velocity component (each within
        // the per-system envelope).
        // The coupled velocity-pressure step runs flexible GMRES on 4n unknowns: the Arnoldi
        // basis keeps r + 1 vectors and the preconditioned basis r more, each 4n values.
        let restart = p.gmres_restart as u64;
        let coupled_basis = mul(mul(n, 4 * 8)?, add(mul(restart, 2)?, 1)?)?;
        let block_columns = match momentum_block_columns {
            Some(columns) => columns,
            None => mul(n, 41)?,
        };
        let solve_workspace = add(
            add(
                ilu.resident_bytes_upper_bound,
                add(mul(block_columns, 9 * (8 + 8 + 8) + 8)?, mul(n, 5 * 216)?)?,
            )?,
            add(mul(ilu.resident_bytes_upper_bound, 3)?, coupled_basis)?,
        )?;
        bytes = add(bytes, solve_workspace)?;
        // The shared GMRES peak is part of the baseline above.
        let mut solver_bytes = add(solve_workspace, gmres.resident_bytes_upper_bound)?;
        // Frozen velocity reconstruction: four source/weight records per cell,
        // doubled payload growth plus the owning Vec header. No field is retained.
        bytes = add(
            bytes,
            add(
                mul(
                    n,
                    2 * super::velocity_gradient::VelocityGradientStencil::RESIDENT_BYTES,
                )?,
                48,
            )?,
        )?;
        // Cell vorticity: one retained and one evaluated vector per cell, doubled payload.
        bytes = add(bytes, add(mul(n, 2 * 2 * 24)?, 48)?)?;
        // Gauge selection/component arrays, original and reduced scalar systems,
        // compatibility traversal and mean-trace normalization coexist with GMRES.
        bytes = add(bytes, add(mul(n, 512)?, mul(f, 64)?)?)?;
        // Exact index-pair plan (unstable in-place sort has no merge allocation),
        // one descriptor per exterior face at most, and two coexisting report arrays
        // during safeguarded recovery. IDs obey the existing256-byte authority limit.
        // Doubled Vec/String payload allowance plus fixed headers.
        // Index pairs exist only for exterior faces (the boundary faces), one report per face
        // at most.
        let exterior = u64::try_from(geometry.boundary_face_count)
            .map_err(|_| Error::Resource("boundary face count"))?;
        bytes = add(
            bytes,
            add(
                mul(
                    exterior,
                    32 * 2
                        + 2 * (256
                            + std::mem::size_of::<super::CanonicalFluidBoundaryBalance>() as u64)
                            * 2,
                )?,
                4096,
            )?,
        )?;
        if p.fixed_point_acceleration.is_some() {
            // Two images/residuals, U/p/q proposal and shared coefficient workspace;
            // one additional complete physical residual/face-force evaluation.
            // Doubled payload growth plus fixed Vec/record headers.
            let acceleration = add(mul(add(mul(n, 26)?, mul(f, 15)?)?, 16)?, 4096)?;
            bytes = add(bytes, acceleration)?;
            solver_bytes = add(solver_bytes, acceleration)?;
        }
        let verification_resident_bytes_upper_bound = bytes
            .checked_sub(solver_bytes)
            .ok_or(Error::Resource("verification resident bytes"))?;
        let solves = mul(
            p.maximum_outer_iterations as u64,
            add(3, p.maximum_nonorthogonal_iterations as u64)?,
        )?;
        let linear_work = add(
            mul(
                add(gmres.operator_applications_upper_bound, 1)?,
                mul(nnz, 2)?,
            )?,
            add(
                mul(gmres.dot_calls_upper_bound, mul(n, 4)?)?,
                gmres.recurrence_scalar_operations_upper_bound,
            )?,
        )?;
        let linear_work = add(
            add(linear_work, mul(n, 4)?)?,
            add(
                ilu.factor_scalar_operations,
                mul(
                    add(gmres.operator_applications_upper_bound, 1)?,
                    ilu.application_scalar_operations,
                )?,
            )?,
        )?;
        let linear_iterations_upper_bound = mul(solves, p.maximum_linear_iterations as u64)?;
        let mut work = add(
            add(
                mul(solves, linear_work)?,
                mul(
                    p.maximum_outer_iterations as u64,
                    add(mul(f, 10000)?, mul(n, 100000)?)?,
                )?,
            )?,
            add(
                add(geometry.traversal_operations as u64, mul(n, 100000)?)?,
                mul(
                    p.maximum_authority_items as u64,
                    add(mul(f, 64)?, mul(n, 64)?)?,
                )?,
            )?,
        )?;
        // Every nonorthogonal pressure pass reconstructs original fields,
        // assembles/reduces the graph and normalizes the authored gauge.
        work = add(
            work,
            mul(
                p.maximum_outer_iterations as u64,
                mul(
                    p.maximum_nonorthogonal_iterations as u64,
                    add(mul(f, 1000)?, mul(n, 10000)?)?,
                )?,
            )?,
        )?;
        // Canonical authority ordering compares bounded256-byte IDs; two word-width
        // comparison passes conservatively cover in-place sorting/traversal. Each
        // outer step may evaluate ordinary and speculative original boundary terms.
        work = add(work, mul(f, 2 * 256 * u64::from(usize::BITS))?)?;
        work = add(
            work,
            mul(p.maximum_outer_iterations as u64, mul(f, 2 * 600)?)?,
        )?;
        if p.fixed_point_acceleration.is_some() {
            work = add(
                work,
                mul(
                    p.maximum_outer_iterations as u64,
                    add(mul(f, 10000)?, mul(n, 100100)?)?,
                )?,
            )?;
        }
        if bytes > p.resident_byte_budget() || bytes > isize::MAX as u64 {
            return Err(Error::Resource("resident bytes"));
        }
        if work > p.operator_work_budget() {
            return Err(Error::Resource("operator work"));
        }
        Ok(CanonicalFluidReservation {
            cells: n,
            faces: f,
            sparse_entries: nnz,
            resident_bytes_upper_bound: bytes,
            verification_resident_bytes_upper_bound,
            operator_work_upper_bound: work,
            linear_iterations_upper_bound,
        })
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or(Error::Resource("arithmetic overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b)
        .ok_or(Error::Resource("arithmetic overflow"))
}

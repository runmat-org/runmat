use runmat_analysis_core::AnalysisField;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_simulation_usage::{
    AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory, UsageContext,
};

use crate::{
    algebra::coercivity::NumericalCapacityError,
    assembly::{
        dofs::StructuralDofKind,
        elements::{
            beam::{local_stiffness_matrix, BeamElementError, BEAM_ELEMENT_DOF_COUNT},
            shell::SHELL_ELEMENT_DOF_COUNT,
        },
        AssemblySummary, BeamRecoveryElementSummary, ShellRecoveryElementSummary,
    },
    contracts::{
        FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
        FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
        FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT, FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
        FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
        FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR, FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
    },
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError},
    post::error::StructuralFieldRecoveryError,
    progress::{cancellation_token, FeaCancellationToken},
};

struct BeamResultValues {
    axial_force: Vec<f64>,
    shear_force: Vec<f64>,
    torsion_moment: Vec<f64>,
    bending_moment: Vec<f64>,
    bending_stress: Vec<f64>,
    torsion_stress: Vec<f64>,
}

/// The grant is for the selected immutable beam schedule. The stiffness kernel
/// owns its 64 operations; only the two matvecs and active stress ratios are
/// reserved by this recovery loop.
pub(super) async fn recover_beam_result_fields_with_capacity<A>(
    summary: &AssemblySummary,
    solution: &[f64],
    plan: LocalExecutionPlan,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Option<Vec<AnalysisField>>, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if summary.structural_beam_recovery.is_empty() {
        return Ok(None);
    }
    let mut work = OperationCount::ZERO;
    for beam in &summary.structural_beam_recovery {
        validate_beam_recovery_input(beam).map_err(NumericalCapacityError::Numerical)?;
        work = work
            .checked_add(OperationCount::from_operations(64 + beam_outer_work(beam)))
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)
            .map_err(NumericalCapacityError::Numerical)?;
    }
    if let Some(owner) = owner {
        owner
            .ensure_capacity(work)
            .await
            .map_err(NumericalCapacityError::Admission)?;
    }
    recover_beam_result_fields(summary, solution, plan).map_err(NumericalCapacityError::Numerical)
}

fn validate_beam_recovery_input(
    beam: &BeamRecoveryElementSummary,
) -> Result<(), StructuralFieldRecoveryError> {
    let source = beam
        .section
        .validate()
        .and_then(|()| beam.material.validate())
        .and_then(|()| {
            let length = beam.length_m.canonical_value();
            if length > 0.0 && length.is_finite() {
                Ok(())
            } else {
                Err(BeamElementError::DegenerateLength)
            }
        });
    source.map_err(|source| StructuralFieldRecoveryError::Beam {
        element_id: beam.element_id.clone(),
        source,
    })
}

/// Two dense 12-term dot products per output: 2 × 12 × (12 products + 11
/// additions) = 552. Each positive section ratio adds one multiply/divide pair.
fn beam_outer_work(beam: &BeamRecoveryElementSummary) -> u64 {
    let section = beam.section;
    let bending_y = (section.outer_fiber_z_m.canonical_value() > 0.0
        && section.iy_m4.canonical_value() > 0.0) as u64;
    let bending_z = (section.outer_fiber_y_m.canonical_value() > 0.0
        && section.iz_m4.canonical_value() > 0.0) as u64;
    let torsion = (section.torsion_outer_radius_m.canonical_value() > 0.0
        && section.torsion_j_m4.canonical_value() > 0.0) as u64;
    552 + 2 * (bending_y + bending_z + torsion)
}

pub(super) fn recover_beam_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
    plan: LocalExecutionPlan,
) -> Result<Option<Vec<AnalysisField>>, StructuralFieldRecoveryError> {
    if summary.structural_beam_recovery.is_empty() {
        return Ok(None);
    }
    let cancellation = cancellation_token();
    let values = if plan.is_serial() {
        recover_beam_result_values(
            summary,
            solution,
            &summary.structural_beam_recovery,
            &cancellation,
        )?
    } else {
        let usage = UsageContext::capture();
        let partitions = execute_scoped_chunks(
            plan,
            summary.structural_beam_recovery.len() as u64,
            |chunk| {
                let _usage = usage.as_ref().map(UsageContext::enter);
                recover_beam_result_values(
                    summary,
                    solution,
                    &summary.structural_beam_recovery
                        [chunk.start() as usize..chunk.end_exclusive() as usize],
                    &cancellation,
                )
            },
        )
        .map_err(scoped_recovery_error)?;
        let mut merged = BeamResultValues::with_capacity(summary.structural_beam_recovery.len());
        for mut partition in partitions {
            merged.append(&mut partition);
        }
        merged
    };
    if cancellation.is_cancelled() {
        return Err(StructuralFieldRecoveryError::Cancelled);
    }
    let element_count = summary.structural_beam_recovery.len();
    Ok(Some(vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            AnalysisFieldLayout::scalar(
                ObservationQuantity::BeamAxialForce,
                FieldLocation::Element,
            ),
            vec![element_count],
            values.axial_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::BeamShearForce,
                FieldLocation::Element,
            ),
            vec![element_count, 2],
            values.shear_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
            AnalysisFieldLayout::scalar(
                ObservationQuantity::BeamTorsionMoment,
                FieldLocation::Element,
            ),
            vec![element_count],
            values.torsion_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::BeamBendingMoment,
                FieldLocation::Element,
            ),
            vec![element_count, 2],
            values.bending_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::BeamBendingStress,
                FieldLocation::Element,
            ),
            vec![element_count, 2],
            values.bending_stress,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
            AnalysisFieldLayout::scalar(
                ObservationQuantity::BeamTorsionStress,
                FieldLocation::Element,
            ),
            vec![element_count],
            values.torsion_stress,
        ),
    ]))
}

impl BeamResultValues {
    fn with_capacity(count: usize) -> Self {
        Self {
            axial_force: Vec::with_capacity(count),
            shear_force: Vec::with_capacity(count * 2),
            torsion_moment: Vec::with_capacity(count),
            bending_moment: Vec::with_capacity(count * 2),
            bending_stress: Vec::with_capacity(count * 2),
            torsion_stress: Vec::with_capacity(count),
        }
    }

    fn append(&mut self, other: &mut Self) {
        self.axial_force.append(&mut other.axial_force);
        self.shear_force.append(&mut other.shear_force);
        self.torsion_moment.append(&mut other.torsion_moment);
        self.bending_moment.append(&mut other.bending_moment);
        self.bending_stress.append(&mut other.bending_stress);
        self.torsion_stress.append(&mut other.torsion_stress);
    }
}

fn recover_beam_result_values(
    summary: &AssemblySummary,
    solution: &[f64],
    beams: &[BeamRecoveryElementSummary],
    cancellation: &FeaCancellationToken,
) -> Result<BeamResultValues, StructuralFieldRecoveryError> {
    let mut values = BeamResultValues::with_capacity(beams.len());
    let usage = UsageContext::capture();

    for beam in beams {
        if cancellation.is_cancelled() {
            return Err(StructuralFieldRecoveryError::Cancelled);
        }
        validate_beam_recovery_input(beam)?;
        let outer_work = beam_outer_work(beam);
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage.ledger().reserve(
                    UsageCategory::Solve,
                    OperationCount::from_operations(outer_work),
                )
            })
            .transpose()?;
        let q_global = beam_global_displacement(summary, solution, beam);
        let q_local = multiply_beam_matrix_vector(&beam.transform_global_to_local, &q_global);
        let k_local = match local_stiffness_matrix(
            beam.section,
            beam.material,
            beam.length_m.canonical_value(),
        ) {
            Ok(matrix) => matrix,
            Err(source) => {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(276))?;
                }
                return Err(StructuralFieldRecoveryError::Beam {
                    element_id: beam.element_id.clone(),
                    source,
                });
            }
        };
        let f_local = multiply_beam_matrix_vector(&k_local, &q_local);

        let axial = signed_peak_pair(f_local[0], f_local[6]);
        let shear_y = signed_peak_pair(f_local[1], f_local[7]);
        let shear_z = signed_peak_pair(f_local[2], f_local[8]);
        let torsion = signed_peak_pair(f_local[3], f_local[9]);
        let bending_y = signed_peak_pair(f_local[4], f_local[10]);
        let bending_z = signed_peak_pair(f_local[5], f_local[11]);

        values.axial_force.push(axial);
        values.shear_force.extend([shear_y, shear_z]);
        values.torsion_moment.push(torsion);
        values.bending_moment.extend([bending_y, bending_z]);
        values.bending_stress.extend([
            bending_stress_from_moment(
                bending_y,
                beam.section.outer_fiber_z_m.canonical_value(),
                beam.section.iy_m4.canonical_value(),
            ),
            bending_stress_from_moment(
                bending_z,
                beam.section.outer_fiber_y_m.canonical_value(),
                beam.section.iz_m4.canonical_value(),
            ),
        ]);
        values.torsion_stress.push(torsion_stress_from_moment(
            torsion,
            beam.section.torsion_outer_radius_m.canonical_value(),
            beam.section.torsion_j_m4.canonical_value(),
        ));
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(outer_work))?;
        }
    }

    Ok(values)
}

fn beam_global_displacement(
    summary: &AssemblySummary,
    solution: &[f64],
    beam: &BeamRecoveryElementSummary,
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let mut q = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for (node_offset, node_index) in [beam.node_i_index, beam.node_j_index].iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            let target = node_offset * StructuralDofKind::ORDER.len() + component;
            q[target] = summary
                .structural_dof_layout
                .index(*node_index, *kind)
                .and_then(|row| solution.get(row).copied())
                .unwrap_or(0.0);
        }
    }
    q
}

fn multiply_beam_matrix_vector(
    matrix: &[[f64; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT],
    vector: &[f64; BEAM_ELEMENT_DOF_COUNT],
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let mut out = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for (row, out_value) in out.iter_mut().enumerate() {
        *out_value = matrix[row]
            .iter()
            .zip(vector.iter())
            .map(|(entry, value)| entry * value)
            .sum();
    }
    out
}

fn signed_peak_pair(a: f64, b: f64) -> f64 {
    if a.abs() >= b.abs() {
        a
    } else {
        b
    }
}

fn bending_stress_from_moment(moment_n_m: f64, outer_fiber_m: f64, inertia_m4: f64) -> f64 {
    if outer_fiber_m > 0.0 && inertia_m4 > 0.0 {
        moment_n_m * outer_fiber_m / inertia_m4
    } else {
        0.0
    }
}

fn torsion_stress_from_moment(moment_n_m: f64, outer_radius_m: f64, torsion_j_m4: f64) -> f64 {
    if outer_radius_m > 0.0 && torsion_j_m4 > 0.0 {
        moment_n_m * outer_radius_m / torsion_j_m4
    } else {
        0.0
    }
}

struct ShellResultValues {
    membrane_force: Vec<f64>,
    bending_moment: Vec<f64>,
    transverse_shear: Vec<f64>,
    von_mises: Vec<f64>,
}

/// The constitutive layer count fixes the outer resultants schedule. Geometry
/// and MITC operators retain their own admitted boundaries (62 + 68 + 597).
/// Constitutive invariant validation is outside this reference numerical
/// kernel, as in the other FEA validation boundaries.
pub(super) async fn recover_shell_result_fields_with_capacity<A>(
    summary: &AssemblySummary,
    solution: &[f64],
    plan: LocalExecutionPlan,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Option<Vec<AnalysisField>>, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    if summary.structural_shell_recovery.is_empty() {
        return Ok(None);
    }
    let mut work = OperationCount::ZERO;
    for shell in &summary.structural_shell_recovery {
        let selected = shell_outer_work(shell)
            .map_err(StructuralFieldRecoveryError::from)
            .map_err(NumericalCapacityError::Numerical)?
            .checked_add(OperationCount::from_operations(62 + 68 + 597))
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)
            .map_err(NumericalCapacityError::Numerical)?;
        work = work
            .checked_add(selected)
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)
            .map_err(NumericalCapacityError::Numerical)?;
    }
    if let Some(owner) = owner {
        owner
            .ensure_capacity(work)
            .await
            .map_err(NumericalCapacityError::Admission)?;
    }
    recover_shell_result_fields(summary, solution, plan).map_err(NumericalCapacityError::Numerical)
}

/// 18 frame dots (90), two B3 contractions (210), B2 (70), four D3 (60),
/// D2 (6), and two add3 (6) = 442. Each layer has two surfaces; each surface
/// uses scale3 (3), add3 (3), D3 (15), and von Mises (9) = 30.
fn shell_outer_work(shell: &ShellRecoveryElementSummary) -> Result<OperationCount, LedgerError> {
    let layers = u64::try_from(shell.constitutive.recovery_layers.len())
        .map_err(|_| LedgerError::CountOverflow)?;
    OperationCount::from_operations(60)
        .checked_mul(layers)
        .and_then(|layers| layers.checked_add(OperationCount::from_operations(442)))
        .map_err(LedgerError::from)
}

pub(super) fn recover_shell_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
    plan: LocalExecutionPlan,
) -> Result<Option<Vec<AnalysisField>>, StructuralFieldRecoveryError> {
    if summary.structural_shell_recovery.is_empty() {
        return Ok(None);
    }
    let cancellation = cancellation_token();
    let values = if plan.is_serial() {
        recover_shell_result_values(
            summary,
            solution,
            &summary.structural_shell_recovery,
            &cancellation,
        )?
    } else {
        let usage = UsageContext::capture();
        let partitions = execute_scoped_chunks(
            plan,
            summary.structural_shell_recovery.len() as u64,
            |chunk| {
                let _usage = usage.as_ref().map(UsageContext::enter);
                recover_shell_result_values(
                    summary,
                    solution,
                    &summary.structural_shell_recovery
                        [chunk.start() as usize..chunk.end_exclusive() as usize],
                    &cancellation,
                )
            },
        )
        .map_err(scoped_recovery_error)?;
        let mut merged = ShellResultValues::with_capacity(summary.structural_shell_recovery.len());
        for mut partition in partitions {
            merged.append(&mut partition);
        }
        merged
    };
    if cancellation.is_cancelled() {
        return Err(StructuralFieldRecoveryError::Cancelled);
    }
    let element_count = summary.structural_shell_recovery.len();
    Ok(Some(vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::ShellMembraneForce,
                FieldLocation::Element,
            ),
            vec![element_count, 3],
            values.membrane_force,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::ShellBendingMoment,
                FieldLocation::Element,
            ),
            vec![element_count, 3],
            values.bending_moment,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            AnalysisFieldLayout::canonical(
                ObservationQuantity::ShellTransverseShear,
                FieldLocation::Element,
            ),
            vec![element_count, 2],
            values.transverse_shear,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
            AnalysisFieldLayout::scalar(ObservationQuantity::ShellVonMises, FieldLocation::Element),
            vec![element_count],
            values.von_mises,
        ),
    ]))
}

fn recover_shell_result_values(
    summary: &AssemblySummary,
    solution: &[f64],
    shells: &[ShellRecoveryElementSummary],
    cancellation: &FeaCancellationToken,
) -> Result<ShellResultValues, StructuralFieldRecoveryError> {
    let mut values = ShellResultValues::with_capacity(shells.len());
    let usage = UsageContext::capture();

    for shell in shells {
        if cancellation.is_cancelled() {
            return Err(StructuralFieldRecoveryError::Cancelled);
        }
        let outer_work = shell_outer_work(shell)?;
        let mut batch = usage
            .as_ref()
            .map(|usage| usage.ledger().reserve(UsageCategory::Solve, outer_work))
            .transpose()?;
        let q = shell_node_values(summary, solution, shell);
        let global_displacement =
            std::array::from_fn::<_, SHELL_ELEMENT_DOF_COUNT, _>(|index| q[index / 6][index % 6]);
        let result = crate::assembly::elements::shell::recover_resultants_for_constitutive(
            &shell.constitutive,
            shell.geometry,
            global_displacement,
        )
        .map_err(|source| StructuralFieldRecoveryError::Shell {
            element_id: shell.element_id.clone(),
            source,
        })?;
        values.membrane_force.extend(result.membrane_force_n_per_m);
        values.bending_moment.extend(result.bending_moment_n);
        values
            .transverse_shear
            .extend(result.transverse_shear_n_per_m);
        values.von_mises.push(result.surface_von_mises_pa);
        if let Some(batch) = &mut batch {
            batch.complete(outer_work)?;
        }
    }

    Ok(values)
}

impl ShellResultValues {
    fn with_capacity(count: usize) -> Self {
        Self {
            membrane_force: Vec::with_capacity(count * 3),
            bending_moment: Vec::with_capacity(count * 3),
            transverse_shear: Vec::with_capacity(count * 2),
            von_mises: Vec::with_capacity(count),
        }
    }

    fn append(&mut self, other: &mut Self) {
        self.membrane_force.append(&mut other.membrane_force);
        self.bending_moment.append(&mut other.bending_moment);
        self.transverse_shear.append(&mut other.transverse_shear);
        self.von_mises.append(&mut other.von_mises);
    }
}

fn scoped_recovery_error(
    error: ScopedLocalExecutionError<StructuralFieldRecoveryError>,
) -> StructuralFieldRecoveryError {
    match error {
        ScopedLocalExecutionError::Work { source, .. } => source,
        other => StructuralFieldRecoveryError::ScopedExecution {
            reason: other.to_string(),
        },
    }
}

fn shell_node_values(
    summary: &AssemblySummary,
    solution: &[f64],
    shell: &ShellRecoveryElementSummary,
) -> [[f64; 6]; 3] {
    let mut out = [[0.0; 6]; 3];
    for (node_offset, node_index) in shell.node_indices.iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            out[node_offset][component] = summary
                .structural_dof_layout
                .index(*node_index, *kind)
                .and_then(|row| solution.get(row).copied())
                .unwrap_or(0.0);
        }
    }
    out
}

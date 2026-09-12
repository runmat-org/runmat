use runmat_analysis_core::AnalysisField;

use crate::{
    contracts::{
        fea_transient_acceleration_field_id, fea_transient_angular_acceleration_field_id,
        fea_transient_angular_velocity_field_id, fea_transient_displacement_field_id,
        fea_transient_rotation_field_id, fea_transient_strain_energy_density_field_id,
        fea_transient_strain_field_id, fea_transient_stress_field_id,
        fea_transient_velocity_field_id, fea_transient_von_mises_field_id, FeaRunError,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_ROTATION,
        FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshot},
    post::fields::recover_deformation_result_fields,
    solve::transient::TransientSolveResult,
};

pub(super) struct TransientFieldRecovery {
    pub final_deformation_fields: Vec<AnalysisField>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub rotation_snapshots: Vec<AnalysisField>,
    pub velocity_snapshots: Vec<AnalysisField>,
    pub angular_velocity_snapshots: Vec<AnalysisField>,
    pub acceleration_snapshots: Vec<AnalysisField>,
    pub angular_acceleration_snapshots: Vec<AnalysisField>,
    pub strain_snapshots: Vec<AnalysisField>,
    pub stress_snapshots: Vec<AnalysisField>,
    pub von_mises_snapshots: Vec<AnalysisField>,
    pub strain_energy_density_snapshots: Vec<AnalysisField>,
}

pub(super) fn recover_transient_fields(
    transient: &TransientSolveResult,
) -> Result<TransientFieldRecovery, FeaRunError> {
    let snapshot_count = transient.time_points_s.len();
    require_snapshot_count(
        snapshot_count,
        "displacement",
        transient.displacement_snapshots.len(),
    )?;
    require_snapshot_count(
        snapshot_count,
        "velocity",
        transient.velocity_snapshots.len(),
    )?;
    require_snapshot_count(
        snapshot_count,
        "acceleration",
        transient.acceleration_snapshots.len(),
    )?;
    require_snapshot_count(
        snapshot_count,
        "assembly",
        transient.assembly_snapshots.len(),
    )?;

    let deformation_fields = transient
        .assembly_snapshots
        .iter()
        .zip(&transient.displacement_snapshots)
        .map(|(summary, displacement)| {
            recover_deformation_result_fields(summary, displacement)
                .map_err(|error| FeaRunError::Assembly(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let final_deformation_fields = deformation_fields
        .last()
        .ok_or_else(|| {
            FeaRunError::Assembly("transient solve produced no accepted state to recover".into())
        })?
        .iter()
        .filter(|field| !field.is_empty())
        .cloned()
        .collect();

    let displacement_snapshots = collect_required_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        fea_transient_displacement_field_id,
    )?;
    let rotation_snapshots = collect_optional_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_ROTATION,
        fea_transient_rotation_field_id,
    )?;
    let strain_snapshots = collect_optional_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_STRAIN,
        fea_transient_strain_field_id,
    )?;
    let stress_snapshots = collect_optional_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_STRESS,
        fea_transient_stress_field_id,
    )?;
    let von_mises_snapshots = collect_optional_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_VON_MISES,
        fea_transient_von_mises_field_id,
    )?;
    let strain_energy_density_snapshots = collect_optional_fields(
        &deformation_fields,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        fea_transient_strain_energy_density_field_id,
    )?;

    let mut velocity_snapshots = Vec::with_capacity(snapshot_count);
    let mut angular_velocity_snapshots = Vec::with_capacity(snapshot_count);
    let mut acceleration_snapshots = Vec::with_capacity(snapshot_count);
    let mut angular_acceleration_snapshots = Vec::with_capacity(snapshot_count);
    for (index, summary) in transient.assembly_snapshots.iter().enumerate() {
        let velocity = project_structural_nodal_snapshot(
            &summary.structural_dof_layout,
            &transient.velocity_snapshots[index],
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let acceleration = project_structural_nodal_snapshot(
            &summary.structural_dof_layout,
            &transient.acceleration_snapshots[index],
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let shape = nodal_vector_shape(&summary.structural_dof_layout);
        velocity_snapshots.push(AnalysisField::host_f64(
            fea_transient_velocity_field_id(index),
            shape.clone(),
            velocity.translations,
        ));
        if summary.structural_dof_layout.rotational_dof_count() > 0 {
            angular_velocity_snapshots.push(AnalysisField::host_f64(
                fea_transient_angular_velocity_field_id(index),
                shape.clone(),
                velocity.rotations,
            ));
        }
        acceleration_snapshots.push(AnalysisField::host_f64(
            fea_transient_acceleration_field_id(index),
            shape.clone(),
            acceleration.translations,
        ));
        if summary.structural_dof_layout.rotational_dof_count() > 0 {
            angular_acceleration_snapshots.push(AnalysisField::host_f64(
                fea_transient_angular_acceleration_field_id(index),
                shape,
                acceleration.rotations,
            ));
        }
    }

    Ok(TransientFieldRecovery {
        final_deformation_fields,
        displacement_snapshots,
        rotation_snapshots,
        velocity_snapshots,
        angular_velocity_snapshots,
        acceleration_snapshots,
        angular_acceleration_snapshots,
        strain_snapshots,
        stress_snapshots,
        von_mises_snapshots,
        strain_energy_density_snapshots,
    })
}

fn collect_required_fields(
    snapshots: &[Vec<AnalysisField>],
    source_id: &str,
    snapshot_id: impl Fn(usize) -> String,
) -> Result<Vec<AnalysisField>, FeaRunError> {
    let fields = collect_fields(snapshots, source_id, snapshot_id);
    if fields.len() != snapshots.len() {
        return Err(FeaRunError::Assembly(format!(
            "transient recovery did not produce required field `{source_id}` for every accepted state"
        )));
    }
    Ok(fields)
}

fn collect_optional_fields(
    snapshots: &[Vec<AnalysisField>],
    source_id: &str,
    snapshot_id: impl Fn(usize) -> String,
) -> Result<Vec<AnalysisField>, FeaRunError> {
    let fields = collect_fields(snapshots, source_id, snapshot_id);
    if !fields.is_empty() && fields.len() != snapshots.len() {
        return Err(FeaRunError::Assembly(format!(
            "transient recovery produced field `{source_id}` for only part of the accepted state history"
        )));
    }
    Ok(fields)
}

fn collect_fields(
    snapshots: &[Vec<AnalysisField>],
    source_id: &str,
    snapshot_id: impl Fn(usize) -> String,
) -> Vec<AnalysisField> {
    snapshots
        .iter()
        .enumerate()
        .filter_map(|(index, fields)| {
            fields
                .iter()
                .find(|field| field.field_id == source_id && !field.is_empty())
                .cloned()
                .map(|mut field| {
                    field.field_id = snapshot_id(index);
                    field
                })
        })
        .collect()
}

pub(super) fn require_snapshot_count(
    expected: usize,
    kind: &str,
    actual: usize,
) -> Result<(), FeaRunError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FeaRunError::Assembly(format!(
            "transient {kind} snapshot count {actual} does not match time-point count {expected}"
        )))
    }
}

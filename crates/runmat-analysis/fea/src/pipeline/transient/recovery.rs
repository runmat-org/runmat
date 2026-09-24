use runmat_analysis_core::AnalysisField;

use crate::{
    connectors::CanonicalStructuralConnectorResult,
    contracts::{
        fea_transient_acceleration_field_id, fea_transient_angular_acceleration_field_id,
        fea_transient_angular_velocity_field_id, fea_transient_connector_force_field_id,
        fea_transient_connector_stored_energy_field_id, fea_transient_displacement_field_id,
        fea_transient_reaction_force_field_id, fea_transient_reaction_moment_field_id,
        fea_transient_rotation_field_id, fea_transient_strain_energy_density_field_id,
        fea_transient_strain_field_id, fea_transient_stress_field_id,
        fea_transient_velocity_field_id, fea_transient_von_mises_field_id, FeaRunError,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_ROTATION,
        FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    pipeline::structural_nodal::{nodal_vector_shape, project_structural_nodal_snapshot},
    post::{fields::recover_deformation_result_fields, recover_connector_observable_fields},
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
    pub reaction_force_snapshots: Vec<AnalysisField>,
    pub reaction_moment_snapshots: Vec<AnalysisField>,
    pub connector_results: Vec<CanonicalStructuralConnectorResult>,
    pub connector_force_snapshots: Vec<AnalysisField>,
    pub connector_stored_energy_snapshots: Vec<AnalysisField>,
    pub strain_snapshots: Vec<AnalysisField>,
    pub stress_snapshots: Vec<AnalysisField>,
    pub von_mises_snapshots: Vec<AnalysisField>,
    pub strain_energy_density_snapshots: Vec<AnalysisField>,
}

pub(super) fn recover_transient_fields(
    transient: &TransientSolveResult,
    empty_connector_plan: Option<&crate::connectors::CanonicalStructuralConnectorPlan>,
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
    require_snapshot_count(
        snapshot_count,
        "dynamic reaction",
        transient.reaction_snapshots.len(),
    )?;

    if snapshot_count == 0 {
        return Err(FeaRunError::Assembly(
            "transient solve produced no accepted state to recover".into(),
        ));
    }
    // Move each recovered field into its history as soon as the snapshot is ready.
    // Only the final state is cloned for the ordinary result-field view; there is
    // no second retained history-sized deformation-field collection.
    let mut final_deformation_fields = Vec::new();
    let mut displacement_snapshots = Vec::with_capacity(snapshot_count);
    let mut rotation_snapshots = Vec::with_capacity(snapshot_count);
    let mut strain_snapshots = Vec::with_capacity(snapshot_count);
    let mut stress_snapshots = Vec::with_capacity(snapshot_count);
    let mut von_mises_snapshots = Vec::with_capacity(snapshot_count);
    let mut strain_energy_density_snapshots = Vec::with_capacity(snapshot_count);
    for (index, (summary, displacement)) in transient
        .assembly_snapshots
        .iter()
        .zip(&transient.displacement_snapshots)
        .enumerate()
    {
        let mut fields = recover_deformation_result_fields(summary, displacement)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        if index + 1 == snapshot_count {
            final_deformation_fields = fields
                .iter()
                .filter(|field| !field.is_empty())
                .cloned()
                .collect();
        }
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            fea_transient_displacement_field_id(index),
            &mut displacement_snapshots,
        );
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_ROTATION,
            fea_transient_rotation_field_id(index),
            &mut rotation_snapshots,
        );
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_STRAIN,
            fea_transient_strain_field_id(index),
            &mut strain_snapshots,
        );
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_STRESS,
            fea_transient_stress_field_id(index),
            &mut stress_snapshots,
        );
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_VON_MISES,
            fea_transient_von_mises_field_id(index),
            &mut von_mises_snapshots,
        );
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            fea_transient_strain_energy_density_field_id(index),
            &mut strain_energy_density_snapshots,
        );
    }
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        displacement_snapshots.len(),
        RecoveredFieldPolicy::Required,
    )?;
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_ROTATION,
        rotation_snapshots.len(),
        RecoveredFieldPolicy::Optional,
    )?;
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_STRAIN,
        strain_snapshots.len(),
        RecoveredFieldPolicy::Optional,
    )?;
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_STRESS,
        stress_snapshots.len(),
        RecoveredFieldPolicy::Optional,
    )?;
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_VON_MISES,
        von_mises_snapshots.len(),
        RecoveredFieldPolicy::Optional,
    )?;
    require_recovered_field_count(
        snapshot_count,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        strain_energy_density_snapshots.len(),
        RecoveredFieldPolicy::Optional,
    )?;

    let mut velocity_snapshots = Vec::with_capacity(snapshot_count);
    let mut angular_velocity_snapshots = Vec::with_capacity(snapshot_count);
    let mut acceleration_snapshots = Vec::with_capacity(snapshot_count);
    let mut angular_acceleration_snapshots = Vec::with_capacity(snapshot_count);
    let mut reaction_force_snapshots = Vec::with_capacity(snapshot_count);
    let mut reaction_moment_snapshots = Vec::with_capacity(snapshot_count);
    let mut connector_results = Vec::with_capacity(snapshot_count);
    let mut connector_force_snapshots = Vec::with_capacity(snapshot_count);
    let mut connector_stored_energy_snapshots = Vec::with_capacity(snapshot_count);
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
        let reaction = project_structural_nodal_snapshot(
            &summary.structural_dof_layout,
            &transient.reaction_snapshots[index],
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
        let connector_plan = summary
            .structural_connector_plan
            .clone()
            .or_else(|| empty_connector_plan.cloned())
            .ok_or_else(|| {
                FeaRunError::Assembly(
                    "transient structural assembly did not publish a canonical connector plan"
                        .into(),
                )
            })?;
        let connector_result = CanonicalStructuralConnectorResult::recover_dynamic(
            connector_plan,
            &summary.structural_dof_layout,
            &transient.displacement_snapshots[index],
            &transient.velocity_snapshots[index],
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let connector_fields = recover_connector_observable_fields(&connector_result)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        if let [force, stored_energy] = connector_fields.as_slice() {
            let mut force = force.clone().into_storage();
            force.field_id = fea_transient_connector_force_field_id(index);
            connector_force_snapshots.push(force);
            let mut stored_energy = stored_energy.clone().into_storage();
            stored_energy.field_id = fea_transient_connector_stored_energy_field_id(index);
            connector_stored_energy_snapshots.push(stored_energy);
        } else if !connector_fields.is_empty() {
            return Err(FeaRunError::Assembly(
                "connector recovery did not publish the canonical force and energy field pair"
                    .into(),
            ));
        }
        connector_results.push(connector_result);
        acceleration_snapshots.push(AnalysisField::host_f64(
            fea_transient_acceleration_field_id(index),
            shape.clone(),
            acceleration.translations,
        ));
        reaction_force_snapshots.push(AnalysisField::host_f64(
            fea_transient_reaction_force_field_id(index),
            shape.clone(),
            reaction.translations,
        ));
        if summary.structural_dof_layout.rotational_dof_count() > 0 {
            angular_acceleration_snapshots.push(AnalysisField::host_f64(
                fea_transient_angular_acceleration_field_id(index),
                shape.clone(),
                acceleration.rotations,
            ));
            reaction_moment_snapshots.push(AnalysisField::host_f64(
                fea_transient_reaction_moment_field_id(index),
                shape,
                reaction.rotations,
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
        reaction_force_snapshots,
        reaction_moment_snapshots,
        connector_results,
        connector_force_snapshots,
        connector_stored_energy_snapshots,
        strain_snapshots,
        stress_snapshots,
        von_mises_snapshots,
        strain_energy_density_snapshots,
    })
}

#[derive(Clone, Copy)]
enum RecoveredFieldPolicy {
    Required,
    Optional,
}

fn require_recovered_field_count(
    snapshot_count: usize,
    source_id: &str,
    field_count: usize,
    policy: RecoveredFieldPolicy,
) -> Result<(), FeaRunError> {
    match policy {
        RecoveredFieldPolicy::Required if field_count != snapshot_count => {
            Err(FeaRunError::Assembly(format!(
                "transient recovery did not produce required field `{source_id}` for every accepted state"
            )))
        }
        RecoveredFieldPolicy::Optional if field_count != 0 && field_count != snapshot_count => {
            Err(FeaRunError::Assembly(format!(
                "transient recovery produced field `{source_id}` for only part of the accepted state history"
            )))
        }
        _ => Ok(()),
    }
}

fn move_snapshot_field(
    fields: &mut Vec<AnalysisField>,
    source_id: &str,
    snapshot_id: String,
    history: &mut Vec<AnalysisField>,
) {
    if let Some(position) = fields
        .iter()
        .position(|field| field.field_id == source_id && !field.is_empty())
    {
        let mut field = fields.remove(position);
        field.field_id = snapshot_id;
        history.push(field);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_snapshot_field_moves_its_storage_without_copying_the_history() {
        let mut fields = vec![AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string(),
            vec![1],
            vec![42.0],
        )];
        let original_storage = fields[0].as_host_f64().unwrap().as_ptr();
        let mut history = Vec::new();
        move_snapshot_field(
            &mut fields,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            fea_transient_displacement_field_id(2),
            &mut history,
        );
        assert!(fields.is_empty());
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].field_id, fea_transient_displacement_field_id(2));
        assert_eq!(history[0].as_host_f64().unwrap().as_ptr(), original_storage);
    }

    #[test]
    fn streamed_recovery_keeps_required_and_optional_history_cardinality_rules() {
        assert!(require_recovered_field_count(
            3,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            3,
            RecoveredFieldPolicy::Required,
        )
        .is_ok());
        assert!(require_recovered_field_count(
            3,
            FEA_FIELD_STRUCTURAL_ROTATION,
            0,
            RecoveredFieldPolicy::Optional,
        )
        .is_ok());
        assert!(require_recovered_field_count(
            3,
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            2,
            RecoveredFieldPolicy::Required,
        )
        .is_err());
        assert!(require_recovered_field_count(
            3,
            FEA_FIELD_STRUCTURAL_ROTATION,
            2,
            RecoveredFieldPolicy::Optional,
        )
        .is_err());
    }
}

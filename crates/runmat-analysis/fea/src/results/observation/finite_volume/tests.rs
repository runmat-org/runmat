use super::*;
use crate::results::{
    field::{
        RecoveredField, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
        RecoveredFieldSample,
    },
    observation::{tests::spec, ResolvedObservationTarget},
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, PersistentEntityId, PersistentEntityKind};

fn fixture() -> (
    AnalysisProjection,
    FiniteVolumeTopology,
    FiniteVolumeFieldTopology,
) {
    let mut mesh = fixtures::slab_mesh(2);
    for element in &mut mesh.topology.volume_elements {
        element.region_id = PersistentEntityId::new(
            PersistentEntityKind::Region,
            format!("cell-region-{}", element.element_id),
            vec!["original".into()],
        );
        element.provenance = vec![element.region_id.clone()];
    }
    // Every exact native Cell region is declared in the canonical mesh inventory,
    // with complete and unique original element ownership. Replacing labels alone
    // would be an invalid mesh, not a region-selection regression witness.
    mesh.topology.regions = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| runmat_meshing_core::MeshRegion {
            region_id: element.region_id.clone(),
            element_ids: vec![element.element_id],
        })
        .collect();
    mesh.topology
        .regions
        .sort_by(|a, b| a.region_id.cmp(&b.region_id));
    for face in &mut mesh.topology.boundary_faces {
        face.provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Face,
            format!("face-{}", face.face_id),
            vec!["original".into()],
        )];
    }
    mesh.seal_canonical_digest().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let geometry = FiniteVolumeTopology::derive(
        &projection,
        crate::topology::FiniteVolumeProjectionPolicy {
            max_cells: 1000,
            max_face_incidences: 4000,
            max_boundary_faces: 4000,
            max_scratch_bytes: 10_000_000,
            max_traversal_operations: 10_000_000,
        },
    )
    .unwrap();
    let layout = FiniteVolumeFieldTopology::derive(&projection, &geometry).unwrap();
    (projection, geometry, layout)
}
fn step(target: RegionTarget, face: bool) -> (ProcedureStep, RecoveredFieldDefinition) {
    let mut spec = spec(
        "native",
        ObservationTarget::Spatial(ExcitationTarget::Region(target)),
    );
    let (quantity, location, domain) = if face {
        (
            ObservationQuantity::VolumeFlowRate,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
        )
    } else {
        (
            ObservationQuantity::FluidPressure,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        )
    };
    spec.quantity = quantity;
    spec.source_location = location;
    spec.output_location = location;
    let definition = RecoveredFieldDefinition::new(
        "native",
        quantity,
        location,
        domain,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut step = ProcedureStep::fluid_static("native");
    step.observations = vec![ObservationRequest::Field(spec)];
    (step, definition)
}

#[test]
fn explicit_cell_selection_preserves_canonical_field_order_and_authored_verification() {
    let (projection, geometry, layout) = fixture();
    let canonical = geometry
        .cells
        .iter()
        .enumerate()
        .find(|(i, cell)| *i != cell.source_element_index)
        .map(|(i, _)| i)
        .expect("fixture must exercise reordered native cells");
    let cell = &geometry.cells[canonical];
    let mut target = RegionTarget::new(
        "exact-cell",
        EntityDimension::Volume,
        ExpectedTopology::Cell,
        TargetCardinality::One,
    );
    target
        .bind_resolved_entities(vec![cell.region.clone()])
        .unwrap();
    let (step, definition) = step(target, false);
    let limits = ObservationLimits::default();
    let plan = resolve_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        std::slice::from_ref(&definition),
        limits,
    )
    .unwrap();
    let ResolvedObservationTarget::Elements {
        indices,
        identities,
        ..
    } = &plan.observations[0].target
    else {
        panic!("native cell target")
    };
    assert_eq!(indices, &[canonical]);
    assert_eq!(identities, &[cell.element_identity]);
    verify_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        std::slice::from_ref(&definition),
        &plan,
        limits,
    )
    .unwrap();
    let storage = AnalysisField::host_f64(
        "native",
        vec![geometry.cells.len()],
        (0..geometry.cells.len())
            .map(|i| i as f64 + 100.0)
            .collect(),
    );
    let catalog = RecoveredFieldCatalog::new_for_topology(
        layout.identity().discretization_digest(),
        layout.entity_counts(),
        vec![
            RecoveredField::new(definition.clone(), RecoveredFieldSample::Final, storage).unwrap(),
        ],
    )
    .unwrap();
    let observed = evaluate_finite_volume_observations(
        &plan,
        &layout,
        &projection,
        &geometry,
        &catalog,
        limits,
    )
    .unwrap();
    assert_eq!(
        observed[0].field.values,
        AnalysisFieldValues::HostF64(vec![canonical as f64 + 100.0])
    );
    let mut swapped = plan.clone();
    let other = (canonical + 1) % geometry.cells.len();
    swapped.observations[0].target = ResolvedObservationTarget::Elements {
        domain: RecoveredFieldEntityDomain::VolumeElements,
        indices: vec![other],
        identities: vec![geometry.cells[other].element_identity],
    };
    swapped.identity = swapped.derive_identity().unwrap();
    assert!(verify_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        std::slice::from_ref(&definition),
        &swapped,
        limits
    )
    .is_err());
    let mut omitted = plan;
    omitted.observations.clear();
    omitted.identity = omitted.derive_identity().unwrap();
    assert!(verify_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        &[definition],
        &omitted,
        limits
    )
    .is_err());
}

#[test]
fn exact_native_face_and_cell_targets_reject_missing_original_entities() {
    let (projection, geometry, layout) = fixture();
    for face in [false, true] {
        let valid = if face {
            projection.boundary_faces[0].provenance[0].clone()
        } else {
            geometry.cells[0].region.clone()
        };
        let mut missing = valid.clone();
        missing.source_topology_id = "missing-original-selection".into();
        let mut target = if face {
            RegionTarget::boundary("exact-face")
        } else {
            RegionTarget::new(
                "exact-cell",
                EntityDimension::Volume,
                ExpectedTopology::Cell,
                TargetCardinality::OneOrMore,
            )
        };
        target.bind_resolved_entities(vec![valid, missing]).unwrap();
        let (step, definition) = step(target, face);
        assert!(resolve_finite_volume_observation_plan(
            &step,
            &layout,
            &projection,
            &geometry,
            &[definition],
            ObservationLimits::default()
        )
        .is_err());
    }
}

#[test]
fn tiny_heap_cap_rejects_before_original_topology_revalidation_or_clone() {
    let (projection, mut geometry, layout) = fixture();
    let mut target = RegionTarget::new(
        "exact-cell",
        EntityDimension::Volume,
        ExpectedTopology::Cell,
        TargetCardinality::One,
    );
    target
        .bind_resolved_entities(vec![geometry.cells[0].region.clone()])
        .unwrap();
    let (step, definition) = step(target, false);
    let definitions = [definition];
    let mut imported = resolve_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    imported.observations.clear(); // Invalid authored coverage and self-identity.
    let metadata_cap = ObservationLimits {
        maximum_peak_resident_bytes: finite_volume_observation_topology_resident_bytes(
            &layout,
            &projection,
            &geometry,
        )
        .unwrap(),
        ..ObservationLimits::default()
    };
    assert!(matches!(
        verify_finite_volume_observation_plan(
            &step,
            &layout,
            &projection,
            &geometry,
            &definitions,
            &imported,
            metadata_cap
        ),
        Err(ObservationError::Limit(
            "finite-volume observation topology residency"
        ))
    ));
    geometry.geometry_digest = runmat_execution::Digest::from_bytes([0; 32]);
    let limits = ObservationLimits {
        maximum_peak_resident_bytes: 1,
        ..ObservationLimits::default()
    };
    assert!(matches!(
        resolve_finite_volume_observation_plan(
            &step,
            &layout,
            &projection,
            &geometry,
            &definitions,
            limits
        ),
        Err(ObservationError::Limit(
            "finite-volume observation topology residency"
        ))
    ));
}

#[test]
fn native_cell_fanout_preserves_exact_indices_and_matches_serial_values() {
    use crate::results::observation::{
        reduce_observation_partitions, ObservationExecutionPolicy, ObservationPartitionPlan,
    };
    use std::num::NonZeroU32;
    let (projection, geometry, layout) = fixture();
    let mut target = RegionTarget::new(
        "cells",
        EntityDimension::Volume,
        ExpectedTopology::Cell,
        TargetCardinality::OneOrMore,
    );
    target
        .bind_resolved_entities(
            geometry
                .cells
                .iter()
                .map(|cell| cell.region.clone())
                .collect(),
        )
        .unwrap();
    let (step, definition) = step(target, false);
    let limits = ObservationLimits::default();
    let plan = resolve_finite_volume_observation_plan(
        &step,
        &layout,
        &projection,
        &geometry,
        std::slice::from_ref(&definition),
        limits,
    )
    .unwrap();
    let field = RecoveredField::new(
        definition,
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "native",
            vec![geometry.cells.len()],
            (0..geometry.cells.len()).map(|i| i as f64 + 2.0).collect(),
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        layout.identity().discretization_digest(),
        layout.entity_counts(),
        vec![field.clone()],
    )
    .unwrap();
    let serial = evaluate_finite_volume_observations(
        &plan,
        &layout,
        &projection,
        &geometry,
        &catalog,
        limits,
    )
    .unwrap()
    .remove(0);
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("native".into()),
        ObservationExecutionPolicy::canonical(NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    let mut contributions = Vec::new();
    for partition in &partition_plan.partitions {
        let indices = finite_volume_observation_partition_source_indices(
            &plan,
            &partition_plan,
            partition,
            FiniteVolumeObservationSource {
                layout: &layout,
                projection: &projection,
                geometry: &geometry,
            },
            limits,
        )
        .unwrap();
        let selected = prepare_finite_volume_observation_field_partition(
            &plan,
            &partition_plan,
            partition,
            FiniteVolumeObservationSource {
                layout: &layout,
                projection: &projection,
                geometry: &geometry,
            },
            &field,
            limits,
        )
        .unwrap();
        assert_eq!(
            selected
                .partitions
                .iter()
                .flat_map(|p| p.source_indices.iter().copied())
                .collect::<Vec<_>>(),
            indices
        );
        assert!(indices.len() < geometry.cells.len());
        let cold = crate::results::field::RecoveredFieldPartitionSet::from_canonical_bytes(
            &selected.canonical_bytes().unwrap(),
        )
        .unwrap();
        contributions.push(
            evaluate_finite_volume_observation_partition(
                &plan,
                &partition_plan,
                partition,
                FiniteVolumeObservationSource {
                    layout: &layout,
                    projection: &projection,
                    geometry: &geometry,
                },
                &cold,
                limits,
            )
            .unwrap(),
        );
    }
    let reduced = reduce_observation_partitions(&plan, &partition_plan, contributions).unwrap();
    assert_eq!(reduced.field, serial.field);
    let mut substituted = geometry.clone();
    substituted.geometry_digest = runmat_execution::Digest::sha256("foreign-native-layout");
    assert!(finite_volume_observation_partition_source_indices(
        &plan,
        &partition_plan,
        &partition_plan.partitions[0],
        FiniteVolumeObservationSource {
            layout: &layout,
            projection: &projection,
            geometry: &substituted
        },
        limits
    )
    .is_err());
}

#[test]
fn whole_model_native_velocity_and_flux_keep_global_components_and_internal_faces() {
    let (projection, geometry, layout) = fixture();
    assert!(geometry.faces.len() > projection.boundary_faces.len());
    for (quantity, location, domain, shape, entities) in [
        (
            ObservationQuantity::FluidVelocity,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::Vector3,
            geometry.cells.len(),
        ),
        (
            ObservationQuantity::VolumeFlowRate,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
            ObservationValueShape::Scalar,
            geometry.faces.len(),
        ),
    ] {
        let mut declaration = spec("native", ObservationTarget::WholeModel);
        declaration.quantity = quantity;
        declaration.source_location = location;
        declaration.output_location = location;
        declaration.value_shape = shape.clone();
        let components = shape.component_count().unwrap();
        let definition = RecoveredFieldDefinition::new(
            "native",
            quantity,
            location,
            domain,
            shape,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap();
        let mut authored = ProcedureStep::fluid_static("native");
        authored.observations = vec![ObservationRequest::Field(declaration)];
        let plan = resolve_finite_volume_observation_plan(
            &authored,
            &layout,
            &projection,
            &geometry,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let values = (0..entities * components)
            .map(|i| i as f64 - 3.0)
            .collect::<Vec<_>>();
        let storage_shape = if components == 1 {
            vec![entities]
        } else {
            vec![entities, components]
        };
        let field = RecoveredField::new(
            definition,
            RecoveredFieldSample::Final,
            AnalysisField::host_f64("native", storage_shape, values.clone()),
        )
        .unwrap();
        let catalog = RecoveredFieldCatalog::new_for_topology(
            layout.identity().discretization_digest(),
            layout.entity_counts(),
            vec![field],
        )
        .unwrap();
        let result = evaluate_finite_volume_observations(
            &plan,
            &layout,
            &projection,
            &geometry,
            &catalog,
            ObservationLimits::default(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(result.field.values, AnalysisFieldValues::HostF64(values));
        assert_eq!(result.field.shape[0], entities);
    }
}

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
        element.region_id = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: format!("cell-region-{}", element.element_id),
            assembly_path: vec!["original".into()],
        };
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
        face.provenance = vec![PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: format!("face-{}", face.face_id),
            assembly_path: vec!["original".into()],
        }];
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

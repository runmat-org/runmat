use super::*;
use runmat_analysis_core::{RecoveryPolicy, RegionTarget};
use runmat_meshing_core::{fixtures, ElementOrder};

#[test]
fn original_native_cell_and_face_selection_fit_counted_scratch_and_reject_short_caps() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => fixtures::slab_mesh(2),
            _ => fixtures::quadratic_slab_mesh(2),
        };
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let topology = ObservationTopology::from_projection(&projection)
            .unwrap()
            .with_projection_faces()
            .unwrap();
        let layout = topology.finite_volume.clone().unwrap();
        let geometry = crate::topology::FiniteVolumeTopology::derive(
            &projection,
            crate::topology::FiniteVolumeProjectionPolicy::covering(&projection),
        )
        .unwrap();
        let mut target = RegionTarget::new(
            "original-volume",
            EntityDimension::Volume,
            ExpectedTopology::Cell,
            TargetCardinality::OneOrMore,
        );
        target
            .bind_resolved_entities(vec![projection.elements[0].region_id.clone()])
            .unwrap();
        let output = selection_output_bytes(layout.cell_identities().len(), usize::MAX).unwrap();
        let bytes = element_selection_resident_bytes(&target, &topology, Some(&projection), output)
            .unwrap();
        let run = |cap| {
            let limits = ObservationLimits {
                maximum_peak_resident_bytes: cap,
                ..Default::default()
            };
            let indices = matching_elements(
                &target,
                RecoveredFieldEntityDomain::VolumeElements,
                &topology,
                limits,
                output,
            )?;
            elements_target(
                selected_indices(indices, limits.maximum_selected_entities)?,
                RecoveredFieldEntityDomain::VolumeElements,
                &topology,
                limits.maximum_selected_entities,
            )
        };
        let (result, _, peak) = crate::topology::measure_test_allocations(|| run(bytes));
        let ResolvedObservationTarget::Elements {
            indices,
            identities,
            ..
        } = result.unwrap()
        else {
            panic!("original cell target");
        };
        let expected = projection
            .elements
            .iter()
            .enumerate()
            .filter(|(_, e)| e.region_id == projection.elements[0].region_id)
            .map(|(i, e)| (i, e.stable_identity))
            .collect::<Vec<_>>();
        assert_eq!(indices, expected.iter().map(|p| p.0).collect::<Vec<_>>());
        assert_eq!(identities, expected.iter().map(|p| p.1).collect::<Vec<_>>());
        assert!(peak as u64 <= bytes, "cell scratch {peak} exceeds {bytes}");
        let (short, retained, peak) = crate::topology::measure_test_allocations(|| run(bytes - 1));
        assert!(matches!(
            short,
            Err(ObservationError::Limit(
                "target selection construction residency"
            ))
        ));
        assert_eq!((retained, peak), (0, 0));

        let mut target = RegionTarget::new(
            "original-boundary",
            EntityDimension::Face,
            ExpectedTopology::BoundaryFace,
            TargetCardinality::OneOrMore,
        );
        target
            .bind_resolved_entities(vec![projection.boundary_faces[0].provenance[0].clone()])
            .unwrap();
        let mut spec = crate::results::observation::tests::spec(
            "original-face",
            ObservationTarget::Spatial(ExcitationTarget::Region(target.clone())),
        );
        spec.output_location = FieldLocation::Face;
        spec.source_location = FieldLocation::Face;
        spec.recovery = RecoveryPolicy::Native;
        let limits = ObservationLimits::default();
        let bytes = face_selection_resident_bytes(&target, &projection, &topology, limits).unwrap();
        let run = |cap| {
            super::super::super::field::resolve_field_target(
                &spec,
                RecoveredFieldEntityDomain::OrientedFaces,
                &topology,
                ObservationLimits {
                    maximum_peak_resident_bytes: cap,
                    ..limits
                },
            )
        };
        let (result, _, peak) = crate::topology::measure_test_allocations(|| run(bytes));
        let ResolvedObservationTarget::Elements {
            indices,
            identities,
            ..
        } = result.unwrap()
        else {
            panic!("original face target");
        };
        let expected = geometry
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                f.boundary_source.is_some_and(|b| {
                    projection.boundary_faces[b.projection_boundary_index]
                        .provenance
                        .contains(&target.resolved_entities()[0])
                })
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        assert_eq!(indices, expected);
        assert_eq!(
            identities,
            indices
                .iter()
                .map(|&i| layout.oriented_faces()[i].identity)
                .collect::<Vec<_>>()
        );
        assert!(peak as u64 <= bytes, "face scratch {peak} exceeds {bytes}");
        let (short, retained, peak) = crate::topology::measure_test_allocations(|| run(bytes - 1));
        assert!(matches!(
            short,
            Err(ObservationError::Limit(
                "target selection construction residency"
            ))
        ));
        assert_eq!((retained, peak), (0, 0));
    }
}

#[test]
fn original_solid_boundary_nodal_union_retains_identities_within_admission() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let topology = ObservationTopology::from_projection(&projection).unwrap();
        let mut target = RegionTarget::new(
            "original-face-nodes",
            EntityDimension::Face,
            ExpectedTopology::BoundaryFace,
            TargetCardinality::OneOrMore,
        );
        target
            .bind_resolved_entities(vec![projection.boundary_faces[0].provenance[0].clone()])
            .unwrap();
        let mut spec = crate::results::observation::tests::spec(
            "face-nodes",
            ObservationTarget::Spatial(ExcitationTarget::Region(target.clone())),
        );
        spec.output_location = FieldLocation::Node;
        let bytes = node_union_resident_bytes(topology.nodes.len(), usize::MAX).unwrap();
        let (result, _, peak) = crate::topology::measure_test_allocations(|| {
            super::super::super::field::resolve_field_target(
                &spec,
                RecoveredFieldEntityDomain::Nodes,
                &topology,
                ObservationLimits {
                    maximum_peak_resident_bytes: bytes,
                    ..Default::default()
                },
            )
        });
        let ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } = result.unwrap()
        else {
            panic!("original nodal target");
        };
        let expected = projection
            .boundary_faces
            .iter()
            .filter(|f| f.provenance.contains(&target.resolved_entities()[0]))
            .flat_map(|f| f.node_indices.iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(indices, expected);
        assert_eq!(
            identities,
            indices
                .iter()
                .map(|&i| projection.nodes[i].stable_identity)
                .collect::<Vec<_>>()
        );
        assert!(peak as u64 <= bytes);
    }
}

use super::*;
use crate::results::field::SupportTopology;
use runmat_meshing_core::fixtures::layered_box_mesh;

/// A solid layer under a fluid layer: the fluid is a region subset of one conformal mesh.
fn coupled() -> (
    AnalysisProjection,
    FiniteVolumeTopology,
    FiniteVolumeTopology,
) {
    let mesh = layered_box_mesh(
        [2, 2, 2],
        [1.0, 1.0, 1.0],
        &[(0.5, "solid"), (1.0, "fluid")],
    );
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let fluid_region = projection
        .elements
        .iter()
        .map(|element| element.region_id.clone())
        .find(|region| region.source_topology_id.contains("fluid"))
        .unwrap();
    let fluid = FiniteVolumeTopology::derive_selected(
        &projection,
        FiniteVolumeProjectionPolicy::covering(&projection),
        FiniteVolumeCellSelection::regions(vec![fluid_region]).unwrap(),
    )
    .unwrap();
    let complete = FiniteVolumeTopology::derive_selected(
        &projection,
        FiniteVolumeProjectionPolicy::covering(&projection),
        FiniteVolumeCellSelection::AllElements,
    )
    .unwrap();
    (projection, fluid, complete)
}

#[test]
fn fluid_subset_is_region_support_on_mesh_elements_and_owner_oriented_faces() {
    let (projection, fluid, complete) = coupled();
    assert!(fluid.cells.len() < projection.elements.len());
    let layout = VolumeMeshLayout::derive(&projection, &fluid).unwrap();
    assert_eq!(layout.counts.volume_elements, projection.elements.len());
    assert_eq!(layout.counts.oriented_faces, complete.faces.len());
    let topology = SupportTopology {
        projection: &projection,
        face_elements: Some(
            &complete
                .faces
                .iter()
                .map(|face| {
                    (
                        complete.cells[face.owner].source_element_index,
                        face.neighbor
                            .map(|n| complete.cells[n].source_element_index),
                    )
                })
                .collect::<Vec<_>>(),
        ),
    };
    // Stored cells are the supported mesh elements, in mesh element order.
    let elements = layout
        .support
        .resolve(RecoveredFieldEntityDomain::VolumeElements, topology)
        .unwrap()
        .unwrap();
    assert_eq!(
        elements,
        layout
            .cells
            .iter()
            .map(|&cell| fluid.cells[cell].source_element_index)
            .collect::<Vec<_>>()
    );
    // Stored faces are the supported mesh faces; a flux keeps its physical direction when
    // the mesh face is owned by the solid cell across the interface.
    let faces = layout
        .faces_support
        .resolve(RecoveredFieldEntityDomain::OrientedFaces, topology)
        .unwrap()
        .unwrap();
    assert_eq!(faces.len(), layout.faces.len());
    let mut reversed = 0;
    for (&mesh_face, &(fluid_face, flip)) in faces.iter().zip(&layout.faces) {
        let mesh_face = &complete.faces[mesh_face];
        let fluid_face = &fluid.faces[fluid_face];
        assert_eq!(mesh_face.corner_identities, fluid_face.corner_identities);
        let sign = if flip { -1.0 } else { 1.0 };
        for axis in 0..3 {
            assert!(
                (mesh_face.owner_area_vector_m2[axis]
                    - sign * fluid_face.owner_area_vector_m2[axis])
                    .abs()
                    < 1e-12
            );
        }
        reversed += usize::from(flip);
    }
    assert!(reversed > 0, "some interface faces are owned by the solid");
    let flux = (0..fluid.faces.len())
        .map(|f| f as f64 + 1.0)
        .collect::<Vec<_>>();
    for (value, &(face, flip)) in layout.face_values(&flux).iter().zip(&layout.faces) {
        assert_eq!(*value, if flip { -flux[face] } else { flux[face] });
    }
}

#[test]
fn fluid_definitions_carry_the_region_support_and_node_fields_stay_complete() {
    let (projection, fluid, _) = coupled();
    let layout = VolumeMeshLayout::derive(&projection, &fluid).unwrap();
    let definitions = layout
        .definitions(super::super::workload::FluidCoupling::ConjugateHeat)
        .unwrap();
    assert_eq!(
        definitions,
        VolumeMeshLayout::source_definitions(
            &projection,
            &fluid,
            super::super::workload::FluidCoupling::ConjugateHeat,
        )
        .unwrap()
    );
    for definition in &definitions {
        let stored = definition.stored_entities(layout.counts);
        match definition.entity_domain() {
            RecoveredFieldEntityDomain::VolumeElements => {
                assert_eq!(stored, fluid.cells.len())
            }
            RecoveredFieldEntityDomain::OrientedFaces => assert_eq!(stored, fluid.faces.len()),
            RecoveredFieldEntityDomain::Nodes => {
                assert!(definition.support().is_complete());
                assert_eq!(stored, projection.nodes.len());
            }
            other => panic!("unexpected fluid domain {other:?}"),
        }
    }
}

#[test]
fn whole_mesh_fluid_is_complete_and_keeps_its_face_order() {
    let mesh = runmat_meshing_core::fixtures::slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let fluid = FiniteVolumeTopology::derive_selected(
        &projection,
        FiniteVolumeProjectionPolicy::covering(&projection),
        FiniteVolumeCellSelection::AllElements,
    )
    .unwrap();
    let layout = VolumeMeshLayout::derive(&projection, &fluid).unwrap();
    assert!(layout.support.is_complete() && layout.faces_support.is_complete());
    assert!(layout
        .faces
        .iter()
        .enumerate()
        .all(|(index, &(face, flip))| index == face && !flip));
}

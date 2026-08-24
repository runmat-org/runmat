use super::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, SolverMeshArtifact,
};

#[test]
fn canonical_tet4_and_tet10_artifacts_assemble_into_solver_csr() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let artifact = artifact(order);
        let topology = solver_solid_topology(&artifact, 3).unwrap();
        assert_eq!(topology.order, order);
        assert_eq!(topology.dof_count, topology.node_count * 3);
        let csr = assemble_solver_solid_stiffness_csr(
            &artifact,
            Some(material(200.0e9)),
            &BTreeMap::new(),
            3,
        )
        .unwrap();
        assert_eq!(csr.row_offsets.len(), topology.dof_count + 1);
        assert_eq!(csr.column_indices.len(), csr.values.len());
        assert!(csr.values.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn canonical_solver_assembly_revalidates_digest_and_uses_region_assignment() {
    let artifact = artifact(ElementOrder::Tet10);
    let default =
        assemble_solver_solid_stiffness_csr(&artifact, Some(material(1.0e6)), &BTreeMap::new(), 3)
            .unwrap();
    let selected = assemble_solver_solid_stiffness_csr(
        &artifact,
        Some(material(1.0e6)),
        &BTreeMap::from([(
            artifact.topology.volume_elements[0]
                .region_id
                .source_topology_id
                .clone(),
            material(2.0e6),
        )]),
        3,
    )
    .unwrap();
    assert_eq!(default.column_indices, selected.column_indices);
    for (default, selected) in default.values.iter().zip(&selected.values) {
        assert!((*selected - 2.0 * default).abs() <= default.abs().max(1.0) * 1.0e-12);
    }

    let mut tampered = artifact;
    tampered.topology.nodes[0].coordinates_m[0] = 0.25;
    assert!(matches!(
        assemble_solver_solid_stiffness_csr(&tampered, Some(material(1.0e6)), &BTreeMap::new(), 3,),
        Err(SolverSolidAssemblyError::InvalidArtifact(_))
    ));
}

#[test]
fn multi_material_assembly_rejects_an_unassigned_mesh_region() {
    let artifact = artifact(ElementOrder::Tet4);
    assert!(matches!(
        assemble_solver_solid_stiffness_csr(&artifact, None, &BTreeMap::new(), 3),
        Err(SolverSolidAssemblyError::UnassignedRegion {
            element_id: 1,
            region_id,
        }) if region_id == "region"
    ));
}

pub(crate) fn artifact(order: ElementOrder) -> SolverMeshArtifact {
    canonical_tetrahedron_solver_mesh(order)
}

fn material(youngs_modulus_pa: f64) -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa,
        poisson_ratio: 0.25,
    }
}

use super::*;
use runmat_analysis_core::{
    MechanicalDampingModel, RayleighMassCoefficient, RayleighStiffnessCoefficient,
};
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh},
    ElementOrder, SolverMeshArtifact,
};

#[test]
fn scoped_solid_assembly_matches_serial_operators_exactly() {
    let artifact = crate::tests::multi_region_tetrahedron_solver_mesh();
    let properties = inventory(&artifact, material(70.0e9, 2_700.0));
    let serial = assemble_solver_solid_operators(&artifact, &properties).unwrap();
    for threads in [2, 3, 4] {
        let plan = LocalExecutionPlan::from_thread_count(
            crate::planning::resource::LocalThreadCount::new(threads).unwrap(),
        )
        .unwrap();
        for _ in 0..3 {
            let parallel =
                assemble_solver_solid_operators_with_execution(&artifact, &properties, plan)
                    .unwrap();
            assert_eq!(parallel, serial);
        }
    }
}

#[test]
fn connected_tet4_affine_patch_balances_independent_boundary_tractions() {
    // u = (0.1 x, 0.2 y, 0.3 z), E = 100 Pa, nu = 0.25 gives the constant
    // stress diag(32, 40, 48) Pa. Analytically integrating sigma*n*N_i over
    // the four outer triangles gives these nodal resultants. The centroid is
    // interior, so its assembled force must cancel across the four elements.
    let mesh = connected_tetrahedron_solver_mesh();
    let operators =
        assemble_solver_solid_operators(&mesh, &inventory(&mesh, material(100.0, 1.0))).unwrap();
    let displacement = mesh
        .topology
        .nodes
        .iter()
        .flat_map(|node| {
            let [x, y, z] = node.coordinates_m;
            [0.1 * x, 0.2 * y, 0.3 * z]
        })
        .collect::<Vec<_>>();
    let internal_force = crate::algebra::spmv(&operators.stiffness, &displacement).unwrap();
    let expected = [
        [-16.0 / 3.0, -20.0 / 3.0, -8.0],
        [16.0 / 3.0, 0.0, 0.0],
        [0.0, 20.0 / 3.0, 0.0],
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 0.0],
    ];
    for (actual, expected) in internal_force.chunks_exact(3).zip(expected) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert_relative(*actual, expected, 1.0e-12);
        }
    }
    let energy = 0.5
        * displacement
            .iter()
            .zip(&internal_force)
            .map(|(u, f)| u * f)
            .sum::<f64>();
    assert_relative(energy, 12.8 / 6.0, 1.0e-12);
}

#[test]
fn scoped_solid_assembly_preserves_shared_row_sum_across_waves() {
    let artifact = artifact(ElementOrder::Tet10);
    let properties = inventory(&artifact, material(70.0e9, 2_700.0));
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, (node.coordinates_m, index * 3)))
        .collect::<NodeIndex>();
    // Repeated contributions deliberately target identical rows. This exercises summation
    // order and a second bounded wave without admitting a duplicate canonical mesh element.
    let elements = vec![artifact.topology.volume_elements[0].clone(); 19];
    let serial = assemble_solid_rows(
        &elements,
        &properties,
        &nodes,
        nodes.len() * 3,
        LocalExecutionPlan::serial(),
    )
    .unwrap();
    let plan = LocalExecutionPlan::from_thread_count(
        crate::planning::resource::LocalThreadCount::new(2).unwrap(),
    )
    .unwrap();
    let parallel =
        assemble_solid_rows(&elements, &properties, &nodes, nodes.len() * 3, plan).unwrap();
    assert_eq!(parallel.stiffness, serial.stiffness);
    assert_eq!(parallel.mass, serial.mass);
    assert_eq!(parallel.damping, serial.damping);
    assert_eq!(parallel.lumped_mass_diag, serial.lumped_mass_diag);
    assert_eq!(parallel.total_mass_kg, serial.total_mass_kg);
}

#[test]
fn scoped_solid_assembly_honors_the_shared_cancellation_token() {
    let artifact = crate::tests::multi_region_tetrahedron_solver_mesh();
    let properties = inventory(&artifact, material(70.0e9, 2_700.0));
    let cancellation = crate::FeaCancellationToken::new();
    let _guard = crate::replace_fea_progress_context(crate::FeaProgressContext {
        cancellation: cancellation.clone(),
        ..crate::FeaProgressContext::default()
    });
    cancellation.cancel();
    let plan = LocalExecutionPlan::from_thread_count(
        crate::planning::resource::LocalThreadCount::new(2).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        assemble_solver_solid_operators_with_execution(&artifact, &properties, plan),
        Err(SolverSolidAssemblyError::Cancelled)
    ));
}

#[test]
fn canonical_tet4_and_tet10_artifacts_assemble_into_solver_csr() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let artifact = artifact(order);
        let topology = solver_solid_topology(&artifact).unwrap();
        assert_eq!(topology.order, order);
        assert_eq!(topology.dof_count, topology.node_count * 3);
        let operators = assemble_solver_solid_operators(
            &artifact,
            &inventory(&artifact, material(200.0e9, 1_000.0)),
        )
        .unwrap();
        let csr = operators.stiffness;
        assert_eq!(csr.pattern().row_offsets().len(), topology.dof_count + 1);
        assert_eq!(csr.pattern().column_indices().len(), csr.values().len());
        assert!(csr.values().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn canonical_solver_assembly_revalidates_digest_and_uses_region_assignment() {
    let artifact = artifact(ElementOrder::Tet10);
    let default =
        assemble_solver_solid_operators(&artifact, &inventory(&artifact, material(1.0e6, 1_000.0)))
            .unwrap();
    let selected = assemble_solver_solid_operators(
        &artifact,
        &assigned_inventory(
            &artifact,
            artifact.topology.volume_elements[0]
                .region_id
                .source_topology_id
                .as_str(),
            material(1.0e6, 1_000.0),
            material(2.0e6, 1_000.0),
        ),
    )
    .unwrap();
    assert_eq!(default.stiffness.pattern(), selected.stiffness.pattern());
    for (default, selected) in default
        .stiffness
        .values()
        .iter()
        .zip(selected.stiffness.values())
    {
        assert!((*selected - 2.0 * default).abs() <= default.abs().max(1.0) * 1.0e-12);
    }

    let properties = inventory(&artifact, material(1.0e6, 1_000.0));
    let mut tampered = artifact;
    tampered.topology.nodes[0].coordinates_m[0] = 0.25;
    assert!(matches!(
        assemble_solver_solid_operators(&tampered, &properties),
        Err(SolverSolidAssemblyError::InvalidArtifact(_))
    ));
}

#[test]
fn multi_material_assembly_rejects_an_unassigned_mesh_region() {
    let artifact = artifact(ElementOrder::Tet4);
    assert!(matches!(
        empty_inventory(&artifact),
        Err(
            crate::properties::PropertyCompilationError::InvalidCoverage {
                assignment_count: 0,
                ..
            }
        )
    ));
}

#[test]
fn solid_material_coverage_rejects_overlap_stale_and_wrong_role() {
    let artifact = artifact(ElementOrder::Tet4);
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let exact_selector = artifact.topology.volume_elements[0]
        .region_id
        .analysis_selection_id()
        .unwrap();
    let mut overlap = model.material_assignments[0].clone();
    overlap.target = runmat_analysis_core::PropertyRegionTarget::solid(exact_selector);
    model.material_assignments.push(overlap);
    assert!(matches!(
        crate::properties::compile_solid_property_map(&model, &artifact),
        Err(
            crate::properties::PropertyCompilationError::InvalidCoverage {
                assignment_count: 2,
                ..
            }
        )
    ));

    model.material_assignments.pop();
    let mut stale = model.material_assignments[0].clone();
    stale.target = runmat_analysis_core::PropertyRegionTarget::solid("stale-region");
    model.material_assignments.push(stale);
    assert!(matches!(
        crate::properties::compile_solid_property_map(&model, &artifact),
        Err(crate::properties::PropertyCompilationError::UnusedAssignment { .. })
    ));

    model.material_assignments.truncate(1);
    model.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::cell("region");
    assert!(matches!(
        crate::properties::compile_solid_property_map(&model, &artifact),
        Err(
            crate::properties::PropertyCompilationError::InvalidCoverage {
                assignment_count: 0,
                ..
            }
        )
    ));
}

#[test]
fn explicit_all_solids_scope_covers_every_solid_and_overlaps_fail_closed() {
    let artifact = artifact(ElementOrder::Tet4);
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    let all_solids = crate::properties::compile_solid_property_map(&model, &artifact).unwrap();
    assemble_solver_solid_operators(&artifact, &all_solids)
        .expect("an explicit all-solids scope should cover the canonical solid mesh");

    let mut selected = model.material_assignments[0].clone();
    selected.target = runmat_analysis_core::PropertyRegionTarget::solid("region");
    model.material_assignments.push(selected);
    assert!(matches!(
        crate::properties::compile_solid_property_map(&model, &artifact),
        Err(
            crate::properties::PropertyCompilationError::InvalidCoverage {
                assignment_count: 2,
                ..
            }
        )
    ));
}

#[test]
fn tet4_and_tet10_mass_preserve_physical_total_and_rigid_translation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let artifact = artifact(order);
        let material = material(70.0e9, 2_700.0);
        let assembled =
            assemble_solver_solid_operators(&artifact, &inventory(&artifact, material)).unwrap();
        let expected_mass = material.density_kg_per_m3 / 6.0;
        assert_relative(assembled.total_mass_kg, expected_mass, 2.0e-12);
        assert!(assembled.lumped_mass_diag.iter().all(|value| *value > 0.0));
        assert_relative(
            assembled.lumped_mass_diag.iter().step_by(3).sum(),
            expected_mass,
            2.0e-12,
        );

        let mut rigid_x = vec![0.0; assembled.lumped_mass_diag.len()];
        for value in rigid_x.iter_mut().step_by(3) {
            *value = 1.0;
        }
        let applied = crate::algebra::spmv(&assembled.consistent_mass, &rigid_x).unwrap();
        let generalized_mass = rigid_x
            .iter()
            .zip(applied)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert_relative(generalized_mass, expected_mass, 2.0e-12);
    }
}

#[test]
fn region_density_controls_mass_without_changing_topology() {
    let artifact = artifact(ElementOrder::Tet4);
    let region = artifact.topology.volume_elements[0]
        .region_id
        .source_topology_id
        .clone();
    let baseline = assemble_solver_solid_operators(
        &artifact,
        &inventory(&artifact, material(70.0e9, 1_000.0)),
    )
    .unwrap();
    let selected = assemble_solver_solid_operators(
        &artifact,
        &assigned_inventory(
            &artifact,
            &region,
            material(70.0e9, 1_000.0),
            material(70.0e9, 2_500.0),
        ),
    )
    .unwrap();
    assert_relative(
        selected.total_mass_kg / baseline.total_mass_kg,
        2.5,
        1.0e-12,
    );
    assert_eq!(
        baseline.consistent_mass.pattern(),
        selected.consistent_mass.pattern()
    );
}

#[test]
fn rayleigh_damping_is_the_exact_element_mass_and_stiffness_sum() {
    let artifact = artifact(ElementOrder::Tet10);
    let alpha = 2.0;
    let beta = 3.0e-6;
    let mut material = material(70.0e9, 2_700.0);
    material.damping = MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(alpha),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(beta),
    };

    let assembled =
        assemble_solver_solid_operators(&artifact, &inventory(&artifact, material)).unwrap();

    for row in 0..assembled.stiffness.rows() {
        for column in 0..assembled.stiffness.columns() {
            let expected = alpha * assembled.consistent_mass.get(row, column)
                + beta * assembled.stiffness.get(row, column);
            assert_relative(assembled.damping.get(row, column), expected, 1.0e-12);
        }
    }
}

pub(crate) fn artifact(order: ElementOrder) -> SolverMeshArtifact {
    canonical_tetrahedron_solver_mesh(order)
}

fn material(youngs_modulus_pa: f64, density_kg_per_m3: f64) -> StructuralElementMaterial {
    StructuralElementMaterial {
        youngs_modulus_pa,
        poisson_ratio: 0.25,
        density_kg_per_m3,
        damping: MechanicalDampingModel::Undamped,
    }
}

fn inventory(
    artifact: &SolverMeshArtifact,
    material: StructuralElementMaterial,
) -> crate::properties::CompiledPropertyMap {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.materials.truncate(1);
    apply_structural_material(&mut model.materials[0], material);
    model.geometry_revision = u32::try_from(artifact.geometry.geometry_revision).unwrap();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    crate::properties::compile_solid_property_map(&model, artifact).unwrap()
}

fn assigned_inventory(
    artifact: &SolverMeshArtifact,
    region_id: &str,
    default: StructuralElementMaterial,
    assigned: StructuralElementMaterial,
) -> crate::properties::CompiledPropertyMap {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.materials.truncate(1);
    apply_structural_material(&mut model.materials[0], default);
    let mut selected = model.materials[0].clone();
    selected.material_id = "selected".into();
    apply_structural_material(&mut selected, assigned);
    model.materials.push(selected);
    *model.material_assignments[0]
        .target
        .selected_region_id_mut()
        .expect("fixture uses selected material region") = region_id.into();
    model.material_assignments[0].expected_material_id = "selected".into();
    model.material_assignments[0].assigned_material_id = "selected".into();
    model.geometry_revision = u32::try_from(artifact.geometry.geometry_revision).unwrap();
    crate::properties::compile_solid_property_map(&model, artifact).unwrap()
}

fn apply_structural_material(
    target: &mut runmat_analysis_core::MaterialModel,
    source: StructuralElementMaterial,
) {
    target.elasticity = Some(
        runmat_analysis_core::MaterialElasticityModel::IsotropicLinear(
            runmat_analysis_core::MaterialMechanicalModel {
                youngs_modulus_pa: source.youngs_modulus_pa,
                poisson_ratio: source.poisson_ratio,
            },
        ),
    );
    target.mass_density_kg_per_m3 = Some(source.density_kg_per_m3);
    target.damping = Some(source.damping);
}

fn empty_inventory(
    artifact: &SolverMeshArtifact,
) -> Result<crate::properties::CompiledPropertyMap, crate::properties::PropertyCompilationError> {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.materials.clear();
    model.material_assignments.clear();
    model.geometry_revision = u32::try_from(artifact.geometry.geometry_revision).unwrap();
    crate::properties::compile_solid_property_map(&model, artifact)
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let relative = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        relative <= tolerance,
        "actual={actual} expected={expected} relative={relative} tolerance={tolerance}"
    );
}

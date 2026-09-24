use super::*;

#[test]
fn native_kernel_executes_conformal_mixed_structural_workload() {
    use runmat_analysis_core::{
        AnalysisConstraint, BeamElementModel, BeamSectionModel, ConformalSolidAttachment,
        CoordinateFrame, PrescribedStructuralDofs, PrescribedValue, RegionTarget, Rotation,
        RotationDofValue, StructuralConstraint, StructuralConstraintKind, StructuralElement,
        StructuralElementKind, StructuralModel, StructuralNode, StructuralRotationDof,
    };

    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m;
    mesh.topology.nodes[0].provenance.push(PersistentEntityId {
        kind: PersistentEntityKind::Vertex,
        source_topology_id: "beam_root".into(),
        assembly_path: vec!["root".into()],
    });
    for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: if face_index == 0 { "root" } else { "tip" }.into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let assignment = model.material_assignments.first().unwrap();
    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: assignment
            .selected_region_id()
            .expect("fixture uses selected material region")
            .to_owned(),
        assembly_path: vec!["root".into()],
    };
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let attached_node_identity = mesh.topology.nodes[0].stable_identity.0;
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.structural = Some(StructuralModel {
        nodes: vec![
            StructuralNode {
                node_id: 101,
                coordinates_m: attached_coordinates,
                solid_attachment: Some(ConformalSolidAttachment::SolverNode {
                    stable_identity: attached_node_identity,
                }),
            },
            StructuralNode {
                node_id: 102,
                coordinates_m: [
                    attached_coordinates[0] + 1.0,
                    attached_coordinates[1],
                    attached_coordinates[2],
                ],
                solid_attachment: None,
            },
        ],
        elements: vec![StructuralElement {
            element_id: "beam".into(),
            region_id: "beam_span".into(),
            kind: StructuralElementKind::Beam(BeamElementModel {
                node_ids: [101, 102],
                section_id: "beam_section".into(),
                reference_axis: [0.0, 1.0, 0.0],
            }),
        }],
        beam_sections: vec![BeamSectionModel {
            section_id: "beam_section".into(),
            area_m2: 1.0e-4,
            iy_m4: 1.0e-9,
            iz_m4: 1.0e-9,
            torsion_j_m4: 1.0e-9,
            outer_fiber_y_m: 0.01,
            outer_fiber_z_m: 0.01,
            torsion_outer_radius_m: 0.01,
        }],
        shell_sections: Vec::new(),
    });
    let mut beam_assignment = model.material_assignments[0].clone();
    beam_assignment.target = runmat_analysis_core::PropertyRegionTarget::beam("beam_span");
    model.material_assignments.push(beam_assignment);
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "beam_root_rotation".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:101"),
                translations: Vec::new(),
                rotations: [
                    StructuralRotationDof::Rx,
                    StructuralRotationDof::Ry,
                    StructuralRotationDof::Rz,
                ]
                .into_iter()
                .map(|dof| RotationDofValue {
                    dof,
                    value: PrescribedValue::Constant(Rotation::new(0.0)),
                })
                .collect(),
            }),
        }));
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("mixed-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:mixed".into(),
            encryption_context: digest("mixed-encryption"),
        },
        workload: solve,
        mesh: Some(mesh),
        initial_guess: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    })
    .unwrap();
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::Solve,
            host: &prepared.host,
            inputs: &prepared.inputs,
        })
        .unwrap();
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
    assert!(result
        .run
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.severity
            != runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Error));
    assert_eq!(
        result
            .run
            .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .shape,
        vec![5, 3]
    );
    let plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&prepared.inputs[2].payload())
            .unwrap();
    assert_eq!(result.constraint_result.topology, plan.topology);
    result.constraint_result.validate_against(&plan).unwrap();
    let rotation_group = result
        .constraint_result
        .groups
        .iter()
        .find(|group| group.constraint_id == "beam_root_rotation")
        .unwrap();
    assert!(rotation_group.row_reactions.iter().all(|row| matches!(
        row.violation,
        runmat_analysis_fea::constraints::ConstraintViolation::RotationRadians(_)
    )));
    let mut substituted = result.constraint_result.clone();
    let CanonicalStructuralTopologyIdentity::ConformalMixed {
        structural_topology_digest,
        ..
    } = &mut substituted.topology
    else {
        panic!("mixed solve must retain mixed topology identity");
    };
    *structural_topology_digest = digest("substituted-structural-topology");
    assert!(substituted.validate_against(&plan).is_err());
}

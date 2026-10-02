//! Actual governed worker and original-field authority evidence.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_fluid_steady_task, AnalysisOutputPolicy, FluidSteadyTaskPreparation,
    PreparedFluidSteadyTask,
};
use runmat_analysis_fea::{
    execution::FluidSteadySolverResult,
    fixtures::{fixture_model, FixtureId},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::fixtures;

mod chain;

fn task(
    nonzero: bool,
    chunk: usize,
) -> (
    FluidSteadyWorkload,
    SolverMeshArtifact,
    PreparedFluidSteadyTask,
) {
    let mesh = fixtures::slab_mesh(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.interfaces.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.frame = CoordinateFrame::Global;
    model.materials[0].mass_density_kg_per_m3 = Some(1.0);
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(1.0),
    });
    model.material_assignments[0].target = PropertyRegionTarget::cell("slab");
    model.cfd = Some(CfdDomain {
        density: MassDensity::new(1.0),
        dynamic_viscosity: DynamicViscosity::new(1.0),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("native-fluid", vec![ProcedureStep::fluid_static("steady")]);
    let velocity = [
        Velocity::new(if nonzero { 1.0 } else { 0.0 }),
        Velocity::new(0.0),
        Velocity::new(0.0),
    ];
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "wall".into(),
            kind: if nonzero {
                FluidConstraintKind::MovingWall {
                    target: RegionTarget::boundary("slab_insulated"),
                    velocity: PrescribedValue::Constant(velocity),
                }
            } else {
                FluidConstraintKind::NoSlipWall {
                    target: RegionTarget::boundary("slab_insulated"),
                }
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary("slab_left"),
                value: PrescribedValue::Constant(velocity),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary("slab_right"),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    });
    for constraint in &mut model.constraints {
        let AnalysisConstraint::Fluid(fluid) = constraint else {
            unreachable!()
        };
        let target = match &mut fluid.kind {
            FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::MovingWall { target, .. }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => target,
            _ => unreachable!(),
        };
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| &f.provenance)
            .filter(|e| target.matches_persistent_entity(e))
            .cloned()
            .collect();
        target.bind_resolved_entities(entities).unwrap();
    }
    for boundary in &mut model.boundaries {
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| &f.provenance)
            .filter(|e| boundary.target.matches_persistent_entity(e))
            .cloned()
            .collect();
        boundary.target.bind_resolved_entities(entities).unwrap();
    }
    let workload = FluidSteadyWorkload {
        model,
        step_id: ProcedureStepId("steady".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        policy: CanonicalFluidPolicy {
            fixed_point_acceleration: None,
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-10,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: 1_000_000_000,
            maximum_operator_work: 1_000_000_000_000,
            maximum_authority_items: 100,
        },
        geometry_policy: FiniteVolumeProjectionPolicy {
            max_cells: 100,
            max_face_incidences: 400,
            max_boundary_faces: 400,
            max_scratch_bytes: 10_000_000,
            max_traversal_operations: 1_000_000,
        },
    };
    let digest = Digest::sha256(b"native fluid original authority fixture");
    let prepared = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-fluid".into(),
            encryption_context: digest,
        },
        workload: workload.clone(),
        mesh: mesh.clone(),
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap();
    (workload, mesh, prepared)
}

fn invoke(task: &PreparedFluidSteadyTask) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::SolveFluidSteady,
        host: &task.host,
        inputs: &task.inputs,
        output_policy: task.output_policy,
    })
}

#[test]
fn native_original_zero_and_analytic_translation_are_chunk_policy_invariant() {
    for nonzero in [false, true] {
        let mut identity = None;
        for chunk in [1024, 8192] {
            let (workload, mesh, task) = task(nonzero, chunk);
            let outputs = invoke(&task).unwrap();
            assert_eq!(outputs.len(), 2);
            let mut outputs = outputs.into_iter();
            let original = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, task.output_policy.inventory_limits())
                .unwrap();
            let bytes = original.payload();
            let result = FluidSteadySolverResult::from_canonical_bytes_for(
                &bytes,
                Digest::sha256(&bytes),
                &workload,
                &mesh,
            )
            .unwrap();
            if let Some(previous) = identity {
                assert_eq!(previous, result.identity().unwrap());
            }
            identity = Some(result.identity().unwrap());
            for velocity in &result.original().velocity_m_per_s {
                assert!((velocity[0] - if nonzero { 1.0 } else { 0.0 }).abs() < 1e-6);
                assert!(velocity[1].abs() < 1e-6 && velocity[2].abs() < 1e-6);
            }
            let receipt = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, task.output_policy.inventory_limits())
                .unwrap();
            let receipt =
                FluidNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload()).unwrap();
            receipt.validate_against(&workload, &result, &mesh).unwrap();
        }
    }
}

#[test]
fn native_rejects_missing_inputs_and_output_policy_substitution() {
    let (_, _, mut task) = task(false, 1024);
    task.output_policy = AnalysisOutputPolicy::new(8192, ObjectInventoryLimits::default()).unwrap();
    assert!(invoke(&task).is_err());
    task.output_policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    task.inputs.pop();
    assert!(invoke(&task).is_err());
}

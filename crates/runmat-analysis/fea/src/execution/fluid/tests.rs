use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_analysis_core::*;
use runmat_canonical_codec::encode_contract;
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures, SolverMeshArtifact};

fn authored() -> (FluidSteadyWorkload, SolverMeshArtifact) {
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
    model.sequence = AnalysisSequence::from_steps(
        "retained-fluid",
        vec![ProcedureStep::fluid_static("steady")],
    );
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "stationary".into(),
            kind: FluidConstraintKind::NoSlipWall {
                target: RegionTarget::boundary("slab_insulated"),
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary("slab_left"),
                value: PrescribedValue::Constant([Velocity::new(0.0); 3]),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary("slab_right"),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    });
    // Bind exactly the original boundary inventory, not a solver-created selector.
    for constraint in &mut model.constraints {
        let AnalysisConstraint::Fluid(c) = constraint else {
            unreachable!()
        };
        let target = match &mut c.kind {
            FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => target,
            _ => unreachable!(),
        };
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| f.provenance.iter())
            .filter(|e| target.matches_persistent_entity(e))
            .cloned()
            .collect();
        target.bind_resolved_entities(entities).unwrap();
    }
    for b in &mut model.boundaries {
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| f.provenance.iter())
            .filter(|e| b.target.matches_persistent_entity(e))
            .cloned()
            .collect();
        b.target.bind_resolved_entities(entities).unwrap();
    }
    let workload = FluidSteadyWorkload {
        model,
        step_id: ProcedureStepId("steady".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        policy: CanonicalFluidPolicy {
            maximum_outer_iterations: 20,
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
    (workload, mesh)
}

#[test]
fn original_zero_flow_cold_fields_and_traction_evidence_roundtrip_without_resolve() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    assert!(result
        .original()
        .velocity_m_per_s
        .iter()
        .flatten()
        .all(|&v| v == 0.0));
    assert!(result.original().pressure_pa.iter().all(|&v| v == 0.0));
    assert!(result
        .original()
        .face_volume_flux_m3_per_s
        .iter()
        .all(|&v| v == 0.0));
    let bytes = result.canonical_bytes().unwrap();
    let cold = FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &workload,
        &mesh,
    )
    .unwrap();
    assert_eq!(cold, result);
    assert_eq!(
        cold.recover_original(&workload, &mesh).unwrap(),
        result.original().residual
    );
    assert_eq!(cold.source_workload(), workload.identity().unwrap());
    assert_eq!(
        cold.producer_identity(),
        workload.producer_identity().unwrap()
    );
    let workload_bytes = workload.canonical_bytes().unwrap();
    assert_eq!(
        FluidSteadyWorkload::from_canonical_bytes(&workload_bytes).unwrap(),
        workload
    );
}

#[test]
fn exact_digest_precedes_decode_and_source_admission() {
    let (mut workload, mesh) = authored();
    workload.model.sequence.steps.clear();
    let error = FluidSteadySolverResult::from_canonical_bytes_for(
        b"not a result",
        Digest::sha256(b"foreign"),
        &workload,
        &mesh,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::execution::AnalysisExecutionError::Identity(_)
    ));
}

#[test]
fn original_nonzero_uniform_flow_cold_roundtrip_preserves_analytic_face_flux() {
    let (mut workload, mesh) = authored();
    workload.policy.maximum_outer_iterations = 200;
    let velocity = [Velocity::new(1.0), Velocity::new(0.0), Velocity::new(0.0)];
    for constraint in &mut workload.model.constraints {
        let AnalysisConstraint::Fluid(fluid) = constraint else {
            unreachable!()
        };
        fluid.kind = match &fluid.kind {
            FluidConstraintKind::NoSlipWall { target } => FluidConstraintKind::MovingWall {
                target: target.clone(),
                velocity: PrescribedValue::Constant(velocity),
            },
            FluidConstraintKind::PrescribedVelocity { target, .. } => {
                FluidConstraintKind::PrescribedVelocity {
                    target: target.clone(),
                    value: PrescribedValue::Constant(velocity),
                }
            }
            _ => unreachable!(),
        };
    }
    // Independent exact rigid translation: div U=0, grad U=0, p=0 and
    // every original owner-oriented face has volume flux S_x. Moving side
    // walls are tangential; no fabricated spatial profile or solver RHS.
    let result = workload.execute(&mesh).unwrap();
    for u in &result.original().velocity_m_per_s {
        assert!((u[0] - 1.0).abs() < 1e-6 && u[1].abs() < 1e-6 && u[2].abs() < 1e-6);
    }
    let source = workload.prepare(&mesh).unwrap();
    for (q, face) in result
        .original()
        .face_volume_flux_m3_per_s
        .iter()
        .zip(&source.topology.faces)
    {
        assert!((q - face.owner_area_vector_m2[0]).abs() < 1e-6);
    }
    let bytes = result.canonical_bytes().unwrap();
    let reopened = FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &workload,
        &mesh,
    )
    .unwrap();
    assert_eq!(reopened, result);
    assert_eq!(
        reopened.recover_original(&workload, &mesh).unwrap(),
        result.original().residual
    );
}

#[test]
fn original_velocity_pressure_flux_traction_and_resource_substitution_reject() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    for path in [
        vec!["original", "velocity_m_per_s", "0", "0"],
        vec!["original", "pressure_pa", "0"],
        vec!["original", "face_volume_flux_m3_per_s", "0"],
        vec![
            "original",
            "residual",
            "face_forces",
            "viscous_traction_n",
            "0",
            "0",
        ],
        vec!["reservation", "resident_bytes_upper_bound"],
    ] {
        let mut wire = serde_json::to_value(&result).unwrap();
        let mut cursor = &mut wire["data"];
        for component in path {
            cursor = if let Ok(index) = component.parse::<usize>() {
                &mut cursor[index]
            } else {
                &mut cursor[component]
            };
        }
        *cursor = cursor
            .as_u64()
            .map(|v| serde_json::json!(v + 1))
            .unwrap_or(serde_json::json!(1.0));
        let bytes =
            encode_contract(RESULT_PREFIX, "steady-fluid-retained-result", &wire, LIMITS).unwrap();
        assert!(FluidSteadySolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh
        )
        .is_err());
    }
}

#[test]
fn missing_original_cell_material_boundary_or_source_and_stale_cfd_reject() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    let mut wrong = workload.clone();
    wrong.model.material_assignments.clear();
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let mut wrong = workload.clone();
    wrong.model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let mut wrong = workload.clone();
    wrong.model.constraints.pop();
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let mut wrong = workload.clone();
    wrong.model.cfd.as_mut().unwrap().dynamic_viscosity = DynamicViscosity::new(2.0);
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let foreign = fixtures::slab_mesh(2);
    assert!(result.validate_for(&workload, &foreign).is_err());
}

#[test]
fn cold_preflight_and_closed_static_scope_reject_before_original_vectors_decode() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    let bytes = result.canonical_bytes().unwrap();
    let geometry =
        crate::topology::FiniteVolumeTopology::reserve_mesh(&mesh, workload.geometry_policy)
            .unwrap();
    let numerical = crate::physics::cfd::canonical::CanonicalFluidReservation::for_counts(
        &geometry,
        geometry.face_incidence_count,
        workload.policy,
    )
    .unwrap();
    let mut low_preparation = workload.clone();
    low_preparation.policy.maximum_resident_bytes = numerical.resident_bytes_upper_bound;
    let mut invalid_geometry = mesh.clone();
    invalid_geometry.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    let error = low_preparation
        .prepare(&invalid_geometry)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("steady fluid preparation requested heap"),
        "{error}"
    );
    let mut wrong = workload.clone();
    wrong.policy.maximum_resident_bytes = 1;
    assert!(FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &wrong,
        &mesh
    )
    .is_err());
    let mut wrong = workload.clone();
    wrong.policy.maximum_operator_work = 1;
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let mut wrong = workload.clone();
    wrong.model.sequence.steps[0].procedure = AnalysisProcedure::TransientResponse;
    assert!(wrong.validate().is_err());
    let mut wrong = workload.clone();
    wrong.formulation_requirement = FormulationRequirement::Production;
    assert!(wrong.validate().is_err());
    let mut wrong = workload.clone();
    wrong.model.sequence.steps[0].state_outputs.clear();
    assert!(wrong.validate().is_err());
    let mut wrong = workload.clone();
    wrong.model.sequence.steps[0].state_outputs[0].kind = StateKind::Thermal;
    assert!(wrong.validate().is_err());
    let mut wrong = workload.clone();
    wrong.model.sequence.steps[0].state_outputs[0].completeness = StateCompleteness::Checkpoint;
    assert!(wrong.validate().is_err());
    let mut wrong = workload;
    wrong.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "no".into(),
        required_state_digest: "no".into(),
    };
    assert!(wrong.validate().is_err());
}

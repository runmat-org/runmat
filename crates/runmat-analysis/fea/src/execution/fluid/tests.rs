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

#[test]
fn fluid_execution_keeps_accounting_and_cancellation_typed() {
    use crate::physics::{
        cfd::canonical::CanonicalFluidError, fluid_structure::FluidStructureError,
    };
    use runmat_simulation_usage::LedgerError;

    let cause = LedgerError::GrantUnavailable;
    assert!(matches!(
        fluid_error(CanonicalFluidError::Accounting(cause)),
        AnalysisExecutionError::Usage(LedgerError::GrantUnavailable)
    ));
    assert!(matches!(
        fluid_structure_error(FluidStructureError::Flow(CanonicalFluidError::Accounting(
            cause
        ))),
        AnalysisExecutionError::Usage(LedgerError::GrantUnavailable)
    ));
    assert!(matches!(
        fluid_error(CanonicalFluidError::Cancelled),
        AnalysisExecutionError::Cancelled
    ));
    assert!(matches!(
        fluid_structure_error(FluidStructureError::Cancelled),
        AnalysisExecutionError::Cancelled
    ));
}

pub(crate) fn authored() -> (FluidSteadyWorkload, SolverMeshArtifact) {
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
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(1.0),
    });
    model.material_assignments[0].target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.cfd = Some(CfdDomain {
        density: MassDensity::new(1.0),
        dynamic_viscosity: DynamicViscosity::new(1.0),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
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
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_insulated",
                )),
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant([Velocity::new(0.0); 3]),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
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
        policy: CanonicalFluidPolicy {
            pressure_velocity_coupling: Default::default(),
            fixed_point_acceleration: None,
            maximum_outer_iterations: 20,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-10,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: Some(1_000_000_000),
            maximum_operator_work: Some(1_000_000_000_000),
            maximum_authority_items: 100,
        },
        geometry_policy: FiniteVolumeProjectionPolicy {
            max_cells: Some(100),
            max_face_incidences: Some(400),
            max_boundary_faces: Some(400),
            max_scratch_bytes: Some(10_000_000),
            max_traversal_operations: Some(1_000_000),
        },
    };
    (workload, mesh)
}

#[test]
fn original_zero_flow_cold_fields_and_traction_evidence_roundtrip_without_resolve() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    assert_eq!(
        workload.result_encoding_profile(&mesh).unwrap(),
        runmat_canonical_codec::profile_serialized_shape(&result, LIMITS).unwrap()
    );
    assert!(result
        .original()
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .flatten()
        .all(|&v| v == 0.0));
    assert!(result
        .original()
        .pressure_pa
        .canonical_values()
        .iter()
        .all(|&v| v == 0.0));
    assert!(result
        .original()
        .face_volume_flux_m3_per_s
        .canonical_values()
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
fn acceleration_controls_bind_portable_workload_result_and_original_reservation() {
    let (mut workload, mesh) = authored();
    let ordinary = workload.identity().unwrap();
    workload.policy.fixed_point_acceleration =
        Some(crate::solve::fixed_point::AndersonOnePolicy::default());
    assert_ne!(workload.identity().unwrap(), ordinary);
    let wire = workload.canonical_bytes().unwrap();
    assert_eq!(
        FluidSteadyWorkload::from_canonical_bytes(&wire).unwrap(),
        workload
    );
    let result = workload.execute(&mesh).unwrap();
    let bytes = result.canonical_bytes().unwrap();
    let cold = FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &workload,
        &mesh,
    )
    .unwrap();
    assert_eq!(cold.original(), result.original());
    assert_eq!(cold.source_workload(), workload.identity().unwrap());
    for acceleration in [
        None,
        Some(crate::solve::fixed_point::AndersonOnePolicy {
            maximum_absolute_coefficient: 16.0,
        }),
    ] {
        let mut substituted = workload.clone();
        substituted.policy.fixed_point_acceleration = acceleration;
        assert!(cold.validate_for(&substituted, &mesh).is_err());
    }
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
    for u in result.original().velocity_m_per_s.canonical_values() {
        assert!((u[0] - 1.0).abs() < 1e-6 && u[1].abs() < 1e-6 && u[2].abs() < 1e-6);
    }
    let source = workload.prepare(&mesh).unwrap();
    let request = result.native_field_request(&workload, &mesh).unwrap();
    let native = result
        .recover_native_fields(&workload, &mesh, &request)
        .unwrap();
    // Cells are stored in mesh element order; the whole mesh is fluid, so faces keep their
    // order and orientation.
    let mut cells = (0..source.topology.cells.len()).collect::<Vec<_>>();
    cells.sort_by_key(|&cell| source.topology.cells[cell].source_element_index);
    for (field, expected) in native.fields().iter().zip([
        cells
            .iter()
            .flat_map(|&cell| result.original().velocity_m_per_s.canonical_values()[cell])
            .collect::<Vec<_>>(),
        cells
            .iter()
            .map(|&cell| result.original().pressure_pa.canonical_values()[cell])
            .collect(),
        result
            .original()
            .face_volume_flux_m3_per_s
            .canonical_values()
            .clone(),
    ]) {
        assert_eq!(
            field.storage().values,
            AnalysisFieldValues::HostF64(expected)
        );
    }
    for (q, face) in result
        .original()
        .face_volume_flux_m3_per_s
        .canonical_values()
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
fn native_cells_and_internal_faces_preserve_original_fields_and_authored_observations() {
    use crate::results::{
        field::{FiniteVolumeFieldTopology, RecoveredFieldCatalog, RecoveredFieldEntityDomain},
        observation::{
            evaluate_observations, resolve_observation_plan, ObservationEntity, ObservationLimits,
        },
    };
    let (mut workload, mesh) = authored();
    for (id, quantity, location) in [
        ("u", ObservationQuantity::FluidVelocity, FieldLocation::Cell),
        ("p", ObservationQuantity::FluidPressure, FieldLocation::Cell),
        (
            "q",
            ObservationQuantity::VolumeFlowRate,
            FieldLocation::Face,
        ),
    ] {
        workload.model.sequence.steps[0]
            .observations
            .push(ObservationRequest::Field(ObservationSpec {
                observation_id: ObservationId(id.into()),
                quantity,
                value_shape: quantity.canonical_value_shape(),
                target: ObservationTarget::WholeModel,
                source_location: location,
                output_location: location,
                recovery: RecoveryPolicy::Native,
                component: ObservationComponent::All,
                frame: CoordinateFrame::Global,
                sign_convention: SignConvention::Native,
                complex_representation: ComplexRepresentation::Real,
                sampling: SamplingPolicy::Final,
                reduction: None,
                storage: ObservationStoragePolicy {
                    precision: SignalPrecision::Real64,
                    compression: CompressionPolicy::Lossless,
                    retention: RetentionPolicy::Canonical,
                    partial_access: PartialAccessPolicy::ManifestAndChunks,
                },
                purpose: ObservationPurpose::Publication,
                cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
                    operation_count: 100_000,
                    peak_resident_bytes: 1_000_000,
                    stored_bytes: 1_000_000,
                    transfer_bytes: 1_000_000,
                    collective_count: 0,
                }),
            }));
    }
    let authored_observations = workload.model.sequence.steps[0].observations.clone();
    let result = workload.execute(&mesh).unwrap();
    assert_eq!(
        workload.model.sequence.steps[0].observations,
        authored_observations
    );
    let source = workload.prepare(&mesh).unwrap();
    let layout = result.native_field_topology(&workload, &mesh).unwrap();
    assert_eq!(layout.oriented_faces().len(), source.topology.faces.len());
    assert!(layout.oriented_faces().iter().any(|f| f.neighbor.is_some()));
    let bytes = layout.canonical_bytes().unwrap();
    let cold = FiniteVolumeFieldTopology::from_canonical_bytes(&bytes).unwrap();
    cold.validate_for(&source.source, &source.topology).unwrap();
    let request = result.native_field_request(&workload, &mesh).unwrap();
    let artifact = result
        .recover_native_fields(&workload, &mesh, &request)
        .unwrap();
    assert_eq!(
        artifact.fields()[0].storage().shape,
        vec![source.topology.cells.len(), 3]
    );
    assert_eq!(
        artifact.fields()[2].definition().entity_domain(),
        RecoveredFieldEntityDomain::OrientedFaces
    );
    assert_eq!(
        artifact.fields()[2].storage().shape,
        vec![source.topology.faces.len()]
    );
    // The whole mesh is fluid: complete support, mesh faces in fluid face order.
    assert!(artifact
        .fields()
        .iter()
        .all(|field| field.definition().support().is_complete()));
    assert_eq!(
        artifact.fields()[2].storage().as_host_f64().unwrap(),
        &result
            .original()
            .face_volume_flux_m3_per_s
            .canonical_values()[..]
    );
    let definitions = artifact
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        source.source.projection_digest,
        artifact.entity_counts(),
        artifact.into_fields(),
    )
    .unwrap();
    let plan = resolve_observation_plan(
        &workload.model.sequence.steps[0],
        &source.source,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let observed = evaluate_observations(&plan, &source.source, &catalog).unwrap();
    assert_eq!(observed[0].location, FieldLocation::Cell);
    assert_eq!(observed[2].entities.len(), source.topology.faces.len());
    assert!(observed[2]
        .entities
        .iter()
        .all(|e| matches!(e, ObservationEntity::OrientedFace(_))));
    let mut substituted = source.topology.clone();
    substituted.faces[0].owner_area_vector_m2[0] += 0.25;
    assert!(cold.validate_for(&source.source, &substituted).is_err());
    let mut wrong = request.clone();
    wrong.plan.requirements[2].cost.output_entity_count -= 1;
    assert!(result
        .recover_native_fields(&workload, &mesh, &wrong)
        .is_err());
}

// witness: FEA-FLUID-EXECUTION-001/resources @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Original payload and authored requested ceilings before expansion; no measured
//   RSS/GPU/distributed bound.
#[test]
fn authored_resource_ceiling_and_original_wire_profile_are_enforced_before_expansion() {
    let (mut workload, mesh) = authored();
    let projection =
        crate::planning::resource::derive_fluid_steady_resource_projection(&workload, &mesh)
            .unwrap();
    let result = workload.execute(&mesh).unwrap();
    assert_eq!(
        workload.result_encoding_profile(&mesh).unwrap(),
        runmat_canonical_codec::profile_serialized_shape(&result, LIMITS).unwrap()
    );
    assert!(
        projection.phase_resident_bytes_upper_bound.0
            > projection.numerical.resident_bytes_upper_bound
    );
    let mut caller_limited = workload.clone();
    caller_limited.policy.maximum_resident_bytes =
        Some(projection.phase_resident_bytes_upper_bound.0 - 1);
    assert!(caller_limited.prepare(&mesh).is_ok());
    assert!(
        crate::planning::resource::derive_fluid_steady_resource_projection(&caller_limited, &mesh)
            .is_err()
    );
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(workload.prepare(&mesh).is_err());
    assert!(
        crate::planning::resource::derive_fluid_steady_resource_projection(&workload, &mesh)
            .is_err()
    );
}

// witness: FEA-FLUID-RECOVERY-001/fields @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Portable original field/traction/resource substitution rejection; no process
//   termination.
#[test]
fn original_velocity_pressure_flux_traction_and_resource_substitution_reject() {
    let (workload, mesh) = authored();
    let result = workload.execute(&mesh).unwrap();
    for path in [
        vec!["original", "steady", "velocity_m_per_s", "0", "0"],
        vec!["original", "steady", "pressure_pa", "0"],
        vec!["original", "steady", "pressure_fluctuation_pa", "0"],
        vec![
            "original",
            "steady",
            "residual",
            "boundary_balances",
            "0",
            "volume_flow_rate_m3_per_s",
        ],
        vec!["original", "steady", "face_volume_flux_m3_per_s", "0"],
        vec!["original", "steady", "pressure_reference_pa", "0"],
        vec![
            "original",
            "steady",
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
    for legacy_schema in 0..4 {
        let mut wire = serde_json::to_value(&result).unwrap();
        if legacy_schema >= 2 {
            wire["data"]["schema_version"] =
                serde_json::json!(if legacy_schema == 2 { 1 } else { 4 });
        } else {
            wire["data"]["original"]["steady"]
                .as_object_mut()
                .unwrap()
                .remove(if legacy_schema == 0 {
                    "pressure_fluctuation_pa"
                } else {
                    "pressure_reference_pa"
                });
        }
        let bytes =
            encode_contract(RESULT_PREFIX, "steady-fluid-retained-result", &wire, LIMITS).unwrap();
        assert!(FluidSteadySolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh,
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
    wrong.model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
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
    low_preparation.policy.maximum_resident_bytes = Some(numerical.resident_bytes_upper_bound);
    let mut invalid_geometry = mesh.clone();
    invalid_geometry.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    let error = low_preparation
        .prepare(&invalid_geometry)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("canonical fluid preparation requested heap"),
        "{error}"
    );
    let mut wrong = workload.clone();
    wrong.policy.maximum_resident_bytes = Some(1);
    assert!(FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &wrong,
        &mesh
    )
    .is_err());
    let mut wrong = workload.clone();
    wrong.policy.maximum_operator_work = Some(1);
    assert!(result.validate_for(&wrong, &mesh).is_err());
    let mut wrong = workload.clone();
    wrong.model.sequence.steps[0].procedure = AnalysisProcedure::TransientResponse;
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

mod closed_reference;

mod checkpoint;
mod transient_workload;

mod mass_flow;
mod pressure_drop;

/// Calibration probe (ignored): the steady-fluid reservation pieces for a channel on an
/// `RM_DIAG_N`³-hexahedra slab (Tet4 cells), then optionally the solve's process peak RSS.
/// Run alone (`--exact --ignored`) so the RSS belongs to this solve.
#[test]
#[ignore]
fn diagnostic_fluid_reservation_at_scale() {
    let n: usize = std::env::var("RM_DIAG_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let (mut workload, _) = authored();
    let mesh = fixtures::slab_mesh(n);
    for constraint in &mut workload.model.constraints {
        let AnalysisConstraint::Fluid(c) = constraint else {
            unreachable!()
        };
        match &mut c.kind {
            FluidConstraintKind::NoSlipWall { target } => {
                rebind(target, &mesh);
            }
            FluidConstraintKind::PrescribedVelocity { target, value } => {
                *value = PrescribedValue::Constant([
                    Velocity::new(0.01),
                    Velocity::new(0.0),
                    Velocity::new(0.0),
                ]);
                rebind(target, &mesh);
            }
            _ => unreachable!(),
        }
    }
    for b in &mut workload.model.boundaries {
        rebind(&mut b.target, &mesh);
    }
    workload.policy.maximum_outer_iterations = 400;
    workload.policy.gmres_restart = 32;
    workload.policy.linear_relative_tolerance = 1e-6;
    workload.policy.physical_relative_tolerance = 1e-6;
    workload.policy.maximum_resident_bytes = Some(64_000_000_000);
    workload.policy.maximum_operator_work = Some(1 << 52);
    workload.geometry_policy = FiniteVolumeProjectionPolicy {
        max_cells: None,
        max_face_incidences: None,
        max_boundary_faces: None,
        max_scratch_bytes: None,
        max_traversal_operations: None,
    };
    let projection =
        crate::planning::resource::derive_fluid_steady_resource_projection(&workload, &mesh)
            .unwrap();
    let r = projection.numerical;
    println!(
        "cells {} faces {} sparse {} numerical {} verification {} preparation {} decode {} \
         result {} materialization {} phase {}",
        r.cells,
        r.faces,
        r.sparse_entries,
        r.resident_bytes_upper_bound,
        r.verification_resident_bytes_upper_bound,
        projection.source_preparation_resident_bytes.0,
        projection.actual_decode_resident_bytes.0,
        projection.result_payload_bytes_upper_bound.0,
        projection.result_materialization_resident_bytes.0,
        projection.phase_resident_bytes_upper_bound.0
    );
    if std::env::var("RM_DIAG_SOLVE").is_ok() {
        let peak = || {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
            unsafe { usage.assume_init() }.ru_maxrss
        };
        let before = peak();
        let outcome = workload.execute(&mesh);
        println!(
            "rss_peak_before_solve {before} rss_peak_after_solve {} ok {}",
            peak(),
            outcome.is_ok()
        );
    }
}

fn rebind(target: &mut RegionTarget, mesh: &SolverMeshArtifact) {
    let entities = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|f| f.provenance.iter())
        .filter(|e| target.matches_persistent_entity(e))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    target.bind_resolved_entities(entities).unwrap();
}

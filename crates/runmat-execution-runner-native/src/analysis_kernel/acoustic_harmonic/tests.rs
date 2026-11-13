use super::*;
use crate::analysis_kernel::NativeAnalysisKernel;
use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisConstraint, Frequency, IndependentDomain,
    MaterialAcousticModel, PrescribedValue, Pressure, ProcedureStep, ProcedureStepId,
    PropertyRegionTarget, RegionTarget,
};
use runmat_analysis_execution::{
    prepare_acoustic_harmonic_task, AcousticHarmonicTaskPreparation, AnalysisKernel,
    AnalysisOutputPolicy, PreparedAcousticHarmonicTask,
};
use runmat_analysis_fea::{
    execution::{AcousticHarmonicSolverResult, AnalysisWorkKind},
    physics::acoustic::AcousticHarmonicPolicy,
};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

mod accuracy;
mod incident;
mod publication;
mod sweep;

fn task(
    order: ElementOrder,
    chunk: usize,
) -> (
    AcousticHarmonicWorkload,
    SolverMeshArtifact,
    PreparedAcousticHarmonicTask,
) {
    task_with_constraint_name(order, chunk, "pressure")
}

fn task_with_constraint_name(
    order: ElementOrder,
    chunk: usize,
    constraint_name: &str,
) -> (
    AcousticHarmonicWorkload,
    SolverMeshArtifact,
    PreparedAcousticHarmonicTask,
) {
    task_with_boundary(order, chunk, constraint_name, false)
}

fn task_with_boundary(
    order: ElementOrder,
    chunk: usize,
    constraint_name: &str,
    radiation: bool,
) -> (
    AcousticHarmonicWorkload,
    SolverMeshArtifact,
    PreparedAcousticHarmonicTask,
) {
    task_with_threads(
        order,
        chunk,
        constraint_name,
        radiation,
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    )
}

fn task_with_threads(
    order: ElementOrder,
    chunk: usize,
    constraint_name: &str,
    radiation: bool,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> (
    AcousticHarmonicWorkload,
    SolverMeshArtifact,
    PreparedAcousticHarmonicTask,
) {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.sequence.steps = vec![step];
    for face in 0..1 {
        model
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: format!("{constraint_name}:{face}"),
                kind: AcousticConstraintKind::PrescribedPressure {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                    value: PrescribedValue::Constant(Pressure::new(1.0)),
                },
            }));
    }
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "matched".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
            operator: if radiation {
                runmat_analysis_core::BoundaryOperator::AcousticRadiation
            } else {
                runmat_analysis_core::BoundaryOperator::AcousticImpedance {
                    impedance: runmat_analysis_core::SpecificAcousticImpedance::new(6.0),
                }
            },
        });
    let workload = AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic".into()),
        frequency: Frequency::new(0.2),
        solver: AcousticHarmonicPolicy::default(),
        power_balance_relative_tolerance: 1e-8,
    };
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let digest = Digest::sha256("acoustic-native-test");
    let prepared = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
        local_execution,
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:acoustic".into(),
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

#[test]
fn native_acoustic_admits_deduplicated_input_transport() {
    let (workload, mesh, task) =
        task_with_constraint_name(ElementOrder::Tet10, 1024, &"p".repeat(20_000));
    let facts = runmat_analysis_execution::prepared_input_resource_facts_without_resource_plan(
        &task.inputs,
    )
    .unwrap();
    assert!(facts.transport_closure_bytes < facts.encoded_payload_bytes);
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default())
                .unwrap(),
        })
        .unwrap();
    let objects = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_prepared(1024, ObjectInventoryLimits::default())
        .unwrap();
    let result = AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload()).unwrap();
    result.validate_for(&workload, &mesh).unwrap();
}

#[test]
fn native_acoustic_cold_inputs_and_result_are_transport_independent() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut identity = None;
        for chunk in [1024, 8192] {
            let (workload, mesh, task) = task(order, chunk);
            let outputs = NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::SolveAcousticHarmonic,
                    host: &task.host,
                    inputs: &task.inputs,
                    output_policy: AnalysisOutputPolicy::new(
                        chunk,
                        ObjectInventoryLimits::default(),
                    )
                    .unwrap(),
                })
                .unwrap();
            assert_eq!(outputs.len(), 3);
            let objects = outputs
                .into_iter()
                .next()
                .unwrap()
                .into_prepared(chunk, ObjectInventoryLimits::default())
                .unwrap();
            let result =
                AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload()).unwrap();
            result.validate_for(&workload, &mesh).unwrap();
            assert_eq!(
                result.source_result_identity().unwrap().digest(),
                objects.manifest.logical_identity
            );
            if let Some(expected) = identity {
                assert_eq!(objects.manifest.logical_identity, expected);
            } else {
                identity = Some(objects.manifest.logical_identity);
            }
            assert!(result.velocity_m_per_s().iter().any(|v| v
                .real
                .iter()
                .chain(&v.imaginary)
                .any(|x| x.abs() > 1e-10)));
            // This driven single-element fixture tests execution, not continuum accuracy.
            assert_eq!(result, workload.execute(&mesh).unwrap());
        }
    }
}

#[test]
fn native_tet4_fields_match_independent_exact_element_integrals() {
    let (_, _, task) = task(ElementOrder::Tet4, 1024);
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default())
                .unwrap(),
        })
        .unwrap();
    let objects = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_prepared(1024, ObjectInventoryLimits::default())
        .unwrap();
    let result = AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload()).unwrap();
    // Unit right tetrahedron, rho=3, c=2, Z=6. N4=z, V=1/6, loaded
    // face x+y+z=1 has area sqrt(3)/2. Integrals give K44=1/18,
    // M44=1/720, M4j=1/1440, B44=sqrt(3)/72 and B42+B43=B44.
    // Nodes1/2/3 are prescribed at1 Pa. This reference is not assembled by RunMat.
    let omega = 2.0 * std::f64::consts::PI * 0.2;
    let ar = 1.0 / 18.0 - omega.powi(2) / 720.0;
    let ai = omega * 3.0_f64.sqrt() / 72.0;
    let br = 1.0 / 18.0 + omega.powi(2) / 480.0;
    let bi = -ai;
    let denominator = ar * ar + ai * ai;
    let real = (br * ar + bi * ai) / denominator;
    let imaginary = (bi * ar - br * ai) / denominator;
    let node = result
        .ordered_node_ids()
        .iter()
        .position(|id| *id == 4)
        .unwrap();
    let pressure = result.pressure_pa()[node];
    assert!((pressure.real - real).abs() < 1e-10);
    assert!((pressure.imaginary - imaginary).abs() < 1e-10);
    // p=1+(p4-1)z; v=i grad(p)/(omega rho), peak phasors.
    let velocity = &result.velocity_m_per_s()[0];
    let vr = -imaginary / (omega * 3.0);
    let vi = (real - 1.0) / (omega * 3.0);
    for axis in 0..2 {
        assert!(velocity.real[axis].abs() < 1e-10);
        assert!(velocity.imaginary[axis].abs() < 1e-10);
    }
    assert!((velocity.real[2] - vr).abs() < 1e-10);
    assert!((velocity.imaginary[2] - vi).abs() < 1e-10);
    let expected_intensity = 0.5 * (((3.0 + real) / 4.0) * vr + imaginary / 4.0 * vi);
    assert!((result.intensity_w_per_m2()[0][2] - expected_intensity).abs() < 1e-10);
    // Exact moments <z>=1/4, <z²>=1/10; integrate products before averaging.
    let pressure_squared =
        1.0 + (real - 1.0) / 2.0 + ((real - 1.0).powi(2) + imaginary.powi(2)) / 10.0;
    let expected_energy = (pressure_squared / 12.0 + 3.0 * (vr * vr + vi * vi)) / 4.0;
    assert!((result.energy_density_j_per_m3()[0] - expected_energy).abs() < 1e-10);
}

#[test]
fn native_acoustic_serial_artifact_transport_round_trip() {
    use runmat_analysis_execution::{
        execute_serial_analysis, import_acoustic_harmonic_result, AnalysisHostResponse,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (workload, mesh, task) = task(order, 1024);
        let limits = ObjectInventoryLimits::default();
        let directory = tempfile::tempdir().unwrap();
        let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            directory.path().join("objects"),
            limits.max_object_bytes,
        )
        .unwrap();
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        let imported = import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
        assert_eq!(imported.result, workload.execute(&mesh).unwrap());
    }
}

#[test]
fn acoustic_task_submission_requires_the_exact_resource_allocation() {
    use runmat_analysis_execution::{
        build_resource_planned_task_submission, AnalysisExecutionContext,
    };
    use runmat_execution::{value::ValuePayload, ExecutionScopeId, PoolId};
    let (_, _, task) = task(ElementOrder::Tet10, 1024);
    let roots = task
        .program
        .request
        .arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) => panic!("canonical inputs must be externalized"),
            ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {
                panic!("canonical analysis inputs must be artifact object roots")
            }
        })
        .collect::<Vec<_>>();
    let mut resources = task.resource_plan.request.clone();
    resources
        .required_capabilities
        .extend(task.host.workload.generic_capabilities().unwrap());
    let context = |resources| AnalysisExecutionContext {
        host: crate::program_host::program_host(
            &task.program.request,
            crate::program_host::local_trust(),
        )
        .expect("compiled acoustic task host contract"),
        scope_id: ExecutionScopeId::derive(&[b"acoustic-study"]),
        pool_id: PoolId::derive(&[b"acoustic-pool"]),
        program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[b"acoustic-program"]),
        resources,
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let admitted = build_resource_planned_task_submission(
        &task.host,
        &task.resource_plan,
        roots.clone(),
        Vec::new(),
        Default::default(),
        context(resources.clone()),
    )
    .unwrap();
    assert_eq!(
        admitted.request.resources.memory_bytes,
        task.resource_plan.request.memory_bytes
    );
    resources.memory_bytes -= 1;
    assert!(build_resource_planned_task_submission(
        &task.host,
        &task.resource_plan,
        roots,
        Vec::new(),
        Default::default(),
        context(resources)
    )
    .is_err());
}

// witness: FEA-ACOUSTIC-EXECUTION-001/admission @ acoustic_harmonic/tet4/native_process;
//   boundary: Tet4 native worker plan substitution rejection; not measured RSS, whole-family
//   lifetime or complete cancellation evidence.
#[test]
fn native_acoustic_rejects_valid_but_substituted_resource_plan() {
    for substitution in 0..2 {
        let (_, _, mut task) = task(ElementOrder::Tet4, 1024);
        let mut input = task.resource_plan.input.clone();
        match substitution {
            0 => input.worker.fixed_memory_bytes.0 += 1,
            1 => {
                let runmat_analysis_fea::planning::resource::FeaResourceDemand::AcousticHarmonic(
                    demand,
                ) = &mut input.demand
                else {
                    panic!("acoustic demand")
                };
                // A larger, still self-consistent reservation is a different plan identity.
                demand.numerical_reservation_bytes.0 += 1;
            }
            _ => unreachable!(),
        }
        let substituted =
            runmat_analysis_fea::planning::resource::build_resource_plan(input).unwrap();
        task.inputs[2] = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            substituted.identity.digest(),
            &substituted.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        task.host.workload.execution_plan = ExecutionPlanIdentity::new(
            NumericalPlanIdentity::from_acoustic_workload(
                AcousticHarmonicWorkload::from_canonical_bytes(&task.inputs[0].payload())
                    .unwrap()
                    .identity()
                    .unwrap(),
                task.inputs[1].manifest.logical_identity,
            )
            .unwrap(),
            substituted.identity.digest(),
        )
        .unwrap();
        let error = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::SolveAcousticHarmonic,
                host: &task.host,
                inputs: &task.inputs,
                output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default())
                    .unwrap(),
            })
            .unwrap_err();
        assert!(
            error.to_string().contains("authoritative inputs"),
            "{error}"
        );
    }
}

#[test]
fn native_acoustic_rejects_wrong_output_and_execution_identity() {
    let (_, _, mut task) = task(ElementOrder::Tet4, 1024);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    task.host
        .workload
        .outputs
        .push(AnalysisNumericalObjectKind::NumericalReceipt);
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: policy
        })
        .is_err());
    task.host.workload.outputs.pop();
    task.host.workload.execution_plan = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_acoustic_workload(
            AcousticHarmonicWorkload::from_canonical_bytes(&task.inputs[0].payload())
                .unwrap()
                .identity()
                .unwrap(),
            Digest::sha256("wrong topology"),
        )
        .unwrap(),
        task.resource_plan.identity.digest(),
    )
    .unwrap();
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: policy
        })
        .is_err());
}

#[test]
fn native_acoustic_plan_carries_admitted_threads_and_kernel_runs_on_that_pool() {
    let threads =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(2_000)
            .unwrap();
    let (workload, mesh, task) =
        task_with_threads(ElementOrder::Tet4, 1024, "pressure", false, threads);
    assert_eq!(task.resource_plan.request.cpu_millicores, 2_000);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    assert_eq!(
        rederive_acoustic_harmonic_resource_plan(&workload, &mesh, &task.inputs, policy).unwrap(),
        task.resource_plan
    );
    let threaded = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: policy,
        })
        .unwrap();
    let objects = threaded
        .into_iter()
        .next()
        .unwrap()
        .into_prepared(1024, ObjectInventoryLimits::default())
        .unwrap();
    let result = AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload()).unwrap();
    result.validate_for(&workload, &mesh).unwrap();
}

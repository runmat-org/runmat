use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract as _, ElementOrder,
};

fn fixture(
    order: ElementOrder,
    samples: usize,
) -> (
    ThermalWorkload,
    SolverMeshArtifact,
    PreparedThermalResourceFacts,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let step = ProcedureStep::thermal_transient("heat");
    let step_id = step.step_id.clone();
    model.sequence = AnalysisSequence::from_steps("thermal_plan", vec![step]);
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler {
            snapshot_count: samples,
        },
        residual_target: 1e-8,
    };
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let profile = workload
        .prepare(&mesh)
        .unwrap()
        .transport_profile(1024)
        .unwrap();
    let bytes =
        (workload.canonical_bytes().unwrap().len() + mesh.canonical_encode().unwrap().len()) as u64;
    let facts = PreparedThermalResourceFacts {
        local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(bytes),
            transport_closure_bytes: ByteCount(bytes + 65536),
            decoded_resident_bytes_admission_budget: ByteCount(bytes * 64),
        },
        result_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(profile.payload_bytes_upper_bound),
            transport_closure_bytes_upper_bound: ByteCount(profile.total_bytes_upper_bound),
            construction_resident_bytes_admission_budget: ByteCount(
                profile.total_bytes_upper_bound * 64,
            ),
        },
        receipt_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(65536),
            transport_closure_bytes_upper_bound: ByteCount(131072),
            construction_resident_bytes_admission_budget: ByteCount(4194304),
        },
        state_profiles_output: {
            let profile = workload
                .state_publication_profiles_encoding_profile(Digest::from_bytes(
                    *mesh.canonical_digest.bytes(),
                ))
                .unwrap();
            PreparedOutputResourceFacts {
                encoded_payload_bytes_upper_bound: ByteCount(profile.encoded_bytes_upper_bound),
                transport_closure_bytes_upper_bound: ByteCount(
                    profile.encoded_bytes_upper_bound + 65536,
                ),
                construction_resident_bytes_admission_budget: ByteCount(
                    profile.materialization_resident_bytes_estimate + 65536,
                ),
            }
        },
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(65536),
        },
    };
    (workload, mesh, facts)
}

#[test]
fn thermal_plan_is_source_bound_host_only_and_accounts_for_outputs() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for samples in [3, 12] {
            let (workload, mesh, facts) = fixture(order, samples);
            let plan = build_thermal_resource_plan(&workload, &mesh, 1024, facts.clone()).unwrap();
            assert_eq!(
                plan.input.phase,
                FeaResourcePhase::ThermalSolve(workload.identity().unwrap())
            );
            assert!(plan.input.structural_operator().is_none());
            assert_eq!(plan.input.partitioning(), &PartitioningPolicy::Monolithic);
            assert!(plan.request.accelerators.is_empty());
            assert_eq!(
                plan.breakdown.workspace_bytes,
                workload
                    .prepare(&mesh)
                    .unwrap()
                    .numerical_reservation_bytes()
                    .unwrap()
            );
            assert_eq!(
                plan.input
                    .demand
                    .output_facts(crate::execution::AnalysisNumericalObjectKind::NumericalReceipt),
                Some(&facts.receipt_output)
            );
            assert_eq!(
                FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
                plan
            );
            assert_eq!(
                plan.input.demand.output_facts(
                    crate::execution::AnalysisNumericalObjectKind::StatePublicationProfiles
                ),
                Some(&facts.state_profiles_output)
            );
            let mut short_metadata = facts.clone();
            short_metadata
                .state_profiles_output
                .encoded_payload_bytes_upper_bound
                .0 -= 1;
            assert!(build_thermal_resource_plan(&workload, &mesh, 1024, short_metadata).is_err());
            let mut larger = facts.clone();
            larger
                .receipt_output
                .construction_resident_bytes_admission_budget
                .0 += 1;
            let extra = build_thermal_resource_plan(&workload, &mesh, 1024, larger).unwrap();
            assert_eq!(extra.request.memory_bytes, plan.request.memory_bytes + 1);
            assert_ne!(extra.identity, plan.identity);
            let mut wrong = plan.input.clone();
            wrong.phase = FeaResourcePhase::StructuralLinearPipeline;
            assert!(build_resource_plan(wrong).is_err());
            let mut short = facts.clone();
            short.result_output.transport_closure_bytes_upper_bound.0 -= 1;
            assert!(build_thermal_resource_plan(&workload, &mesh, 1024, short).is_err());
            let mut wrong = facts.clone();
            wrong.inputs.encoded_payload_bytes.0 += 1;
            assert!(build_thermal_resource_plan(&workload, &mesh, 1024, wrong).is_err());
            let mut bounded = plan.input.clone();
            bounded.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
            assert!(build_resource_plan(bounded).is_err());
            let mut bounded = plan.input.clone();
            bounded.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes);
            assert!(build_resource_plan(bounded).is_ok());
        }
    }
}

#[test]
fn retained_source_fact_is_additive_identity_bound_and_exactly_ceiling_admitted() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (workload, mesh, facts) = fixture(order, 3);
        let baseline = build_thermal_resource_plan(&workload, &mesh, 1024, facts.clone()).unwrap();
        let mut retained = facts.clone();
        retained.field_source = Some(ThermalFieldSourceResourceFacts {
            identity: Digest::sha256("original-retained-source"),
            encoded_payload_bytes: ByteCount(64),
            validation_workspace_bytes: ByteCount(16384),
        });
        retained.inputs.encoded_payload_bytes.0 += 64;
        retained.inputs.transport_closure_bytes.0 += 64;
        retained.inputs.decoded_resident_bytes_admission_budget.0 += 4096;
        let plan = build_thermal_resource_plan(&workload, &mesh, 1024, retained.clone()).unwrap();
        assert_ne!(baseline.identity, plan.identity);
        assert_eq!(
            plan.breakdown.workspace_bytes.0,
            baseline.breakdown.workspace_bytes.0 + 16384
        );
        assert_eq!(
            plan.request.memory_bytes,
            baseline.request.memory_bytes + 16384 + 4096 + 64
        );
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
            plan
        );
        let mut below = plan.input.clone();
        below.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
        assert!(build_resource_plan(below).is_err());
        let mut exact = plan.input.clone();
        exact.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes);
        assert!(build_resource_plan(exact).is_ok());
        let mut changed = retained.clone();
        changed
            .field_source
            .as_mut()
            .unwrap()
            .validation_workspace_bytes
            .0 += 1;
        let grown = build_thermal_resource_plan(&workload, &mesh, 1024, changed).unwrap();
        assert_eq!(grown.request.memory_bytes, plan.request.memory_bytes + 1);
        assert_ne!(grown.identity, plan.identity);
        for invalid_field in 0..4 {
            let mut bad = retained.clone();
            let source = bad.field_source.as_mut().unwrap();
            match invalid_field {
                0 => source.identity = Digest::sha256([]),
                1 => source.encoded_payload_bytes = ByteCount(0),
                2 => source.validation_workspace_bytes = ByteCount(0),
                _ => source.validation_workspace_bytes = ByteCount(u64::MAX),
            }
            assert!(build_thermal_resource_plan(&workload, &mesh, 1024, bad).is_err());
        }
        let mut absent = retained;
        absent.field_source = None;
        assert!(build_thermal_resource_plan(&workload, &mesh, 1024, absent).is_err());
    }
}

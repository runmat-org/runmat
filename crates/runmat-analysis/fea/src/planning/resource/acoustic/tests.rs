use super::*;
use crate::{
    execution::AnalysisNumericalObjectKind,
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    physics::acoustic::AcousticHarmonicPolicy,
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
};
use runmat_analysis_core::*;
mod periodic;
mod source;

fn workload() -> AcousticHarmonicWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 =
        (Some(3.0)).map(runmat_types::physical::MassDensity::new);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.sequence.steps = vec![step];
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic".into()),
        frequency: Frequency::new(0.2),
        solver: AcousticHarmonicPolicy {
            relative_tolerance: 1e-10,
            ..Default::default()
        },
        power_balance_relative_tolerance: 1e-8,
    }
}

fn facts(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
) -> PreparedAcousticResourceFacts {
    let input =
        (workload.canonical_bytes().unwrap().len() + mesh.canonical_encode().unwrap().len()) as u64;
    let output = acoustic_harmonic_result_payload_bytes_upper_bound(workload, mesh).unwrap();
    let profiles = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))
        .unwrap()
        .encoded_bytes_upper_bound;
    PreparedAcousticResourceFacts {
        local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(input),
            transport_closure_bytes: ByteCount(input + 2048),
            decoded_resident_bytes_admission_budget: ByteCount(input * 16),
        },
        result_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(output),
            transport_closure_bytes_upper_bound: ByteCount(output + 2048),
            construction_resident_bytes_admission_budget: ByteCount(output * 16),
        },
        state_profiles_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(profiles),
            transport_closure_bytes_upper_bound: ByteCount(profiles + 2048),
            construction_resident_bytes_admission_budget: ByteCount(profiles * 16),
        },
        receipt_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(
                crate::execution::ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES,
            ),
            transport_closure_bytes_upper_bound: ByteCount(
                crate::execution::ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES + 2048,
            ),
            construction_resident_bytes_admission_budget: ByteCount(
                crate::execution::ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES * 16,
            ),
        },
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
    }
}

#[test]
fn canonical_results_fit_admitted_tet4_tet10_payload_and_policy_identity_is_deterministic() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let workload = workload();
        let prepared = facts(&workload, &mesh);
        let result = workload.execute(&mesh).unwrap();
        let receipt =
            crate::execution::AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh)
                .unwrap();
        assert!(
            receipt.canonical_bytes().unwrap().len() as u64
                <= prepared.receipt_output.encoded_payload_bytes_upper_bound.0
        );
        assert!(
            result.canonical_bytes().unwrap().len() as u64
                <= prepared.result_output.encoded_payload_bytes_upper_bound.0
        );
        let profiles = result
            .state_publication_profiles(Digest::sha256(result.canonical_bytes().unwrap()))
            .unwrap();
        assert!(
            profiles.canonical_bytes().unwrap().len() as u64
                <= prepared
                    .state_profiles_output
                    .encoded_payload_bytes_upper_bound
                    .0
        );
        let plan = build_acoustic_resource_plan(&workload, &mesh, prepared.clone()).unwrap();
        assert_eq!(
            plan.input
                .demand
                .output_facts(AnalysisNumericalObjectKind::NumericalReceipt),
            Some(&prepared.receipt_output)
        );
        assert_eq!(
            plan,
            build_acoustic_resource_plan(&workload, &mesh, prepared).unwrap()
        );
        assert_eq!(
            plan.input.phase,
            FeaResourcePhase::AcousticHarmonic(workload.identity().unwrap())
        );
        assert_eq!(plan.input.partitioning(), &PartitioningPolicy::Monolithic);
        assert!(plan.breakdown.workspace_bytes.0 > 0);
        let mut changed = workload.clone();
        changed.solver.contraction_limit = 0.25;
        let changed_plan =
            build_acoustic_resource_plan(&changed, &mesh, facts(&changed, &mesh)).unwrap();
        assert_ne!(plan.identity, changed_plan.identity);
    }
}

#[test]
fn acoustic_prepared_byte_facts_fail_closed() {
    let workload = workload();
    let mesh = slab_mesh(1);
    for field in 0..11 {
        let mut prepared = facts(&workload, &mesh);
        match field {
            0 => prepared.inputs.encoded_payload_bytes.0 -= 1,
            1 => prepared.inputs.transport_closure_bytes = ByteCount(0),
            2 => prepared.inputs.decoded_resident_bytes_admission_budget = ByteCount(0),
            3 => prepared.result_output.encoded_payload_bytes_upper_bound.0 -= 1,
            4 => {
                prepared
                    .result_output
                    .construction_resident_bytes_admission_budget = ByteCount(0)
            }
            5 => {
                prepared
                    .state_profiles_output
                    .encoded_payload_bytes_upper_bound
                    .0 -= 1
            }
            6 => {
                prepared
                    .state_profiles_output
                    .transport_closure_bytes_upper_bound
                    .0 = u64::MAX
            }
            7 => {
                prepared
                    .state_profiles_output
                    .construction_resident_bytes_admission_budget = ByteCount(0)
            }
            8 => prepared.receipt_output.encoded_payload_bytes_upper_bound.0 -= 1,
            9 => {
                prepared
                    .receipt_output
                    .construction_resident_bytes_admission_budget = ByteCount(0)
            }
            _ => {
                prepared
                    .receipt_output
                    .transport_closure_bytes_upper_bound
                    .0 = u64::MAX
            }
        }
        assert!(build_acoustic_resource_plan(&workload, &mesh, prepared).is_err());
    }
}

#[test]
fn factorization_reservation_is_included_and_a_short_budget_rejects() {
    let mesh = slab_mesh(1);
    let workload = workload();
    let planned = build_acoustic_resource_plan(&workload, &mesh, facts(&workload, &mesh)).unwrap();
    let FeaResourceDemand::AcousticHarmonic(demand) = &planned.input.demand else {
        panic!("acoustic demand")
    };
    assert!(
        demand.sparse_solve_reservation_bytes.0
            > crate::solve::certificate::nonsingularity_reservation_bytes(
                demand.block_dimension_upper_bound.0 as usize,
                0
            )
    );
    let mut bad = planned.input.clone();
    let FeaResourceDemand::AcousticHarmonic(demand) = &mut bad.demand else {
        panic!("acoustic demand")
    };
    demand.numerical_reservation_bytes.0 = demand.sparse_solve_reservation_bytes.0 - 1;
    assert!(build_resource_plan(bad).is_err());
}

#[test]
fn singleton_and_sweep_entry_publications_have_exact_admitted_transport_envelopes() {
    let mesh = slab_mesh(1);
    let mut declared = workload();
    declared.model.sequence.steps[0].state_outputs[0].state_id.0 = "declared_pressure".repeat(16);
    let mut sweep_entry = declared.clone();
    sweep_entry.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.3]);
    for workload in [declared, sweep_entry] {
        let prepared = facts(&workload, &mesh);
        let result = workload.execute(&mesh).unwrap();
        let bytes = result.canonical_bytes().unwrap();
        let profiles = result
            .state_publication_profiles(Digest::sha256(&bytes))
            .unwrap();
        assert_eq!(profiles.entries().len(), 1);
        assert!(bytes.len() as u64 <= prepared.result_output.encoded_payload_bytes_upper_bound.0);
        assert!(
            profiles.canonical_bytes().unwrap().len() as u64
                <= prepared
                    .state_profiles_output
                    .encoded_payload_bytes_upper_bound
                    .0
        );
        let plan = build_acoustic_resource_plan(&workload, &mesh, prepared.clone()).unwrap();
        assert_eq!(
            plan.input
                .demand
                .output_facts(AnalysisNumericalObjectKind::StatePublicationProfiles),
            Some(&prepared.state_profiles_output)
        );
        let FeaResourceDemand::AcousticHarmonic(demand) = &plan.input.demand else {
            panic!("acoustic demand was not retained");
        };
        assert!(demand.state_profiles_resident_bytes.0 > 0);
        assert!(plan.breakdown.workspace_bytes.0 >= demand.state_profiles_resident_bytes.0);
    }
}

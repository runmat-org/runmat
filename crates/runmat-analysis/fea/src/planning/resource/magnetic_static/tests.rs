use super::*;
use crate::execution::{MagneticNumericalReceiptArtifact, MAGNETIC_RECEIPT_MAX_ENCODED_BYTES};
use crate::physics::magnetic::test_support::authored_model;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::{
    ProcedureStepId, StateCompleteness, StateId, StateKind, StateOutput, StatePersistence,
};
use runmat_meshing_core::ElementOrder;
fn workload(current: f64) -> MagnetostaticWorkload {
    let mut model = authored_model(current);
    if let runmat_analysis_core::AnalysisConstraint::Magnetic(c) = &mut model.constraints[0] {
        c.constraint_id = "ground".repeat(256);
    }
    model.sequence.steps[0].state_outputs = ["first", "second"]
        .into_iter()
        .map(|id| StateOutput {
            state_id: StateId(id.repeat(256)),
            kind: StateKind::ElectromagneticStatic,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::Canonical,
        })
        .collect();
    MagnetostaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
}
fn output(bytes: u64) -> PreparedOutputResourceFacts {
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(bytes),
        transport_closure_bytes_upper_bound: ByteCount(bytes + 2048),
        construction_resident_bytes_admission_budget: ByteCount((bytes + 2048) * 16),
    }
}
fn facts(w: &MagnetostaticWorkload, m: &SolverMeshArtifact) -> PreparedMagnetostaticResourceFacts {
    let input = (w.canonical_bytes().unwrap().len() + m.canonical_encode().unwrap().len()) as u64;
    PreparedMagnetostaticResourceFacts {
        local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(input),
            transport_closure_bytes: ByteCount(input + 2048),
            decoded_resident_bytes_admission_budget: ByteCount(input * 16),
        },
        result_output: output(magnetostatic_result_payload_bytes_upper_bound(w, m).unwrap()),
        receipt_output: output(MAGNETIC_RECEIPT_MAX_ENCODED_BYTES),
        state_profiles_output: output(
            w.state_publication_profiles_encoding_profile(Digest::from_bytes(
                *m.canonical_digest.bytes(),
            ))
            .unwrap()
            .encoded_bytes_upper_bound,
        ),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
    }
}
#[test]
fn magnetic_admitted_chain_exact_fields_receipt_and_all_outputs_fit_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for current in [-2.0, 0.0, 2.0] {
            let w = workload(current);
            let m = match order {
                ElementOrder::Tet4 => slab_mesh(2),
                _ => quadratic_slab_mesh(2),
            };
            let f = facts(&w, &m);
            let plan = build_magnetostatic_resource_plan(&w, &m, f.clone()).unwrap();
            assert_eq!(
                FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
                plan
            );
            let result = w.execute_with_resource_plan(&m, &plan).unwrap();
            assert_eq!(result, w.execute(&m).unwrap());
            assert!(
                result.canonical_bytes().unwrap().len() as u64
                    <= f.result_output.encoded_payload_bytes_upper_bound.0
            );
            let receipt = MagneticNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
            assert!(
                receipt.canonical_bytes().unwrap().len() as u64
                    <= MAGNETIC_RECEIPT_MAX_ENCODED_BYTES
            );
            let cold = MagneticNumericalReceiptArtifact::from_canonical_bytes(
                &receipt.canonical_bytes().unwrap(),
            )
            .unwrap();
            cold.validate_against(&w, &result, &m).unwrap();
            let profiles = result
                .state_publication_profiles(Digest::sha256(result.canonical_bytes().unwrap()))
                .unwrap();
            assert_eq!(profiles.entries().len(), 2);
            assert!(
                profiles.canonical_bytes().unwrap().len() as u64
                    <= f.state_profiles_output.encoded_payload_bytes_upper_bound.0
            );
            let FeaResourceDemand::Magnetostatic(d) = &plan.input.demand else {
                panic!("magnetic demand")
            };
            assert!(d.krylov.dimension.0 > 0);
            assert!(d.physical_nonzeros.0 > 0);
            assert!(d.saddle_nonzeros.0 >= d.physical_nonzeros.0);
            assert_eq!(
                d.quadrature_sample_count.0,
                (m.topology.volume_elements.len() * 15) as u64
            );
        }
    }
}
#[test]
fn magnetic_resource_authority_and_low_ceiling_fail_closed() {
    let w = workload(2.0);
    let m = slab_mesh(2);
    let f = facts(&w, &m);
    let plan = build_magnetostatic_resource_plan(&w, &m, f.clone()).unwrap();
    let foreign = workload(-2.0);
    assert!(foreign.execute_with_resource_plan(&m, &plan).is_err());
    let result = w.execute_with_resource_plan(&m, &plan).unwrap();
    let receipt = MagneticNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
    assert!(receipt.validate_against(&foreign, &result, &m).is_err());
    let mut understated = f;
    understated
        .result_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(build_magnetostatic_resource_plan(&w, &m, understated).is_err());
    let mut altered = plan.input.clone();
    let FeaResourceDemand::Magnetostatic(d) = &mut altered.demand else {
        panic!("magnetic")
    };
    d.krylov.operator_applications.0 -= 1;
    assert!(build_resource_plan(altered).is_err());
    let mut low = w;
    low.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(magnetostatic_result_payload_bytes_upper_bound(&low, &m).is_err());
    assert!(low.execute_with_resource_plan(&m, &plan).is_err());
}

#[test]
fn magnetic_shared_envelope_is_tolerance_independent_and_transport_can_deduplicate() {
    let m = slab_mesh(2);
    let mut earlier = None;
    for tolerance in [1e-9, 1e-11] {
        let mut w = workload(2.0);
        w.relative_residual_target = tolerance;
        let mut f = facts(&w, &m);
        // Synthetic prepared facts exercise this invariant only: immutable chunk
        // deduplication may make transport smaller than reconstructed payload.
        f.inputs.transport_closure_bytes = ByteCount(f.inputs.encoded_payload_bytes.0 / 2);
        let plan = build_magnetostatic_resource_plan(&w, &m, f).unwrap();
        let FeaResourceDemand::Magnetostatic(d) = &plan.input.demand else {
            panic!("magnetic")
        };
        if let Some(e) = &earlier {
            assert_eq!(e, &d.krylov);
        }
        earlier = Some(d.krylov.clone());
        let result = w.execute_with_resource_plan(&m, &plan).unwrap();
        let receipt = MagneticNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
        assert_eq!(receipt.receipt().solver_relative_tolerance, tolerance);
        let mut wire = serde_json::to_value(&receipt).unwrap();
        wire["receipt"]["magnetic_energy_j"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<MagneticNumericalReceiptArtifact>(wire).is_err());
    }
}

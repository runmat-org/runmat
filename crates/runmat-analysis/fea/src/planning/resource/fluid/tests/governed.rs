use super::*;
use crate::planning::resource::*;
use runmat_meshing_core::CanonicalMeshingContract;

fn solve_facts(
    w: &FluidSteadyWorkload,
    m: &SolverMeshArtifact,
) -> PreparedFluidSteadyResourceFacts {
    let projection = derive_fluid_steady_resource_projection(w, m).unwrap();
    PreparedFluidSteadyResourceFacts {
        local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: projection.canonical_input_bytes,
            transport_closure_bytes: ByteCount(projection.canonical_input_bytes.0 + 2048),
            decoded_resident_bytes_admission_budget: projection.actual_decode_resident_bytes,
        },
        result_output: projected_canonical_output_resource_facts(
            projection.result_payload_bytes_upper_bound.0,
        )
        .unwrap(),
        receipt_output: projected_canonical_output_resource_facts(
            crate::execution::FLUID_RECEIPT_MAX_ENCODED_BYTES,
        )
        .unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
    }
}

#[test]
fn original_cpu_plan_codec_execution_and_request_substitution() {
    let (workload, mesh) = crate::execution::fluid_test_workload();
    let facts = solve_facts(&workload, &mesh);
    let plan = build_fluid_steady_resource_plan(&workload, &mesh, facts.clone()).unwrap();
    assert_eq!(
        FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
        plan
    );
    assert!(plan.input.worker.accelerators.is_empty());
    let result = workload.execute_with_resource_plan(&mesh, &plan).unwrap();
    result.validate_for(&workload, &mesh).unwrap();
    let mut changed = workload.clone();
    changed.policy.maximum_outer_iterations += 1;
    assert!(changed.execute_with_resource_plan(&mesh, &plan).is_err());
    let mut short = facts;
    short.inputs.decoded_resident_bytes_admission_budget.0 -= 1;
    assert!(build_fluid_steady_resource_plan(&workload, &mesh, short).is_err());
    let mut malformed = plan.input.clone();
    let FeaResourceDemand::FluidSteadySolve(demand) = &mut malformed.demand else {
        unreachable!()
    };
    demand.receipt_output.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(build_resource_plan(malformed).is_err());
}

#[test]
fn minimum_numeric_ceiling_cannot_be_bypassed_by_higher_authored_ceiling() {
    let (mut workload, mesh) = crate::execution::fluid_test_workload();
    let projection = derive_fluid_steady_resource_projection(&workload, &mesh).unwrap();
    workload.policy.maximum_resident_bytes =
        Some(projection.phase_resident_bytes_upper_bound.0 - 1);
    assert!(derive_fluid_steady_resource_projection(&workload, &mesh).is_err());
}

#[test]
fn native_recovery_binds_all_four_original_wires_and_actual_cell_face_storage() {
    let (workload, mesh) = crate::execution::fluid_test_workload();
    let result = workload.execute(&mesh).unwrap();
    let request = result.native_field_request(&workload, &mesh).unwrap();
    let bytes = [
        workload.canonical_bytes().unwrap(),
        mesh.canonical_encode().unwrap(),
        result.canonical_bytes().unwrap(),
        request.canonical_bytes().unwrap(),
    ];
    let encoded = bytes.iter().map(|b| b.len() as u64).sum();
    let decoded = bytes
        .iter()
        .map(|b| {
            runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(b)
                .unwrap()
        })
        .sum();
    let output = fluid_field_recovery_shape(&request)
        .unwrap()
        .0
        .into_iter()
        .sum();
    let facts = PreparedPhaseResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(encoded),
            transport_closure_bytes: ByteCount(encoded + 2048),
            decoded_resident_bytes_admission_budget: ByteCount(decoded),
        },
        output: projected_canonical_output_resource_facts(output).unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
    };
    let plan = build_fluid_field_recovery_resource_plan(
        &workload,
        &mesh,
        &result,
        &request,
        facts.clone(),
    )
    .unwrap();
    assert_eq!(
        FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
        plan
    );
    let mut short = facts.clone();
    short.inputs.encoded_payload_bytes.0 -= request.canonical_bytes().unwrap().len() as u64;
    assert!(
        build_fluid_field_recovery_resource_plan(&workload, &mesh, &result, &request, short)
            .is_err()
    );
    let mut short = facts;
    short.output.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(
        build_fluid_field_recovery_resource_plan(&workload, &mesh, &result, &request, short)
            .is_err()
    );
}

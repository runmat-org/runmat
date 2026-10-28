use super::*;
use crate::solve::nonlinear::{solid::continuum_qualification::fixture, NonlinearSolveOptions};

#[test]
fn block_admission_binds_inputs_output_and_cumulative_history() {
    for quadratic in [false, true] {
        let (model, mesh, _) = fixture::prepare(1, quadratic);
        let workload = NonlinearSolidWorkload {
            model,
            controls: NonlinearSolveOptions {
                increment_count: 2,
                max_accepted_increments: 8,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                ..NonlinearSolveOptions::default()
            },
        };
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 1,
            predecessor: None,
        };
        let bytes =
            nonlinear_solid_block_payload_bytes_upper_bound(&workload, &mesh, &request).unwrap();
        let facts = PreparedNonlinearSolidBlockResourceFacts {
            local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
            inputs: PreparedInputResourceFacts {
                encoded_payload_bytes: ByteCount(100_000),
                transport_closure_bytes: ByteCount(110_000),
                decoded_resident_bytes_admission_budget: ByteCount(20_000_000),
            },
            output: projected_canonical_output_resource_facts(bytes).unwrap(),
            receipt_output: projected_canonical_output_resource_facts(
                crate::execution::NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES,
            )
            .unwrap(),
            output_set: PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: ByteCount(65_536),
            },
        };
        let plan = build_nonlinear_solid_block_resource_plan(
            &workload,
            &mesh,
            &request,
            None,
            facts.clone(),
        )
        .unwrap();
        assert_eq!(
            plan.breakdown.input_decoded_resident_bytes,
            facts.inputs.decoded_resident_bytes_admission_budget
        );
        let threads =
            crate::planning::resource::LocalExecutionPlan::from_cpu_millicores(3_000).unwrap();
        let threaded = build_nonlinear_solid_block_resource_plan(
            &workload,
            &mesh,
            &request,
            None,
            PreparedNonlinearSolidBlockResourceFacts {
                local_execution: threads,
                ..facts.clone()
            },
        )
        .unwrap();
        assert_eq!(threaded.request.cpu_millicores, 3_000);
        assert_eq!(
            crate::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                threaded.request.cpu_millicores
            )
            .unwrap(),
            threads
        );
        assert!(plan.input.demand.governs_prepared_outputs());
        let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &plan.input.demand else {
            panic!("missing original nonlinear demand");
        };
        assert_eq!(demand.maximum_accepted_increments.0, 8);
        assert_eq!(demand.maximum_retained_output_increments.0, 1);
        let mut cumulative = workload.clone();
        cumulative.controls.max_accepted_increments = 4096;
        let cumulative_request =
            NonlinearSolidBlockRequest::initial(&cumulative, request.topology, 1).unwrap();
        let cumulative_bytes = nonlinear_solid_block_payload_bytes_upper_bound(
            &cumulative,
            &mesh,
            &cumulative_request,
        )
        .unwrap();
        // The cumulative controller cap changes identity/work, not this task's live fields.
        assert_eq!(cumulative_bytes, bytes);
        assert_ne!(cumulative_request.workload, request.workload);
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
            plan
        );
        let block = workload
            .execute_block(mesh.clone(), &request, None)
            .unwrap();
        let actual = runmat_canonical_codec::profile_serialized(
            &block,
            runmat_canonical_codec::CanonicalLimits::new(
                1024 * 1024 * 1024,
                100_000_000,
                65_536,
                64,
            ),
        )
        .unwrap();
        assert!(bytes >= actual.encoded_bytes_upper_bound);
        // Blocks are encoded directly into reserved output; no value tree is built.
        let encoding = runmat_canonical_codec::profile_canonical_encoding(
            &block,
            runmat_canonical_codec::CanonicalLimits::new(
                1024 * 1024 * 1024,
                100_000_000,
                65_536,
                64,
            ),
        )
        .unwrap();
        assert!(
            facts.output.construction_resident_bytes_admission_budget.0
                >= encoding.resident_workspace_bytes_estimate
        );
        for altered in 0..3 {
            let mut insufficient = facts.clone();
            match altered {
                0 => insufficient.inputs.decoded_resident_bytes_admission_budget = ByteCount(1),
                1 => {
                    insufficient
                        .output
                        .construction_resident_bytes_admission_budget = ByteCount(1)
                }
                _ => insufficient.output.encoded_payload_bytes_upper_bound.0 -= 1,
            }
            assert!(build_nonlinear_solid_block_resource_plan(
                &workload,
                &mesh,
                &request,
                None,
                insufficient
            )
            .is_err());
        }
        let mut ceiling = workload.clone();
        ceiling.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
        let scoped = NonlinearSolidBlockRequest {
            workload: ceiling.identity().unwrap(),
            ..request
        };
        assert!(
            build_nonlinear_solid_block_resource_plan(&ceiling, &mesh, &scoped, None, facts)
                .is_err()
        );
    }
}

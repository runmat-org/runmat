use super::*;
use runmat_analysis_execution::{AnalysisHostWorkload, AnalysisWorkloadRequest};
use runmat_analysis_fea::planning::resource::build_resource_plan;

#[test]
fn worker_rejects_rehashed_understatements_before_numerical_acceptance() {
    let (workload, mesh) = fixture(false);
    let request = NonlinearSolidBlockRequest {
        workload: workload.identity().unwrap(),
        topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        block_ordinal: 0,
        first_output_ordinal: 1,
        accepted_block_limit: 9,
        predecessor: None,
    };
    let task = prepare(&workload, &mesh, request, None);
    for substitution in 0..5 {
        let mut input = task.resource_plan.input.clone();
        let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &mut input.demand else {
            panic!("expected nonlinear demand");
        };
        match substitution {
            0 => {
                demand
                    .block_transport
                    .as_mut()
                    .unwrap()
                    .inputs
                    .decoded_resident_bytes_admission_budget
                    .0 -= 1
            }
            1 => demand.point_count.0 -= 1,
            2 => {
                demand
                    .block_transport
                    .as_mut()
                    .unwrap()
                    .output
                    .encoded_payload_bytes_upper_bound
                    .0 -= 1
            }
            3 => {
                demand
                    .block_transport
                    .as_mut()
                    .unwrap()
                    .receipt_output
                    .construction_resident_bytes_admission_budget
                    .0 -= 1
            }
            _ => demand.linear_solve_limit.0 -= 1,
        }
        // Rehash the entire valid plan and host closure: rejection must derive the
        // actual numerical/transport authority, not merely notice a stale hash.
        let altered = build_resource_plan(input).unwrap();
        let mut inputs = task.inputs.clone();
        let carried = inputs
            .iter_mut()
            .find(|input| input.manifest.kind == Kind::ResourcePlan)
            .unwrap();
        *carried = prepare_analysis_objects(
            Kind::ResourcePlan,
            altered.identity.digest(),
            &altered.canonical_bytes().unwrap(),
            task.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            task.output_policy.inventory_limits(),
        )
        .unwrap();
        let source = &task.host.workload;
        let host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                source.study,
                source.step,
                ExecutionPlanIdentity::new(
                    NumericalPlanIdentity::from_nonlinear_solid_block(&task.request).unwrap(),
                    altered.identity.digest(),
                )
                .unwrap(),
                source.kind,
                inputs
                    .iter()
                    .map(|input| AnalysisWorkloadInput {
                        kind: input.manifest.kind,
                        logical_identity: input.manifest.logical_identity,
                        payload_digest: input.manifest.payload_digest,
                        manifest_digest: input.root.digest,
                        manifest_encoded_length: input.root.encoded_length,
                        reference_kind: ValueRefKind::DriverObject,
                    })
                    .collect(),
                source.outputs.clone(),
                source.required_capabilities.clone(),
            )
            .unwrap(),
            task.host.artifact_access.clone(),
            task.host.output_policy,
        )
        .unwrap();
        let error = super::super::super::NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                host: &host,
                kind: source.kind,
                inputs: &inputs,
                output_policy: task.output_policy,
            })
            .unwrap_err();
        assert!(error.to_string().contains("nonlinear"), "{error}");
    }
}

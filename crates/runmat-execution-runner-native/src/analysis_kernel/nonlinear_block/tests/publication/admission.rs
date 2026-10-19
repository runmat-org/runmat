//! Self-consistent host/resource rehashes must still converge from actual source bytes.
use super::*;
use runmat_analysis_execution::{prepare_analysis_objects, AnalysisWorkloadRequest};
use runmat_analysis_fea::planning::resource::{
    build_resource_plan, FeaResourceDemand, FeaResourcePlan,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::ObjectNamespace;

pub(super) fn reject_understated_decode(
    plan: &FeaResourcePlan,
    inputs: &[runmat_analysis_execution::PreparedAnalysisObjects],
    host: &AnalysisHostWorkload,
    policy: AnalysisOutputPolicy,
) {
    let mut input = plan.input.clone();
    let facts = match &mut input.demand {
        FeaResourceDemand::AccuracyEvaluation(d) => &mut d.inputs,
        FeaResourceDemand::SemanticStatePublication(d) => &mut d.inputs,
        _ => panic!("expected shared post-process admission"),
    };
    facts.decoded_resident_bytes_admission_budget.0 -= 1;
    let altered = build_resource_plan(input).unwrap();
    let mut inputs = inputs.to_vec();
    let position = inputs
        .iter()
        .position(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    inputs[position] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        altered.identity.digest(),
        &altered.canonical_bytes().unwrap(),
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )
    .unwrap();
    let source = &host.workload;
    let mut closure = source.inputs.clone();
    let index = closure
        .iter()
        .position(|i| i.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    closure[index] = runmat_analysis_execution::AnalysisWorkloadInput {
        kind: inputs[position].manifest.kind,
        logical_identity: inputs[position].manifest.logical_identity,
        payload_digest: inputs[position].manifest.payload_digest,
        manifest_digest: inputs[position].root.digest,
        manifest_encoded_length: inputs[position].root.encoded_length,
        reference_kind: ValueRefKind::DriverObject,
    };
    let host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            source.study,
            source.step,
            source.execution_plan,
            source.kind,
            closure,
            source.outputs.clone(),
            source.required_capabilities.clone(),
        )
        .unwrap(),
        host.artifact_access.clone(),
        host.output_policy,
    )
    .unwrap();
    assert!(super::super::super::super::NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            host: &host,
            kind: source.kind,
            inputs: &inputs,
            output_policy: policy
        })
        .is_err());
}

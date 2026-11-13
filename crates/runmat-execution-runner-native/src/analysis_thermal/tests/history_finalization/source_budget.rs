//! Preserve the real source closure while understating its carried construction reservation.
use super::*;
use runmat_analysis_fea::planning::resource::{build_resource_plan, ByteCount, FeaResourceDemand};

pub(super) fn reject_understated_constructor(
    accuracy: &PreparedAccuracy,
    output_policy: AnalysisOutputPolicy,
) {
    let mut input = accuracy.resource_plan.input.clone();
    let FeaResourceDemand::AccuracyEvaluation(demand) = &mut input.demand else {
        panic!("accuracy family")
    };
    demand.evaluator_workspace_bytes_upper_bound =
        ByteCount(demand.evaluator_workspace_bytes_upper_bound.0 - 1);
    let understated = build_resource_plan(input).unwrap();
    understated.validate_identity().unwrap();
    let mut inputs = accuracy.inputs.clone();
    let carried = inputs
        .iter_mut()
        .find(|input| input.manifest.kind == Kind::ResourcePlan)
        .unwrap();
    *carried = prepare_analysis_objects(
        Kind::ResourcePlan,
        understated.identity.digest(),
        &understated.canonical_bytes().unwrap(),
        output_policy.chunk_bytes(),
        runmat_execution_artifact::ObjectNamespace::InputValue,
        output_policy.inventory_limits(),
    )
    .unwrap();
    require_accuracy_source_decode_budget(&inputs, &understated, output_policy.inventory_limits())
        .unwrap();
    let error = crate::NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy,
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy.host,
            inputs: &inputs,
        })
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("thermal source construction workspace")
            || error.contains("thermal schedule resident bytes"),
        "{error}"
    );
}

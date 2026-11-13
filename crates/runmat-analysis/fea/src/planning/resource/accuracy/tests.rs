use super::*;
use crate::execution::{AccuracySourceKind, SequenceIdentity, SolverResultIdentity, StepIdentity};
use crate::planning::resource::{
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
};
use runmat_analysis_core::AccuracyPlan;
use runmat_execution::Digest;

fn request(kind: AccuracySourceKind, maximum: u64) -> AccuracyEvaluationRequest {
    let sequence = SequenceIdentity::new(Digest::sha256("accuracy admission sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("accuracy admission step")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap();
    let mut plan = AccuracyPlan::baseline("admission");
    plan.resources.maximum_resident_bytes = Some(maximum);
    plan.resources.maximum_wall_time_milliseconds = 60_000;
    AccuracyEvaluationRequest::new(producer, kind, result, None, plan).unwrap()
}

fn build(request: &AccuracyEvaluationRequest) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let profile = accuracy_evaluation_resource_profile(request, &[]).unwrap();
    let output = profile.output_encoded_bytes_upper_bound;
    build_accuracy_resource_plan(
        request,
        &[],
        PreparedPhaseResourceFacts {
            inputs: PreparedInputResourceFacts {
                encoded_payload_bytes: ByteCount(1024),
                transport_closure_bytes: ByteCount(2048),
                decoded_resident_bytes_admission_budget: ByteCount(4096),
            },
            output: PreparedOutputResourceFacts {
                encoded_payload_bytes_upper_bound: ByteCount(output),
                transport_closure_bytes_upper_bound: ByteCount(output + 65536),
                construction_resident_bytes_admission_budget: ByteCount(output * 2),
            },
            output_set: PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: ByteCount(65536),
            },
        },
    )
}

#[test]
fn every_accuracy_source_honors_its_authored_resident_ceiling() {
    for kind in [
        AccuracySourceKind::NonlinearSolid,
        AccuracySourceKind::StructuralLinear,
        AccuracySourceKind::StructuralModalResponseSpectrum,
        AccuracySourceKind::StructuralRandomResponse,
        AccuracySourceKind::Thermal,
        AccuracySourceKind::AcousticHarmonic,
    ] {
        let baseline = build(&request(kind, 8 * 1024 * 1024 * 1024)).unwrap();
        let required = baseline.request.memory_bytes;
        let exact_request = request(kind, required);
        let exact = build(&exact_request).unwrap();
        assert_eq!(exact.input.accuracy_ceiling, exact_request.plan.resources);
        assert_eq!(exact.input.worker.maximum_wall_milliseconds, 60_000);
        assert_eq!(exact.request.memory_bytes, required);
        assert_ne!(exact.identity, baseline.identity);
        for maximum in [1, required - 1] {
            assert_eq!(
                build(&request(kind, maximum)).unwrap_err(),
                ResourcePlanningError::AccuracyCeiling {
                    resource: "resident memory bytes",
                    required,
                    maximum,
                }
            );
        }
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&exact.canonical_bytes().unwrap()).unwrap(),
            exact
        );
    }
}

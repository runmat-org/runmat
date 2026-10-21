use super::*;
use runmat_analysis_core::AccuracyPlan;
use runmat_analysis_fea::{
    execution::{
        AccuracySourceKind, AnalysisNumericalObjectKind, SequenceIdentity, SolverResultIdentity,
        StepIdentity,
    },
    planning::resource::{build_accuracy_resource_plan, ByteCount, PreparedPhaseResourceFacts},
};
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

#[test]
fn shared_accuracy_authentication_and_actual_decode_profiles_precede_typed_decode() {
    let sequence = SequenceIdentity::new(Digest::sha256("accuracy auth source")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("original")).unwrap();
    let request = AccuracyEvaluationRequest::new(
        producer,
        AccuracySourceKind::FluidSteady,
        result,
        None,
        AccuracyPlan::baseline("auth"),
    )
    .unwrap();
    let input = crate::prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
        request.identity().digest(),
        &request.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let inputs = vec![input];
    let facts = prepared_input_resource_facts_without_resource_plan(&inputs).unwrap();
    let profile = accuracy_evaluation_resource_profile(&request, &[]).unwrap();
    let make = |input_facts| {
        build_accuracy_resource_plan(
            &request,
            &[],
            PreparedPhaseResourceFacts {
                inputs: input_facts,
                output: projected_output_resource_facts(profile.output_encoded_bytes_upper_bound)
                    .unwrap(),
                output_set: projected_output_set_resource_facts(),
            },
        )
        .unwrap()
    };
    let original = make(facts.clone());
    require_accuracy_source_decode_budget(&inputs, &original, ObjectInventoryLimits::default())
        .unwrap();
    let mut understated = facts.clone();
    understated.decoded_resident_bytes_admission_budget = ByteCount(facts.encoded_payload_bytes.0);
    assert_ne!(understated, facts);
    let rehashed = make(understated);
    assert!(require_accuracy_source_decode_budget(
        &inputs,
        &rehashed,
        ObjectInventoryLimits::default()
    )
    .is_err());
    let mut corrupted = inputs;
    corrupted[0].objects[1].bytes[0] ^= 1;
    assert!(require_accuracy_source_decode_budget(
        &corrupted,
        &original,
        ObjectInventoryLimits::default()
    )
    .is_err());
}

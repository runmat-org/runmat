//! Adversarial checks use the real solve/observation fixture and freshly rehashed requests.
use crate::{validate_accuracy_source_context, PreparedAccuracy};
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AccuracySourceReceipt, AnalysisNumericalObjectKind as Kind,
    },
    results::observation::ObservationPublicationCatalog,
};

pub(super) fn reject_substituted_structural_authority(
    prepared: &PreparedAccuracy,
    catalog: &ObservationPublicationCatalog,
    source_host: &crate::AnalysisHostWorkload,
) {
    let receipt = AccuracySourceReceipt::from_canonical_bytes(
        prepared.request.source_kind,
        &prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == Kind::NumericalReceipt)
            .unwrap()
            .payload(),
    )
    .unwrap();
    let context: Vec<_> = prepared
        .inputs
        .iter()
        .filter(|input| {
            matches!(
                input.manifest.kind,
                Kind::AccuracySourceContext | Kind::SolverInvocation | Kind::Mesh
            )
        })
        .collect();
    validate_accuracy_source_context(&prepared.request, &receipt, Some(catalog), &context).unwrap();

    for omitted in 0..context.len() {
        let incomplete: Vec<_> = context
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != omitted)
            .map(|(_, input)| *input)
            .collect();
        assert!(validate_accuracy_source_context(
            &prepared.request,
            &receipt,
            Some(catalog),
            &incomplete
        )
        .is_err());
    }
    let mut duplicate = context.clone();
    duplicate.push(context[0]);
    assert!(validate_accuracy_source_context(
        &prepared.request,
        &receipt,
        Some(catalog),
        &duplicate
    )
    .is_err());

    // Every request below has a valid NEW identity. Rejection must come from the producing
    // authored requirements, not a stale request digest or a missing result reference.
    let mut weakened = prepared.request.plan.clone();
    weakened.criteria[0].tolerance = runmat_analysis_core::AccuracyTolerance::Relative(0.99);
    let request = AccuracyEvaluationRequest::new(
        prepared.request.producer,
        prepared.request.source_kind,
        prepared.request.source_result,
        Some(catalog),
        prepared.request.formulation_requirement,
        weakened,
    )
    .unwrap();
    request.validate().unwrap();
    assert!(
        validate_accuracy_source_context(&request, &receipt, Some(catalog), &context)
            .unwrap_err()
            .to_string()
            .contains("authored requirements")
    );

    let omitted = AccuracyEvaluationRequest::new(
        prepared.request.producer,
        prepared.request.source_kind,
        prepared.request.source_result,
        None,
        prepared.request.formulation_requirement,
        prepared.request.plan.clone(),
    )
    .unwrap();
    omitted.validate().unwrap();
    assert!(
        validate_accuracy_source_context(&omitted, &receipt, None, &context)
            .unwrap_err()
            .to_string()
            .contains("authored requirements")
    );

    // The source descriptor must itself carry the receipt-bound workload identity.
    let descriptor = context
        .iter()
        .find(|input| input.manifest.kind == Kind::AccuracySourceContext)
        .unwrap();
    let mut wrong = (**descriptor).clone();
    wrong.manifest.logical_identity = runmat_execution::Digest::sha256("foreign-source");
    let substituted: Vec<_> = context
        .iter()
        .map(|input| {
            if input.manifest.kind == Kind::AccuracySourceContext {
                &wrong
            } else {
                *input
            }
        })
        .collect();
    assert!(validate_accuracy_source_context(
        &prepared.request,
        &receipt,
        Some(catalog),
        &substituted
    )
    .unwrap_err()
    .to_string()
    .contains("numerical receipt"));

    // Transport layout is not scientific identity. Rechunk the same invocation and update
    // its descriptor roots; receipt binding and authored validation must still succeed.
    let original = context
        .iter()
        .find(|input| input.manifest.kind == Kind::SolverInvocation)
        .unwrap();
    let reencoded = crate::prepare_analysis_objects(
        Kind::SolverInvocation,
        original.manifest.logical_identity,
        &original.payload(),
        127,
        runmat_execution_artifact::ObjectNamespace::InputValue,
        runmat_execution_artifact::object::ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut source = source_host.workload.clone();
    let input = source
        .inputs
        .iter_mut()
        .find(|input| input.kind == Kind::SolverInvocation)
        .unwrap();
    input.manifest_digest = reencoded.root.digest;
    input.manifest_encoded_length = reencoded.root.encoded_length;
    source.validate().unwrap();
    let descriptor = crate::accuracy_source_context::prepare_accuracy_source_contract(
        &source,
        4096,
        runmat_execution_artifact::object::ObjectInventoryLimits::default(),
    )
    .unwrap();
    let rechunked: Vec<_> = context
        .iter()
        .map(|input| match input.manifest.kind {
            Kind::SolverInvocation => &reencoded,
            Kind::AccuracySourceContext => &descriptor,
            _ => *input,
        })
        .collect();
    validate_accuracy_source_context(&prepared.request, &receipt, Some(catalog), &rechunked)
        .unwrap();

    // A fully rehashed source contract with different scientific bytes cannot retain the
    // numerical receipt's workload identity, even if all its own hashes are internally valid.
    source
        .inputs
        .iter_mut()
        .find(|input| input.kind == Kind::SolverInvocation)
        .unwrap()
        .payload_digest = runmat_execution::Digest::sha256("foreign-scientific-content");
    let foreign = crate::AnalysisWorkloadRequest::new(
        source.study,
        source.step,
        source.execution_plan,
        source.kind,
        source.inputs,
        source.outputs,
        source.required_capabilities,
    )
    .unwrap();
    let foreign = crate::accuracy_source_context::prepare_accuracy_source_contract(
        &foreign,
        4096,
        runmat_execution_artifact::object::ObjectInventoryLimits::default(),
    )
    .unwrap();
    let substituted: Vec<_> = context
        .iter()
        .map(|input| {
            if input.manifest.kind == Kind::AccuracySourceContext {
                &foreign
            } else {
                *input
            }
        })
        .collect();
    assert!(validate_accuracy_source_context(
        &prepared.request,
        &receipt,
        Some(catalog),
        &substituted
    )
    .unwrap_err()
    .to_string()
    .contains("numerical receipt"));
}

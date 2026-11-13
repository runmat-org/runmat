//! Real source fixtures distinguish resource-plan rejection from source reconstruction errors.
use super::*;
use runmat_analysis_fea::planning::resource::{
    build_resource_plan, ByteCount, FeaResourceDemand, FeaResourcePlan,
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

fn carry_plan(inputs: &mut [crate::PreparedAnalysisObjects], plan: &FeaResourcePlan) {
    let carried = inputs
        .iter_mut()
        .find(|input| input.manifest.kind == Kind::ResourcePlan)
        .unwrap();
    *carried = crate::prepare_analysis_objects(
        Kind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
}

pub(super) fn verify_original_source_admission(
    prepared: &PreparedAccuracy,
    catalog: &ObservationPublicationCatalog,
    receipt: &AccuracySourceReceipt,
) {
    crate::validate_admitted_accuracy_source(
        &prepared.request,
        receipt,
        Some(catalog),
        &prepared.inputs,
        &prepared.resource_plan,
    )
    .unwrap();
    let required = catalog.metadata_validation_resident_bytes().unwrap();
    assert!(required > 0);
    catalog
        .validate_source_result_admitted(
            prepared.request.producer,
            prepared.request.source_kind,
            prepared.request.source_result,
            required,
        )
        .unwrap();
    let short = catalog
        .validate_source_result_admitted(
            prepared.request.producer,
            prepared.request.source_kind,
            prepared.request.source_result,
            required - 1,
        )
        .unwrap_err()
        .to_string();
    assert!(
        short.contains("catalog metadata validation workspace"),
        "{short}"
    );
    for kind in [
        Kind::AccuracyEvaluationRequest,
        Kind::NumericalReceipt,
        Kind::ObservationCatalog,
    ] {
        for duplicate in [false, true] {
            let mut inputs = prepared.inputs.clone();
            let index = inputs
                .iter()
                .position(|input| input.manifest.kind == kind)
                .unwrap();
            if duplicate {
                // Distinct chunk roots preserve a valid transport inventory while carrying
                // the same semantic phase input twice; do not rely on deduplication rejection.
                let original = &inputs[index];
                inputs.push(
                    crate::prepare_analysis_objects(
                        kind,
                        original.manifest.logical_identity,
                        &original.payload(),
                        127,
                        original.root.namespace,
                        ObjectInventoryLimits::default(),
                    )
                    .unwrap(),
                );
            } else {
                inputs.remove(index);
            }
            let plan =
                crate::build_prepared_accuracy_resource_plan(&prepared.request, receipt, &inputs)
                    .unwrap();
            carry_plan(&mut inputs, &plan);
            crate::require_accuracy_source_decode_budget(
                &inputs,
                &plan,
                ObjectInventoryLimits::default(),
            )
            .unwrap();
            let error = crate::validate_admitted_accuracy_source(
                &prepared.request,
                receipt,
                Some(catalog),
                &inputs,
                &plan,
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains("accuracy phase input inventory"),
                "{kind:?}/{duplicate}: {error}"
            );
        }
    }
    // A fully authenticated source descriptor retains its payload but lies about its
    // receipt-bound logical identity. Its wire/decode profile is unchanged.
    let mut inputs = prepared.inputs.clone();
    let descriptor = inputs
        .iter_mut()
        .find(|input| input.manifest.kind == Kind::AccuracySourceContext)
        .unwrap();
    let chunk_bytes = descriptor.manifest.chunks.first().unwrap().encoded_length as usize;
    *descriptor = crate::prepare_analysis_objects(
        Kind::AccuracySourceContext,
        runmat_execution::Digest::sha256("foreign-admitted-source"),
        &descriptor.payload(),
        chunk_bytes,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let original =
        crate::build_prepared_accuracy_resource_plan(&prepared.request, receipt, &inputs).unwrap();
    carry_plan(&mut inputs, &original);
    crate::require_accuracy_source_decode_budget(
        &inputs,
        &original,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let source_error = crate::validate_admitted_accuracy_source(
        &prepared.request,
        receipt,
        Some(catalog),
        &inputs,
        &original,
    )
    .unwrap_err()
    .to_string();
    assert!(source_error.contains("numerical receipt"), "{source_error}");

    // Rehash and carry an internally valid smaller demand. The exact plan gate must
    // reject it before the independently invalid source descriptor is reconstructed.
    let mut input = original.input.clone();
    let FeaResourceDemand::AccuracyEvaluation(demand) = &mut input.demand else {
        panic!("accuracy family")
    };
    demand.evaluator_workspace_bytes_upper_bound = ByteCount(1);
    let understated = build_resource_plan(input).unwrap();
    understated.validate_identity().unwrap();
    carry_plan(&mut inputs, &understated);
    crate::require_accuracy_source_decode_budget(
        &inputs,
        &understated,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let error = crate::validate_admitted_accuracy_source(
        &prepared.request,
        receipt,
        Some(catalog),
        &inputs,
        &understated,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("accuracy original-source resource rederivation differs"),
        "{error}"
    );
}

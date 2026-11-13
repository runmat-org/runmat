//! Semantic closure validation for the original structural publication.
use super::*;

impl ExactStructuralStudyAnalysisClosure {
    pub(super) fn objects(&self) -> &[ValueRef] {
        match self {
            Self::LinearSequence { objects, .. }
            | Self::StructuralTransient { objects, .. }
            | Self::StructuralModal { objects, .. }
            | Self::StructuralModalTransient { objects, .. }
            | Self::StructuralModalResponseSpectrum { objects, .. }
            | Self::StructuralRandomResponse { objects, .. }
            | Self::StructuralBuckling { objects, .. }
            | Self::StructuralPrestressedModal { objects, .. }
            | Self::StructuralPrestressedFrequency { objects, .. }
            | Self::StructuralPrestressedModalFrequency { objects, .. }
            | Self::StructuralFrequency { objects, .. } => objects,
        }
    }

    pub(super) fn validate(&self) -> NativeExecutionResult<()> {
        match self {
            Self::LinearSequence {
                closures, objects, ..
            } => {
                let mut nodes = BTreeSet::new();
                for closure in closures {
                    if !nodes.insert(closure.node.clone()) {
                        return Err(protocol(
                            "study publication contains a duplicate sequence node",
                        ));
                    }
                    validate_externalized_response(&closure.response, objects)?;
                }
                let mut closure_objects = Vec::new();
                for closure in closures {
                    closure_objects.extend(response_objects(&closure.response)?);
                }
                let closure_objects = unique_references(closure_objects)?;
                if reference_inventory(&closure_objects)? != reference_inventory(objects)? {
                    return Err(protocol(
                        "analysis inventory contains unreferenced or missing sequence objects",
                    ));
                }
            }
            Self::StructuralTransient {
                result_root,
                receipt_root,
                objects,
            } => {
                if result_root == receipt_root
                    || !objects.contains(result_root)
                    || !objects.contains(receipt_root)
                {
                    return Err(protocol(
                        "structural numerical closure omits or aliases a declared root",
                    ));
                }
                reference_inventory(objects)?;
            }
            Self::StructuralModal {
                result_root,
                receipt_root,
                objects,
            } => {
                validate_declared_roots(&[result_root, receipt_root], objects)?;
            }
            Self::StructuralModalTransient {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                transient_result_root,
                transient_receipt_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        transient_result_root,
                        transient_receipt_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralModalResponseSpectrum {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_publication_baseline_usage_root,
                response_resource_usage_root,
                response_publication_root,
                field_recovery_root,
                envelope_state_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        response_result_root,
                        response_receipt_root,
                        response_accuracy_root,
                        response_publication_baseline_usage_root,
                        response_resource_usage_root,
                        response_publication_root,
                        field_recovery_root,
                        envelope_state_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralRandomResponse {
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                response_result_root,
                response_receipt_root,
                response_accuracy_root,
                response_publication_baseline_usage_root,
                response_resource_usage_root,
                response_publication_root,
                statistics_state_root,
                objects,
            } => {
                validate_declared_roots(
                    &[
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        response_result_root,
                        response_receipt_root,
                        response_accuracy_root,
                        response_publication_baseline_usage_root,
                        response_resource_usage_root,
                        response_publication_root,
                        statistics_state_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralBuckling {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                buckling_result_root,
                buckling_receipt_root,
                objects,
            } => validate_declared_roots(
                &[
                    preload_result_root,
                    preload_receipt_root,
                    prestress_state_root,
                    buckling_result_root,
                    buckling_receipt_root,
                ],
                objects,
            )?,
            Self::StructuralPrestressedModal {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                objects,
            } => validate_declared_roots(
                &[
                    preload_result_root,
                    preload_receipt_root,
                    prestress_state_root,
                    modal_result_root,
                    modal_receipt_root,
                ],
                objects,
            )?,
            Self::StructuralPrestressedFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                frequency,
                objects,
            } => {
                validate_frequency_closure(frequency, objects)?;
                validate_declared_roots(
                    &[
                        preload_result_root,
                        preload_receipt_root,
                        prestress_state_root,
                        &frequency.result_root,
                        &frequency.receipt_root,
                        &frequency.resource_usage_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralPrestressedModalFrequency {
                preload_result_root,
                preload_receipt_root,
                prestress_state_root,
                modal_result_root,
                modal_receipt_root,
                basis_state_root,
                frequency,
                objects,
            } => {
                validate_frequency_closure(frequency, objects)?;
                validate_declared_roots(
                    &[
                        preload_result_root,
                        preload_receipt_root,
                        prestress_state_root,
                        modal_result_root,
                        modal_receipt_root,
                        basis_state_root,
                        &frequency.result_root,
                        &frequency.receipt_root,
                        &frequency.resource_usage_root,
                    ],
                    objects,
                )?;
            }
            Self::StructuralFrequency { frequency, objects } => {
                validate_frequency_closure(frequency, objects)?
            }
        }
        Ok(())
    }
}

fn validate_frequency_closure(
    frequency: &ExactStructuralFrequencyClosure,
    objects: &[ValueRef],
) -> NativeExecutionResult<()> {
    frequency.publication.validate().map_err(protocol)?;
    if frequency
        .objects
        .iter()
        .any(|reference| !objects.contains(reference))
    {
        return Err(protocol(
            "structural frequency object closure escapes its enclosing result",
        ));
    }
    let objects = frequency.objects.as_slice();
    if frequency.result_root == frequency.receipt_root
        || frequency.entries.is_empty()
        || !objects.contains(&frequency.result_root)
        || !objects.contains(&frequency.receipt_root)
        || !objects.contains(&frequency.resource_usage_root)
        || frequency.entries.iter().any(|entry| {
            !objects.contains(&entry.fields_root)
                || entry
                    .observations_root
                    .as_ref()
                    .is_some_and(|root| !objects.contains(root))
        })
        || frequency
            .histories
            .iter()
            .any(|history| !objects.contains(&history.checkpoint_root))
    {
        return Err(protocol(
            "structural frequency closure omits a declared artifact root",
        ));
    }
    let history_ids = frequency
        .histories
        .iter()
        .map(|history| &history.observation_id)
        .collect::<BTreeSet<_>>();
    if history_ids.len() != frequency.histories.len() {
        return Err(protocol(
            "structural frequency closure contains duplicate histories",
        ));
    }
    reference_inventory(objects)?;
    Ok(())
}

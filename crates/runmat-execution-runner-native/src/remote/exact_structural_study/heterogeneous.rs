//! Execution of structural producer/consumer sequences after exact input resolution.
use super::*;

mod consumers;

pub(super) async fn heterogeneous(
    input: Composition<'_, '_>,
    workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let authored = StructuralStudyWorkloads::HeterogeneousSequence { workloads };
    let step_ids = authored.step_ids().map_err(protocol)?;
    if let [preload_step, modal_step, frequency_step] = step_ids.as_slice() {
        let is_modal_frequency = match &authored {
            StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                (
                    workloads.get(preload_step),
                    workloads.get(modal_step),
                    workloads.get(frequency_step),
                ),
                (
                    Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. }),
                    Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. }),
                    Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency { .. }),
                )
            ),
            _ => false,
        };
        if !is_modal_frequency {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "three-step structural execution is not a prestressed modal-frequency sequence"
                    .into(),
            ));
        }
        if matches!(
            &input.recovery,
            StructuralTransientRecoveryAdmission::Available {
                restored: Some(_),
                ..
            }
        ) {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "prestressed modal-frequency execution cannot consume a transient recovery closure"
                    .into(),
            ));
        }
        return Box::pin(consumers::modal_frequency(input, authored)).await;
    }
    let [producer_step, consumer_step] = step_ids.as_slice() else {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "heterogeneous structural execution requires exactly two steps".into(),
        ));
    };
    let producer_is_linear = match &authored {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
            workloads.get(producer_step),
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. })
        ),
        _ => false,
    };
    let consumer_is_prestressed_modal = match &authored {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
            workloads.get(consumer_step),
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. })
        ),
        _ => false,
    };
    let consumer_is_prestressed_frequency = match &authored {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
            workloads.get(consumer_step),
            Some(
                runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency { .. }
            )
        ),
        _ => false,
    };
    let consumer_is_response_spectrum = match &authored {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
            workloads.get(consumer_step),
            Some(
                runmat_analysis_fea::execution::StructuralStepWorkload::ModalResponseSpectrum { .. }
            )
        ),
        _ => false,
    };
    let consumer_is_random_response = match &authored {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
            workloads.get(consumer_step),
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::RandomResponse { .. })
        ),
        _ => false,
    };
    if producer_is_linear {
        if matches!(
            &input.recovery,
            StructuralTransientRecoveryAdmission::Available {
                restored: Some(_),
                ..
            }
        ) {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "prestress execution cannot consume a transient recovery closure".into(),
            ));
        }
        if consumer_is_prestressed_frequency {
            return Box::pin(consumers::frequency(input, authored)).await;
        }
        if consumer_is_prestressed_modal {
            return Box::pin(consumers::modal(input, authored)).await;
        }
        return Box::pin(consumers::buckling(input, authored)).await;
    }
    if consumer_is_response_spectrum {
        if matches!(
            &input.recovery,
            StructuralTransientRecoveryAdmission::Available {
                restored: Some(_),
                ..
            }
        ) {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "response-spectrum execution cannot consume a transient recovery closure".into(),
            ));
        }
        return Box::pin(consumers::spectrum(input, authored)).await;
    }
    if consumer_is_random_response {
        if matches!(
            &input.recovery,
            StructuralTransientRecoveryAdmission::Available {
                restored: Some(_),
                ..
            }
        ) {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "random-response execution cannot consume a transient recovery closure".into(),
            ));
        }
        return Box::pin(consumers::random(input, authored)).await;
    }
    Box::pin(consumers::transient(input, authored)).await
}

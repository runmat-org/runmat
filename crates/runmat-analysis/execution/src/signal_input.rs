use runmat_analysis_core::SignalArtifactRef;
use runmat_analysis_fea::execution::SignalInputAdmission;
use runmat_execution::value::{ElementType, InlineValue, ValueAdmissionReceipt, ValuePayload};
use runmat_execution::Digest;

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSignalArtifact {
    pub value: ValuePayload,
    pub admission: ValueAdmissionReceipt,
}

pub fn resolve_signal_artifact(
    artifact: &SignalArtifactRef,
    resolved: &ResolvedSignalArtifact,
) -> AnalysisBridgeResult<SignalInputAdmission> {
    let logical_digest = artifact
        .content_digest
        .parse::<Digest>()
        .map_err(AnalysisBridgeError::Platform)?;
    let ValuePayload::Object(reference) = &resolved.value else {
        return Err(AnalysisBridgeError::Invalid(
            "resolved immutable signal input is not externalized".into(),
        ));
    };
    if reference.logical_digest != logical_digest || reference.value_schema != artifact.schema {
        return Err(AnalysisBridgeError::Identity(
            "resolved immutable signal input differs from its declaration",
        ));
    }
    let admission = SignalInputAdmission {
        artifact_id: artifact.artifact_id.clone(),
        logical_digest,
        value_schema: artifact.schema.clone(),
        receipt: resolved.admission.clone(),
    };
    admission.validate()?;
    Ok(admission)
}

pub fn validate_signal_input_payload(
    admission: &SignalInputAdmission,
    payload: &ValuePayload,
) -> AnalysisBridgeResult<()> {
    admission.validate()?;
    let (element_type, shape) = match payload {
        ValuePayload::Inline(value) => match value.as_ref() {
            InlineValue::F64Bits(_) => (ElementType::F64, vec![1_usize, 1]),
            InlineValue::Dense(dense) => (
                dense.element_type,
                dense
                    .shape
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            AnalysisBridgeError::Invalid(
                                "immutable signal input shape exceeds this host".into(),
                            )
                        })
                    })
                    .collect::<AnalysisBridgeResult<Vec<_>>>()?,
            ),
            _ => {
                return Err(AnalysisBridgeError::Invalid(
                    "immutable signal input is not a real64 scalar or dense array".into(),
                ))
            }
        },
        ValuePayload::Object(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {
            return Err(AnalysisBridgeError::Invalid(
                "immutable signal input payload is not closed".into(),
            ))
        }
    };
    if element_type != ElementType::F64
        || !admission
            .receipt
            .target
            .shape
            .is_proven_equivalent(&runmat_types::ShapeFact::from(
                shape.into_iter().map(Some).collect::<Vec<_>>(),
            ))
    {
        return Err(AnalysisBridgeError::Identity(
            "immutable signal input payload differs from its admission receipt",
        ));
    }
    Ok(())
}

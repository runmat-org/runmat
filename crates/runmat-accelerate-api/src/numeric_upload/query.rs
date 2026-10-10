//! One canonical typed transfer-upload query for class-preserving publication.
use crate::{
    ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderWorkload,
};

/// Build the owner query for exact host-to-device result publication. This
/// describes representations only; provider feasibility and physical evidence
/// still decide whether a particular route is available.
pub fn class_preserving_upload_query(
    outputs: Vec<ProviderRepresentation>,
) -> Option<ProviderFeasibilityQuery> {
    let requested_outputs = u32::try_from(outputs.len()).ok()?;
    if outputs.is_empty()
        || outputs.iter().any(|output| {
            output.residency != ProviderResidency::Device || output.checked_byte_len().is_none()
        })
    {
        return None;
    }
    let inputs = outputs
        .iter()
        .cloned()
        .map(|mut input| {
            input.residency = ProviderResidency::Host;
            input
        })
        .collect();
    Some(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs,
        outputs,
        workload: ProviderWorkload {
            requested_outputs: Some(requested_outputs),
            ..ProviderWorkload::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProviderElementType, ProviderLayout, ProviderStorage};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exact_class_and_shape_are_preserved_and_invalid_outputs_decline() {
        let output = ProviderRepresentation {
            element_type: ProviderElementType::I32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 3],
            residency: ProviderResidency::Device,
        };
        let query = class_preserving_upload_query(vec![output.clone()]).unwrap();
        assert_eq!(query.outputs, vec![output.clone()]);
        assert_eq!(query.inputs[0].shape, output.shape);
        assert_eq!(query.inputs[0].residency, ProviderResidency::Host);
        assert_eq!(query.workload.requested_outputs, Some(1));
        assert!(class_preserving_upload_query(Vec::new()).is_none());
        assert!(class_preserving_upload_query(vec![ProviderRepresentation {
            residency: ProviderResidency::Host,
            ..output.clone()
        }])
        .is_none());
        assert!(class_preserving_upload_query(vec![ProviderRepresentation {
            shape: vec![u64::MAX, 2],
            ..output
        }])
        .is_none());
    }
}

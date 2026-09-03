use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
use runmat_analysis_fea::{ComputeBackend, FeaRunResult};

use super::contracts::FallbackEvent;

pub(crate) fn promote_run_fields_to_device_refs(
    run: &mut FeaRunResult,
    fallback_events: &mut Vec<FallbackEvent>,
) {
    if run.requested_backend != ComputeBackend::Gpu {
        return;
    }

    for field in &mut run.fields {
        let field_label = field.field_id.clone();
        promote_field_to_device_ref(&field_label, field, fallback_events);
    }
}

fn promote_field_to_device_ref(
    field_label: &str,
    field: &mut AnalysisField,
    fallback_events: &mut Vec<FallbackEvent>,
) {
    let host_values = match &field.values {
        AnalysisFieldValues::HostF64(values) => values.clone(),
        AnalysisFieldValues::DeviceRef(_) => return,
    };

    let Some(provider) = runmat_accelerate_api::provider() else {
        fallback_events.push(FallbackEvent::field_provider_unavailable(field_label));
        return;
    };

    let shape = field.shape.clone();
    let view = runmat_accelerate_api::HostTensorView {
        data: &host_values,
        shape: &shape,
    };
    match provider.upload(&view) {
        Ok(handle) => {
            let backend = provider
                .device_info_struct()
                .backend
                .unwrap_or_else(|| "gpu".to_string());
            field.values = AnalysisFieldValues::DeviceRef(DeviceFieldRef {
                backend,
                token: format!("device:{}:buffer:{}", handle.device_id, handle.buffer_id),
                element_count: host_values.len(),
            });
        }
        Err(error) => fallback_events.push(FallbackEvent::field_upload_failed(
            field_label,
            error.to_string(),
        )),
    }
}

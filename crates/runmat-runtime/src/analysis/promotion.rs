use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
use runmat_analysis_fea::{ComputeBackend, FeaRunResult};

pub(crate) fn promote_run_fields_to_device_refs(
    run: &mut FeaRunResult,
    fallback_events: &mut Vec<String>,
) {
    if run.backend != ComputeBackend::Gpu {
        return;
    }

    promote_field_to_device_ref("displacement", &mut run.displacement_field, fallback_events);
    promote_field_to_device_ref("von_mises", &mut run.von_mises_field, fallback_events);
}

fn promote_field_to_device_ref(
    field_label: &str,
    field: &mut AnalysisField,
    fallback_events: &mut Vec<String>,
) {
    let host_values = match &field.values {
        AnalysisFieldValues::HostF64(values) => values.clone(),
        AnalysisFieldValues::DeviceRef(_) => return,
    };

    let Some(provider) = runmat_accelerate_api::provider() else {
        fallback_events.push(format!(
            "BACKEND_NO_PROVIDER:{field_label}:retained_host_field"
        ));
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
        Err(error) => {
            fallback_events.push(format!("BACKEND_UPLOAD_FAILED:{field_label}:{}", error))
        }
    }
}

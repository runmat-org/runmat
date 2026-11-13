//! Admit exactly the requested QR result from the input handle's physical owner.

use runmat_accelerate_api::{
    same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderQrResult,
};

use super::{
    gpu_helpers, qr_internal_error, BuiltinResult, QrEval, QrMode, QrOptions, QrOutputRequest,
};

pub(super) fn admit(
    mut result: ProviderQrResult,
    input: &GpuTensorHandle,
    owner: &dyn AccelProvider,
    options: &QrOptions,
    request: QrOutputRequest,
) -> BuiltinResult<QrEval> {
    let rows = input.shape().first().copied().unwrap_or(1);
    let cols = input.shape().get(1).copied().unwrap_or(1);
    let reduced = rows.min(cols);
    let q_shape = [
        rows,
        if matches!(options.mode, QrMode::Economy) && rows >= cols {
            reduced
        } else {
            rows
        },
    ];
    let r_shape = [
        if matches!(options.mode, QrMode::Economy) && rows >= cols {
            reduced
        } else {
            rows
        },
        cols,
    ];
    let matrix_shape = [cols, cols];
    let vector_shape = [cols, 1];
    let factor_precision = runmat_accelerate_api::handle_precision(input);
    let permutation_precision = Some(runmat_accelerate_api::ProviderPrecision::F64);
    let slots = match (&result, request) {
        (ProviderQrResult::R { r }, QrOutputRequest::R) => {
            Some([Some((r, r_shape, factor_precision)), None, None, None])
        }
        (ProviderQrResult::Qr { q, r }, QrOutputRequest::Qr) => Some([
            Some((q, q_shape, factor_precision)),
            Some((r, r_shape, factor_precision)),
            None,
            None,
        ]),
        (ProviderQrResult::Qrp { q, r, permutation }, QrOutputRequest::Qrp) => {
            let perm_shape = match options.format.pivot_mode() {
                super::PivotMode::Matrix => matrix_shape,
                super::PivotMode::Vector => vector_shape,
            };
            Some([
                Some((q, q_shape, factor_precision)),
                Some((r, r_shape, factor_precision)),
                Some((permutation, perm_shape, permutation_precision)),
                None,
            ])
        }
        (
            ProviderQrResult::FullEnvelope {
                q,
                r,
                perm_matrix,
                perm_vector,
            },
            QrOutputRequest::FullEnvelope | QrOutputRequest::FullEnvelopeUnpivoted,
        ) => Some([
            Some((q, q_shape, factor_precision)),
            Some((r, r_shape, factor_precision)),
            Some((perm_matrix, matrix_shape, permutation_precision)),
            Some((perm_vector, vector_shape, permutation_precision)),
        ]),
        _ => None,
    };
    let valid = factor_precision.is_some()
        && slots.is_some_and(|slots| {
            slots
                .into_iter()
                .flatten()
                .all(|(output, expected_shape, expected_precision)| {
                    output.shape() == expected_shape
                        && output.device_id() == input.device_id()
                        && !output.shares_allocation_with(input)
                        && output.allocation_lease().validate_live().is_ok()
                        && !runmat_accelerate_api::handle_is_transposed(output)
                        && !runmat_accelerate_api::handle_is_logical(output)
                        && runmat_accelerate_api::handle_integer_type(output).is_none()
                        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_precision(output) == expected_precision
                        && gpu_helpers::exact_provider_for_handle(output)
                            .is_some_and(|candidate| same_provider_instance(candidate, owner))
                })
                && slots
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .all(|(index, (output, _, _))| {
                        slots
                            .into_iter()
                            .flatten()
                            .skip(index + 1)
                            .all(|(other, _, _)| !output.shares_allocation_with(other))
                    })
        });
    if !valid {
        return Err(reject(
            result,
            "qr: provider returned invalid or mismatched factor outputs",
        ));
    }

    let outputs: [Option<&mut GpuTensorHandle>; 4] = match &mut result {
        ProviderQrResult::R { r } => [Some(r), None, None, None],
        ProviderQrResult::Qr { q, r } => [Some(q), Some(r), None, None],
        ProviderQrResult::Qrp { q, r, permutation } => [Some(q), Some(r), Some(permutation), None],
        ProviderQrResult::FullEnvelope {
            q,
            r,
            perm_matrix,
            perm_vector,
        } => [Some(q), Some(r), Some(perm_matrix), Some(perm_vector)],
    };
    let mut provenance_error = None;
    for output in outputs.into_iter().flatten() {
        if let Err(error) = gpu_helpers::propagate_output_provenance(output, [input]) {
            provenance_error = Some(error);
            break;
        }
    }
    if let Some(error) = provenance_error {
        return Err(reject_with_cause(result, error));
    }
    QrEval::from_provider(result, options, request)
}

fn reject(result: ProviderQrResult, message: &'static str) -> super::RuntimeError {
    reject_with_cause(result, qr_internal_error(message))
}

fn reject_with_cause(result: ProviderQrResult, error: super::RuntimeError) -> super::RuntimeError {
    let handles = match result {
        ProviderQrResult::R { r } => vec![r],
        ProviderQrResult::Qr { q, r } => vec![q, r],
        ProviderQrResult::Qrp { q, r, permutation } => vec![q, r, permutation],
        ProviderQrResult::FullEnvelope {
            q,
            r,
            perm_matrix,
            perm_vector,
        } => {
            vec![q, r, perm_matrix, perm_vector]
        }
    };
    gpu_helpers::reject_gpu_outputs(handles, error)
}

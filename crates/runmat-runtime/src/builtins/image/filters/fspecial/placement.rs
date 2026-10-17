//! Typed whole-call admission for the optional resident fspecial constructor.

use super::FspecialFilterSpec;
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactProviderOperation, SelectedProviderOperation,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibilityQuery, ProviderFspecialFilter, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceBoundUnavailable, ResourceByteBound};

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    spec: &FspecialFilterSpec,
    materialize_on_gpu: bool,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    if !materialize_on_gpu || !spec.is_gpu_supported() {
        return Ok(None);
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    let request = spec.to_request()?;
    let filter = ProviderFspecialFilter::from(&request);
    let Some([rows, cols]) = filter.output_shape() else {
        return Ok(None);
    };
    let output = ProviderRepresentation {
        element_type: ProviderElementType::from(provider.precision()),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![rows, cols],
        residency: ProviderResidency::Device,
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fspecial"),
        family: ProviderOperationFamily::Library,
        inputs: Vec::new(),
        outputs: vec![output],
        workload: ProviderWorkload {
            fspecial: Some(filter),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    // The public implementation constructs and retains one f64 host tensor
    // while the optional provider hook runs. Its vector is moved into Tensor;
    // there is no second materialized payload for these supported forms.
    let host_scratch = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(8))
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ));
    let domain_facts = Digest::sha256(
        serde_json::to_vec(&query.workload.fspecial).expect("typed fspecial request serializes"),
    );
    select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation: "fspecial",
            variant: "fspecial.optional_resident_constructor.v1",
            domain_facts,
            calibration_facts: None,
            host_scratch,
            provider_host_scratch: ResourceByteBound::Known(0),
            host_cpu_millicores: 1_000,
            already_live_bytes: 0,
            readbacks: &[],
            produced_readback: None,
            internal_input_stages: None,
            output_queue_slots: None,
            conditional_host_fallback: Some(host_scratch),
            conditional_host_scratch: ResourceByteBound::Known(0),
            conditional_provider_upload: None,
            provider_host_mediated: false,
        },
    )
}

//! One admitted dimensional extrema provider route. The provider owns the
//! two-output physical recipe; Runtime owns public output demand and feedback.

use super::{ExtremaDemand, ExtremaKind};
use crate::context::{
    provider_operation::{
        select_exact_provider_operation, ExactProducedProviderReadback, ExactProviderOperation,
        SelectedProviderOperation,
    },
    RuntimeContext,
};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    handle_provenance, AccelProvider, GpuTensorHandle, NumericElementType, ProviderElementType,
    ProviderExtremaDemand, ProviderExtremaInputBacking, ProviderExtremaKind,
    ProviderExtremaWorkload, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};

pub(super) struct SelectedExtrema {
    pub(super) placement: SelectedProviderOperation,
    backing: ProviderExtremaInputBacking,
}

impl SelectedExtrema {
    pub(super) fn revalidate(&self, input: &GpuTensorHandle) -> BuiltinResult<bool> {
        self.backing.matches_live_handle(input).map_err(|source| {
            crate::build_runtime_error("dimensional extrema input lease changed before dispatch")
                .with_identifier("RunMat:Placement:ExpiredExtremaInput")
                .with_source(source)
                .build()
        })
    }
}

fn element_type(source: NumericElementType) -> ProviderElementType {
    match source {
        NumericElementType::F32 => ProviderElementType::F32,
        NumericElementType::F64 => ProviderElementType::F64,
        NumericElementType::I8 => ProviderElementType::I8,
        NumericElementType::I16 => ProviderElementType::I16,
        NumericElementType::I32 => ProviderElementType::I32,
        NumericElementType::I64 => ProviderElementType::I64,
        NumericElementType::U8 => ProviderElementType::U8,
        NumericElementType::U16 => ProviderElementType::U16,
        NumericElementType::U32 => ProviderElementType::U32,
        NumericElementType::U64 => ProviderElementType::U64,
    }
}

fn representation(class: ProviderElementType, shape: Vec<u64>) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    }
}

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    kind: ExtremaKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    axis: usize,
    demand: ExtremaDemand,
) -> BuiltinResult<Option<SelectedExtrema>> {
    // Public min/max validate the selected dimension before reaching this
    // physical owner. Preserve that applicability exit before resource probes.
    if input.shape().get(axis).is_none() {
        return Ok(None);
    }
    let Some(backing) =
        ProviderExtremaInputBacking::from_full_live_handle(input).map_err(|source| {
            crate::build_runtime_error("dimensional extrema input allocation is unavailable")
                .with_identifier("RunMat:Placement:ExpiredExtremaInput")
                .with_source(source)
                .build()
        })?
    else {
        return crate::context::provider_operation::unavailable_provider_route(runtime, provider);
    };
    let Some(shape) = input
        .shape()
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
    else {
        return crate::context::provider_operation::unavailable_provider_route(runtime, provider);
    };
    let Some(&axis_extent) = shape.get(axis) else {
        return Ok(None);
    };
    let mut output_shape = shape.clone();
    if !output_shape.contains(&0) {
        output_shape[axis] = 1;
    }
    let class = element_type(input.physical().element_type());
    let index_class = ProviderElementType::from(provider.precision());
    let mut outputs = vec![representation(class, output_shape.clone())];
    let demand = match demand {
        ExtremaDemand::ValuesOnly => ProviderExtremaDemand::ValuesOnly,
        ExtremaDemand::ValuesAndIndices => {
            outputs.push(representation(index_class, output_shape.clone()));
            ProviderExtremaDemand::ValuesAndIndices
        }
    };
    let kind = match kind {
        ExtremaKind::Min => ProviderExtremaKind::Minimum,
        ExtremaKind::Max => ProviderExtremaKind::Maximum,
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(match kind {
            ProviderExtremaKind::Minimum => runmat_builtins::MIN_CATALOG_ENTRY.identity.name,
            ProviderExtremaKind::Maximum => runmat_builtins::MAX_CATALOG_ENTRY.identity.name,
        }),
        family: ProviderOperationFamily::Reduction,
        inputs: vec![representation(class, shape)],
        outputs,
        workload: ProviderWorkload {
            requested_outputs: Some(match demand {
                ProviderExtremaDemand::ValuesOnly => 1,
                ProviderExtremaDemand::ValuesAndIndices => 2,
            }),
            extrema: Some(ProviderExtremaWorkload {
                kind,
                axis: u64::try_from(axis).unwrap_or(u64::MAX),
                demand,
                input_backing: Some(backing),
            }),
            ..ProviderWorkload::default()
        },
    };
    // Only F32 physical indices need a real native readback and public
    // Double conversion. Values-only still budgets the provider's discarded
    // physical index in its two-output peak, but performs no readback.
    let produced_readback = (demand == ProviderExtremaDemand::ValuesAndIndices
        && index_class == ProviderElementType::F32)
        .then(|| {
            let indices = output_shape
                .iter()
                .try_fold(1_u64, |count, extent| count.checked_mul(*extent))?;
            Some(ExactProducedProviderReadback {
                output_index: 1,
                public_host_bytes: ResourceByteBound::Known(indices.checked_mul(8)?),
                // The F64 conversion vector becomes the public host index
                // result; only its F32 source coexists as scratch.
                conversion_host_workspace: ResourceByteBound::Known(0),
            })
        })
        .flatten();
    if demand == ProviderExtremaDemand::ValuesAndIndices
        && index_class == ProviderElementType::F32
        && produced_readback.is_none()
    {
        return Err(crate::build_runtime_error(
            "dimensional extrema index conversion geometry overflows",
        )
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let identity = input.view_identity();
    let domain_facts = Digest::sha256(
        serde_json::to_vec(&(
            identity.allocation(),
            identity.view(),
            handle_provenance(input),
            axis_extent,
            demand,
        ))
        .expect("typed extrema facts serialize"),
    );
    let selected = select_exact_provider_operation(
        runtime,
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation: match kind {
                ProviderExtremaKind::Minimum => "min",
                ProviderExtremaKind::Maximum => "max",
            },
            variant: "extrema.dimensional.native.v1",
            domain_facts,
            calibration_facts: None,
            host_scratch: ResourceByteBound::Known(0),
            provider_host_scratch: ResourceByteBound::Known(0),
            host_cpu_millicores: 1_000,
            already_live_bytes: input.physical().requested_bytes(),
            readbacks: &[],
            produced_readback,
            internal_input_stages: None,
            output_queue_slots: None,
            conditional_host_fallback: None,
            conditional_host_scratch: ResourceByteBound::Known(0),
            conditional_provider_upload: None,
            provider_host_mediated: false,
        },
    )?;
    Ok(selected.map(|placement| SelectedExtrema { placement, backing }))
}

//! Dimension selection and NaN-aware mean on one provider.
use super::{
    admitted_real, expected_shape, reject_output, Failure, ObserveResidentHook,
    ResidentStageProgress,
};
use crate::builtins::stats::summary::statistical_means::options::{
    Axes, MissingMode, ReductionKind,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderPrecision};

pub(super) fn selected_axes(shape: &[usize], axes: &Axes) -> Vec<usize> {
    match axes {
        Axes::Default => vec![shape.iter().position(|extent| *extent != 1).unwrap_or(0)],
        Axes::All => (0..shape.len()).collect(),
        Axes::Dims(dimensions) => dimensions
            .iter()
            .filter_map(|axis| {
                let zero_based = axis - 1;
                (zero_based < shape.len()).then_some(zero_based)
            })
            .collect(),
    }
}

pub(super) fn protect_live_operands(
    kind: ReductionKind,
    output: GpuTensorHandle,
    operands: &[&GpuTensorHandle],
    stage: &'static str,
) -> Result<GpuTensorHandle, Failure> {
    if operands
        .iter()
        .any(|operand| output.shares_allocation_with(operand))
    {
        Err(reject_output(kind, output, stage))
    } else {
        Ok(output)
    }
}

async fn sum(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    axes: &[usize],
    protected: &[&GpuTensorHandle],
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let mut current = input.clone();
    for axis in axes {
        let next = provider
            .reduce_sum_dim(&current, *axis)
            .await
            .observe(progress, kind)?;
        let expected = expected_shape(current.shape(), &Axes::Dims(vec![axis + 1]));
        let next = admitted_real(
            kind,
            provider,
            input,
            next,
            &expected,
            ProviderPrecision::F64,
        )?;
        // A singleton reduction may legitimately return a view of its input.
        // A nontrivial reduction must allocate, and neither case may return
        // another live operand from the surrounding mean computation.
        let next =
            protect_live_operands(kind, next, protected, "resident sum aliased a live operand")?;
        if current.shape()[*axis] != 1 && next.shares_allocation_with(&current) {
            return Err(reject_output(kind, next, "resident sum aliased its input"));
        }
        current = next;
    }
    Ok(current)
}

pub(super) async fn mean(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    transformed: &GpuTensorHandle,
    axes: &Axes,
    missing: MissingMode,
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let selected = selected_axes(original.shape(), axes);
    let precision = runmat_accelerate_api::handle_precision(original).ok_or_else(|| {
        progress.unavailable(kind, "resident mean input has no floating precision")
    })?;
    if selected.is_empty() {
        return Ok(transformed.clone());
    }
    if missing == MissingMode::Include {
        let mut current = transformed.clone();
        for axis in &selected {
            let next = progress.observe_optional(
                kind,
                provider.reduce_mean_dim(&current, *axis).await,
                "resident mean dimensional stage unavailable",
            )?;
            let expected = expected_shape(current.shape(), &Axes::Dims(vec![axis + 1]));
            current = admitted_real(kind, provider, original, next, &expected, precision)?;
        }
        return Ok(current);
    }
    let cleaned = provider
        .map_nan_to_zero(transformed)
        .observe(progress, kind)?;
    let cleaned = admitted_real(
        kind,
        provider,
        original,
        cleaned,
        transformed.shape(),
        precision,
    )?;
    let cleaned = protect_live_operands(
        kind,
        cleaned,
        &[original, transformed],
        "resident NaN map aliased an input",
    )?;
    let mask = provider.not_nan_mask(original).observe(progress, kind)?;
    let mask = admitted_real(kind, provider, original, mask, original.shape(), precision)?;
    let mask = protect_live_operands(
        kind,
        mask,
        &[original, transformed, &cleaned],
        "resident NaN mask aliased a live operand",
    )?;
    let numerator = sum(
        kind,
        provider,
        &cleaned,
        &selected,
        &[original, transformed, &mask],
        progress,
    )
    .await?;
    let denominator = sum(
        kind,
        provider,
        &mask,
        &selected,
        &[original, transformed, &cleaned, &numerator],
        progress,
    )
    .await?;
    let result = provider
        .elem_div(&numerator, &denominator)
        .await
        .observe(progress, kind)?;
    let result = admitted_real(
        kind,
        provider,
        original,
        result,
        &expected_shape(original.shape(), axes),
        ProviderPrecision::F64,
    )?;
    let result = protect_live_operands(
        kind,
        result,
        &[
            original,
            transformed,
            &cleaned,
            &mask,
            &numerator,
            &denominator,
        ],
        "resident division aliased a live operand",
    )?;
    if precision == ProviderPrecision::F64 {
        return Ok(result);
    }
    let output = provider
        .unary_single(&result)
        .await
        .observe(progress, kind)?;
    let output = admitted_real(
        kind,
        provider,
        original,
        output,
        result.shape(),
        ProviderPrecision::F32,
    )?;
    protect_live_operands(
        kind,
        output,
        &[
            original,
            transformed,
            &cleaned,
            &mask,
            &numerator,
            &denominator,
            &result,
        ],
        "resident class cast aliased a live operand",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, HostNumericDataView,
        HostNumericTensorView, NumericElementType, ProviderPrecision,
    };
    use runmat_value::Value;

    #[test]
    fn completed_resident_mean_stage_rejects_unsupported_replay() {
        let unsupported = || {
            runmat_accelerate_api::unsupported_provider_operation(
                "fixture rejects a later mean stage",
            )
        };
        let mut progress = ResidentStageProgress::default();
        assert!(matches!(
            progress.observe::<(), _>(ReductionKind::Harmonic, Err(unsupported())),
            Err(Failure::Unsupported)
        ));
        progress
            .observe(ReductionKind::Harmonic, Ok::<(), anyhow::Error>(()))
            .unwrap();
        let Err(Failure::Execution(error)) =
            progress.observe::<(), _>(ReductionKind::Harmonic, Err(unsupported()))
        else {
            panic!("completed mean stage must not replay")
        };
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }

    #[test]
    fn omitted_single_observations_accumulate_in_double_and_publish_single() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let samples = [16_777_216.0_f32, 1.0, 1.0, f32::NAN];
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&samples),
                    shape: &[4, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            provider.reset_telemetry();
            let result = block_on(mean(
                ReductionKind::Harmonic,
                provider,
                &input,
                &input,
                &Axes::Default,
                MissingMode::Omit,
                &mut ResidentStageProgress::default(),
            ))
            .unwrap();
            assert_eq!(
                runmat_accelerate_api::handle_precision(&result),
                Some(ProviderPrecision::F32)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(
                crate::builtins::common::test_support::gather(Value::GpuTensor(result))
                    .unwrap()
                    .materialize_f64(),
                vec![5_592_406.0],
            );
            input.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn sibling_and_cast_stage_aliases_reject_only_the_returned_reference() {
        let owner = TestAllocationOwner::new();
        let sibling = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let rejected = protect_live_operands(
            ReductionKind::Harmonic,
            sibling.handle.clone(),
            &[&sibling.handle],
            "resident division aliased a live operand",
        );
        assert!(matches!(rejected, Err(Failure::Execution(_))));
        assert_eq!(sibling.probe.retirements(), 0);
        sibling.handle.allocation_lease().validate_live().unwrap();

        let cast_operand = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let rejected = protect_live_operands(
            ReductionKind::Harmonic,
            cast_operand.handle.clone(),
            &[&cast_operand.handle],
            "resident class cast aliased a live operand",
        );
        assert!(matches!(rejected, Err(Failure::Execution(_))));
        assert_eq!(cast_operand.probe.retirements(), 0);
        cast_operand
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
    }
}

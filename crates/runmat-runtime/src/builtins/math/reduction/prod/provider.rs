//! A reduction chain may fall back only before its first attempted stage.
use super::super::numerical_execution::NumericalCall;
use super::*;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

pub(super) async fn execute(
    input: &GpuTensorHandle,
    parsed: &ParsedArguments,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(input).ok_or_else(|| {
        prod_internal_error("resident product input has no exact owning provider")
    })?;
    input.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("product input allocation is unavailable")
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let integer = runmat_accelerate_api::handle_integer_type(input);
    let logical = runmat_accelerate_api::handle_is_logical(input);
    let native_integer = integer.is_some() && parsed.output == OutputTemplate::Native;
    let precision = if native_integer
        || (parsed.output != OutputTemplate::Double && integer.is_none() && !logical)
    {
        runmat_accelerate_api::handle_precision(input)
    } else {
        Some(ProviderPrecision::F64)
    };
    // Integer default arithmetic must convert before multiplying, never after
    // a saturated native product. The current floating-cast owner does not
    // yet admit packed integer sources.
    if (integer.is_some() && !native_integer)
        || (logical && parsed.output == OutputTemplate::Native)
        || (!native_integer && precision != Some(provider.precision()))
        || parsed.nan_mode == ReductionNaN::Omit
    {
        return fallback(input, provider, parsed, call).await;
    }
    let axes = ResolvedDims::resolve(input.shape(), &parsed.selection);
    let mut owned = None;
    if runmat_accelerate_api::handle_precision(input) != precision || logical {
        call.host()?; // selected floating-cast metadata has no finite owner here
        call.check()?;
        let cast = provider.unary_double(input).await;
        let cast = match cast {
            Ok(cast) => cast,
            Err(source) if source.is_unsupported() => {
                return fallback(input, provider, parsed, call).await
            }
            Err(source) => {
                return Err(crate::runtime_error::with_provider_execution_source(
                    prod_internal_error("product class conversion failed"),
                    source.with_unproven_effects(),
                ))
            }
        };
        if !valid(&cast, input, input, provider, None, precision, false) {
            return Err(gpu_helpers::reject_gpu_output(
                cast,
                prod_internal_error("product class conversion returned invalid output"),
            ));
        }
        owned = Some(cast);
    }
    if axes.len() == 0 {
        return match owned {
            Some(output) => gpu_helpers::resident_gpu_value(output),
            None => gpu_helpers::resident_gpu_value(input.clone()),
        };
    }
    if axes.len() == input.shape().len() {
        let source = owned.as_ref().unwrap_or(input);
        let pending = if native_integer {
            provider.reduce_integer_prod_native(source)
        } else {
            provider.reduce_prod(source)
        };
        let output = match call.attempt_handle(pending).await {
            Ok(output) => output,
            Err(error) => return Err(reject(owned, error)),
        };
        if let Some(output) = output {
            if !valid(
                &output,
                input,
                source,
                provider,
                Some(Shape::Scalar),
                precision,
                native_integer,
            ) {
                return Err(reject(
                    owned,
                    gpu_helpers::reject_gpu_output(
                        output,
                        prod_internal_error("provider returned an invalid global product"),
                    ),
                ));
            }
            retire(owned, output)
                .and_then(|output| gpu_helpers::propagate_owned_output_provenance(output, [input]))
                .and_then(gpu_helpers::resident_gpu_value)
        } else if owned.is_some() {
            Err(reject(
                owned,
                prod_internal_error("product declined after its class conversion"),
            ))
        } else {
            // A global NoAttempt is safe; the independent dimension pipeline
            // may still be available without replaying a completed operation.
            dimensions(
                input,
                provider,
                parsed,
                call,
                &axes,
                None,
                precision,
                native_integer,
            )
            .await
        }
    } else {
        dimensions(
            input,
            provider,
            parsed,
            call,
            &axes,
            owned,
            precision,
            native_integer,
        )
        .await
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Scalar,
    Axis(usize),
}
fn valid(
    output: &GpuTensorHandle,
    original: &GpuTensorHandle,
    source: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    shape: Option<Shape>,
    precision: Option<ProviderPrecision>,
    integer: bool,
) -> bool {
    let geometry = match shape {
        Some(Shape::Scalar) => !output.shape().is_empty() && output.shape().iter().all(|n| *n == 1),
        Some(Shape::Axis(axis)) => {
            output.shape().len() == source.shape().len()
                && output
                    .shape()
                    .iter()
                    .enumerate()
                    .all(|(i, n)| *n == if i == axis { 1 } else { source.shape()[i] })
        }
        None => output.shape() == source.shape(),
    };
    let integer_class = if integer {
        runmat_accelerate_api::handle_integer_type(original)
    } else {
        None
    };
    geometry
        && !output.shares_allocation_with(original)
        && !output.shares_allocation_with(source)
        && output.device_id() == original.device_id()
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == precision
        && runmat_accelerate_api::handle_integer_type(output) == integer_class
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, precision, integer_class, false)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}
async fn dimensions(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    parsed: &ParsedArguments,
    call: &NumericalCall<'_>,
    axes: &ResolvedDims,
    mut owned: Option<GpuTensorHandle>,
    precision: Option<ProviderPrecision>,
    integer: bool,
) -> BuiltinResult<Value> {
    for axis in axes.iter() {
        let source = owned.as_ref().unwrap_or(input);
        let pending = if integer {
            provider.reduce_integer_prod_native_dim(source, axis)
        } else {
            provider.reduce_prod_dim(source, axis)
        };
        let output = match call.attempt_handle(pending).await {
            Ok(Some(output)) => output,
            Ok(None) if owned.is_none() => return fallback(input, provider, parsed, call).await,
            Ok(None) => {
                return Err(reject(
                    owned,
                    prod_internal_error("provider declined a product after an earlier stage"),
                ))
            }
            Err(error) => return Err(reject(owned, error)),
        };
        if !valid(
            &output,
            input,
            source,
            provider,
            Some(Shape::Axis(axis)),
            precision,
            integer,
        ) {
            return Err(reject(
                owned,
                gpu_helpers::reject_gpu_output(
                    output,
                    prod_internal_error("provider returned invalid dimensional product"),
                ),
            ));
        }
        owned = Some(retire(owned, output)?);
    }
    let output = owned.expect("nonempty dimensional reduction");
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}
fn reject(output: Option<GpuTensorHandle>, error: RuntimeError) -> RuntimeError {
    match output {
        Some(output) => gpu_helpers::reject_gpu_output(output, error),
        None => error,
    }
}
fn retire(
    previous: Option<GpuTensorHandle>,
    output: GpuTensorHandle,
) -> BuiltinResult<GpuTensorHandle> {
    if let Some(previous) = previous {
        if let Err(source) = previous.release() {
            let error = crate::build_runtime_error("product intermediate retirement failed")
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build();
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
    }
    Ok(output)
}
async fn fallback(
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    parsed: &ParsedArguments,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Value> {
    call.fallback([&Value::GpuTensor(input.clone())])?;
    let host = gpu_helpers::download_value_preserving_residency_async(provider, input).await?;
    call.check()?;
    let host = host::execute(host, parsed)?;
    // Automatic placement may return a host result when the owner cannot
    // physically represent its required public class.
    gpu_helpers::restore_class_preserving_value(input, host, NAME)
}

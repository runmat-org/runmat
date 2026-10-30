//! Checked resident axis preparation and grid publication.
use super::*;

#[cfg(test)]
#[path = "gpu/tests.rs"]
mod tests;

pub(super) fn try_meshgrid_gpu_from_vector_axes(
    x_axis: &AxisData,
    y_axis: &AxisData,
    z_axis: Option<&AxisData>,
) -> crate::BuiltinResult<Option<Vec<MeshgridOutput>>> {
    let Some(x_handle) = x_axis.gpu_real() else {
        return Ok(None);
    };
    let Some(y_handle) = y_axis.gpu_real() else {
        return Ok(None);
    };

    let z_handle = match z_axis {
        Some(axis) => match axis.gpu_real() {
            Some(h) => Some(h),
            None => return Ok(None),
        },
        None => None,
    };

    let invalid_owner = || builtin_error("meshgrid: no exact provider owns the resident axes");
    if let Some(z) = z_handle {
        gpu_helpers::require_resident_compute_owners(
            &[x_handle, y_handle, z],
            false,
            invalid_owner,
        )?;
    } else {
        gpu_helpers::require_resident_compute_owners(&[x_handle, y_handle], false, invalid_owner)?;
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(x_handle) else {
        return Ok(None);
    };
    let Some(y_provider) = gpu_helpers::exact_provider_for_handle(y_handle) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(y_provider, provider) {
        return Ok(None);
    }
    if let Some(z) = z_handle {
        let Some(z_provider) = gpu_helpers::exact_provider_for_handle(z) else {
            return Ok(None);
        };
        if !runmat_accelerate_api::same_provider_instance(z_provider, provider) {
            return Ok(None);
        }
    }

    let nx = x_axis.len;
    let ny = y_axis.len;
    let nz = z_axis.map(|axis| axis.len).unwrap_or(1);

    let allocated = std::cell::Cell::new(false);
    let failure = |message: String, source: runmat_accelerate_api::ProviderExecutionError| {
        let source = if allocated.get() {
            source.after_effect()
        } else {
            source.with_unproven_effects()
        };
        crate::runtime_error::with_provider_execution_source(builtin_error(message), source)
    };
    let reshape =
        |original: &GpuTensorHandle, input: &GpuTensorHandle, shape: &[usize], label: &str| {
            if original.allocation_lease().validate_live().is_err()
                || original.view().validate(original.physical()).is_err()
                || gpu_helpers::expected_handle_numeric_element_type(original).is_err()
                || !gpu_helpers::reshape_output_matches(original, input, provider, input.shape())
            {
                return Err(failure(
                    "meshgrid: invalid provider axis input".into(),
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "meshgrid: invalid provider axis input"
                    )),
                ));
            }
            let output = provider.reshape(input, shape).map_err(|source| {
                let message = format!("meshgrid: reshape {label} failed: {source}");
                failure(message, source.into())
            })?;
            if ![Some(x_handle), Some(y_handle), z_handle]
                .iter()
                .flatten()
                .any(|borrowed| output.shares_allocation_with(borrowed))
            {
                allocated.set(true);
            }
            if !gpu_helpers::reshape_output_matches(original, &output, provider, shape) {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    failure(
                        format!("meshgrid: invalid provider reshape {label} output"),
                        runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                            "meshgrid: invalid provider axis reshape"
                        )),
                    ),
                ));
            }
            Ok(output)
        };

    // Every returned reshape is an owned view, even when it aliases a borrowed
    // axis. Keep exactly those six possible views until all grids are ready.
    let mut views: [Option<GpuTensorHandle>; 6] = std::array::from_fn(|_| None);
    let mut outputs = Vec::with_capacity(if z_handle.is_some() { 3 } else { 2 });
    let protected = [Some(x_handle), Some(y_handle), z_handle];
    let expected = if z_handle.is_some() {
        vec![ny, nx, nz]
    } else {
        vec![ny, nx]
    };
    let replicate = |input: &GpuTensorHandle,
                     reps: &[usize],
                     prior: &[MeshgridOutput]|
     -> crate::BuiltinResult<GpuTensorHandle> {
        let tiling_failure = |source: runmat_accelerate_api::ProviderExecutionError| {
            failure(
                format!("meshgrid: provider tiling failed: {source}"),
                source,
            )
        };
        let mut output = provider
            .repmat(input, reps)
            .map_err(tiling_failure)?
            .ok_or_else(|| {
                tiling_failure(
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "meshgrid: required tiling unavailable after axis preparation"
                    ))
                    .with_unproven_effects(),
                )
            })?;
        if !output.shares_allocation_with(input)
            && !protected.iter().flatten().any(|borrowed| output.shares_allocation_with(borrowed))
            && !prior.iter().any(|value| matches!(value, MeshgridOutput::Gpu(other) if output.shares_allocation_with(other))) {
            allocated.set(true);
        }
        if !gpu_helpers::replication_output_matches(input, &output, provider, &expected)
            || protected.iter().flatten().any(|borrowed| output.shares_allocation_with(borrowed))
            || prior.iter().any(|value| matches!(value, MeshgridOutput::Gpu(other) if output.shares_allocation_with(other))) {
            return Err(gpu_helpers::reject_gpu_output(output, tiling_failure(
                runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!("meshgrid: invalid provider grid output")))));
        }
        if let Err(error) = gpu_helpers::propagate_output_provenance(
            &mut output,
            protected.iter().flatten().copied(),
        ) {
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        Ok(output)
    };
    let build = (|| -> crate::BuiltinResult<()> {
        views[0] = Some(reshape(x_handle, x_handle, &[1, nx], "X")?);
        views[1] = Some(reshape(y_handle, y_handle, &[ny, 1], "Y")?);
        if let Some(z) = z_handle {
            views[2] = Some(reshape(
                x_handle,
                views[0].as_ref().expect("X row was prepared"),
                &[1, nx, 1],
                "X(3d)",
            )?);
            views[3] = Some(reshape(
                y_handle,
                views[1].as_ref().expect("Y column was prepared"),
                &[ny, 1, 1],
                "Y(3d)",
            )?);

            let x_grid = replicate(
                views[2].as_ref().expect("X base was prepared"),
                &[ny, 1, nz],
                &outputs,
            )?;
            outputs.push(MeshgridOutput::Gpu(x_grid));
            let y_grid = replicate(
                views[3].as_ref().expect("Y base was prepared"),
                &[1, nx, nz],
                &outputs,
            )?;

            outputs.push(MeshgridOutput::Gpu(y_grid));
            views[4] = Some(reshape(z, z, &[1, nz], "Z")?);
            views[5] = Some(reshape(
                z,
                views[4].as_ref().expect("Z row was prepared"),
                &[1, 1, nz],
                "Z(3d)",
            )?);
            let z_grid = replicate(
                views[5].as_ref().expect("Z base was prepared"),
                &[ny, nx, 1],
                &outputs,
            )?;
            outputs.push(MeshgridOutput::Gpu(z_grid));
        } else {
            let x_grid = replicate(
                views[0].as_ref().expect("X row was prepared"),
                &[ny, 1],
                &outputs,
            )?;
            outputs.push(MeshgridOutput::Gpu(x_grid));
            let y_grid = replicate(
                views[1].as_ref().expect("Y column was prepared"),
                &[1, nx],
                &outputs,
            )?;
            outputs.push(MeshgridOutput::Gpu(y_grid));
        }

        Ok(())
    })();
    if let Err(error) = build {
        return Err(gpu_helpers::reject_gpu_outputs(
            views
                .into_iter()
                .flatten()
                .chain(outputs.into_iter().filter_map(|output| match output {
                    MeshgridOutput::Gpu(handle) => Some(handle),
                    MeshgridOutput::Host(_) => None,
                })),
            error,
        ));
    }

    let mut remaining_views = views.into_iter().flatten();
    while let Some(view) = remaining_views.next() {
        if let Err(source) = view.release() {
            let error = build_runtime_error("meshgrid: intermediate view retirement failed")
                .with_builtin("meshgrid")
                .with_source(source)
                .build();
            return Err(gpu_helpers::reject_gpu_outputs(
                remaining_views.chain(outputs.into_iter().filter_map(|output| match output {
                    MeshgridOutput::Gpu(handle) => Some(handle),
                    MeshgridOutput::Host(_) => None,
                })),
                error,
            ));
        }
    }

    Ok(Some(outputs))
}

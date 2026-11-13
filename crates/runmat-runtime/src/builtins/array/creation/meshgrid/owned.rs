//! Consuming meshgrid publication retains each GPU result until moved or retired.
use super::*;

impl MeshgridOutput {
    async fn into_value(
        self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> crate::BuiltinResult<Value> {
        match self {
            MeshgridOutput::Host(host) => host.to_value(class, residency),
            MeshgridOutput::Gpu(handle)
                if matches!(class, PrototypeClass::Real)
                    && matches!(residency, DevicePreference::Gpu(_)) =>
            {
                admit_direct_gpu_output(handle, false)
            }
            MeshgridOutput::Gpu(handle)
                if matches!(class, PrototypeClass::Complex)
                    && matches!(residency, DevicePreference::Gpu(_))
                    && runmat_accelerate_api::handle_storage(&handle)
                        == GpuTensorStorage::ComplexInterleaved =>
            {
                admit_direct_gpu_output(handle, true)
            }
            MeshgridOutput::Gpu(handle) => {
                let result = MeshgridOutput::Gpu(handle.clone())
                    .to_value(class, residency)
                    .await;
                match result {
                    Ok(value) => match handle.release() {
                        Ok(_) => Ok(value),
                        Err(source) => Err(reject_meshgrid_value(
                            value,
                            build_runtime_error("meshgrid: intermediate output retirement failed")
                                .with_builtin("meshgrid")
                                .with_source(source)
                                .build(),
                        )),
                    },
                    Err(error) => Err(gpu_helpers::reject_gpu_output(handle, error)),
                }
            }
        }
    }
}

fn admit_direct_gpu_output(
    handle: GpuTensorHandle,
    require_complex: bool,
) -> crate::BuiltinResult<Value> {
    if require_complex && runmat_accelerate_api::handle_is_logical(&handle) {
        return Err(gpu_helpers::reject_gpu_output(
            handle,
            builtin_error("meshgrid: complex provider output cannot be logical"),
        ));
    }
    if let Err(source) = handle.allocation_lease().validate_live() {
        return Err(gpu_helpers::reject_gpu_output(
            handle,
            build_runtime_error("meshgrid: provider output is no longer live")
                .with_builtin("meshgrid")
                .with_source(source)
                .build(),
        ));
    }
    runmat_accelerate_api::mark_residency(&handle);
    Ok(Value::GpuTensor(handle))
}

impl MeshgridEval {
    pub(super) async fn into_first(self) -> crate::BuiltinResult<Value> {
        let mut remaining = self.outputs.into_iter();
        let first = remaining.next().expect("meshgrid has a first output");
        let value = match first
            .into_value(self.target_class, self.target_residency)
            .await
        {
            Ok(value) => value,
            Err(error) => return Err(reject_meshgrid_outputs(remaining, error)),
        };
        while let Some(output) = remaining.next() {
            if let Err(error) = release_meshgrid_output(output) {
                let error = reject_meshgrid_value(value, error);
                return Err(reject_meshgrid_outputs(remaining, error));
            }
        }
        Ok(value)
    }

    pub(super) async fn into_output_list(self, requested: usize) -> crate::BuiltinResult<Value> {
        if requested > self.outputs.len() {
            let message = if self.outputs.len() == 2 {
                "meshgrid with two inputs supports at most two outputs"
            } else {
                "meshgrid supports at most three outputs"
            };
            return Err(reject_meshgrid_outputs(
                self.outputs,
                builtin_error(message),
            ));
        }
        let mut values = Vec::with_capacity(requested);
        let mut remaining = self.outputs.into_iter().enumerate();
        while let Some((index, output)) = remaining.next() {
            if index < requested {
                match output
                    .into_value(self.target_class, self.target_residency)
                    .await
                {
                    Ok(value) => values.push(value),
                    Err(error) => {
                        let error = reject_meshgrid_values(values, error);
                        return Err(reject_meshgrid_outputs(
                            remaining.map(|(_, output)| output),
                            error,
                        ));
                    }
                }
            } else if let Err(error) = release_meshgrid_output(output) {
                let error = reject_meshgrid_values(values, error);
                return Err(reject_meshgrid_outputs(
                    remaining.map(|(_, output)| output),
                    error,
                ));
            }
        }
        Ok(Value::output_list(values))
    }
}

fn release_meshgrid_output(output: MeshgridOutput) -> crate::BuiltinResult<()> {
    match output {
        MeshgridOutput::Host(_) => Ok(()),
        MeshgridOutput::Gpu(handle) => handle.release().map(|_| ()).map_err(|source| {
            build_runtime_error("meshgrid: unrequested provider output retirement failed")
                .with_builtin("meshgrid")
                .with_source(source)
                .build()
        }),
    }
}

fn reject_meshgrid_outputs(
    outputs: impl IntoIterator<Item = MeshgridOutput>,
    error: crate::RuntimeError,
) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_outputs(
        outputs.into_iter().filter_map(|output| match output {
            MeshgridOutput::Gpu(handle) => Some(handle),
            MeshgridOutput::Host(_) => None,
        }),
        error,
    )
}

fn reject_meshgrid_values(values: Vec<Value>, error: crate::RuntimeError) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_outputs(
        values.into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }),
        error,
    )
}

fn reject_meshgrid_value(value: Value, error: crate::RuntimeError) -> crate::RuntimeError {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::reject_gpu_output(handle, error),
        _ => error,
    }
}

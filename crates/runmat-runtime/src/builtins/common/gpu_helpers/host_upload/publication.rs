//! Publication and owning-result validation for selected uploads.

use super::*;

impl<'a> GpuHostUpload<'a> {
    pub(crate) fn publish_complex(
        &mut self,
        tensor: &ComplexTensor,
        shape: &[usize],
    ) -> BuiltinResult<GpuTensorHandle> {
        let source = self.source;
        self.publish_complex_protecting(tensor, shape, &[source])
    }

    pub(crate) fn publish_complex_protecting(
        &mut self,
        tensor: &ComplexTensor,
        shape: &[usize],
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<GpuTensorHandle> {
        let lanes = tensor
            .len()
            .checked_mul(2)
            .ok_or_else(|| error(self.label, "complex upload lane extent overflow"))?;
        match tensor.numeric_dtype() {
            NumericDType::F64 => {
                if let Some(mut workspace) = self.sample_workspace(NumericDType::F64, lanes)? {
                    let values = workspace
                        .as_f64_slice_mut()
                        .expect("admitted complex lanes");
                    for (index, pair) in tensor
                        .as_f64_slice()
                        .expect("double complex tensor")
                        .iter()
                        .enumerate()
                    {
                        if index.is_multiple_of(4096) {
                            self.checkpoint()?;
                        }
                        values[2 * index] = pair.0;
                        values[2 * index + 1] = pair.1;
                    }
                    return self.publish_protecting(
                        &HostNumericTensorView {
                            data: HostNumericDataView::F64(values),
                            shape,
                            storage: GpuTensorStorage::ComplexInterleaved,
                        },
                        protected,
                    );
                }
                let values = gpu_helpers::complex_transfer::interleave(
                    tensor.as_f64_slice().expect("double complex tensor"),
                )
                .map_err(|source| error(self.label, source))?;
                self.publish_protecting(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F64(&values),
                        shape,
                        storage: GpuTensorStorage::ComplexInterleaved,
                    },
                    protected,
                )
            }
            NumericDType::F32 => {
                if let Some(mut workspace) = self.sample_workspace(NumericDType::F32, lanes)? {
                    let values = workspace
                        .as_f32_slice_mut()
                        .expect("admitted complex lanes");
                    for (index, pair) in tensor
                        .as_f32_slice()
                        .expect("single complex tensor")
                        .iter()
                        .enumerate()
                    {
                        if index.is_multiple_of(4096) {
                            self.checkpoint()?;
                        }
                        values[2 * index] = pair.0;
                        values[2 * index + 1] = pair.1;
                    }
                    return self.publish_protecting(
                        &HostNumericTensorView {
                            data: HostNumericDataView::F32(values),
                            shape,
                            storage: GpuTensorStorage::ComplexInterleaved,
                        },
                        protected,
                    );
                }
                let values = gpu_helpers::complex_transfer::interleave(
                    tensor.as_f32_slice().expect("single complex tensor"),
                )
                .map_err(|source| error(self.label, source))?;
                self.publish_protecting(
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(&values),
                        shape,
                        storage: GpuTensorStorage::ComplexInterleaved,
                    },
                    protected,
                )
            }
            _ => Err(error(self.label, "complex integer upload is unsupported")),
        }
    }

    /// Caller declares this after the plan so its native sample buffer drops
    /// first. This function does not consume/refund the plan before that drop.
    pub(crate) fn publish(
        &mut self,
        view: &HostNumericTensorView<'_>,
    ) -> BuiltinResult<GpuTensorHandle> {
        let source = self.source;
        self.publish_protecting(view, &[source])
    }

    /// `protected` contains the source and every still-live sibling result.
    pub(crate) fn publish_protecting(
        &mut self,
        view: &HostNumericTensorView<'_>,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<GpuTensorHandle> {
        if !protected
            .iter()
            .any(|handle| handle.shares_allocation_with(self.source))
        {
            return Err(error(self.label, "native upload source is not protected"));
        }
        self.check_source_before_upload()?;
        if view.shape != self.shape
            || self.class
                != (HostUploadClass::Numeric {
                    element: view.data.element_type(),
                    storage: view.storage,
                })
        {
            return Err(error(
                self.label,
                format!("{}: native result contract mismatch", self.label),
            ));
        }
        let recorder = self
            .selection
            .as_ref()
            .map(HostPresentationReceipt::recorder);
        let upload =
            runmat_accelerate_api::observation::with_provider_observation(recorder, || {
                upload_numeric_protecting_admitted(
                    self.provider,
                    view,
                    protected,
                    self.permit.as_deref(),
                )
            });
        let output = match upload {
            Ok(output) => output,
            Err(source) => {
                if let Some(selected) = self.selection.take() {
                    selected.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed));
                }
                return Err(crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!("{}: native upload failed: {source}", self.label),
                    ),
                    source.into_execution_error(),
                ));
            }
        };
        self.adopt_uploaded_output(output)
    }

    pub(crate) fn publish_logical(
        &mut self,
        view: &HostLogicalTensorView<'_>,
    ) -> BuiltinResult<GpuTensorHandle> {
        let source = self.source;
        self.publish_logical_protecting(view, &[source])
    }

    pub(crate) fn publish_logical_protecting(
        &mut self,
        view: &HostLogicalTensorView<'_>,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<GpuTensorHandle> {
        self.check_source_before_upload()?;
        if self.class != HostUploadClass::Logical || view.shape != self.shape {
            return Err(error(self.label, "logical upload result contract mismatch"));
        }
        let dtype = match self.provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
            runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
        };
        let mut workspace = self.sample_workspace(dtype, view.data.len())?;
        let scratch = workspace.as_mut().map(|workspace| match dtype {
            NumericDType::F32 => HostLogicalUploadScratch::F32(
                workspace
                    .as_f32_slice_mut()
                    .expect("admitted logical F32 workspace"),
            ),
            NumericDType::F64 => HostLogicalUploadScratch::F64(
                workspace
                    .as_f64_slice_mut()
                    .expect("admitted logical F64 workspace"),
            ),
            _ => unreachable!("provider floating precision"),
        });
        let recorder = self
            .selection
            .as_ref()
            .map(HostPresentationReceipt::recorder);
        let upload =
            runmat_accelerate_api::observation::with_provider_observation(recorder, || {
                upload_logical_protecting_admitted(
                    self.provider,
                    view,
                    protected,
                    self.permit.as_deref(),
                    scratch,
                )
            });
        let output = match upload {
            Ok(output) => output,
            Err(source) => {
                if let Some(selected) = self.selection.take() {
                    selected.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed));
                }
                return Err(crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!("{}: logical upload failed: {source}", self.label),
                    ),
                    source.into_execution_error(),
                ));
            }
        };
        self.adopt_uploaded_output(output)
    }

    fn check_source_before_upload(&self) -> BuiltinResult<()> {
        self.checkpoint()?;
        self.source
            .allocation_lease()
            .validate_live()
            .map_err(|cause| {
                crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!(
                            "{}: GPU prototype retired before upload: {cause}",
                            self.label
                        ),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                        .with_unproven_effects(),
                )
            })
    }

    fn adopt_uploaded_output(
        &mut self,
        mut output: GpuTensorHandle,
    ) -> BuiltinResult<GpuTensorHandle> {
        let valid_class = match self.class {
            HostUploadClass::Numeric { element, storage } => {
                !runmat_accelerate_api::handle_is_logical(&output)
                    && runmat_accelerate_api::handle_class_identity(&output)
                        == Some(element.class_identity())
                    && output.physical().storage() == storage
            }
            HostUploadClass::Logical => {
                runmat_accelerate_api::handle_is_logical(&output)
                    && output.physical().storage() == GpuTensorStorage::Real
            }
        };
        if !valid_class
            || !gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, self.provider)
            })
            || output.physical().requested_bytes() != self.physical_bytes
        {
            self.reject_selected_publication();
            return Err(gpu_helpers::reject_gpu_output(
                output,
                error(
                    self.label,
                    format!(
                        "{}: uploaded result owner/class/physical bound mismatch",
                        self.label
                    ),
                ),
            ));
        }
        if let Err(cause) = gpu_helpers::propagate_output_provenance(&mut output, [self.source]) {
            self.reject_selected_publication();
            return Err(gpu_helpers::reject_gpu_output(output, cause));
        }
        if let Err(cause) = self.source.allocation_lease().validate_live() {
            self.reject_selected_publication();
            return Err(gpu_helpers::reject_gpu_output(
                output,
                crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!(
                            "{}: GPU prototype retired during upload: {cause}",
                            self.label
                        ),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                        .with_unproven_effects(),
                ),
            ));
        }
        if let Err(cause) = self.checkpoint() {
            self.reject_selected_publication();
            return Err(gpu_helpers::reject_gpu_output(output, cause));
        }
        if let Some(mut selected) = self.selection.take() {
            if let Err(cause) = selected.validate_published_outputs(&[&output]) {
                selected.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed));
                return Err(gpu_helpers::reject_gpu_output(output, cause));
            }
            selected.finish(Ok(true));
        }
        Ok(output)
    }
}

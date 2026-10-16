use super::*;

macro_rules! unary {
    ($hook:ident) => {
        fn $hook<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { self.fail(stringify!($hook)) })
        }
    };
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "math boundary source witness".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        assert!(
            matches!(self.failure, "upload_x" | "upload_query" | "interp1"),
            "terminal hook failure must not upload"
        );
        let hook = if self.trace.lock().unwrap().is_empty() {
            "upload_x"
        } else {
            "upload_query"
        };
        if hook == self.failure {
            return self.fail(hook);
        }
        self.trace.lock().unwrap().push(hook);
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(view.data.to_vec()),
            shape: view.shape.to_vec(),
            storage: GpuTensorStorage::Real,
        })?;
        self.probes
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe));
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostTensorOwned {
                data: self.read_probe(handle)?,
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(self.read_probe(handle)?),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "reduce_min" {
                return self.fail("reduce_min").map_err(Into::into);
            }
            self.trace.lock().unwrap().push("reduce_min");
            let result = self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })?;
            self.probes
                .lock()
                .unwrap()
                .push((result.handle.buffer_id(), result.probe));
            Ok(result.handle)
        })
    }
    unary!(unary_erf);
    unary!(unary_erfcinv);
    unary!(unary_factorial);
    unary!(unary_gamma);
    unary!(unary_gammaln);
    unary!(unary_sqrt);
    unary!(unary_pow2);
    unary!(unary_nextpow2);
    unary!(unary_log);
    unary!(unary_log2);
    unary!(unary_log10);
    unary!(unary_log1p);
    unary!(unary_real);
    unary!(unary_imag);
    fn unary_conj<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.fail("unary_conj").map(Some).map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })
        })
    }
    unary!(complex_from_real);
    fn interp1<'a>(
        &'a self,
        _: &'a ProviderInterp1Request<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("interp1") })
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_sin", input) })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_cos", input) })
    }
    fn unary_tan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_tan", input) })
    }
    fn reduce_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("reduce_max").map_err(Into::into) })
    }
    fn transpose(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        if self.failure == "transpose" {
            return self.fail("transpose");
        }
        assert_eq!(self.failure, "unary_conj");
        self.trace.lock().unwrap().push("transpose");
        let mut shape = input.shape().to_vec();
        shape.swap(0, 1);
        self.allocate(shape, handle_storage(input))
    }
    fn permute(
        &self,
        _: &GpuTensorHandle,
        _: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = self.fail("permute");
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn scalar_mul(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fail("scalar_mul")
    }
    fn matmul<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("matmul") })
    }
    fn dot<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("dot") })
    }
    fn cross(
        &self,
        _: &GpuTensorHandle,
        _: &GpuTensorHandle,
        _: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.fail("cross")
    }
    fn cond<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: ProviderCondNorm,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "cond_after_unsupported" {
                self.trace.lock().unwrap().push("cond");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "cond became unavailable after staging",
                ))
                .after_effect());
            }
            if self.failure == "cond_malformed" || self.failure == "cond_valid" {
                self.trace.lock().unwrap().push("cond");
                let shape = if self.failure == "cond_valid" {
                    vec![1, 1]
                } else {
                    vec![1, 2]
                };
                return self
                    .allocate(shape, GpuTensorStorage::Real)
                    .map_err(|source| ProviderExecutionError::from(source).after_effect());
            }
            self.fail("cond")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn inv<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: ProviderInvOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "inv_after_unsupported" {
                self.trace.lock().unwrap().push("inv");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "inv became unavailable after staging",
                ))
                .after_effect());
            }
            self.fail("inv")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn pinv<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: ProviderPinvOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "pinv_after_unsupported" {
                self.trace.lock().unwrap().push("pinv");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "pinv became unavailable after staging",
                ))
                .after_effect());
            }
            self.fail("pinv")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn norm<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: ProviderNormOrder,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "norm_after_unsupported" {
                self.trace.lock().unwrap().push("norm");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "norm became unavailable after staging",
                ))
                .after_effect());
            }
            self.fail("norm")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn rcond<'a>(&'a self, _: &'a GpuTensorHandle) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "rcond_after_unsupported" {
                self.trace.lock().unwrap().push("rcond");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "rcond became unavailable after staging",
                ))
                .after_effect());
            }
            self.fail("rcond")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn rank<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: Option<f64>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "rank_after_unsupported" {
                self.trace.lock().unwrap().push("rank");
                return Err(ProviderExecutionError::from(unsupported_provider_operation(
                    "rank became unavailable after staging",
                ))
                .after_effect());
            }
            if self.failure == "rank_malformed" || self.failure == "rank_valid" {
                self.trace.lock().unwrap().push("rank");
                let shape = if self.failure == "rank_valid" {
                    vec![1, 1]
                } else {
                    vec![1, 2]
                };
                return self
                    .allocate(shape, GpuTensorStorage::Real)
                    .map_err(|source| ProviderExecutionError::from(source).after_effect());
            }
            self.fail("rank")
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    fn complex_from_real_imag<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("complex_from_real_imag") })
    }
}

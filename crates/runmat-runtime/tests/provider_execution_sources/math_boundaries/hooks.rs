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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.failure == "reduce_min" {
                return self.fail("reduce_min");
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
    unary!(unary_conj);
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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("reduce_max") })
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
    fn permute(&self, _: &GpuTensorHandle, _: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.fail("permute")
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
    fn complex_from_real_imag<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("complex_from_real_imag") })
    }
}

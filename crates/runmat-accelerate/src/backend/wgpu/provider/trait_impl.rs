use super::*;

impl AccelProvider for WgpuProvider {
    fn export_context(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        match kind {
            AccelContextKind::Plotting => Some(AccelContextHandle::Wgpu(WgpuContextHandle {
                instance: self.instance.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                adapter: self.adapter.clone(),
                adapter_info: self.adapter_info.clone(),
                limits: self.adapter_limits.clone(),
                features: self.device.features(),
            })),
        }
    }

    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(&self, handle: &GpuTensorHandle) -> Option<WgpuBufferRef> {
        self.get_entry(handle).ok().map(|entry| WgpuBufferRef {
            buffer: entry.buffer,
            len: entry.len,
            shape: entry.shape,
            element_size: self.element_size,
            precision: match entry.precision {
                NumericPrecision::F32 => ProviderPrecision::F32,
                NumericPrecision::F64 => ProviderPrecision::F64,
            },
        })
    }

    fn device_id(&self) -> u32 {
        self.runtime_device_id
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        SpawnHandleConcurrency::SynchronizedMutation
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.gather_linear_exec(source, indices, output_shape)
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        self.scatter_linear_exec(target, indices, values)
    }

    fn zeros(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.zeros_exec(shape)
    }

    fn zeros_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.zeros_exec(&prototype.shape)
    }
    fn precision(&self) -> ProviderPrecision {
        match self.precision {
            NumericPrecision::F32 => ProviderPrecision::F32,
            NumericPrecision::F64 => ProviderPrecision::F64,
        }
    }

    fn fill(&self, shape: &[usize], value: f64) -> Result<GpuTensorHandle> {
        self.fill_exec(shape, value)
    }

    fn fill_like(&self, prototype: &GpuTensorHandle, value: f64) -> Result<GpuTensorHandle> {
        self.fill_exec(&prototype.shape, value)
    }

    fn eye(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.eye_exec(shape)
    }

    fn eye_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.eye_exec(&prototype.shape)
    }

    fn meshgrid(&self, axes: &[MeshgridAxisView<'_>]) -> Result<ProviderMeshgridResult> {
        self.meshgrid_exec(axes)
    }

    fn linspace(&self, start: f64, stop: f64, count: usize) -> Result<GpuTensorHandle> {
        self.linspace_exec(start, stop, count)
    }

    fn random_uniform(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_uniform_exec(shape)
    }

    fn random_normal(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normal_exec(shape)
    }

    fn random_exponential(&self, mu: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_exponential_exec(mu, shape)
    }

    fn random_normrnd(&self, mu: f64, sigma: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normrnd_exec(mu, sigma, shape)
    }

    fn random_unifrnd(&self, a: f64, b: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_unifrnd_exec(a, b, shape)
    }

    fn stochastic_evolution(
        &self,
        state: &GpuTensorHandle,
        drift: f64,
        scale: f64,
        steps: u32,
    ) -> Result<GpuTensorHandle> {
        self.stochastic_evolution_exec(state, drift, scale, steps)
    }

    fn fspecial(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        self.fspecial_exec(request)
    }

    fn peaks(&self, n: usize) -> Result<GpuTensorHandle> {
        self.peaks_exec(n)
    }

    fn peaks_xy(&self, x: &GpuTensorHandle, y: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.peaks_xy_exec(x, y)
    }

    fn hann_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hann, len, periodic)
    }

    fn hamming_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hamming, len, periodic)
    }

    fn blackman_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Blackman, len, periodic)
    }

    fn imfilter<'a>(
        &'a self,
        image: &'a GpuTensorHandle,
        kernel: &'a GpuTensorHandle,
        options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.imfilter_exec(image, kernel, options).await })
    }

    fn random_integer_range(
        &self,
        lower: i64,
        upper: i64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.random_integer_range_exec(lower, upper, shape)
    }

    fn set_rng_state(&self, state: u64) -> Result<()> {
        let mut guard = rng_state()
            .lock()
            .map_err(|_| anyhow::anyhow!("wgpu provider: RNG mutex poisoned"))?;
        *guard = if state == 0 { RNG_DEFAULT_SEED } else { state };
        Ok(())
    }

    fn random_permutation(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        self.randperm_exec(n, k)
    }

    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> Result<GpuTensorHandle> {
        self.randperm_exec(n, k)
    }

    fn polyval(
        &self,
        coeffs: &GpuTensorHandle,
        points: &GpuTensorHandle,
        options: &ProviderPolyvalOptions,
    ) -> Result<GpuTensorHandle> {
        self.polyval_exec(coeffs, points, options)
    }

    fn polyfit<'a>(
        &'a self,
        x: &'a GpuTensorHandle,
        y: &'a GpuTensorHandle,
        degree: usize,
        weights: Option<&'a GpuTensorHandle>,
    ) -> AccelProviderFuture<'a, ProviderPolyfitResult> {
        Box::pin(async move {
            let x_host = <Self as AccelProvider>::download(self, x).await?;
            let y_host = <Self as AccelProvider>::download(self, y).await?;
            ensure!(
                x_host.data.len() == y_host.data.len(),
                "polyfit: X and Y vectors must match in length"
            );
            let weights_host = match weights {
                Some(handle) => Some(<Self as AccelProvider>::download(self, handle).await?),
                None => None,
            };
            let weights_slice = weights_host.as_ref().map(|w| w.data.as_slice());
            let host_result =
                polyfit_host_real_for_provider(&x_host.data, &y_host.data, degree, weights_slice)
                    .map_err(|err| anyhow!(err))?;
            Ok(ProviderPolyfitResult {
                coefficients: host_result.coefficients,
                r_matrix: host_result.r_matrix,
                normr: host_result.normr,
                df: host_result.df,
                mu: host_result.mu,
            })
        })
    }

    fn polyder_single<'a>(
        &'a self,
        polynomial: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_exec(polynomial).await })
    }

    fn polyder_product<'a>(
        &'a self,
        p: &'a GpuTensorHandle,
        q: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_product_exec(p, q).await })
    }

    fn polyder_quotient<'a>(
        &'a self,
        u: &'a GpuTensorHandle,
        v: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, ProviderPolyderQuotient> {
        Box::pin(async move { self.polyder_quotient_exec(u, v).await })
    }

    fn polyint(&self, polynomial: &GpuTensorHandle, constant: f64) -> Result<GpuTensorHandle> {
        self.polyint_exec(polynomial, constant)
    }

    fn diag_from_vector(&self, vector: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_from_vector_exec(vector, offset)
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_extract_exec(matrix, offset)
    }

    fn tril<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.tril_exec(matrix, offset).await })
    }

    fn triu<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.triu_exec(matrix, offset).await })
    }

    fn reduce_mean_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_nd_mean_exec(a, dims_zero_based).await })
    }

    fn reduce_moments_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::ProviderMoments2> {
        Box::pin(async move { self.reduce_moments_nd_exec(a, dims_zero_based) })
    }

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Add, a, b)
        })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, b)
        })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, a, b)
        })
    }

    fn elem_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Max, a, b)
        })
    }

    fn elem_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Min, a, b)
        })
    }

    fn elem_div<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Div, a, b)
        })
    }

    fn elem_pow<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Pow, a, b)
        })
    }

    fn elem_ge<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ge_exec(a, b) })
    }

    fn elem_le<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_le_exec(a, b) })
    }

    fn elem_lt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_lt_exec(a, b) })
    }

    fn elem_gt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_gt_exec(a, b) })
    }

    fn elem_eq<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_eq_exec(a, b) })
    }

    fn elem_ne<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ne_exec(a, b) })
    }

    fn logical_and(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_and_exec(a, b)
    }

    fn logical_or(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_or_exec(a, b)
    }

    fn logical_xor(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_xor_exec(a, b)
    }

    fn logical_not(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_not_exec(a)
    }

    fn logical_islogical(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(runmat_accelerate_api::handle_is_logical(a))
    }

    fn logical_isreal(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(true)
    }

    fn logical_isfinite(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isfinite_exec(a)
    }

    fn logical_isnan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isnan_exec(a)
    }

    fn logical_isinf(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isinf_exec(a)
    }

    fn elem_hypot<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Hypot, a, b)
        })
    }

    fn elem_atan2<'a>(
        &'a self,
        y: &'a GpuTensorHandle,
        x: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Atan2, y, x)
        })
    }

    fn unary_sin<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sin, a) },
        )
    }

    fn unary_sinc<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinc, a) },
        )
    }

    fn unary_gamma<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Gamma, a) },
        )
    }

    fn unary_factorial<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Factorial, a)
        })
    }

    fn unary_asinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asinh, a) },
        )
    }

    fn unary_sinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinh, a) },
        )
    }

    fn unary_cosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cosh, a) },
        )
    }

    fn unary_asin<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asin, a) },
        )
    }

    fn unary_acos<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acos, a) },
        )
    }

    fn unary_acosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acosh, a) },
        )
    }

    fn unary_tan<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tan, a) },
        )
    }

    fn unary_tanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tanh, a) },
        )
    }

    fn unary_atan<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atan, a) },
        )
    }
    fn unary_atanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atanh, a) },
        )
    }

    fn unary_ceil<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Ceil, a) },
        )
    }

    fn unary_floor<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Floor, a) },
        )
    }

    fn unary_fix<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Fix, a) },
        )
    }

    fn unary_cos<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cos, a) },
        )
    }

    fn unary_abs<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a) },
        )
    }

    fn unary_conj<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Conj, a) },
        )
    }

    fn unary_exp<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Exp, a) },
        )
    }

    fn unary_log<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log, a) },
        )
    }

    fn unary_log1p<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log1p, a) },
        )
    }

    fn unary_sqrt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, a) },
        )
    }

    fn unary_double<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.precision != NumericPrecision::F64 {
                return Err(anyhow!(
                    "wgpu provider: shader-f64 unavailable; cannot materialise double precision"
                ));
            }
            let entry = self.get_entry(a)?;
            Ok(self.register_existing_buffer(entry.buffer, entry.shape, entry.len))
        })
    }

    fn unary_single<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Single, a) },
        )
    }

    fn unary_pow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let out = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, a)?;
            // Record squared->base mapping for later reduction fusion (moments reuse)
            if let Ok(mut map) = self.pow2_of.lock() {
                map.insert(out.buffer_id, a.buffer_id);
            }
            Ok(out)
        })
    }

    fn unary_nextpow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::NextPow2, a)
        })
    }

    fn pow2_scale(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if mantissa.shape != exponent.shape {
            return Err(anyhow!("pow2_scale requires matching shapes"));
        }
        let pow = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, exponent)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            mantissa,
            &pow,
        );
        let _ = self.free(&pow);
        result
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RSub, a, scalar)
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RDiv, a, scalar)
    }

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Add, a, scalar)
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Sub, a, scalar)
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Mul, a, scalar)
    }

    fn scalar_max(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Max, a, scalar)
    }

    fn scalar_min(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Min, a, scalar)
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Div, a, scalar)
    }

    fn sort_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        order: SortOrder,
        comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let shape = host.shape.clone();
            let (values, indices) =
                sort_host_tensor(&host.data, &host.shape, dim, order, comparison)?;
            Ok(SortResult {
                values: HostTensorOwned {
                    data: values,
                    shape: shape.clone(),
                    storage: GpuTensorStorage::Real,
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            })
        })
    }
    fn sort_rows<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        columns: &'a [SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let SortRowsHostOutputs {
                values,
                indices,
                indices_shape,
            } = sort_rows_host(&host.data, &host.shape, columns, comparison)?;
            Ok(SortResult {
                values: HostTensorOwned {
                    data: values,
                    shape: host.shape.clone(),
                    storage: GpuTensorStorage::Real,
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape: indices_shape,
                    storage: GpuTensorStorage::Real,
                },
            })
        })
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.transpose_exec(a)
    }

    fn permute(&self, handle: &GpuTensorHandle, order: &[usize]) -> Result<GpuTensorHandle> {
        self.permute_exec(handle, order)
    }

    fn flip(&self, handle: &GpuTensorHandle, axes: &[usize]) -> Result<GpuTensorHandle> {
        self.flip_exec(handle, axes)
    }

    fn conv1d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        self.conv1d_exec(signal, kernel, options)
    }
    fn iir_filter<'a>(
        &'a self,
        b: &'a GpuTensorHandle,
        a: &'a GpuTensorHandle,
        x: &'a GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async move { self.iir_filter_exec(b, a, x, options).await })
    }
    fn conv2d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        Err(anyhow!("conv2d not implemented for the WGPU provider yet"))
    }

    fn diff_dim(
        &self,
        handle: &GpuTensorHandle,
        order: usize,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.diff_exec(handle, dim, order)
    }

    fn gradient_dim(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> Result<GpuTensorHandle> {
        self.gradient_exec(handle, dim, spacing)
    }

    fn cumsum_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.cumsum_exec(input, dim, direction, nan_mode)
    }

    fn cumprod_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.cumprod_exec(input, dim, direction, nan_mode)
    }

    fn cummin_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<runmat_accelerate_api::ProviderCumminResult> {
        self.cummin_exec(input, dim, direction, nan_mode)
    }

    fn cummax_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<runmat_accelerate_api::ProviderCummaxResult> {
        self.cummax_exec(input, dim, direction, nan_mode)
    }

    fn circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> Result<GpuTensorHandle> {
        self.circshift_exec(handle, shifts)
    }

    fn fft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_dim_exec(handle, len, dim).await })
    }

    fn ifft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.ifft_dim_exec(handle, len, dim).await })
    }

    fn fft_extract_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_extract_real_exec(handle) })
    }

    fn unique<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: &'a UniqueOptions,
    ) -> AccelProviderFuture<'a, UniqueResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let HostTensorOwned { data, shape, .. } = host;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("unique: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
                tensor, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("unique: {err}"));
                }
            };
            match eval.into_numeric_unique_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("unique: {err}")),
            }
        })
    }
    fn ismember<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a IsMemberOptions,
    ) -> AccelProviderFuture<'a, IsMemberResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("ismember: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("ismember: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::ismember::ismember_numeric_from_tensors(
                tensor_a,
                tensor_b,
                options.rows,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("ismember: {err}"));
                }
            };
            match eval.into_numeric_ismember_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("ismember: {err}")),
            }
        })
    }

    fn union<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a UnionOptions,
    ) -> AccelProviderFuture<'a, UnionResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("union: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("union: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::union::union_numeric_from_tensors(
                tensor_a, tensor_b, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("union: {err}"));
                }
            };
            match eval.into_numeric_union_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("union: {err}")),
            }
        })
    }
    fn setdiff<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a SetdiffOptions,
    ) -> AccelProviderFuture<'a, SetdiffResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::setdiff::setdiff_numeric_from_tensors(
                tensor_a, tensor_b, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("setdiff: {err}"));
                }
            };
            match eval.into_numeric_setdiff_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("setdiff: {err}")),
            }
        })
    }

    fn cat(&self, dim: usize, inputs: &[GpuTensorHandle]) -> Result<GpuTensorHandle> {
        self.cat_exec(dim, inputs)
    }

    fn repmat(&self, handle: &GpuTensorHandle, reps: &[usize]) -> Result<GpuTensorHandle> {
        self.repmat_exec(handle, reps)
    }

    fn kron(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.kron_exec(a, b)
    }
    fn cross(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        self.cross_exec(lhs, rhs, dim)
    }
    fn reshape(&self, handle: &GpuTensorHandle, new_shape: &[usize]) -> Result<GpuTensorHandle> {
        let new_len = if new_shape.is_empty() {
            1
        } else {
            product_checked(new_shape)
                .ok_or_else(|| anyhow!("reshape: dimension product exceeds GPU limits"))?
        };
        let mut buffers = self.buffers.lock().expect("buffer mutex poisoned");
        let entry = buffers
            .get_mut(&handle.buffer_id)
            .ok_or_else(|| anyhow!("reshape: unknown buffer {}", handle.buffer_id))?;
        ensure!(
            entry.len == new_len,
            "reshape: product of dimensions ({}) must equal original tensor length ({})",
            new_len,
            entry.len
        );
        entry.shape = new_shape.to_vec();
        let mut updated = handle.clone();
        updated.shape = new_shape.to_vec();
        Ok(updated)
    }

    fn lu<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let LuHostFactors {
                combined,
                lower,
                upper,
                perm_matrix,
                pivot_vector,
                combined_shape,
                lower_shape,
                upper_shape,
                perm_shape,
                pivot_shape,
            } = lu_factor_host(&host.data, &host.shape)?;
            let combined = self.upload(&HostTensorView {
                data: &combined,
                shape: &combined_shape,
            })?;
            let lower = self.upload(&HostTensorView {
                data: &lower,
                shape: &lower_shape,
            })?;
            let upper = self.upload(&HostTensorView {
                data: &upper,
                shape: &upper_shape,
            })?;
            let perm_matrix = self.upload(&HostTensorView {
                data: &perm_matrix,
                shape: &perm_shape,
            })?;
            let perm_vector = self.upload(&HostTensorView {
                data: &pivot_vector,
                shape: &pivot_shape,
            })?;
            Ok(ProviderLuResult {
                combined,
                lower,
                upper,
                perm_matrix,
                perm_vector,
            })
        })
    }

    fn chol<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        lower: bool,
    ) -> AccelProviderFuture<'a, ProviderCholResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("chol: {e}"))?;
            let mut args = Vec::new();
            if lower {
                args.push(Value::from("lower"));
            }
            let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
                Value::Tensor(tensor),
                &args,
            )
            .await
            .map_err(|err| runtime_flow_to_anyhow("chol", err))?;
            let factor_tensor = host_tensor_from_value("chol", eval.factor())?;
            let factor = self.upload(&HostTensorView {
                data: &factor_tensor.data,
                shape: &factor_tensor.shape,
            })?;
            Ok(ProviderCholResult {
                factor,
                info: eval.flag_index() as u32,
            })
        })
    }
    fn qr<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> AccelProviderFuture<'a, ProviderQrResult> {
        Box::pin(async move {
            if let Some(result) = self.try_qr_device(handle, &options)? {
                return Ok(result);
            }
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("qr: {e}"))?;
            self.qr_host_result(tensor, &options).await
        })
    }

    fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        let res = self.kernel_resources.take_matmul_sources(product);
        log::debug!(
            "take_matmul_sources: product={} found={} active_fusion={:?}",
            product.buffer_id,
            res.is_some(),
            active_fusion()
        );
        res
    }

    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> AccelProviderFuture<'a, Option<ProviderQrPowerIterResult>> {
        Box::pin(async move {
            let debug_qr = std::env::var("RUNMAT_DEBUG_QR").is_ok();
            if !options.economy {
                return Ok(None);
            }

            let product_entry = self.get_entry(product)?;
            if product_entry.shape.len() != 2 {
                return Ok(None);
            }
            let rows = product_entry.shape[0];
            let cols = product_entry.shape[1];
            if rows == 0 || cols == 0 {
                return Ok(None);
            }
            if cols > QR_DEVICE_MAX_COLS {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: column count {} exceeds device kernel limit {}; falling back",
                        cols,
                        QR_DEVICE_MAX_COLS
                    );
                }
                return Ok(None);
            }
            if self.precision() != ProviderPrecision::F32 {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: precision {:?} unsupported for device QR kernel; falling back",
                        self.precision()
                    );
                }
                return Ok(None);
            }
            let q_entry = self.get_entry(q_handle)?;
            if q_entry.shape != product_entry.shape {
                return Ok(None);
            }
            let k = cols;

            let mut pre_product_max = match <Self as AccelProvider>::download(self, product).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter pre-download failed: {err}");
                    None
                }
            };

            let pre_q_max = match <Self as AccelProvider>::download(self, q_handle).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter q-handle pre-download failed: {err}");
                    None
                }
            };

            const PRODUCT_EPS: f64 = 1.0e-12;
            const Q_EPS: f64 = 1.0e-6;
            if pre_product_max.unwrap_or(0.0) <= PRODUCT_EPS && pre_q_max.unwrap_or(0.0) > Q_EPS {
                let debug_zero_host = std::env::var("RUNMAT_DEBUG_QR_ZEROHOST").is_ok();
                if debug_zero_host {
                    if let Some(lhs_handle) = product_lhs {
                        let lhs_download =
                            <Self as AccelProvider>::download(self, lhs_handle).await;
                        let q_download = <Self as AccelProvider>::download(self, q_handle).await;
                        match (lhs_download, q_download) {
                            (Ok(lhs_host), Ok(q_host)) => {
                                let lhs_rows = lhs_host.shape.first().copied().unwrap_or(0);
                                let lhs_cols = lhs_host.shape.get(1).copied().unwrap_or(0);
                                let q_rows = q_host.shape.first().copied().unwrap_or(0);
                                let q_cols = q_host.shape.get(1).copied().unwrap_or(0);
                                if lhs_rows == q_rows
                                    && lhs_cols == q_rows
                                    && q_rows == rows
                                    && q_cols == cols
                                {
                                    let mut max_host_product = 0.0f64;
                                    for col in 0..cols {
                                        for row in 0..rows {
                                            let mut sum = 0.0f64;
                                            for k_idx in 0..lhs_cols {
                                                let lhs_idx = row + k_idx * lhs_rows;
                                                let q_idx = k_idx + col * q_rows;
                                                sum += lhs_host.data[lhs_idx] * q_host.data[q_idx];
                                            }
                                            max_host_product = max_host_product.max(sum.abs());
                                        }
                                    }
                                    log::info!(
                                    "qr_power_iter host check: rows={} cols={} host_max_product={:.6e}",
                                    rows,
                                    cols,
                                    max_host_product
                                );
                                } else {
                                    log::info!(
                                    "qr_power_iter host check skipped: lhs_shape={:?} q_shape={:?} rows={} cols={}",
                                    lhs_host.shape,
                                    q_host.shape,
                                    rows,
                                    cols
                                );
                                }
                            }
                            (lhs_res, q_res) => {
                                log::info!(
                                    "qr_power_iter host check download failed: lhs={:?} q={:?} product_id={}",
                                    lhs_res.err(),
                                    q_res.err(),
                                    product.buffer_id
                                );
                            }
                        }
                    } else {
                        log::info!(
                            "qr_power_iter host check skipped: product_lhs unavailable (product_id={})",
                            product.buffer_id
                        );
                    }
                }
                if let Some(lhs_handle) = product_lhs {
                    log::warn!(
                        "qr_power_iter: detected zero matmul product (product_id={} max_product_abs_pre={:?} max_q_abs_pre={:?}); recomputing",
                        product.buffer_id,
                        pre_product_max,
                        pre_q_max
                    );
                    if let Ok(lhs_entry) = self.get_entry(lhs_handle) {
                        if let Ok(rhs_entry) = self.get_entry(q_handle) {
                            let lhs_view = build_matrix_operand_view(lhs_handle, &lhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            let rhs_view = build_matrix_operand_view(q_handle, &rhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            log::info!(
                                "qr_power_iter recompute operands: product_id={} lhs_shape={:?} rhs_shape={:?} lhs_view={{rows:{} cols:{} lda:{} transpose:{}}} rhs_view={{rows:{} cols:{} lda:{} transpose:{}}}",
                                product.buffer_id,
                                lhs_entry.shape,
                                rhs_entry.shape,
                                lhs_view.rows,
                                lhs_view.cols,
                                lhs_view.lda,
                                lhs_view.transpose,
                                rhs_view.rows,
                                rhs_view.cols,
                                rhs_view.lda,
                                rhs_view.transpose
                            );
                            log::info!(
                                "qr_power_iter recompute buffers: product_id={} lhs_ptr={:p} rhs_ptr={:p}",
                                product.buffer_id,
                                lhs_entry.buffer.as_ref(),
                                rhs_entry.buffer.as_ref()
                            );
                        }
                    }
                    let recomputed = self.matmul_exec_with_usage(
                        lhs_handle,
                        q_handle,
                        BufferUsageClass::FusionOut,
                    )?;
                    let mut recomputed_max: Option<f64> = None;
                    if debug_zero_host {
                        match <Self as AccelProvider>::download(self, &recomputed).await {
                            Ok(host) => {
                                let max_recomputed = host
                                    .data
                                    .iter()
                                    .fold(0.0f64, |acc, value| acc.max(value.abs()));
                                log::info!(
                                    "qr_power_iter recompute check: product_id={} max_recomputed_abs={:.6e}",
                                    product.buffer_id,
                                    max_recomputed
                                );
                                recomputed_max = Some(max_recomputed);
                            }
                            Err(err) => {
                                log::info!(
                                    "qr_power_iter recompute check failed: product_id={} err={}",
                                    product.buffer_id,
                                    err
                                );
                            }
                        }
                    }
                    let recomputed_entry = self.get_entry(&recomputed)?;
                    log::info!(
                        "qr_power_iter recompute start: product_id={} original_len={} recomputed_len={}",
                        product.buffer_id,
                        product_entry.len,
                        recomputed_entry.len
                    );
                    let bytes = (recomputed_entry.len as u64) * self.element_size as u64;
                    log::info!(
                        "qr_power_iter recompute copy detail: product_id={} product_ptr={:p} recomputed_ptr={:p}",
                        product.buffer_id,
                        product_entry.buffer.as_ref(),
                        recomputed_entry.buffer.as_ref()
                    );
                    if bytes > 0 {
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("runmat-qr-product-recompute"),
                                });
                        encoder.copy_buffer_to_buffer(
                            recomputed_entry.buffer.as_ref(),
                            0,
                            product_entry.buffer.as_ref(),
                            0,
                            bytes,
                        );
                        self.submit(encoder);
                    }

                    let max_val = if let Some(val) = recomputed_max {
                        val
                    } else {
                        match <Self as AccelProvider>::download(self, product).await {
                            Ok(host) => host
                                .data
                                .iter()
                                .fold(0.0f64, |acc, value| acc.max(value.abs())),
                            Err(err) => {
                                log::warn!("qr_power_iter recompute verification failed: {err}");
                                0.0
                            }
                        }
                    };
                    log::info!(
                        "qr_power_iter recompute copy: product_id={} bytes={} post_max={:.6e}",
                        product.buffer_id,
                        bytes,
                        max_val
                    );
                    if max_val == 0.0 {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let q_download =
                                <Self as AccelProvider>::download(self, q_handle).await;
                            if let Ok(lhs_dump) =
                                <Self as AccelProvider>::download(self, lhs_handle).await
                            {
                                if let Ok(ref q_dump) = q_download {
                                    let dump_dir = Path::new("target/matmul_zero");
                                    let _ = fs::create_dir_all(dump_dir);
                                    let lhs_path = dump_dir.join(format!(
                                        "lhs_{}_{}.bin",
                                        product.buffer_id,
                                        lhs_dump.data.len()
                                    ));
                                    let rhs_path = dump_dir.join(format!(
                                        "rhs_{}_{}.bin",
                                        product.buffer_id,
                                        q_dump.data.len()
                                    ));
                                    let _ =
                                        fs::write(&lhs_path, cast_slice(lhs_dump.data.as_slice()));
                                    let _ =
                                        fs::write(&rhs_path, cast_slice(q_dump.data.as_slice()));
                                    log::warn!(
                                        "qr_power_iter dump written: product_id={} lhs_path={} rhs_path={}",
                                        product.buffer_id,
                                        lhs_path.display(),
                                        rhs_path.display()
                                    );
                                }
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            log::warn!("qr_power_iter: skipping matmul dump because filesystem APIs are unavailable on wasm");
                        }
                        log::warn!(
                            "qr_power_iter: recomputed product is still zero; falling back to host QR"
                        );
                        let _ = self.free(&recomputed);
                        if let Some(handle) = self.qr_power_iter_host(product, options).await? {
                            return Ok(Some(handle));
                        }
                        return Ok(None);
                    }
                    pre_product_max = Some(max_val);

                    let _ = self.free(&recomputed);
                } else {
                    log::warn!(
                        "qr_power_iter: zero product detected for buffer {} without lhs handle; proceeding with existing data",
                        product.buffer_id
                    );
                }
            }

            let (q_result, r_handle, mut r_inv_opt) = self.qr_factor_device(
                product,
                rows,
                cols,
                Some(q_handle),
                "runmat-qr-power",
                true,
            )?;

            let mut fallback_needed = false;
            if let Ok(host_r) = <Self as AccelProvider>::download(self, &r_handle).await {
                for col in 0..cols {
                    let diag = host_r.data[col + col * cols];
                    if !diag.is_finite() || diag.abs() <= 1.0e-12 {
                        fallback_needed = true;
                        break;
                    }
                }
            }

            if fallback_needed {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if pre_product_max.unwrap_or(0.0) <= 1.0e-8 {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if debug_qr {
                if let Err(err) = self
                    .debug_qr_power_iter(
                        product,
                        &product_entry,
                        pre_product_max,
                        pre_q_max,
                        &q_result,
                        &r_handle,
                        r_inv_opt
                            .as_ref()
                            .expect("retain_r_inv=true must provide inverse handle"),
                        None::<&runmat_accelerate_api::HostTensorOwned>,
                        rows,
                        cols,
                    )
                    .await
                {
                    log::warn!("qr_power_iter debug failed: {err}");
                }
            }

            if let Some(handle) = r_inv_opt.take() {
                let _ = self.free(&handle);
            }

            let mut perm_matrix = vec![0.0f64; k * k];
            for i in 0..k {
                perm_matrix[i + i * k] = 1.0;
            }
            let perm_vector: Vec<f64> = (1..=k).map(|v| v as f64).collect();

            let perm_matrix_shape = [k, k];
            let perm_matrix_handle = self.upload(&HostTensorView {
                data: &perm_matrix,
                shape: &perm_matrix_shape,
            })?;
            let perm_vector_shape = vec![k, 1];
            let perm_vector_handle = self.upload(&HostTensorView {
                data: &perm_vector,
                shape: &perm_vector_shape,
            })?;

            let _ = self.free(product);

            Ok(Some(ProviderQrPowerIterResult {
                q: q_result,
                r: r_handle,
                perm_matrix: perm_matrix_handle,
                perm_vector: perm_vector_handle,
            }))
        })
    }
    fn matmul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_exec(a, b) })
    }

    fn syrk(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.syrk_exec(a)
    }
    fn matmul_epilogue<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        ep: &'a runmat_accelerate_api::MatmulEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            use runmat_accelerate_api::ProviderPrecision;
            let entry_a = self.get_entry(a)?;
            let entry_b = self.get_entry(b)?;
            if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
                return Err(anyhow!("matmul_epilogue: only 2D tensors supported"));
            }
            let view_a = build_matrix_operand_view(a, &entry_a)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;
            let view_b = build_matrix_operand_view(b, &entry_b)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;

            if view_a.cols != view_b.rows {
                return Err(anyhow!("matmul_epilogue: inner dimensions must match"));
            }
            let m = view_a.rows;
            let n = view_b.cols;
            let k = view_a.cols;

            let out_shape = vec![m, n];
            let len = m * n;
            let out_buffer =
                self.create_storage_buffer_checked(len, "runmat-matmul-epilogue-out")?;
            if len == 0 {
                return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
            }

            let start = Instant::now();

            let m_u32 =
                u32::try_from(m).map_err(|_| anyhow!("matmul_epilogue: m exceeds GPU limits"))?;
            let n_u32 =
                u32::try_from(n).map_err(|_| anyhow!("matmul_epilogue: n exceeds GPU limits"))?;
            let k_u32 =
                u32::try_from(k).map_err(|_| anyhow!("matmul_epilogue: k exceeds GPU limits"))?;
            let mut flags = 0u32;
            if view_a.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
            }
            if view_b.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
            }

            let params = crate::backend::wgpu::params::MatmulParams {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                lda: view_a.lda,
                ldb: view_b.lda,
                ldc: m_u32,
                offset_a: 0,
                offset_b: 0,
                offset_out: 0,
                flags,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-matmul-epilogue-params");

            // Resolve optional scales and epilogue params by precision
            use crate::backend::wgpu::params::{
                MATMUL_EPILOGUE_FLAG_CLAMP_MAX, MATMUL_EPILOGUE_FLAG_CLAMP_MIN,
                MATMUL_EPILOGUE_FLAG_COL_DIV, MATMUL_EPILOGUE_FLAG_COL_SCALE,
                MATMUL_EPILOGUE_FLAG_DIAG_WRITE, MATMUL_EPILOGUE_FLAG_POW,
                MATMUL_EPILOGUE_FLAG_ROW_DIV, MATMUL_EPILOGUE_FLAG_ROW_SCALE,
            };
            let has_row = ep.row_scale.is_some();
            let has_col = ep.col_scale.is_some();
            let dummy_rowcol = self.create_storage_buffer(1, "runmat-matmul-epilogue-dummy-scale");
            let row_buf = match &ep.row_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };
            let col_buf = match &ep.col_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };

            let (diag_rows, diag_stride, diag_offset, has_diag) = match &ep.diag_output {
                Some(_) => {
                    return Err(anyhow!(
                        "matmul_epilogue: diag_output is not supported by the WGPU provider yet"
                    ));
                }
                None => (0u32, 1u32, 0u32, false),
            };

            let mut flags: u32 = 0;
            if has_row {
                flags |= MATMUL_EPILOGUE_FLAG_ROW_SCALE;
                if matches!(ep.row_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_ROW_DIV;
                }
            }
            if has_col {
                flags |= MATMUL_EPILOGUE_FLAG_COL_SCALE;
                if matches!(ep.col_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_COL_DIV;
                }
            }

            let mut clamp_min = 0.0f64;
            if let Some(v) = ep.clamp_min {
                clamp_min = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MIN;
            }
            let mut clamp_max = 0.0f64;
            if let Some(v) = ep.clamp_max {
                clamp_max = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MAX;
            }
            let mut pow_exponent = 1.0f64;
            if let Some(v) = ep.pow_exponent {
                pow_exponent = v;
                flags |= MATMUL_EPILOGUE_FLAG_POW;
            }
            if has_diag {
                flags |= MATMUL_EPILOGUE_FLAG_DIAG_WRITE;
            }

            let tile = crate::backend::wgpu::config::effective_matmul_tile();
            let groups_x =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(n as u32, tile);
            let groups_y =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile);

            // Build a layout tag incorporating the epilogue mask for cache keying
            let layout_tag = format!("runmat-matmul-epilogue-layout-flags-{flags:08x}");

            // Create module from the static WGSL (token substitution handled inside)
            let (shader_src, ep_buf, pipeline_layout) = match self.precision() {
                ProviderPrecision::F64 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF64 {
                        alpha: ep.alpha,
                        beta: ep.beta,
                        clamp_min,
                        clamp_max,
                        pow_exponent,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                        _pad2: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F64,
                        ep_buf,
                        pl,
                    )
                }
                ProviderPrecision::F32 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF32 {
                        alpha: ep.alpha as f32,
                        beta: ep.beta as f32,
                        clamp_min: clamp_min as f32,
                        clamp_max: clamp_max as f32,
                        pow_exponent: pow_exponent as f32,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F32,
                        ep_buf,
                        pl,
                    )
                }
            };

            let module = crate::backend::wgpu::pipelines::create_shader_module(
                self.device_ref(),
                "runmat-matmul-epilogue-module",
                shader_src,
            );
            let key =
                self.compute_pipeline_hash_bytes(shader_src.as_bytes(), &layout_tag, Some(tile));
            let pipeline = self.get_or_create_pipeline(
                key,
                &pipeline_layout,
                &module,
                "runmat-matmul-epilogue",
                Some(shader_src.as_bytes()),
                Some(&layout_tag),
                Some(tile),
            );

            let bg = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-matmul-epilogue-bind"),
                    layout: &self.pipelines.matmul_epilogue.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: row_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: col_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: ep_buf.as_entire_binding(),
                        },
                    ],
                });
            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bg,
                groups_x,
                groups_y,
            );
            let handle = self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                len,
                BufferUsageClass::FusionOut,
            );

            self.telemetry.record_matmul_duration(start.elapsed());

            Ok(handle)
        })
    }
    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let entry = self.get_entry(input)?;
            ensure!(
                entry.shape.len() == 3,
                "image_normalize: expected 3-D tensor, got {:?}",
                entry.shape
            );
            ensure!(
                entry.shape[0] == desc.batch
                    && entry.shape[1] == desc.height
                    && entry.shape[2] == desc.width,
                "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
                (desc.batch, desc.height, desc.width),
                entry.shape
            );

            if entry.len == 0 {
                return self.image_normalize_cpu_fallback(input, desc).await;
            }

            match self.precision {
                NumericPrecision::F64 => self.image_normalize_cpu_fallback(input, desc).await,
                NumericPrecision::F32 => {
                    ensure!(
                        desc.epsilon.is_finite(),
                        "image_normalize: epsilon must be finite"
                    );
                    ensure!(
                        desc.epsilon >= 0.0,
                        "image_normalize: epsilon must be non-negative"
                    );

                    let batches = entry.shape[0];
                    let height = entry.shape[1];
                    let width = entry.shape[2];
                    let plane = height
                        .checked_mul(width)
                        .ok_or_else(|| anyhow!("image_normalize: height*width overflow"))?;
                    ensure!(
                        entry.len == plane * batches,
                        "image_normalize: inconsistent tensor length {} vs dims {:?}",
                        entry.len,
                        entry.shape
                    );

                    let stride_h = batches;
                    let stride_w = batches
                        .checked_mul(height)
                        .ok_or_else(|| anyhow!("image_normalize: stride overflow"))?;

                    let batches_u32 = u32::try_from(batches)
                        .map_err(|_| anyhow!("image_normalize: batch size too large"))?;
                    let height_u32 = u32::try_from(height)
                        .map_err(|_| anyhow!("image_normalize: height too large"))?;
                    let width_u32 = u32::try_from(width)
                        .map_err(|_| anyhow!("image_normalize: width too large"))?;
                    let plane_u32 = u32::try_from(plane)
                        .map_err(|_| anyhow!("image_normalize: plane size too large"))?;
                    let stride_h_u32 = u32::try_from(stride_h)
                        .map_err(|_| anyhow!("image_normalize: stride_h too large"))?;
                    let stride_w_u32 = u32::try_from(stride_w)
                        .map_err(|_| anyhow!("image_normalize: stride_w too large"))?;
                    let (tuning, cache_hit) =
                        self.resolve_image_normalize_tuning(batches_u32, plane_u32);
                    log::debug!(
                    "image_normalize tuning batches={} plane={} lane={} spatial={} values/thread={} cache_hit={}",
                    batches_u32,
                    plane_u32,
                    tuning.lane_count,
                    tuning.spatial_tile,
                    tuning.values_per_thread,
                    cache_hit
                );
                    let pipeline = self.image_normalize_pipeline(&tuning)?;

                    let mut flags = 0u32;
                    if desc.gain.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAIN;
                    }
                    if desc.bias.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_BIAS;
                    }
                    if desc.gamma.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAMMA;
                    }

                    let mut uniforms = ImageNormalizeUniforms {
                        batch_count: 0,
                        height: height_u32,
                        width: width_u32,
                        plane: plane_u32,
                        stride_h: stride_h_u32,
                        stride_w: stride_w_u32,
                        flags,
                        batch_stride: batches_u32,
                        batch_offset: 0,
                        _pad0: 0,
                        epsilon: desc.epsilon as f32,
                        gain: desc.gain.unwrap_or(1.0) as f32,
                        bias: desc.bias.unwrap_or(0.0) as f32,
                        gamma: desc.gamma.unwrap_or(1.0) as f32,
                        _pad1: 0,
                    };

                    let out_buffer = self.create_storage_buffer_checked_with_usage(
                        entry.len,
                        "runmat-image-normalize-out",
                        BufferUsageClass::FusionOut,
                    )?;
                    let uniform_buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ImageNormalizeUniforms,
                        std::mem::size_of::<ImageNormalizeUniforms>() as u64,
                        "runmat-image-normalize-uniform",
                    );
                    let stream_hot_cap = self
                        .image_normalize_hot_stream_cap(plane_u32, batches_u32)
                        .max(1);
                    let cold_cap =
                        stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
                    let chunk_limit = if cache_hit {
                        stream_hot_cap
                    } else {
                        cold_cap.max(1)
                    };

                    let bind_entries = [
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                    ];
                    let layout = &self.pipelines.image_normalize.layout;
                    let bind_group =
                        self.bind_group_cache
                            .get_or_create(layout, &bind_entries, || {
                                Arc::new(self.device_ref().create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        label: Some("runmat-image-normalize-bind"),
                                        layout,
                                        entries: &bind_entries,
                                    },
                                ))
                            });

                    let mut offset = 0u32;
                    while offset < batches_u32 {
                        let remaining = batches_u32 - offset;
                        let chunk = remaining.min(chunk_limit).max(1);
                        uniforms.batch_count = chunk;
                        uniforms.batch_offset = offset;
                        self.queue
                            .write_buffer(uniform_buf.as_ref(), 0, bytes_of(&uniforms));
                        crate::backend::wgpu::dispatch::image_normalize::run(
                            self.device_ref(),
                            self.queue_ref(),
                            pipeline.as_ref(),
                            bind_group.as_ref(),
                            chunk,
                            tuning.batch_tile,
                        );
                        offset += chunk;
                    }

                    Ok(self.register_existing_buffer_with_usage(
                        out_buffer,
                        entry.shape.clone(),
                        entry.len,
                        BufferUsageClass::FusionOut,
                    ))
                }
            }
        })
    }
    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a runmat_accelerate_api::PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let rhs_entry = self.get_entry(rhs)?;
            let product = self.matmul_exec(lhs, rhs)?;
            let squared = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Mul,
                &product,
                &product,
            )?;
            let mut sum_sq = self.reduce_dim_sum_mean_exec(
                &squared,
                0,
                crate::backend::wgpu::types::DimReduceOp::Sum,
            )?;
            let _ = self.free(&squared);
            if epilogue.epsilon != 0.0 {
                let eps = self.fill_exec(&sum_sq.shape, epilogue.epsilon)?;
                let adjusted = self.binary_op_exec(
                    crate::backend::wgpu::types::BinaryOpCode::Add,
                    &sum_sq,
                    &eps,
                )?;
                let _ = self.free(&sum_sq);
                let _ = self.free(&eps);
                sum_sq = adjusted;
            }
            let norms =
                self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
            let _ = self.free(&sum_sq);
            let normalized = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Div,
                &product,
                &norms,
            )?;
            let _ = self.free(&product);
            let _ = self.free(&norms);

            let mut reused = false;
            let rhs_shape_match = rhs_entry.shape == normalized.shape;
            let rhs_transposed = runmat_accelerate_api::handle_transpose_info(rhs).is_some();
            let rhs_ref_count = Arc::strong_count(&rhs_entry.buffer);
            if rhs_shape_match && !rhs_transposed && rhs_entry.len > 0 && rhs_ref_count <= 2 {
                if let Ok(normalized_entry) = self.get_entry(&normalized) {
                    let bytes = (rhs_entry.len as u64) * self.element_size as u64;
                    if bytes > 0 {
                        let mut encoder = self.device_ref().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-power-step-copy"),
                            },
                        );
                        encoder.copy_buffer_to_buffer(
                            normalized_entry.buffer.as_ref(),
                            0,
                            rhs_entry.buffer.as_ref(),
                            0,
                            bytes,
                        );
                        self.submit(encoder);
                    }
                    let _ = self.free(&normalized);
                    self.mark_buffer_usage(rhs, BufferUsageClass::FusionOut);
                    log::debug!(
                        "matmul_power_step: reused rhs buffer {} for normalized output (len={})",
                        rhs.buffer_id,
                        rhs_entry.len
                    );
                    reused = true;
                }
            }

            if reused {
                Ok(rhs.clone())
            } else {
                log::debug!(
                "matmul_power_step: fallback reuse (shape_match={} transpose={} len={} ref_count={})",
                rhs_shape_match,
                rhs_transposed,
                rhs_entry.len,
                rhs_ref_count
            );
                Ok(normalized)
            }
        })
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if options.rows != CovRows::All {
                return Err(anyhow!(
                    "covariance: rows option {:?} not supported by WGPU provider",
                    options.rows
                ));
            }
            if options.has_weight_vector || weights.is_some() {
                return Err(anyhow!(
                    "covariance: weight vectors are not supported by WGPU provider"
                ));
            }

            let combined = if let Some(rhs) = second {
                let left_entry = self.get_entry(matrix)?;
                let right_entry = self.get_entry(rhs)?;

                let rows_left = match left_entry.shape.len() {
                    0 => 1usize,
                    1 => left_entry.shape[0],
                    2 => left_entry.shape[0],
                    _ => {
                        return Err(anyhow!(
                            "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                            left_entry.shape
                        ))
                    }
                };
                let rows_right = match right_entry.shape.len() {
                    0 => 1usize,
                    1 => right_entry.shape[0],
                    2 => right_entry.shape[0],
                    _ => {
                        return Err(anyhow!(
                            "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                            right_entry.shape
                        ))
                    }
                };

                ensure!(
                    rows_left == rows_right,
                    "covariance: inputs must have the same number of rows (got {} and {})",
                    rows_left,
                    rows_right
                );

                let cat_inputs = vec![matrix.clone(), rhs.clone()];
                Some(self.cat_exec(2, &cat_inputs)?)
            } else {
                None
            };

            let result = {
                let source = combined.as_ref().unwrap_or(matrix);
                self.covariance_exec(source, options).await
            };

            if let Some(handle) = combined {
                let _ = self.free(&handle);
            }

            result
        })
    }
    fn corrcoef<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: &'a CorrcoefOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.corrcoef_exec(matrix, options).await })
    }
    fn linsolve<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        options: &'a ProviderLinsolveOptions,
    ) -> AccelProviderFuture<'a, ProviderLinsolveResult> {
        Box::pin(async move {
            if let Some(result) = self.try_linsolve_device(lhs, rhs, options).await? {
                return Ok(result);
            }
            let start = Instant::now();
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;

            let (solution, rcond) =
                linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
                    .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_linsolve_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("linsolve:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &solution.data,
                shape: &solution.shape,
            })?;

            Ok(ProviderLinsolveResult {
                solution: handle,
                reciprocal_condition: rcond,
            })
        })
    }
    fn inv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("inv: {e}"))?;
            let result = inv_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn pinv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("pinv: {e}"))?;
            let result = pinv_host_real_for_provider(&tensor, options.tolerance)
                .map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn cond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("cond: {e}"))?;
            let cond_value =
                cond_host_real_for_provider(&tensor, norm).map_err(|e| anyhow!("{e}"))?;
            let scalar = [cond_value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn norm<'a>(
        &'a self,
        tensor: &'a GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, tensor).await?;
            let host_tensor = Tensor::new(data, shape).map_err(|e| anyhow!("norm: {e}"))?;
            let value =
                norm_host_real_for_provider(&host_tensor, order).map_err(|e| anyhow!("{e}"))?;
            let scalar = [value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rank<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rank: {e}"))?;
            let rank =
                rank_host_real_for_provider(&tensor, tolerance).map_err(|e| anyhow!("{e}"))? as f64;
            let scalar = [rank];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rcond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rcond: {e}"))?;
            let estimate = rcond_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            let scalar = [estimate];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn mldivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self
                .try_linsolve_device(lhs, rhs, &ProviderLinsolveOptions::default())
                .await?
            {
                self.telemetry.record_mldivide_duration(start.elapsed());
                return Ok(result.solution);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;

            let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mldivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mldivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self.try_mrdivide_device(lhs, rhs).await? {
                self.telemetry.record_mrdivide_duration(start.elapsed());
                return Ok(result);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;

            let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mrdivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mrdivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn dot<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.dot_exec(lhs, rhs, dim) })
    }
    fn eig<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        compute_left: bool,
    ) -> AccelProviderFuture<'a, ProviderEigResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("eig: {e}"))?;
            let eval = runmat_runtime::builtins::math::linalg::factor::eig::evaluate(
                Value::Tensor(tensor),
                &[],
                compute_left,
            )
            .await
            .map_err(|err| runtime_flow_to_anyhow("eig", err))?;

            let eigenvalues_tensor = host_tensor_from_value("eig", eval.eigenvalues())?;
            let diagonal_tensor = host_tensor_from_value("eig", eval.diagonal_matrix())?;
            let right_tensor = host_tensor_from_value("eig", eval.right())?;

            let left_value = if compute_left {
                Some(
                    eval.left()
                        .map_err(|err| runtime_flow_to_anyhow("eig", err))?,
                )
            } else {
                None
            };

            let left_tensor = match left_value {
                Some(value) => Some(host_tensor_from_value("eig", value)?),
                None => None,
            };

            let eigenvalues = self.upload(&HostTensorView {
                data: &eigenvalues_tensor.data,
                shape: &eigenvalues_tensor.shape,
            })?;
            let diagonal = self.upload(&HostTensorView {
                data: &diagonal_tensor.data,
                shape: &diagonal_tensor.shape,
            })?;
            let right = self.upload(&HostTensorView {
                data: &right_tensor.data,
                shape: &right_tensor.shape,
            })?;
            let left = match left_tensor {
                Some(tensor) => Some(self.upload(&HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })?),
                None => None,
            };

            if compute_left && left.is_none() {
                return Err(anyhow!(
                    "eig: left eigenvectors are not available for the requested matrix"
                ));
            }

            Ok(ProviderEigResult {
                eigenvalues,
                diagonal,
                right,
                left,
            })
        })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
        })
    }
    fn reduce_nnz_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(
                a,
                dim,
                crate::backend::wgpu::types::DimReduceOp::CountNonZero,
            )
        })
    }
    fn reduce_prod_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Prod)
        })
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Mean)
        })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AnyOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AnyInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }
    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AnyOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AnyInclude
            };
            let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
            match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                Ok(handle) => {
                    let _ = self.free(&first);
                    Ok(handle)
                }
                Err(err) => {
                    let _ = self.free(&first);
                    Err(err)
                }
            }
        })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AllOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AllInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }

    fn reduce_all<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AllOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AllInclude
            };
            let total_elems = if a.shape.is_empty() {
                1
            } else {
                product_checked(&a.shape)
                    .ok_or_else(|| anyhow!("reduce_all: tensor size exceeds GPU limits"))?
            };
            if total_elems == 0 {
                return self.fill(&[1usize, 1usize], f64::NAN);
            }
            if a.shape.len() <= 2 {
                let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
                match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                    Ok(handle) => {
                        let _ = self.free(&first);
                        Ok(handle)
                    }
                    Err(err) => {
                        let _ = self.free(&first);
                        Err(err)
                    }
                }
            } else {
                let original_shape = a.shape.clone();
                let flattened_shape = vec![total_elems, 1usize];
                let flattened = self.reshape(a, &flattened_shape)?;
                let result = self.reduce_dim_sum_mean_exec(&flattened, 0, op);
                let _ = self.reshape(a, &original_shape);
                result
            }
        })
    }

    fn reduce_median<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let median = median_from_slice(&host.data);
            let data = [median];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
        })
    }

    fn reduce_median_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            if host.shape.len() != 2 {
                return Err(anyhow!("reduce_median_dim: only 2D supported"));
            }
            let rows = host.shape[0];
            let cols = host.shape[1];
            let mut scratch = Vec::<f64>::with_capacity(rows.max(cols));
            let (out, shape) = if dim <= 1 {
                let mut values = vec![f64::NAN; cols];
                for (c, value) in values.iter_mut().enumerate().take(cols) {
                    scratch.clear();
                    let mut saw_nan = false;
                    for r in 0..rows {
                        let v = host.data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            scratch.clear();
                            break;
                        }
                        scratch.push(v);
                    }
                    *value = if saw_nan || scratch.is_empty() {
                        f64::NAN
                    } else {
                        compute_median_inplace(&mut scratch)
                    };
                }
                (values, vec![1usize, cols])
            } else {
                let mut values = vec![f64::NAN; rows];
                for (r, value) in values.iter_mut().enumerate().take(rows) {
                    scratch.clear();
                    let mut saw_nan = false;
                    for c in 0..cols {
                        let v = host.data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            scratch.clear();
                            break;
                        }
                        scratch.push(v);
                    }
                    *value = if saw_nan || scratch.is_empty() {
                        f64::NAN
                    } else {
                        compute_median_inplace(&mut scratch)
                    };
                }
                (values, vec![rows, 1usize])
            };
            self.upload(&HostTensorView {
                data: &out,
                shape: &shape,
            })
        })
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
        })
    }

    fn reduce_nnz<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::CountNonZero)
        })
    }

    fn reduce_prod<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Prod)
        })
    }

    fn reduce_mean<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            // Mean over all elements: compute via single-pass sum then divide by len
            let sum_handle =
                self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
            let total_elems: usize = self.get_entry(a)?.len.max(1);
            let scalar = 1.0 / (total_elems as f64);
            let out = self.scalar_op_exec(
                crate::backend::wgpu::types::ScalarOpCode::Mul,
                &sum_handle,
                scalar,
            )?;
            // Free temporary sum buffer
            let _ = self.free(&sum_handle);
            Ok(out)
        })
    }
    fn reduce_std<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_exec(a, normalization, nan_mode) })
    }

    fn reduce_std_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_dim_exec(a, dim, normalization, nan_mode) })
    }

    fn reduce_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Min)
        })
    }

    fn reduce_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Max)
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Min)
        })
    }

    fn reduce_max_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Max)
        })
    }

    fn find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        self.find_exec(a, limit, direction)
    }
    fn issymmetric(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_symmetry_shape(&entry.shape).map_err(|e| anyhow!("issymmetric: {e}"))?;
        if rows != cols {
            return Ok(false);
        }
        if rows == 0 || cols == 0 {
            return Ok(true);
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("issymmetric: matrix dimensions too large"))?;
        if total > entry.len {
            return Err(anyhow!(
                "issymmetric: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("issymmetric: matrix exceeds GPU limits"));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "issymmetric: tolerance must be finite and non-negative"
            ));
        }

        let mode = match kind {
            ProviderSymmetryKind::Symmetric => 0u32,
            ProviderSymmetryKind::Skew => 1u32,
        };

        let output_init = [1u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-issymmetric-output"),
                contents: cast_slice(&output_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let pipeline = &self.pipelines.symmetry;
        match entry.precision {
            NumericPrecision::F64 => {
                let params = SymmetryParamsF64 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance,
                    _pad: 0.0,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f64"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
            NumericPrecision::F32 => {
                let tol32 = tolerance.min(f32::MAX as f64).max(0.0) as f32;
                let params = SymmetryParamsF32 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance: tol32,
                    _pad: [0.0; 3],
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f32"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
        }

        let staging_size = std::mem::size_of::<u32>() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-issymmetric-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-issymmetric-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "issymmetric")?;
        let words: &[u32] = cast_slice(&bytes);
        let flag = words.first().copied().unwrap_or(0);

        Ok(flag != 0)
    }

    fn ishermitian<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(anyhow!(
                    "ishermitian: tolerance must be finite and non-negative"
                ));
            }
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            let skew = matches!(kind, ProviderHermitianKind::Skew);
            ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance)
                .map_err(|e| anyhow!(e))
        })
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm<'a>(&'a self, matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
        })
    }
    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        let entry = self.get_entry(h)?;
        let elem_size = match entry.precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<f32>() as u64,
        };
        let total_bytes = (linear_index as u64)
            .checked_mul(elem_size)
            .ok_or_else(|| anyhow!("read_scalar: index overflow"))?;
        if (linear_index + 1) > entry.len {
            return Err(anyhow!(
                "read_scalar: index {} out of bounds (len {})",
                linear_index + 1,
                entry.len
            ));
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-read-scalar-staging"),
            size: elem_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-read-scalar-enc"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), total_bytes, &staging, 0, elem_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, elem_size, "read_scalar")?;
        let value = match entry.precision {
            NumericPrecision::F64 => {
                let words: &[f64] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0)
            }
            NumericPrecision::F32 => {
                let words: &[f32] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0) as f64
            }
        };
        Ok(value)
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_exec(shader, inputs, output_shape, len);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    fn fused_elementwise_multi(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let result =
            self.fused_elementwise_multi_exec(shader, inputs, output_shape, len, num_outputs);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
                ("num_outputs", num_outputs as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise_multi", &shape, &tuning);
        }
        result
    }

    fn map_nan_to_zero(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-nan-to-zero-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }
    fn not_nan_mask(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-not-nan-mask-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }

    fn fused_reduction(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        );
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_reduction_duration(elapsed);
            let actual_wg = if workgroup_size == 0 {
                self.default_reduction_workgroup_size()
            } else {
                workgroup_size
            } as u64;
            let flavor_tag = match flavor {
                ReductionFlavor::Sum => 0,
                ReductionFlavor::Mean => 1,
                ReductionFlavor::CustomScale(_) => 2,
            };
            let shape = [
                ("reduce_len", reduce_len as u64),
                ("slices", num_slices as u64),
                ("rank", output_shape.len() as u64),
            ];
            let tuning = [("wg", actual_wg), ("flavor", flavor_tag)];
            self.record_kernel_launch_basic("fused_reduction", &shape, &tuning);
        }
        result
    }
    fn warmup(&self) {
        if std::env::var("RUNMAT_WGPU_SKIP_WARMUP")
            .ok()
            .and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    None
                } else if trimmed.eq_ignore_ascii_case("1")
                    || trimmed.eq_ignore_ascii_case("true")
                    || trimmed.eq_ignore_ascii_case("yes")
                {
                    Some(true)
                } else if trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("no")
                {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false)
        {
            log::info!("RunMat Accelerate: skipping wgpu warmup (RUNMAT_WGPU_SKIP_WARMUP=1)");
            return;
        }

        let start = Instant::now();
        self.warmup_from_disk();
        // Proactively warm common pipelines used by normalization and reduction chains
        let pl = &self.pipelines;
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary_broadcast.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.unary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.scalar.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_dim_sum_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_nd_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_global.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.fill.pipeline,
        );

        let ms = start.elapsed().as_millis() as u64;
        self.metrics.set_last_warmup_millis(ms);
    }
    fn fused_cache_counters(&self) -> (u64, u64) {
        self.metrics.counters()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        Some(self.metrics.last_warmup_millis())
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        let (fusion_hits, fusion_misses) = self.metrics.counters();
        let (bind_hits, bind_misses) = self.bind_group_cache.counters();
        // Build per-layout telemetry by resolving layout pointers to tags
        let mut by_layout: Vec<runmat_accelerate_api::BindGroupLayoutTelemetry> = Vec::new();
        let per = self.bind_group_cache.per_layout_counters();
        if let Ok(tags) = self.bind_group_layout_tags.lock() {
            for (ptr, (h, m)) in per {
                let tag = tags
                    .get(&ptr)
                    .cloned()
                    .unwrap_or_else(|| format!("layout_ptr_{:#x}", ptr));
                by_layout.push(runmat_accelerate_api::BindGroupLayoutTelemetry {
                    tag,
                    hits: h,
                    misses: m,
                });
            }
        }
        self.telemetry.snapshot(
            fusion_hits,
            fusion_misses,
            bind_hits,
            bind_misses,
            Some(by_layout),
        )
    }

    fn reset_telemetry(&self) {
        self.telemetry.reset();
        self.metrics.reset();
        self.bind_group_cache.reset_counters();
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.reduction_workgroup_size_default
    }

    fn two_pass_threshold(&self) -> usize {
        self.reduction_two_pass_threshold
    }

    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode
    }

    fn scatter_column(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_column_exec(matrix, col_index, values)
    }
    fn scatter_row(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_row_exec(matrix, row_index, values)
    }

    fn sub2ind(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.sub2ind_exec(dims, strides, inputs, scalar_mask, len, output_shape)
    }

    fn supports_ind2sub(&self) -> bool {
        true
    }

    fn ind2sub(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> Result<Vec<GpuTensorHandle>> {
        self.ind2sub_exec(dims, strides, indices, total, len, output_shape)
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let _span = info_span!(
            "gpu.transfer.upload",
            shape = ?host.shape,
            len = host.data.len()
        )
        .entered();
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                "upload",
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let buffer =
            if len == 0 {
                self.create_storage_buffer(0, "runmat-upload-empty")
            } else {
                match self.precision {
                    NumericPrecision::F64 => {
                        let contents = cast_slice(host.data);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                        let contents = cast_slice(&data_f32);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                }
            };
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer(buffer, shape, len))
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let span = info_span!(
                "gpu.transfer.download",
                shape = ?h.shape,
                buffer_id = h.buffer_id
            );
            let entry = {
                let _guard = span.enter();
                log::trace!("wgpu download id={} shape={:?}", h.buffer_id, &h.shape);
                self.get_entry(h)?
            };
            if let Some(last) = entry.last_submission_id {
                log::trace!(
                    "wgpu download id={} last_submission_id={}",
                    h.buffer_id,
                    last
                );
            } else {
                log::trace!("wgpu download id={} last_submission_id=<none>", h.buffer_id);
            }
            if entry.len == 0 {
                return Ok(HostTensorOwned {
                    data: Vec::new(),
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                });
            }

            let size_bytes = (entry.len * self.element_size) as u64;

            // Shared post-map readback logic: decode mapped bytes, unmap, record telemetry,
            // apply transpose metadata, and return host tensor.
            let finish_readback =
                |staging: wgpu::Buffer, size_bytes: u64| -> Result<HostTensorOwned> {
                    let slice = staging.slice(..);
                    let data = slice.get_mapped_range();
                    log::trace!(
                        "wgpu download copying data id={} len={} bytes={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes
                    );

                    let mut out = vec![0.0f64; entry.len];
                    match entry.precision {
                        NumericPrecision::F64 => out.copy_from_slice(cast_slice(&data)),
                        NumericPrecision::F32 => {
                            let f32_slice: &[f32] = cast_slice(&data);
                            for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                                *dst = *src as f64;
                            }
                        }
                    }
                    drop(data);
                    staging.unmap();
                    log::trace!("wgpu download finished copy id={}", h.buffer_id);
                    self.telemetry.record_download_bytes(size_bytes);

                    let mut shape = h.shape.clone();
                    if let Some(info) = runmat_accelerate_api::handle_transpose_info(h) {
                        let base_rows = info.base_rows;
                        let base_cols = info.base_cols;
                        if base_rows * base_cols != out.len() {
                            return Err(anyhow!(
                                "download: transpose metadata mismatch for buffer {}",
                                h.buffer_id
                            ));
                        }
                        if shape.len() == 2 {
                            let rows_t = base_cols;
                            let cols_t = base_rows;
                            let mut transposed = vec![0.0f64; out.len()];
                            for col in 0..base_cols {
                                for row in 0..base_rows {
                                    let src_idx = row + col * base_rows;
                                    let dst_idx = col + row * base_cols;
                                    transposed[dst_idx] = out[src_idx];
                                }
                            }
                            out = transposed;
                            shape[0] = rows_t;
                            shape[1] = cols_t;
                        }
                    }

                    log::trace!(
                        "wgpu download complete id={} final_shape={:?}",
                        h.buffer_id,
                        shape
                    );

                    Ok(HostTensorOwned {
                        data: out,
                        shape,
                        storage: runmat_accelerate_api::handle_storage(h),
                    })
                };

            log::trace!(
                "wgpu download creating staging buffer id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-download-staging"),
                size: size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-download-encoder"),
                });
            encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, &staging, 0, size_bytes);
            self.submit(encoder);
            let slice = staging.slice(..);
            let (tx, rx) = oneshot::channel();

            let map_buffer_id = h.buffer_id;
            slice.map_async(wgpu::MapMode::Read, move |res| {
                log::trace!(
                    "wgpu download map_async callback id={} status={:?}",
                    map_buffer_id,
                    res
                );
                let _ = tx.send(res);
            });
            log::trace!(
                "wgpu download awaiting map_async completion id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.device.poll(wgpu::Maintain::Wait);
            }
            let map_result = rx
                .await
                .map_err(|_| anyhow!("map_async callback dropped for buffer {}", h.buffer_id))?;

            log::trace!("wgpu download map_async success id={}", h.buffer_id);
            map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
            finish_readback(staging, size_bytes)
        })
    }
    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        // Remove from handle table and return buffer to pool for reuse
        log::trace!("wgpu free id={}", h.buffer_id);
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&h.buffer_id);
        if let Some(entry) = entry {
            if entry.len > 0 {
                let size_bytes = (entry.len as u64).saturating_mul(self.element_size as u64);
                let poolable_by_size = self.buffer_residency_max_poolable_bytes > 0
                    && size_bytes <= self.buffer_residency_max_poolable_bytes;
                let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                // Always invalidate bind-group cache first so cache-held references
                // do not pin dropped buffers across loop iterations.
                self.bind_group_cache.invalidate_buffer(buffer_ptr);
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    self.buffer_residency
                        .release(entry.usage, entry.len, entry.buffer.clone());
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} bytes={} strong_count={} poolable_by_size={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes,
                        strong_count,
                        poolable_by_size
                    );
                }
            }
        }
        self.kernel_resources.clear_matmul_source(h.buffer_id);
        runmat_accelerate_api::clear_handle_logical(h);
        runmat_accelerate_api::clear_handle_storage(h);
        runmat_accelerate_api::clear_handle_transpose(h);
        Ok(())
    }

    fn device_info(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        let backend = format!("{:?}", self.adapter_info.backend).to_ascii_lowercase();
        let memory_bytes = if self.adapter_limits.max_buffer_size > 0 {
            Some(self.adapter_limits.max_buffer_size)
        } else {
            None
        };
        ApiDeviceInfo {
            device_id: self.runtime_device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            memory_bytes,
            backend: Some(backend),
        }
    }
}


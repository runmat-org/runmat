    fn export_context(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        self.export_context_exec(kind)
    }

    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(&self, handle: &GpuTensorHandle) -> Option<WgpuBufferRef> {
        self.export_wgpu_buffer_exec(handle)
    }

    fn device_id(&self) -> u32 {
        self.device_id_exec()
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        self.spawn_handle_concurrency_exec()
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
        self.provider_precision_exec()
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
        self.set_rng_state_exec(state)
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
        Box::pin(async move { self.polyfit_exec(x, y, degree, weights).await })
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

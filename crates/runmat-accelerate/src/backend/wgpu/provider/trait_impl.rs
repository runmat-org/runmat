use super::*;
use runmat_accelerate_api::{ProviderEnvelopeRequest, ProviderEnvelopeResult};

impl AccelProvider for WgpuProvider {
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
        self.logical_islogical_exec(a)
    }

    fn logical_isreal(&self, a: &GpuTensorHandle) -> Result<bool> {
        self.logical_isreal_exec(a)
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
        Box::pin(async move { self.unary_abs_exec(a) })
    }

    fn unary_heaviside<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Heaviside, a)
        })
    }

    fn unary_conj<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_conj_exec(a) })
    }

    fn unary_imag<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_imag_exec(a) })
    }

    fn unary_real<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_real_exec(a) })
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
        Box::pin(async move { self.unary_double_exec(a) })
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
        Box::pin(async move { self.unary_pow2_exec(a) })
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
        self.pow2_scale_exec(mantissa, exponent)
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
        Box::pin(async move { self.sort_dim_exec(a, dim, order, comparison).await })
    }
    fn sort_rows<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        columns: &'a [SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        Box::pin(async move { self.sort_rows_exec(a, columns, comparison).await })
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
    fn uniform_spectral_estimate<'a>(
        &'a self,
        request: &'a ProviderSpectralRequest<'a>,
    ) -> AccelProviderFuture<'a, ProviderSpectralResult> {
        Box::pin(async move { self.uniform_spectral_estimate_exec(request).await })
    }
    fn signal_envelope<'a>(
        &'a self,
        request: &'a ProviderEnvelopeRequest<'a>,
    ) -> AccelProviderFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async move { self.signal_envelope_exec(request).await })
    }
    fn conv2d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        self.conv2d_exec(signal, kernel, mode)
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
        Box::pin(async move { self.unique_exec(handle, options).await })
    }
    fn ismember<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a IsMemberOptions,
    ) -> AccelProviderFuture<'a, IsMemberResult> {
        Box::pin(async move { self.ismember_exec(a, b, options).await })
    }

    fn union<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a UnionOptions,
    ) -> AccelProviderFuture<'a, UnionResult> {
        Box::pin(async move { self.union_exec(a, b, options).await })
    }
    fn setdiff<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a SetdiffOptions,
    ) -> AccelProviderFuture<'a, SetdiffResult> {
        Box::pin(async move { self.setdiff_exec(a, b, options).await })
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
        self.reshape_exec(handle, new_shape)
    }

    fn lu<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move { self.lu_exec(a).await })
    }

    fn chol<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        lower: bool,
    ) -> AccelProviderFuture<'a, ProviderCholResult> {
        Box::pin(async move { self.chol_exec(a, lower).await })
    }
    fn qr<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> AccelProviderFuture<'a, ProviderQrResult> {
        Box::pin(async move { self.qr_exec(handle, options).await })
    }

    fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.take_matmul_sources_exec(product)
    }

    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> AccelProviderFuture<'a, Option<ProviderQrPowerIterResult>> {
        Box::pin(async move {
            self.qr_power_iter_exec(product, product_lhs, q_handle, options)
                .await
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
        Box::pin(async move { self.matmul_epilogue_exec(a, b, ep).await })
    }

    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.image_normalize_exec(input, desc).await })
    }
    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a runmat_accelerate_api::PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_power_step_exec(lhs, rhs, epilogue).await })
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.covariance_with_optional_exec(matrix, second, weights, options)
                .await
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
        Box::pin(async move { self.linsolve_exec(lhs, rhs, options).await })
    }
    fn inv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.inv_exec(matrix).await })
    }

    fn pinv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.pinv_exec(matrix, options).await })
    }

    fn cond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.cond_exec(matrix, norm).await })
    }

    fn norm<'a>(
        &'a self,
        tensor: &'a GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.norm_exec(tensor, order).await })
    }

    fn rank<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.rank_exec(matrix, tolerance).await })
    }

    fn rcond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.rcond_exec(matrix).await })
    }

    fn mldivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.mldivide_exec(lhs, rhs).await })
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.mrdivide_exec(lhs, rhs).await })
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
        Box::pin(async move { self.eig_exec(handle, compute_left).await })
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
        Box::pin(async move { self.reduce_any_dim_exec(a, dim, omit_nan) })
    }
    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_any_exec(a, omit_nan) })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_all_dim_exec(a, dim, omit_nan) })
    }

    fn reduce_all<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_all_exec(a, omit_nan) })
    }

    fn reduce_median<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_median_exec(a).await })
    }

    fn reduce_median_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_median_dim_exec(a, dim).await })
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
        Box::pin(async move { self.reduce_mean_global_exec(a) })
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
        self.issymmetric_exec(matrix, kind, tolerance)
    }

    fn ishermitian<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move { self.ishermitian_exec(matrix, kind, tolerance).await })
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm<'a>(&'a self, matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move { self.sym_rcm_exec(matrix).await })
    }
    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        self.read_scalar_exec(h, linear_index)
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_with_telemetry_exec(shader, inputs, output_shape, len)
    }

    fn fused_elementwise_multi(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        self.fused_elementwise_multi_with_telemetry_exec(
            shader,
            inputs,
            output_shape,
            len,
            num_outputs,
        )
    }

    fn map_nan_to_zero(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.map_nan_to_zero_exec(a)
    }
    fn not_nan_mask(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.not_nan_mask_exec(a)
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
        self.fused_reduction_with_telemetry_exec(
            crate::backend::wgpu::provider::backend::telemetry::FusedReductionTelemetryRequest {
                shader,
                inputs,
                output_shape,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            },
        )
    }
    fn warmup(&self) {
        self.warmup_exec();
    }
    fn fused_cache_counters(&self) -> (u64, u64) {
        self.fused_cache_counters_exec()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        self.last_warmup_millis_exec()
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        self.telemetry_snapshot_exec()
    }

    fn reset_telemetry(&self) {
        self.reset_telemetry_exec();
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.default_reduction_workgroup_size_exec()
    }

    fn two_pass_threshold(&self) -> usize {
        self.two_pass_threshold_exec()
    }

    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode_exec()
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
        self.upload_exec(host)
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move { self.download_exec(h).await })
    }
    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        self.free_exec(h)
    }

    fn device_info(&self) -> String {
        self.device_info_exec()
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        self.device_info_struct_exec()
    }
}

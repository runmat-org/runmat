macro_rules! linalg_advanced_pagefun_methods {
    () => {
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

    };
}

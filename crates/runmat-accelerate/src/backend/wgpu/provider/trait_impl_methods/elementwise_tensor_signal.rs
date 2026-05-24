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


macro_rules! indexing_io_telemetry_methods {
    () => {
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
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
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

    };
}

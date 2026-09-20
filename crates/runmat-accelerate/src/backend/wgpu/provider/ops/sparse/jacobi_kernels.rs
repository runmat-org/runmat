use std::borrow::Cow;
use std::sync::Arc;

use super::super::WgpuProvider;

pub(crate) struct SparseJacobiPreparePipeline {
    pub(super) layout: Arc<wgpu::BindGroupLayout>,
    pub(super) pipeline: wgpu::ComputePipeline,
}

pub(crate) struct SparseJacobiApplyPipeline {
    pub(super) layout: Arc<wgpu::BindGroupLayout>,
    pub(super) pipeline: wgpu::ComputePipeline,
}

pub(super) fn prepare_pipeline(provider: &WgpuProvider) -> Arc<SparseJacobiPreparePipeline> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .jacobi_prepare
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create_prepare_pipeline(provider)))
        .clone()
}

pub(super) fn apply_pipeline(provider: &WgpuProvider) -> Arc<SparseJacobiApplyPipeline> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .jacobi_apply
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create_apply_pipeline(provider)))
        .clone()
}

fn create_prepare_pipeline(provider: &WgpuProvider) -> SparseJacobiPreparePipeline {
    let entries = [
        crate::backend::wgpu::bindings::storage_read_entry(0),
        crate::backend::wgpu::bindings::storage_read_entry(1),
        crate::backend::wgpu::bindings::storage_read_entry(2),
        crate::backend::wgpu::bindings::storage_read_write_entry(3),
        crate::backend::wgpu::bindings::storage_read_write_entry(4),
    ];
    let (layout, pipeline) = create_pipeline(
        provider,
        "runmat-sparse-jacobi-prepare",
        &entries,
        prepare_shader(provider),
    );
    SparseJacobiPreparePipeline { layout, pipeline }
}

fn create_apply_pipeline(provider: &WgpuProvider) -> SparseJacobiApplyPipeline {
    let entries = [
        crate::backend::wgpu::bindings::storage_read_entry(0),
        crate::backend::wgpu::bindings::storage_read_entry(1),
        crate::backend::wgpu::bindings::storage_read_write_entry(2),
    ];
    let (layout, pipeline) = create_pipeline(
        provider,
        "runmat-sparse-jacobi-apply",
        &entries,
        apply_shader(provider),
    );
    SparseJacobiApplyPipeline { layout, pipeline }
}

fn create_pipeline(
    provider: &WgpuProvider,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
    source: String,
) -> (Arc<wgpu::BindGroupLayout>, wgpu::ComputePipeline) {
    let layout = Arc::new(provider.device_ref().create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        },
    ));
    let pipeline_layout =
        provider
            .device_ref()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout.as_ref()],
                push_constant_ranges: &[],
            });
    let module = provider
        .device_ref()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    let pipeline = provider.device_ref().create_compute_pipeline(
        &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor!(
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
        ),
    );
    (layout, pipeline)
}

fn prepare_shader(provider: &WgpuProvider) -> String {
    let scalar = provider.precision.as_str();
    let finite_limit = match provider.precision {
        super::super::NumericPrecision::F32 => "3.402823466e38",
        super::super::NumericPrecision::F64 => "1.7976931348623157e308",
    };
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    format!(
        r#"
struct IndexBuffer {{ data: array<u32>, }};
struct ScalarBuffer {{ data: array<{scalar}>, }};
struct StatusBuffer {{ data: array<atomic<u32>>, }};
@group(0) @binding(0) var<storage, read> RowOffsets: IndexBuffer;
@group(0) @binding(1) var<storage, read> ColumnIndices: IndexBuffer;
@group(0) @binding(2) var<storage, read> Values: ScalarBuffer;
@group(0) @binding(3) var<storage, read_write> InverseDiagonal: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Status: StatusBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let rows = arrayLength(&RowOffsets.data) - 1u;
    if (row >= rows) {{ return; }}
    var diagonal: {scalar} = 0.0;
    var found = false;
    for (var index = RowOffsets.data[row]; index < RowOffsets.data[row + 1u]; index = index + 1u) {{
        if (ColumnIndices.data[index] == row) {{
            diagonal = Values.data[index];
            found = true;
        }}
    }}
    let inverse = 1.0 / diagonal;
    if (!found || diagonal == 0.0 || inverse != inverse || abs(inverse) > {finite_limit}) {{
        atomicOr(&Status.data[0], 1u);
        InverseDiagonal.data[row] = 0.0;
        return;
    }}
    InverseDiagonal.data[row] = inverse;
}}
"#
    )
}

fn apply_shader(provider: &WgpuProvider) -> String {
    let scalar = provider.precision.as_str();
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    format!(
        r#"
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> InverseDiagonal: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Input: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> Output: ScalarBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < arrayLength(&Input.data)) {{
        Output.data[index] = InverseDiagonal.data[index] * Input.data[index];
    }}
}}
"#
    )
}

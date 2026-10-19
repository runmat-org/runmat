use super::execute::OrderedMeanCheckpoint;
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderOrderedRealMeanControl, ProviderOrderedRealMeanResources,
    ProviderOrderedRealMeanSource,
};

#[test]
fn ordered_axis_shader_has_distinct_storage_classes_and_serial_f64_accumulation() {
    let first = crate::backend::wgpu::shaders::reduction::ordered_mean::source(
        NumericPrecision::F32,
        NumericPrecision::F64,
    );
    assert!(first.contains("array<f32>"));
    assert!(first.contains("array<f64>"));
    assert!(first.contains("var sum = f64(0.0)"));
    assert!(first.contains("k = k + 1u"));
    assert!(!first.contains("var<workgroup> tile"));
    let final_stage = crate::backend::wgpu::shaders::reduction::ordered_mean::source(
        NumericPrecision::F64,
        NumericPrecision::F32,
    );
    assert!(final_stage.contains("OutBuf.data[col] = f32(result)"));
}

#[path = "tests/vecdim.rs"]
mod vecdim;
#[path = "tests/singleton.rs"]
mod singleton;
#[path = "tests/faults.rs"]
mod faults;

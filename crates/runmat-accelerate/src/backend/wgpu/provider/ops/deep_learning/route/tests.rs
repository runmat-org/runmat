use super::*;
use runmat_accelerate_api::{
    AccelProvider, DeepLearningFeedForwardStage, DeepLearningForwardBufferBound,
    DeepLearningForwardParameter, DeepLearningForwardParameters, DeepLearningForwardStep,
    GpuTensorView, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ProviderViewStep,
};

#[path = "tests/basic.rs"]
mod basic;
#[path = "tests/pooling_view.rs"]
mod pooling_view;
#[path = "tests/normalization_dense.rs"]
mod normalization_dense;
#[path = "tests/limits.rs"]
mod limits;

struct Fixture {
    provider: WgpuProvider,
    class: NumericElementType,
    resident: Option<GpuTensorHandle>,
    small_input: Option<GpuTensorHandle>,
    shared_parameter: Option<GpuTensorHandle>,
}
impl Fixture {
    fn new() -> Self {
        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let class = match provider.precision() {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        let input_f32 = [1.0_f32; 6];
        let input_f64 = [1.0_f64; 6];
        let resident = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&input_f32),
                    NumericElementType::F64 => HostNumericDataView::F64(&input_f64),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let parameter_f32 = [2.0_f32];
        let parameter_f64 = [2.0_f64];
        let shared_parameter = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&parameter_f32),
                    NumericElementType::F64 => HostNumericDataView::F64(&parameter_f64),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let small_input = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&input_f32[..2]),
                    NumericElementType::F64 => HostNumericDataView::F64(&input_f64[..2]),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        Self {
            provider,
            class,
            resident: Some(resident),
            small_input: Some(small_input),
            shared_parameter: Some(shared_parameter),
        }
    }
    fn resident(&self) -> &GpuTensorHandle {
        self.resident.as_ref().unwrap()
    }
    fn small_input(&self) -> &GpuTensorHandle {
        self.small_input.as_ref().unwrap()
    }
    fn shared_parameter(&self) -> &GpuTensorHandle {
        self.shared_parameter.as_ref().unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.resident.take().unwrap().release().unwrap();
        self.small_input.take().unwrap().release().unwrap();
        self.shared_parameter.take().unwrap().release().unwrap();
    }
}
fn route<'a>(source: DeepLearningForwardSource<'a>) -> DeepLearningForwardRoute<'a> {
    DeepLearningForwardRoute {
        input: source,
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Relu { elements: 6 },
            ),
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![2, 3]),
    }
}

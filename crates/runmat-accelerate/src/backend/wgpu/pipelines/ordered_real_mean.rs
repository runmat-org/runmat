//! Lazily compiled, class-specific Double-accumulated mean kernels.

use super::*;

pub(crate) struct OrderedRealMeanPipelines {
    first_f32_to_f64: PipelineBundle,
    f64_to_f64: PipelineBundle,
    final_f32_to_f32: PipelineBundle,
    final_f64_to_f32: PipelineBundle,
}

impl OrderedRealMeanPipelines {
    pub(crate) fn new(access: &runmat_accelerate_api::WgpuDeviceAccess) -> Self {
        let make = |layout, shader, pipeline, input, output| {
            create_pipeline(
                access,
                layout,
                shader,
                pipeline,
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                crate::backend::wgpu::shaders::reduction::ordered_mean::source(input, output),
            )
            .requiring(wgpu::Features::SHADER_INT64)
        };
        Self {
            first_f32_to_f64: make(
                "runmat-ordered-mean-32-64-layout",
                "runmat-ordered-mean-32-64-shader",
                "runmat-ordered-mean-32-64-pipeline",
                NumericPrecision::F32,
                NumericPrecision::F64,
            ),
            f64_to_f64: make(
                "runmat-ordered-mean-64-64-layout",
                "runmat-ordered-mean-64-64-shader",
                "runmat-ordered-mean-64-64-pipeline",
                NumericPrecision::F64,
                NumericPrecision::F64,
            ),
            final_f32_to_f32: make(
                "runmat-ordered-mean-32-32-layout",
                "runmat-ordered-mean-32-32-shader",
                "runmat-ordered-mean-32-32-pipeline",
                NumericPrecision::F32,
                NumericPrecision::F32,
            ),
            final_f64_to_f32: make(
                "runmat-ordered-mean-64-32-layout",
                "runmat-ordered-mean-64-32-shader",
                "runmat-ordered-mean-64-32-pipeline",
                NumericPrecision::F64,
                NumericPrecision::F32,
            ),
        }
    }

    pub(crate) fn select(
        &self,
        input: NumericPrecision,
        output: NumericPrecision,
    ) -> &PipelineBundle {
        match (input, output) {
            (NumericPrecision::F32, NumericPrecision::F64) => &self.first_f32_to_f64,
            (NumericPrecision::F64, NumericPrecision::F64) => &self.f64_to_f64,
            (NumericPrecision::F32, NumericPrecision::F32) => &self.final_f32_to_f32,
            (NumericPrecision::F64, NumericPrecision::F32) => &self.final_f64_to_f32,
        }
    }
}

use super::*;
use crate::backend::wgpu::shaders::elementwise::{binary_shader_f64, RealBinaryShaderKind};
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[test]
fn binary_variants_bind_shader_size_to_the_device_launch() {
    for workgroup in [64, 256, 512] {
        let launch = FrozenLinearLaunch::new(workgroup, 3).unwrap();
        for broadcast in [false, true] {
            let f32_source = if broadcast {
                BINARY_BROADCAST_SHADER_F32
            } else {
                BINARY_SHADER_F32
            };
            let mut variants = vec![(f32_source.to_string(), RequiredCaps::None)];
            for kind in [
                RealBinaryShaderKind::General,
                RealBinaryShaderKind::SignedZeroAtan2,
            ] {
                let caps = match kind {
                    RealBinaryShaderKind::General => RequiredCaps::Float64,
                    RealBinaryShaderKind::SignedZeroAtan2 => RequiredCaps::Float64Int64,
                };
                variants.push((binary_shader_f64(kind, broadcast), caps));
            }
            for precision in [NumericPrecision::F32, NumericPrecision::F64] {
                variants.push((crate::backend::wgpu::shaders::real_max::binary_shader(
                    precision, broadcast,
                    crate::backend::wgpu::shaders::real_max::RealMaxPolicy::IncludeNaNPreferPositiveZero,
                ), if precision == NumericPrecision::F32 { RequiredCaps::None } else { RequiredCaps::Float64Int64 }));
            }
            for (source, caps) in variants {
                let source = freeze_binary_source(&source, launch);
                assert!(source.contains(&format!("@workgroup_size({workgroup})")));
                validate_generated("device-bound-binary", &source, workgroup, 8, caps);
            }
        }
        assert_eq!(launch.chunk_capacity(), Some(workgroup as usize * 3));
        assert_eq!(launch.dispatch_groups(workgroup as usize + 1), Some(2));
    }
}

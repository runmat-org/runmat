use super::substitute_tokens;
use runmat_accelerate_api::{ProviderMatmulGeometry, ProviderPrecision};

#[test]
fn matrix_and_diagonal_sources_keep_the_provider_recipe_during_lazy_compilation() {
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        for geometry in [
            ProviderMatmulGeometry {
                tile_size: 8,
                diagonal_workgroup_size: 64,
            },
            ProviderMatmulGeometry {
                tile_size: 16,
                diagonal_workgroup_size: 128,
            },
        ] {
            let matrix = crate::backend::wgpu::shaders::matmul::epilogue_shader(
                precision,
                geometry.tile_size,
            );
            let diagonal = crate::backend::wgpu::shaders::diag::extract_shader(
                precision,
                geometry.diagonal_workgroup_size,
            );
            assert!(matrix.contains(&format!(
                "@workgroup_size({}, {}, 1)",
                geometry.tile_size, geometry.tile_size
            )));
            assert!(diagonal.contains(&format!(
                "@workgroup_size({})",
                geometry.diagonal_workgroup_size
            )));
            for source in [&matrix, &diagonal] {
                assert!(!source.contains("@MT@"));
                assert!(!source.contains("@WG@"));
                // Lazy compilation may observe another provider's old globals.
                // Fully resolved source (also hashed by the dynamic cache)
                // must remain byte-identical under that generic patch step.
                assert_eq!(
                    substitute_tokens(source, 512, 32, None).unwrap().as_str(),
                    source.as_str()
                );
            }
            let changed = crate::backend::wgpu::shaders::matmul::epilogue_shader(
                precision,
                geometry.tile_size * 2,
            );
            assert_ne!(matrix, changed);
        }
    }
}

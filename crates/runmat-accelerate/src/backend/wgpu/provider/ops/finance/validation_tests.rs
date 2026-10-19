//! Naga validation of the actual six-input pricing source, without GPU handles.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[derive(Clone, Copy, Debug)]
enum GeometryCase {
    Scalar,
    Broadcast,
    Contiguous,
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn black_scholes_source_preserves_scalar_broadcast_and_contiguous_indexing() {
    let output_shape = [2, 3];
    let scalar_shape = [1, 1];
    let scalar_strides = [1, 1];
    let price_shape = [2, 1];
    let price_strides = [1, 2];
    let strike_shape = [1, 3];
    let strike_strides = [1, 1];
    let dense_strides = [1, 2];
    let scalar = BlackScholesShaderInputGeometry {
        shape: &scalar_shape,
        strides: &scalar_strides,
    };
    let dense = BlackScholesShaderInputGeometry {
        shape: &output_shape,
        strides: &dense_strides,
    };
    let geometries = [
        (
            GeometryCase::Scalar,
            BlackScholesShaderGeometry {
                output_shape: &output_shape,
                inputs: [scalar; 6],
            },
        ),
        (
            GeometryCase::Broadcast,
            BlackScholesShaderGeometry {
                output_shape: &output_shape,
                inputs: [
                    BlackScholesShaderInputGeometry {
                        shape: &price_shape,
                        strides: &price_strides,
                    },
                    BlackScholesShaderInputGeometry {
                        shape: &strike_shape,
                        strides: &strike_strides,
                    },
                    scalar,
                    scalar,
                    scalar,
                    scalar,
                ],
            },
        ),
        (
            GeometryCase::Contiguous,
            BlackScholesShaderGeometry {
                output_shape: &output_shape,
                inputs: [dense; 6],
            },
        ),
    ];
    for (precision, caps) in [
        (NumericPrecision::F32, RequiredCaps::None),
        (NumericPrecision::F64, RequiredCaps::Float64Int64),
    ] {
        let mut sources = Vec::new();
        for (case, geometry) in &geometries {
            let source = black_scholes_price_shader(precision, geometry, 6, 1, 5);
            assert!(source.contains("if (gid.x >= 5u)"));
            assert!(source.contains("let linear = gid.x + 1u;"));
            assert!(source.contains("if (linear >= 6u)"));
            match *case {
                GeometryCase::Scalar => {
                    assert!(source.contains("return price_buf.data[0u];"));
                    assert!(source.contains("return strike_buf.data[0u];"));
                }
                GeometryCase::Broadcast => {
                    assert!(source.contains("return price_buf.data[(((linear / 1u) % 2u) * 1u)];"));
                    assert!(source.contains("return strike_buf.data[(((linear / 2u) % 3u) * 1u)];"));
                }
                GeometryCase::Contiguous => {
                    assert!(source.contains(
                        "return price_buf.data[(((linear / 1u) % 2u) * 1u) + (((linear / 2u) % 3u) * 2u)];"
                    ));
                }
            }
            validate_generated(
                &format!("black-scholes-{precision:?}-{case:?}"),
                &source,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
                8,
                caps,
            );
            sources.push(source);
        }
        assert_ne!(sources[0], sources[1]);
        assert_ne!(sources[1], sources[2]);
        assert_ne!(sources[0], sources[2]);
    }
}

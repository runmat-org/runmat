//! GPU-free checks of selected signal generators, not the entire shader inventory.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::signal::{
    self, SpectralFrameShaderConfig, SpectralFrameShaderMode, SpectralRangeShaderMode,
};
use crate::backend::wgpu::types::NumericPrecision;
use std::collections::BTreeSet;

const WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8)];

fn validate_rows(rows: Vec<(&'static str, String)>, expected: usize, precision: NumericPrecision) {
    let mut labels = BTreeSet::new();
    for (name, source) in rows {
        assert!(labels.insert(name), "duplicate signal shader case: {name}");
        let required = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64,
        };
        for &(workgroup, tile) in WORKGROUPS {
            validate_generated(
                &format!("signal-{name}-{precision:?}-wg{workgroup}-tile{tile}"),
                &source,
                workgroup,
                tile,
                required,
            );
        }
    }
    assert_eq!(labels.len(), expected, "signal shader case count changed");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn spectral_source_variants_validate_with_declared_precision() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let frame = |mode, input_complex| {
            signal::spectral_frame_shader(
                &SpectralFrameShaderConfig {
                    input_complex,
                    mode,
                    window_len: 4,
                    nfft: 8,
                    frame_count: 3,
                    input_len: 24,
                },
                precision,
            )
        };
        validate_rows(
            vec![
                (
                    "frame-sliding-real",
                    frame(SpectralFrameShaderMode::Sliding { hop: 2 }, false),
                ),
                (
                    "frame-sliding-complex",
                    frame(SpectralFrameShaderMode::Sliding { hop: 2 }, true),
                ),
                (
                    "frame-column-real",
                    frame(
                        SpectralFrameShaderMode::ColumnSliding {
                            hop: 2,
                            input_rows: 8,
                            frames_per_column: 3,
                        },
                        false,
                    ),
                ),
                (
                    "frame-column-complex",
                    frame(
                        SpectralFrameShaderMode::ColumnSliding {
                            hop: 2,
                            input_rows: 8,
                            frames_per_column: 3,
                        },
                        true,
                    ),
                ),
                (
                    "frame-folded-real",
                    frame(
                        SpectralFrameShaderMode::FoldedColumns { input_rows: 8 },
                        false,
                    ),
                ),
                (
                    "frame-folded-complex",
                    frame(
                        SpectralFrameShaderMode::FoldedColumns { input_rows: 8 },
                        true,
                    ),
                ),
            ],
            6,
            precision,
        );

        let mut rows = Vec::new();
        for (label, range, selected_rows) in [
            ("select-onesided", SpectralRangeShaderMode::Onesided, 5),
            ("select-twosided", SpectralRangeShaderMode::Twosided, 8),
            ("select-centered", SpectralRangeShaderMode::Centered, 8),
        ] {
            rows.push((
                label,
                signal::spectral_select_shader(8, selected_rows, range, precision),
            ));
        }
        validate_rows(rows, 3, precision);

        validate_rows(
            vec![
                (
                    "power-onesided-even",
                    signal::spectral_power_shader(
                        5,
                        SpectralRangeShaderMode::Onesided,
                        true,
                        8.0,
                        precision,
                    ),
                ),
                (
                    "power-onesided-odd",
                    signal::spectral_power_shader(
                        5,
                        SpectralRangeShaderMode::Onesided,
                        false,
                        8.0,
                        precision,
                    ),
                ),
                (
                    "power-twosided-even",
                    signal::spectral_power_shader(
                        8,
                        SpectralRangeShaderMode::Twosided,
                        true,
                        8.0,
                        precision,
                    ),
                ),
                (
                    "power-twosided-odd",
                    signal::spectral_power_shader(
                        8,
                        SpectralRangeShaderMode::Twosided,
                        false,
                        8.0,
                        precision,
                    ),
                ),
                (
                    "power-centered-even",
                    signal::spectral_power_shader(
                        8,
                        SpectralRangeShaderMode::Centered,
                        true,
                        8.0,
                        precision,
                    ),
                ),
                (
                    "power-centered-odd",
                    signal::spectral_power_shader(
                        8,
                        SpectralRangeShaderMode::Centered,
                        false,
                        8.0,
                        precision,
                    ),
                ),
            ],
            6,
            precision,
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn envelope_source_variants_validate_with_declared_precision() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        validate_rows(
            vec![
                (
                    "center-real-to-complex",
                    signal::envelope_center_real_to_complex_shader(8, 3, precision),
                ),
                (
                    "analytic-mask",
                    signal::envelope_analytic_mask_shader(8, 3, precision),
                ),
                (
                    "analytic-signal-mask",
                    signal::analytic_signal_mask_shader(8, 2, 3, precision),
                ),
                (
                    "analytic-bounds",
                    signal::envelope_analytic_bounds_shader(8, 3, precision),
                ),
                (
                    "rms-bounds",
                    signal::envelope_rms_bounds_shader(8, 3, 5, precision),
                ),
                (
                    "analytic-fir-bounds",
                    signal::envelope_analytic_fir_bounds_shader(8, 3, 5, precision),
                ),
            ],
            6,
            precision,
        );
    }
}

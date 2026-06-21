use crate::backend::wgpu::types::NumericPrecision;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpectralFrameShaderConfig {
    pub input_complex: bool,
    pub mode: SpectralFrameShaderMode,
    pub window_len: usize,
    pub nfft: usize,
    pub frame_count: usize,
    pub input_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpectralFrameShaderMode {
    Sliding { hop: usize },
    FoldedColumns { input_rows: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpectralRangeShaderMode {
    Onesided,
    Twosided,
    Centered,
}

pub(crate) fn spectral_frame_shader(
    config: &SpectralFrameShaderConfig,
    precision: NumericPrecision,
) -> String {
    let (ty, zero, _) = spectral_shader_numeric_fragments(precision);
    let input_complex = u32::from(config.input_complex);
    let (mode, hop, input_rows) = match config.mode {
        SpectralFrameShaderMode::Sliding { hop } => (0u32, hop, 0usize),
        SpectralFrameShaderMode::FoldedColumns { input_rows } => (2u32, 0usize, input_rows),
    };
    let window_len = config.window_len;
    let nfft = config.nfft;
    let frame_count = config.frame_count;
    let input_len = config.input_len;
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad1: u32,
    _pad2: u32,
}};

@group(0) @binding(0) var<storage, read> Signal: Tensor;
@group(0) @binding(1) var<storage, read> Window: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {nfft}u * {frame_count}u * 2u;
    if idx >= total {{
        return;
    }}

    let complex_idx = idx / 2u;
    let part = idx % 2u;
    let row = complex_idx % {nfft}u;
    let col = complex_idx / {nfft}u;
    var value = {zero};
    if row < {window_len}u {{
        var source_index: u32;
        if {mode}u == 0u {{
            source_index = col * {hop}u + row;
        }} else if {mode}u == 1u {{
            source_index = col * {input_rows}u + row;
        }} else {{
            source_index = col * {input_rows}u + row;
        }}
        if source_index < {input_len}u {{
            let window_value = Window.data[row];
            if {input_complex}u != 0u {{
                let base = source_index * 2u + part;
                value = Signal.data[base] * window_value;
            }} else if part == 0u {{
                value = Signal.data[source_index] * window_value;
            }}
        }}
    }}
    if {mode}u == 2u {{
        value = {zero};
        var folded_row = row;
        loop {{
            if folded_row >= {window_len}u {{
                break;
            }}
            let source_index = col * {input_rows}u + folded_row;
            if source_index < {input_len}u {{
                let window_value = Window.data[folded_row];
                if {input_complex}u != 0u {{
                    let base = source_index * 2u + part;
                    value = value + Signal.data[base] * window_value;
                }} else if part == 0u {{
                    value = value + Signal.data[source_index] * window_value;
                }}
            }}
            folded_row = folded_row + {nfft}u;
        }}
    }}
    Out.data[idx] = value;
}}
"#,
        ty = ty,
        zero = zero,
        nfft = nfft,
        frame_count = frame_count,
        window_len = window_len,
        mode = mode,
        hop = hop,
        input_rows = input_rows,
        input_len = input_len,
        input_complex = input_complex,
    )
}

pub(crate) fn spectral_select_shader(
    nfft: usize,
    rows: usize,
    range: SpectralRangeShaderMode,
    precision: NumericPrecision,
) -> String {
    let (ty, _, _) = spectral_shader_numeric_fragments(precision);
    let mode = match range {
        SpectralRangeShaderMode::Onesided => 0u32,
        SpectralRangeShaderMode::Centered => 1u32,
        SpectralRangeShaderMode::Twosided => 2u32,
    };
    let centered_shift = spectral_centered_shift(nfft);
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad1: u32,
    _pad2: u32,
}};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let complex_idx = idx / 2u;
    let part = idx % 2u;
    let row = complex_idx % {rows}u;
    let col = complex_idx / {rows}u;
    var src_row = row;
    if {mode}u == 1u {{
        src_row = (row + {centered_shift}u) % {nfft}u;
    }}
    let src_complex = col * {nfft}u + src_row;
    Out.data[idx] = Input.data[src_complex * 2u + part];
}}
"#,
        ty = ty,
        rows = rows,
        mode = mode,
        centered_shift = centered_shift,
        nfft = nfft,
    )
}

pub(crate) fn spectral_power_shader(
    rows: usize,
    range: SpectralRangeShaderMode,
    has_nyquist: bool,
    denominator: f64,
    precision: NumericPrecision,
) -> String {
    let (ty, _, cast) = spectral_shader_numeric_fragments(precision);
    let double_one_sided = u32::from(matches!(range, SpectralRangeShaderMode::Onesided));
    let has_nyquist = u32::from(has_nyquist);
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad1: u32,
    _pad2: u32,
}};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let row = idx % {rows}u;
    let re = Input.data[idx * 2u];
    let im = Input.data[idx * 2u + 1u];
    var scale = {cast}(1.0);
    if {double_one_sided}u != 0u {{
        let is_dc = row == 0u;
        let is_nyquist = {has_nyquist}u != 0u && row == {last_row}u;
        if !is_dc && !is_nyquist {{
            scale = {cast}(2.0);
        }}
    }}
    Out.data[idx] = ((re * re) + (im * im)) * scale / {cast}({denominator});
}}
"#,
        ty = ty,
        rows = rows,
        cast = cast,
        double_one_sided = double_one_sided,
        has_nyquist = has_nyquist,
        last_row = rows.saturating_sub(1),
        denominator = denominator,
    )
}

fn spectral_centered_shift(nfft: usize) -> usize {
    if nfft.is_multiple_of(2) {
        nfft / 2 + 1
    } else {
        nfft.div_ceil(2)
    }
}

fn spectral_shader_numeric_fragments(
    precision: NumericPrecision,
) -> (&'static str, &'static str, &'static str) {
    match precision {
        NumericPrecision::F64 => ("f64", "0.0", "f64"),
        NumericPrecision::F32 => ("f32", "0.0", "f32"),
    }
}

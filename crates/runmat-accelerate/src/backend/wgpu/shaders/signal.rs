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
    Sliding {
        hop: usize,
    },
    ColumnSliding {
        hop: usize,
        input_rows: usize,
        frames_per_column: usize,
    },
    FoldedColumns {
        input_rows: usize,
    },
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
    let (mode, hop, input_rows, frames_per_column) = match config.mode {
        SpectralFrameShaderMode::Sliding { hop } => (0u32, hop, 0usize, 1usize),
        SpectralFrameShaderMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        } => (1u32, hop, input_rows, frames_per_column),
        SpectralFrameShaderMode::FoldedColumns { input_rows } => (2u32, 0usize, input_rows, 1usize),
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
            let segment = col % {frames_per_column}u;
            let source_col = col / {frames_per_column}u;
            source_index = source_col * {input_rows}u + segment * {hop}u + row;
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
        frames_per_column = frames_per_column,
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

pub(crate) fn envelope_center_real_to_complex_shader(
    channel_len: usize,
    channel_count: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, zero, cast) = spectral_shader_numeric_fragments(precision);
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

fn channel_mean(col: u32) -> {ty} {{
    var sum = {zero};
    var row = 0u;
    loop {{
        if row >= {channel_len}u {{
            break;
        }}
        sum = sum + Input.data[col * {channel_len}u + row];
        row = row + 1u;
    }}
    return sum / {cast}({channel_len}.0);
}}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {channel_len}u * {channel_count}u * 2u;
    if idx >= total {{
        return;
    }}
    let complex_idx = idx / 2u;
    let part = idx % 2u;
    let row = complex_idx % {channel_len}u;
    let col = complex_idx / {channel_len}u;
    if part == 0u {{
        Out.data[idx] = Input.data[col * {channel_len}u + row] - channel_mean(col);
    }} else {{
        Out.data[idx] = {zero};
    }}
}}
"#,
        ty = ty,
        zero = zero,
        cast = cast,
        channel_len = channel_len,
        channel_count = channel_count,
    )
}

pub(crate) fn envelope_analytic_mask_shader(
    channel_len: usize,
    channel_count: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, _, cast) = spectral_shader_numeric_fragments(precision);
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

fn analytic_multiplier(row: u32) -> {ty} {{
    if row == 0u {{
        return {cast}(1.0);
    }}
    if ({channel_len}u % 2u) == 0u {{
        if row < ({channel_len}u / 2u) {{
            return {cast}(2.0);
        }}
        if row == ({channel_len}u / 2u) {{
            return {cast}(1.0);
        }}
        return {cast}(0.0);
    }}
    if row <= ({channel_len}u / 2u) {{
        return {cast}(2.0);
    }}
    return {cast}(0.0);
}}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {channel_len}u * {channel_count}u * 2u;
    if idx >= total {{
        return;
    }}
    let row = (idx / 2u) % {channel_len}u;
    Out.data[idx] = Input.data[idx] * analytic_multiplier(row);
}}
"#,
        ty = ty,
        cast = cast,
        channel_len = channel_len,
        channel_count = channel_count,
    )
}

pub(crate) fn analytic_signal_mask_shader(
    transform_len: usize,
    inner_stride: usize,
    total_lanes: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, _, cast) = spectral_shader_numeric_fragments(precision);
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

fn analytic_multiplier(freq: u32) -> {ty} {{
    if freq == 0u {{
        return {cast}(1.0);
    }}
    if ({transform_len}u % 2u) == 0u {{
        if freq < ({transform_len}u / 2u) {{
            return {cast}(2.0);
        }}
        if freq == ({transform_len}u / 2u) {{
            return {cast}(1.0);
        }}
        return {cast}(0.0);
    }}
    if freq <= ({transform_len}u / 2u) {{
        return {cast}(2.0);
    }}
    return {cast}(0.0);
}}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= {total_lanes}u {{
        return;
    }}
    let element = idx / 2u;
    let freq = (element / {inner_stride}u) % {transform_len}u;
    Out.data[idx] = Input.data[idx] * analytic_multiplier(freq);
}}
"#,
        ty = ty,
        cast = cast,
        transform_len = transform_len,
        inner_stride = inner_stride,
        total_lanes = total_lanes,
    )
}

pub(crate) fn envelope_analytic_bounds_shader(
    channel_len: usize,
    channel_count: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, zero, cast) = spectral_shader_numeric_fragments(precision);
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
@group(0) @binding(1) var<storage, read> Analytic: Tensor;
@group(0) @binding(2) var<storage, read_write> Upper: Tensor;
@group(0) @binding(3) var<storage, read_write> Lower: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

fn channel_mean(col: u32) -> {ty} {{
    var sum = {zero};
    var row = 0u;
    loop {{
        if row >= {channel_len}u {{
            break;
        }}
        sum = sum + Input.data[col * {channel_len}u + row];
        row = row + 1u;
    }}
    return sum / {cast}({channel_len}.0);
}}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {channel_len}u * {channel_count}u;
    if idx >= total {{
        return;
    }}
    let row = idx % {channel_len}u;
    let col = idx / {channel_len}u;
    let base = (col * {channel_len}u + row) * 2u;
    let re = Analytic.data[base];
    let im = Analytic.data[base + 1u];
    let mag = sqrt(re * re + im * im);
    let mean = channel_mean(col);
    Upper.data[idx] = mean + mag;
    Lower.data[idx] = mean - mag;
}}
"#,
        ty = ty,
        zero = zero,
        cast = cast,
        channel_len = channel_len,
        channel_count = channel_count,
    )
}

pub(crate) fn envelope_rms_bounds_shader(
    channel_len: usize,
    channel_count: usize,
    window_len: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, zero, cast) = spectral_shader_numeric_fragments(precision);
    let half_before = (window_len - 1) / 2;
    let half_after = window_len / 2;
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
@group(0) @binding(1) var<storage, read_write> Upper: Tensor;
@group(0) @binding(2) var<storage, read_write> Lower: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {channel_len}u * {channel_count}u;
    if idx >= total {{
        return;
    }}
    let row = idx % {channel_len}u;
    let col = idx / {channel_len}u;
    let start = select(0u, row - {half_before}u, row >= {half_before}u);
    let end_raw = row + {half_after}u + 1u;
    let end = min(end_raw, {channel_len}u);
    var sum = {zero};
    var source = start;
    loop {{
        if source >= end {{
            break;
        }}
        let value = Input.data[col * {channel_len}u + source];
        sum = sum + value * value;
        source = source + 1u;
    }}
    let count = {cast}(end - start);
    let upper = sqrt(sum / count);
    Upper.data[idx] = upper;
    Lower.data[idx] = -upper;
}}
"#,
        ty = ty,
        zero = zero,
        cast = cast,
        channel_len = channel_len,
        channel_count = channel_count,
        half_before = half_before,
        half_after = half_after,
    )
}

pub(crate) fn envelope_analytic_fir_bounds_shader(
    channel_len: usize,
    channel_count: usize,
    filter_len: usize,
    precision: NumericPrecision,
) -> String {
    let (ty, zero, cast) = spectral_shader_numeric_fragments(precision);
    let center = filter_len / 2;
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
@group(0) @binding(1) var<storage, read> Kernel: Tensor;
@group(0) @binding(2) var<storage, read_write> Upper: Tensor;
@group(0) @binding(3) var<storage, read_write> Lower: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

fn channel_mean(col: u32) -> {ty} {{
    var sum = {zero};
    var row = 0u;
    loop {{
        if row >= {channel_len}u {{
            break;
        }}
        sum = sum + Input.data[col * {channel_len}u + row];
        row = row + 1u;
    }}
    return sum / {cast}({channel_len}.0);
}}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    let total = {channel_len}u * {channel_count}u;
    if idx >= total {{
        return;
    }}
    let row = idx % {channel_len}u;
    let col = idx / {channel_len}u;
    let mean = channel_mean(col);
    let centered = Input.data[col * {channel_len}u + row] - mean;
    var quadrature = {zero};
    var tap = 0u;
    loop {{
        if tap >= {filter_len}u {{
            break;
        }}
        let shifted = i32(row) + i32(tap) - i32({center}u);
        if shifted >= 0 && shifted < i32({channel_len}u) {{
            let source = u32(shifted);
            let source_value = Input.data[col * {channel_len}u + source] - mean;
            quadrature = quadrature + source_value * Kernel.data[tap];
        }}
        tap = tap + 1u;
    }}
    let mag = sqrt(centered * centered + quadrature * quadrature);
    Upper.data[idx] = mean + mag;
    Lower.data[idx] = mean - mag;
}}
"#,
        ty = ty,
        zero = zero,
        cast = cast,
        channel_len = channel_len,
        channel_count = channel_count,
        filter_len = filter_len,
        center = center,
    )
}

fn spectral_centered_shift(nfft: usize) -> usize {
    if nfft.is_multiple_of(2) {
        nfft / 2
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

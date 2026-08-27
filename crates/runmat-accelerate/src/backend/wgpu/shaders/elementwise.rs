use crate::backend::wgpu::types::NumericPrecision;
use crate::backend::wgpu::types::UnaryOpCode;

fn wgsl_function(source: &str, name: &str) -> String {
    let marker = format!("fn {name}(");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing unary WGSL helper {name}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for unary WGSL helper {name}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=open + offset].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated unary WGSL helper {name}");
}

fn wgsl_switch_case(source: &str, op: UnaryOpCode) -> String {
    let marker = format!("case {}u:", op as u32);
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing unary WGSL case {}", op as u32));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for unary WGSL case {}", op as u32));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source[open + 1..open + offset].trim().to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated unary WGSL case {}", op as u32);
}

fn unary_helper_names(op: UnaryOpCode, precision: NumericPrecision) -> &'static [&'static str] {
    match op {
        UnaryOpCode::Expm1 => &["expm1_precise"],
        UnaryOpCode::Log1p => &["log1p_precise"],
        UnaryOpCode::Gamma => match precision {
            NumericPrecision::F64 => &[
                "lanczos_gamma",
                "is_non_positive_integer",
                "is_nan64",
                "pos_inf_f64",
                "neg_inf_f64",
                "nan_f64",
                "is_inf64",
                "gamma_real",
            ],
            NumericPrecision::F32 => &[
                "lanczos_gamma",
                "is_non_positive_integer",
                "is_nan32",
                "pos_inf_f32",
                "neg_inf_f32",
                "nan_f32",
                "is_inf32",
                "gamma_real",
            ],
        },
        UnaryOpCode::Factorial => match precision {
            NumericPrecision::F64 => &[
                "is_nan64",
                "pos_inf_f64",
                "neg_inf_f64",
                "nan_f64",
                "is_inf64",
                "factorial_real",
            ],
            NumericPrecision::F32 => &[
                "is_nan32",
                "pos_inf_f32",
                "neg_inf_f32",
                "nan_f32",
                "is_inf32",
                "factorial_real",
            ],
        },
        UnaryOpCode::Sinc => match precision {
            NumericPrecision::F64 => &[
                "is_nan64",
                "pos_inf_f64",
                "neg_inf_f64",
                "is_inf64",
                "sinc_real",
            ],
            NumericPrecision::F32 => &[
                "is_nan32",
                "pos_inf_f32",
                "neg_inf_f32",
                "is_inf32",
                "sinc_real",
            ],
        },
        UnaryOpCode::Heaviside => match precision {
            NumericPrecision::F64 => &["is_nan64", "heaviside_real"],
            NumericPrecision::F32 => &["is_nan32", "heaviside_real"],
        },
        UnaryOpCode::Round => match precision {
            NumericPrecision::F64 => &["is_nan64", "pos_inf_f64", "neg_inf_f64", "is_inf64"],
            NumericPrecision::F32 => &["is_nan32", "pos_inf_f32", "neg_inf_f32", "is_inf32"],
        },
        UnaryOpCode::Erf => match precision {
            NumericPrecision::F64 => &[
                "is_nan64",
                "pos_inf_f64",
                "neg_inf_f64",
                "is_inf64",
                "erf_real",
            ],
            NumericPrecision::F32 => &[
                "is_nan32",
                "pos_inf_f32",
                "neg_inf_f32",
                "is_inf32",
                "erf_real",
            ],
        },
        UnaryOpCode::Gammaln => match precision {
            NumericPrecision::F64 => {
                &["lanczos_gammaln", "is_nan64", "pos_inf_f64", "gammaln_real"]
            }
            NumericPrecision::F32 => {
                &["lanczos_gammaln", "is_nan32", "pos_inf_f32", "gammaln_real"]
            }
        },
        UnaryOpCode::Erfcinv => match precision {
            NumericPrecision::F64 => &[
                "is_nan64",
                "pos_inf_f64",
                "neg_inf_f64",
                "is_inf64",
                "erf_real",
                "erfc_positive_tail_real",
                "erfcinv_positive_tail_real",
                "erfcinv_real",
            ],
            NumericPrecision::F32 => &[
                "is_nan32",
                "pos_inf_f32",
                "neg_inf_f32",
                "is_inf32",
                "erf_real",
                "erfc_positive_tail_real",
                "erfcinv_positive_tail_real",
                "erfcinv_real",
            ],
        },
        _ => &[],
    }
}

fn inline_gammaln_body(precision: NumericPrecision) -> String {
    let (ty, max_finite, small_cutoff, epsilon) = match precision {
        NumericPrecision::F64 => ("f64", "1.7976931348623157e308", "1.0e-305", "1.0e-12"),
        NumericPrecision::F32 => ("f32", "3.4028234663852886e38", "1.0e-30", "1.0e-5"),
    };
    format!(
        r#"
if a != a {{ return a; }}
if a == {ty}(0.0) || a > {ty}({max_finite}) {{ return {ty}(1.0) / {ty}(0.0); }}
if a < {ty}(0.0) {{ return {ty}(0.0) / {ty}(0.0); }}
if a < {ty}({small_cutoff}) {{ return -log(a); }}
var z = a;
if a < {ty}(0.5) {{ z = {ty}(1.0) - a; }}
let z_minus_one = z - {ty}(1.0);
var sum: {ty} = {ty}(0.99999999999980993);
sum = sum + {ty}(676.5203681218851) / (z_minus_one + {ty}(1.0));
sum = sum + {ty}(-1259.1392167224028) / (z_minus_one + {ty}(2.0));
sum = sum + {ty}(771.3234287776531) / (z_minus_one + {ty}(3.0));
sum = sum + {ty}(-176.6150291621406) / (z_minus_one + {ty}(4.0));
sum = sum + {ty}(12.507343278686905) / (z_minus_one + {ty}(5.0));
sum = sum + {ty}(-0.13857109526572012) / (z_minus_one + {ty}(6.0));
sum = sum + {ty}(0.000009984369578019572) / (z_minus_one + {ty}(7.0));
sum = sum + {ty}(0.00000015056327351493116) / (z_minus_one + {ty}(8.0));
let t = z_minus_one + {ty}(7.5);
let value = {ty}(0.9189385332046727) + (z_minus_one + {ty}(0.5)) * log(t) - t + log(sum);
if a < {ty}(0.5) {{
    let sin_term = sin({ty}(3.141592653589793) * a);
    if abs(sin_term) <= {ty}({epsilon}) {{ return {ty}(1.0) / {ty}(0.0); }}
    return log({ty}(3.141592653589793)) - log(sin_term) - value;
}}
return value;
"#
    )
}

fn inline_erfcinv_body(precision: NumericPrecision) -> String {
    // Some D3D12 drivers reject kernels that construct non-finite floating-point
    // values in shader arithmetic. Encode every result as its IEEE storage bits;
    // ordinary finite lanes are still computed by the GPU before the bitcast.
    let (ty, encode, positive_infinity, negative_infinity, nan) = match precision {
        NumericPrecision::F64 => (
            "f64",
            "bitcast<vec2<u32>>",
            "vec2<u32>(0u, 0x7ff00000u)",
            "vec2<u32>(0u, 0xfff00000u)",
            "vec2<u32>(0u, 0x7ff80000u)",
        ),
        NumericPrecision::F32 => (
            "f32",
            "bitcast<u32>",
            "0x7f800000u",
            "0xff800000u",
            "0x7fc00000u",
        ),
    };
    format!(
        r#"
if a != a {{ return {encode}(a); }}
if a < {ty}(0.0) || a > {ty}(2.0) {{ return {nan}; }}
if a == {ty}(0.0) {{ return {positive_infinity}; }}
if a == {ty}(2.0) {{ return {negative_infinity}; }}
if a == {ty}(1.0) {{ return {encode}({ty}(0.0)); }}
let p = a * {ty}(0.5);
var normal: {ty};
if p < {ty}(0.02425) {{
    let q = sqrt(-{ty}(2.0) * log(p));
    var numerator = {ty}(-0.007784894002430293);
    numerator = numerator * q + {ty}(-0.3223964580411365);
    numerator = numerator * q + {ty}(-2.400758277161838);
    numerator = numerator * q + {ty}(-2.549732539343734);
    numerator = numerator * q + {ty}(4.374664141464968);
    numerator = numerator * q + {ty}(2.938163982698783);
    var denominator = {ty}(0.007784695709041462);
    denominator = denominator * q + {ty}(0.3224671290700398);
    denominator = denominator * q + {ty}(2.445134137142996);
    denominator = denominator * q + {ty}(3.754408661907416);
    denominator = denominator * q + {ty}(1.0);
    normal = numerator / denominator;
}} else if p > {ty}(0.97575) {{
    let q = sqrt(-{ty}(2.0) * log({ty}(1.0) - p));
    var numerator = {ty}(-0.007784894002430293);
    numerator = numerator * q + {ty}(-0.3223964580411365);
    numerator = numerator * q + {ty}(-2.400758277161838);
    numerator = numerator * q + {ty}(-2.549732539343734);
    numerator = numerator * q + {ty}(4.374664141464968);
    numerator = numerator * q + {ty}(2.938163982698783);
    var denominator = {ty}(0.007784695709041462);
    denominator = denominator * q + {ty}(0.3224671290700398);
    denominator = denominator * q + {ty}(2.445134137142996);
    denominator = denominator * q + {ty}(3.754408661907416);
    denominator = denominator * q + {ty}(1.0);
    normal = -(numerator / denominator);
}} else {{
    let q = p - {ty}(0.5);
    let r = q * q;
    var numerator = {ty}(-39.69683028665376);
    numerator = numerator * r + {ty}(220.9460984245205);
    numerator = numerator * r + {ty}(-275.9285104469687);
    numerator = numerator * r + {ty}(138.3577518672690);
    numerator = numerator * r + {ty}(-30.66479806614716);
    numerator = numerator * r + {ty}(2.506628277459239);
    var denominator = {ty}(-54.47609879822406);
    denominator = denominator * r + {ty}(161.5858368580409);
    denominator = denominator * r + {ty}(-155.6989798598866);
    denominator = denominator * r + {ty}(66.80131188771972);
    denominator = denominator * r + {ty}(-13.28068155288572);
    denominator = denominator * r + {ty}(1.0);
    normal = (numerator * q) / denominator;
}}
return {encode}(-normal * {ty}(0.7071067811865476));
"#
    )
}

pub(crate) fn real_unary_shader(op: UnaryOpCode, precision: NumericPrecision) -> String {
    let (template, ty) = match precision {
        NumericPrecision::F64 => (UNARY_SHADER_F64, "f64"),
        NumericPrecision::F32 => (UNARY_SHADER_F32, "f32"),
    };
    let constants_start = template
        .find("const PI:")
        .expect("unary WGSL constants must start with PI");
    let constants_end = template[constants_start..]
        .find("fn expm1_precise")
        .map(|offset| constants_start + offset)
        .expect("unary WGSL constants must precede expm1");
    let constants = &template[constants_start..constants_end];
    let helper_names: &[&str] = if matches!(op, UnaryOpCode::Gammaln | UnaryOpCode::Erfcinv) {
        &[]
    } else {
        unary_helper_names(op, precision)
    };
    let helpers = helper_names
        .iter()
        .map(|name| wgsl_function(template, name))
        .collect::<Vec<_>>()
        .join("\n\n");
    let case_body = if matches!(op, UnaryOpCode::Gammaln) {
        inline_gammaln_body(precision)
    } else if matches!(op, UnaryOpCode::Erfcinv) {
        inline_erfcinv_body(precision)
    } else {
        wgsl_switch_case(template, op)
    };
    let output_ty = match (op, precision) {
        (UnaryOpCode::Erfcinv, NumericPrecision::F64) => "vec2<u32>",
        (UnaryOpCode::Erfcinv, NumericPrecision::F32) => "u32",
        _ => ty,
    };
    format!(
        "struct InputTensor {{ data: array<{ty}> }};\nstruct OutputTensor {{ data: array<{output_ty}> }};\nstruct Params {{ len: u32, offset: u32, _pad1: u32, _pad2: u32 }};\n{constants}\n@group(0) @binding(0) var<storage, read> A: InputTensor;\n@group(0) @binding(1) var<storage, read_write> Out: OutputTensor;\n@group(0) @binding(2) var<uniform> params: Params;\n{helpers}\nfn apply(a: {ty}) -> {output_ty} {{ {case_body} }}\n@compute @workgroup_size(@WG@)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let local = gid.x;\n    if local >= params.len {{ return; }}\n    let idx = params.offset + local;\n    Out.data[idx] = apply(A.data[idx]);\n}}\n"
    )
}

#[cfg(test)]
mod real_unary_tests {
    use super::*;

    const OPS: [UnaryOpCode; 39] = [
        UnaryOpCode::Sin,
        UnaryOpCode::Cos,
        UnaryOpCode::Abs,
        UnaryOpCode::Exp,
        UnaryOpCode::Log,
        UnaryOpCode::Sqrt,
        UnaryOpCode::Sign,
        UnaryOpCode::Real,
        UnaryOpCode::Imag,
        UnaryOpCode::Conj,
        UnaryOpCode::Angle,
        UnaryOpCode::Expm1,
        UnaryOpCode::Log1p,
        UnaryOpCode::Log10,
        UnaryOpCode::Log2,
        UnaryOpCode::Pow2,
        UnaryOpCode::Floor,
        UnaryOpCode::Ceil,
        UnaryOpCode::Fix,
        UnaryOpCode::Tan,
        UnaryOpCode::Asin,
        UnaryOpCode::Acos,
        UnaryOpCode::Atan,
        UnaryOpCode::Sinh,
        UnaryOpCode::Cosh,
        UnaryOpCode::Tanh,
        UnaryOpCode::Asinh,
        UnaryOpCode::Acosh,
        UnaryOpCode::Atanh,
        UnaryOpCode::Gamma,
        UnaryOpCode::Factorial,
        UnaryOpCode::Single,
        UnaryOpCode::NextPow2,
        UnaryOpCode::Sinc,
        UnaryOpCode::Heaviside,
        UnaryOpCode::Erf,
        UnaryOpCode::Gammaln,
        UnaryOpCode::Round,
        UnaryOpCode::Erfcinv,
    ];

    #[test]
    fn every_real_unary_opcode_specializes_for_both_precisions() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            for op in OPS {
                let shader = real_unary_shader(op, precision);
                assert!(shader.contains("fn apply("));
                assert!(shader.contains("@workgroup_size(@WG@)"));
                assert!(!shader.contains("switch params.op"));
            }
        }
    }

    #[test]
    fn simple_real_unary_shader_excludes_special_function_helpers() {
        let shader = real_unary_shader(UnaryOpCode::Abs, NumericPrecision::F32);
        assert!(!shader.contains("fn lanczos_"));
        assert!(!shader.contains("fn erf_real"));
        assert!(!shader.contains("fn factorial_real"));
    }

    #[test]
    fn erfcinv_shader_writes_portable_ieee_storage_bits() {
        let f32_shader = real_unary_shader(UnaryOpCode::Erfcinv, NumericPrecision::F32);
        assert!(f32_shader.contains("struct OutputTensor { data: array<u32> }"));
        assert!(f32_shader.contains("return 0x7f800000u"));
        assert!(f32_shader.contains("return 0xff800000u"));
        assert!(f32_shader.contains("return 0x7fc00000u"));

        let f64_shader = real_unary_shader(UnaryOpCode::Erfcinv, NumericPrecision::F64);
        assert!(f64_shader.contains("struct OutputTensor { data: array<vec2<u32>> }"));
        assert!(f64_shader.contains("return vec2<u32>(0u, 0x7ff00000u)"));
        assert!(f64_shader.contains("return vec2<u32>(0u, 0xfff00000u)"));
        assert!(f64_shader.contains("return vec2<u32>(0u, 0x7ff80000u)"));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComplexUnaryOp {
    Real,
    Imag,
    Abs,
    Conj,
    Angle,
    Sin,
    Sinc,
    Cos,
    Sinh,
    Cosh,
    Tan,
    Sign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComplexBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ComplexBinaryOp {
    pub(crate) fn try_from_binary_op(
        op: crate::backend::wgpu::types::BinaryOpCode,
    ) -> Option<Self> {
        match op {
            crate::backend::wgpu::types::BinaryOpCode::Add => Some(Self::Add),
            crate::backend::wgpu::types::BinaryOpCode::Sub => Some(Self::Sub),
            crate::backend::wgpu::types::BinaryOpCode::Mul => Some(Self::Mul),
            crate::backend::wgpu::types::BinaryOpCode::Div => Some(Self::Div),
            _ => None,
        }
    }
}

pub(crate) fn complex_unary_shader(op: ComplexUnaryOp, precision: NumericPrecision) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let half_max_finite = match precision {
        NumericPrecision::F64 => "8.988465674311579e307",
        NumericPrecision::F32 => "1.7014117331926443e38",
    };
    let sign_inf_helper = format!(
        r#"
const MAX_FINITE_SIGN_COMPLEX_UNARY: {ty} = {ty}({max_finite});

fn isinf_complex_unary(x: {ty}) -> bool {{
    return abs(x) > MAX_FINITE_SIGN_COMPLEX_UNARY;
}}
"#,
        ty = ty,
        max_finite = max_finite,
    );
    let extra_helpers = match op {
        ComplexUnaryOp::Abs => format!(
            r#"
fn complex_abs_lane(out_idx: u32) -> {ty} {{
    let re = abs(A.data[out_idx * 2u]);
    let im = abs(A.data[out_idx * 2u + 1u]);
    if re > {ty}({max_finite}) {{
        return re;
    }}
    if im > {ty}({max_finite}) {{
        return im;
    }}
    let scale = max(re, im);
    if scale == {ty}(0.0) {{
        return {ty}(0.0);
    }}
    if scale > {ty}({max_finite}) {{
        return scale;
    }}
    let sr = re / scale;
    let si = im / scale;
    return scale * sqrt((sr * sr) + (si * si));
}}
"#,
            ty = ty,
            max_finite = max_finite,
        ),
        ComplexUnaryOp::Sinc => format!(
            r#"
const PI_COMPLEX_UNARY: {ty} = {ty}(3.141592653589793);
const MAX_FINITE_COMPLEX_UNARY: {ty} = {ty}({max_finite});

fn isfinite_complex_unary(x: {ty}) -> bool {{
    return (x == x) && (abs(x) < MAX_FINITE_COMPLEX_UNARY);
}}

fn signum_nonzero_complex_unary(x: {ty}) -> {ty} {{
    return select(-{ty}(1.0), {ty}(1.0), x > {ty}(0.0));
}}

fn finite_product_complex_unary(a: {ty}, b: {ty}) -> {ty} {{
    if a == {ty}(0.0) || b == {ty}(0.0) {{
        return {ty}(0.0);
    }}
    let product = a * b;
    if isfinite_complex_unary(product) || product != product {{
        return product;
    }}
    return signum_nonzero_complex_unary(a) * signum_nonzero_complex_unary(b) * MAX_FINITE_COMPLEX_UNARY;
}}

fn finite_sum_complex_unary(a: {ty}, b: {ty}) -> {ty} {{
    let sum = a + b;
    if isfinite_complex_unary(sum) || sum != sum {{
        return sum;
    }}
    return signum_nonzero_complex_unary(a) * MAX_FINITE_COMPLEX_UNARY;
}}

fn sinc_real_complex_unary(x: {ty}) -> {ty} {{
    if x == {ty}(0.0) {{
        return {ty}(1.0);
    }}
    let abs_x = abs(x);
    if isfinite_complex_unary(x) && floor(abs_x) == abs_x {{
        return {ty}(0.0);
    }}
    let scaled = PI_COMPLEX_UNARY * x;
    return sin(scaled) / scaled;
}}

fn sinc_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    if im == {ty}(0.0) {{
        if (out_idx % 2u) == 0u {{
            return sinc_real_complex_unary(re);
        }}
        return {ty}(0.0);
    }}

    let scaled_re = PI_COMPLEX_UNARY * re;
    let scaled_im = PI_COMPLEX_UNARY * im;
    let num_re = finite_product_complex_unary(sin(scaled_re), cosh(scaled_im));
    let num_im = finite_product_complex_unary(cos(scaled_re), sinh(scaled_im));
    let denom_norm = (scaled_re * scaled_re) + (scaled_im * scaled_im);
    let out_re = finite_sum_complex_unary(
        finite_product_complex_unary(num_re, scaled_re),
        finite_product_complex_unary(num_im, scaled_im),
    ) / denom_norm;
    let out_im = finite_sum_complex_unary(
        finite_product_complex_unary(num_im, scaled_re),
        -finite_product_complex_unary(num_re, scaled_im),
    ) / denom_norm;
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}
"#,
            ty = ty,
            max_finite = max_finite,
        ),
        ComplexUnaryOp::Sin
        | ComplexUnaryOp::Cos
        | ComplexUnaryOp::Sinh
        | ComplexUnaryOp::Cosh
        | ComplexUnaryOp::Tan => format!(
            r#"
const MAX_FINITE_COMPLEX_UNARY: {ty} = {ty}({max_finite});

fn isfinite_complex_unary(x: {ty}) -> bool {{
    return (x == x) && (abs(x) < MAX_FINITE_COMPLEX_UNARY);
}}

fn signum_nonzero_complex_unary(x: {ty}) -> {ty} {{
    return select(-{ty}(1.0), {ty}(1.0), x > {ty}(0.0));
}}

fn finite_product_complex_unary(a: {ty}, b: {ty}) -> {ty} {{
    if a == {ty}(0.0) || b == {ty}(0.0) {{
        return {ty}(0.0);
    }}
    let product = a * b;
    if isfinite_complex_unary(product) || product != product {{
        return product;
    }}
    return signum_nonzero_complex_unary(a) * signum_nonzero_complex_unary(b) * MAX_FINITE_COMPLEX_UNARY;
}}

fn sin_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    let out_re = finite_product_complex_unary(sin(re), cosh(im));
    let out_im = finite_product_complex_unary(cos(re), sinh(im));
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}

fn cos_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    let out_re = finite_product_complex_unary(cos(re), cosh(im));
    let out_im = -finite_product_complex_unary(sin(re), sinh(im));
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}

fn sinh_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    let out_re = finite_product_complex_unary(sinh(re), cos(im));
    let out_im = finite_product_complex_unary(cosh(re), sin(im));
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}

fn cosh_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    let out_re = finite_product_complex_unary(cosh(re), cos(im));
    let out_im = finite_product_complex_unary(sinh(re), sin(im));
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}

fn tan_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    let two_re = {ty}(2.0) * re;
    let two_im = {ty}(2.0) * im;
    if abs(two_im) > {ty}(80.0) {{
        return select({ty}(0.0), select(-{ty}(1.0), {ty}(1.0), im > {ty}(0.0)), (out_idx % 2u) == 1u);
    }}
    let inv_cosh = {ty}(1.0) / cosh(two_im);
    let denom = {ty}(1.0) + (cos(two_re) * inv_cosh);
    let out_re = (sin(two_re) * inv_cosh) / denom;
    let out_im = tanh(two_im) / denom;
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}
"#,
            ty = ty,
            max_finite = max_finite,
        ),
        ComplexUnaryOp::Sign => format!(
            r#"
const HALF_MAX_FINITE_COMPLEX_UNARY: {ty} = {ty}({half_max_finite});

fn nan_complex_unary() -> {ty} {{
    return {ty}(0.0) / {ty}(0.0);
}}

{sign_inf_helper}

fn signum_complex_unary(x: {ty}) -> {ty} {{
    if x > {ty}(0.0) {{
        return {ty}(1.0);
    }}
    if x < {ty}(0.0) {{
        return -{ty}(1.0);
    }}
    if x == {ty}(0.0) {{
        return {ty}(0.0);
    }}
    return x;
}}

fn sign_complex_lane(out_idx: u32) -> {ty} {{
    let elem = out_idx / 2u;
    let re = A.data[elem * 2u];
    let im = A.data[elem * 2u + 1u];
    if (re != re) || (im != im) {{
        return nan_complex_unary();
    }}
    if (re == {ty}(0.0)) && (im == {ty}(0.0)) {{
        return {ty}(0.0);
    }}

    let re_inf = isinf_complex_unary(re);
    let im_inf = isinf_complex_unary(im);
    if re_inf || im_inf {{
        let real = select({ty}(0.0), signum_complex_unary(re), re_inf);
        let imag = select({ty}(0.0), signum_complex_unary(im), im_inf);
        let norm = sqrt((real * real) + (imag * imag));
        if norm == {ty}(0.0) {{
            return select(real, imag, (out_idx % 2u) == 1u);
        }}
        let out_re = real / norm;
        let out_im = imag / norm;
        return select(out_re, out_im, (out_idx % 2u) == 1u);
    }}

    let abs_re = abs(re);
    let abs_im = abs(im);
    var out_re: {ty};
    var out_im: {ty};
    if abs_re >= abs_im {{
        var denom_re = re;
        var numer_im = im;
        if abs_re > HALF_MAX_FINITE_COMPLEX_UNARY {{
            denom_re = re * {ty}(0.5);
            numer_im = im * {ty}(0.5);
        }}
        let ratio = numer_im / denom_re;
        let denom = sqrt({ty}(1.0) + (ratio * ratio));
        let sign_re = signum_complex_unary(re);
        out_re = sign_re / denom;
        out_im = (sign_re * ratio) / denom;
    }} else {{
        var denom_im = im;
        var numer_re = re;
        if abs_im > HALF_MAX_FINITE_COMPLEX_UNARY {{
            denom_im = im * {ty}(0.5);
            numer_re = re * {ty}(0.5);
        }}
        let ratio = numer_re / denom_im;
        let denom = sqrt({ty}(1.0) + (ratio * ratio));
        let sign_im = signum_complex_unary(im);
        out_re = (sign_im * ratio) / denom;
        out_im = sign_im / denom;
    }}
    return select(out_re, out_im, (out_idx % 2u) == 1u);
}}
"#,
            ty = ty,
            half_max_finite = half_max_finite,
            sign_inf_helper = sign_inf_helper,
        ),
        _ => String::new(),
    };
    let expression = match op {
        ComplexUnaryOp::Real => "A.data[idx * 2u]",
        ComplexUnaryOp::Imag => "A.data[idx * 2u + 1u]",
        ComplexUnaryOp::Abs => "complex_abs_lane(idx)",
        ComplexUnaryOp::Conj => "select(A.data[idx], -A.data[idx], (idx % 2u) == 1u)",
        ComplexUnaryOp::Angle => "atan2(A.data[idx * 2u + 1u], A.data[idx * 2u])",
        ComplexUnaryOp::Sin => "sin_complex_lane(idx)",
        ComplexUnaryOp::Sinc => "sinc_complex_lane(idx)",
        ComplexUnaryOp::Cos => "cos_complex_lane(idx)",
        ComplexUnaryOp::Sinh => "sinh_complex_lane(idx)",
        ComplexUnaryOp::Cosh => "cosh_complex_lane(idx)",
        ComplexUnaryOp::Tan => "tan_complex_lane(idx)",
        ComplexUnaryOp::Sign => "sign_complex_lane(idx)",
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

{extra_helpers}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    Out.data[idx] = {expression};
}}
"#,
        ty = ty,
        expression = expression,
        extra_helpers = extra_helpers,
    )
}

pub(crate) fn complex_from_real_shader(precision: NumericPrecision) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> Real: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    if (idx % 2u) == 0u {{
        Out.data[idx] = Real.data[elem];
    }} else {{
        Out.data[idx] = 0.0;
    }}
}}
"#,
        ty = ty,
    )
}

pub(crate) fn complex_from_real_imag_shader(
    precision: NumericPrecision,
    real_scalar: bool,
    imag_scalar: bool,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let real_index = if real_scalar { "0u" } else { "elem" };
    let imag_index = if imag_scalar { "0u" } else { "elem" };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> Real: Tensor;
@group(0) @binding(1) var<storage, read> Imag: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    if (idx % 2u) == 0u {{
        Out.data[idx] = Real.data[{real_index}];
    }} else {{
        Out.data[idx] = Imag.data[{imag_index}];
    }}
}}
"#,
        ty = ty,
        real_index = real_index,
        imag_index = imag_index,
    )
}

pub(crate) fn complex_binary_shader(
    op: ComplexBinaryOp,
    precision: NumericPrecision,
    lhs_complex: bool,
    rhs_complex: bool,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let lhs_real = if lhs_complex {
        "A.data[elem * 2u]"
    } else {
        "A.data[elem]"
    };
    let lhs_imag = if lhs_complex {
        "A.data[elem * 2u + 1u]"
    } else {
        "0.0"
    };
    let rhs_real = if rhs_complex {
        "B.data[elem * 2u]"
    } else {
        "B.data[elem]"
    };
    let rhs_imag = if rhs_complex {
        "B.data[elem * 2u + 1u]"
    } else {
        "0.0"
    };
    let body = match op {
        ComplexBinaryOp::Add => "let out_re = ar + br;\n    let out_im = ai + bi;",
        ComplexBinaryOp::Sub => "let out_re = ar - br;\n    let out_im = ai - bi;",
        ComplexBinaryOp::Mul => {
            "let out_re = (ar * br) - (ai * bi);\n    let out_im = (ar * bi) + (ai * br);"
        }
        ComplexBinaryOp::Div => {
            "let scale = max(abs(br), abs(bi));\n    let sr = br / scale;\n    let si = bi / scale;\n    let denom = scale * ((sr * sr) + (si * si));\n    let out_re = ((ar * sr) + (ai * si)) / denom;\n    let out_im = ((ai * sr) - (ar * si)) / denom;"
        }
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    let ar = {lhs_real};
    let ai = {lhs_imag};
    let br = {rhs_real};
    let bi = {rhs_imag};
    {body}
    Out.data[idx] = select(out_re, out_im, (idx % 2u) == 1u);
}}
"#,
        ty = ty,
        lhs_real = lhs_real,
        lhs_imag = lhs_imag,
        rhs_real = rhs_real,
        rhs_imag = rhs_imag,
        body = body,
    )
}

pub(crate) fn complex_binary_broadcast_shader(
    op: ComplexBinaryOp,
    precision: NumericPrecision,
    lhs_complex: bool,
    rhs_complex: bool,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let lhs_real = if lhs_complex {
        "A.data[ia * 2u]"
    } else {
        "A.data[ia]"
    };
    let lhs_imag = if lhs_complex {
        "A.data[ia * 2u + 1u]"
    } else {
        "0.0"
    };
    let rhs_real = if rhs_complex {
        "B.data[ib * 2u]"
    } else {
        "B.data[ib]"
    };
    let rhs_imag = if rhs_complex {
        "B.data[ib * 2u + 1u]"
    } else {
        "0.0"
    };
    let body = match op {
        ComplexBinaryOp::Add => "let out_re = ar + br;\n    let out_im = ai + bi;",
        ComplexBinaryOp::Sub => "let out_re = ar - br;\n    let out_im = ai - bi;",
        ComplexBinaryOp::Mul => {
            "let out_re = (ar * br) - (ai * bi);\n    let out_im = (ar * bi) + (ai * br);"
        }
        ComplexBinaryOp::Div => {
            "let scale = max(abs(br), abs(bi));\n    let sr = br / scale;\n    let si = bi / scale;\n    let denom = scale * ((sr * sr) + (si * si));\n    let out_re = ((ar * sr) + (ai * si)) / denom;\n    let out_im = ((ai * sr) - (ar * si)) / denom;"
        }
    };
    format!(
        r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {{
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}};

alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor {{ data: array<{ty}>, }};

struct Params {{
    len: u32,
    offset: u32,
    rank: u32,
    op: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    a_stride: PackedArray,
    b_shape: PackedArray,
    b_stride: PackedArray,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{ return; }}
    let idx = params.offset + local;
    let elem = idx / 2u;

    var coord: array<u32, MAX_RANK>;
    var tmp: u32 = elem;
    var d: u32 = 0u;
    loop {{
        if d >= params.rank {{ break; }}
        let dim = params.out_shape[d].value;
        if dim == 0u {{ coord[d] = 0u; }}
        else {{ coord[d] = tmp % dim; tmp = tmp / dim; }}
        d = d + 1u;
    }}

    var ia: u32 = 0u;
    var ib: u32 = 0u;
    d = 0u;
    loop {{
        if d >= params.rank {{ break; }}
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        let ca = select(coord[d], 0u, ad == 1u);
        let cb = select(coord[d], 0u, bd == 1u);
        ia = ia + ca * params.a_stride[d].value;
        ib = ib + cb * params.b_stride[d].value;
        d = d + 1u;
    }}

    let ar = {lhs_real};
    let ai = {lhs_imag};
    let br = {rhs_real};
    let bi = {rhs_imag};
    {body}
    Out.data[idx] = select(out_re, out_im, (idx % 2u) == 1u);
}}
"#,
        ty = ty,
        lhs_real = lhs_real,
        lhs_imag = lhs_imag,
        rhs_real = rhs_real,
        rhs_imag = rhs_imag,
        body = body,
    )
}

pub const BINARY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f64, b: f64) -> f64 {
    return sqrt((a * a) + (b * b));
}

fn apply(a: f64, b: f64) -> f64 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return select(atan2(a, b), select(a, 0.0, bitcast<u64>(b) == 0x8000000000000000u), (a == 0.0) && (b == 0.0)); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    Out.data[idx] = apply(A.data[idx], B.data[idx]);
}
"#;

pub const BINARY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f32, b: f32) -> f32 {
    return sqrt((a * a) + (b * b));
}

fn apply(a: f32, b: f32) -> f32 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return select(atan2(a, b), select(a, 0.0, bitcast<u32>(b) == 0x80000000u), (a == 0.0) && (b == 0.0)); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    Out.data[idx] = apply(A.data[idx], B.data[idx]);
}
"#;

// Broadcast-aware binary shader (N-D implicit expansion)
pub const BINARY_BROADCAST_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f64>, };

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    op: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_stride: PackedArray,
    b_stride: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f64, b: f64) -> f64 { return sqrt((a * a) + (b * b)); }

fn apply(a: f64, b: f64) -> f64 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return select(atan2(a, b), select(a, 0.0, bitcast<u64>(b) == 0x8000000000000000u), (a == 0.0) && (b == 0.0)); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let idx = params.offset + local;

    // Compute N-D coordinates from linear index (column-major order)
    var coord: array<u32, MAX_RANK>;
    var tmp: u32 = idx;
    var d: u32 = 0u;
    loop {
        if d >= params.rank { break; }
        let dim = params.out_shape[d].value;
        if dim == 0u { coord[d] = 0u; }
        else { coord[d] = tmp % dim; tmp = tmp / dim; }
        d = d + 1u;
    }

    // Map to A and B indices with broadcasting
    var ia: u32 = 0u;
    var ib: u32 = 0u;
    d = 0u;
    loop {
        if d >= params.rank { break; }
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        let ca = select(coord[d], 0u, ad == 1u);
        let cb = select(coord[d], 0u, bd == 1u);
        ia = ia + ca * params.a_stride[d].value;
        ib = ib + cb * params.b_stride[d].value;
        d = d + 1u;
    }

    Out.data[idx] = apply(A.data[ia], B.data[ib]);
}
"#;

pub const BINARY_BROADCAST_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f32>, };

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    op: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_stride: PackedArray,
    b_stride: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f32, b: f32) -> f32 { return sqrt((a * a) + (b * b)); }

fn apply(a: f32, b: f32) -> f32 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return select(atan2(a, b), select(a, 0.0, bitcast<u32>(b) == 0x80000000u), (a == 0.0) && (b == 0.0)); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let idx = params.offset + local;

    var coord: array<u32, MAX_RANK>;
    var tmp: u32 = idx;
    var d: u32 = 0u;
    loop {
        if d >= params.rank { break; }
        let dim = params.out_shape[d].value;
        if dim == 0u { coord[d] = 0u; }
        else { coord[d] = tmp % dim; tmp = tmp / dim; }
        d = d + 1u;
    }

    var ia: u32 = 0u;
    var ib: u32 = 0u;
    d = 0u;
    loop {
        if d >= params.rank { break; }
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        let ca = select(coord[d], 0u, ad == 1u);
        let cb = select(coord[d], 0u, bd == 1u);
        ia = ia + ca * params.a_stride[d].value;
        ib = ib + cb * params.b_stride[d].value;
        d = d + 1u;
    }

    Out.data[idx] = apply(A.data[ia], B.data[ib]);
}
"#;

pub const UNARY_LAYOUT_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params { len: u32, op: u32, offset: u32, total: u32, };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = params.offset + gid.x;
    if gid.x < params.len && idx < params.total { Out.data[idx] = A.data[idx]; }
}
"#;

pub const UNARY_LAYOUT_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params { len: u32, op: u32, offset: u32, total: u32, };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = params.offset + gid.x;
    if gid.x < params.len && idx < params.total { Out.data[idx] = A.data[idx]; }
}
"#;

pub const UNARY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

const PI: f64 = 3.141592653589793;
const SQRT_TWO_PI: f64 = 2.5066282746310002;
const LN_SQRT_TWO_PI: f64 = 0.9189385332046727;
const LANCZOS_G: f64 = 7.0;
const EPSILON: f64 = 1.0e-12;
const SMALL_REFLECTION_CUTOFF: f64 = 1.0e-305;
const FACTORIAL_MAX: u32 = 170u;
const FACTORIAL_INT_TOL: f64 = 1.0e-10;
fn expm1_precise(a: f64) -> f64 {
    let abs_a = abs(a);
    if abs_a < 1.0e-6 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return (((a5 * (1.0 / 120.0)) + (a4 * (1.0 / 24.0)) + (a3 * (1.0 / 6.0))) + (a2 * 0.5)) + a;
    }
    return exp(a) - 1.0;
}

fn log1p_precise(a: f64) -> f64 {
    let abs_a = abs(a);
    if abs_a < 1.0e-6 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return ((((a5 * (1.0 / 5.0)) - (a4 * 0.25)) + (a3 * (1.0 / 3.0))) - (a2 * 0.5)) + a;
    }
    return log(1.0 + a);
}

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn lanczos_gamma(z: f64) -> f64 {
    let z_minus_one = z - 1.0;
    var sum = 0.99999999999980993;
    sum = sum + 676.5203681218851 / (z_minus_one + 1.0);
    sum = sum + -1259.1392167224028 / (z_minus_one + 2.0);
    sum = sum + 771.3234287776531 / (z_minus_one + 3.0);
    sum = sum + -176.6150291621406 / (z_minus_one + 4.0);
    sum = sum + 12.507343278686905 / (z_minus_one + 5.0);
    sum = sum + -0.13857109526572012 / (z_minus_one + 6.0);
    sum = sum + 0.000009984369578019572 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327351493116 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return SQRT_TWO_PI * pow(t, z_minus_one + 0.5) * exp(-t) * sum;
}

fn lanczos_gammaln(z: f64) -> f64 {
    let z_minus_one = z - 1.0;
    var sum = 0.99999999999980993;
    sum = sum + 676.5203681218851 / (z_minus_one + 1.0);
    sum = sum + -1259.1392167224028 / (z_minus_one + 2.0);
    sum = sum + 771.3234287776531 / (z_minus_one + 3.0);
    sum = sum + -176.6150291621406 / (z_minus_one + 4.0);
    sum = sum + 12.507343278686905 / (z_minus_one + 5.0);
    sum = sum + -0.13857109526572012 / (z_minus_one + 6.0);
    sum = sum + 0.000009984369578019572 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327351493116 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return LN_SQRT_TWO_PI + (z_minus_one + 0.5) * log(t) - t + log(sum);
}

fn is_non_positive_integer(x: f64) -> bool {
    if (x > 0.0) {
        return false;
    }
    let nearest = round(x);
    return abs(x - nearest) <= EPSILON * (1.0 + abs(x));
}

fn is_nan64(x: f64) -> bool {
    let bits = bitcast<u64>(x);
    return (bits & 0x7ff0000000000000u) == 0x7ff0000000000000u &&
        (bits & 0x000fffffffffffffu) != 0u;
}

fn pos_inf_f64() -> f64 {
    var bits: u64 = 0x7ff0000000000000u;
    return bitcast<f64>(bits);
}

fn neg_inf_f64() -> f64 {
    var bits: u64 = 0xfff0000000000000u;
    return bitcast<f64>(bits);
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

fn is_inf64(x: f64) -> bool {
    let inf = pos_inf_f64();
    let neg_inf = neg_inf_f64();
    return x == inf || x == neg_inf;
}

fn gamma_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (is_inf64(a)) {
        if (a > 0.0) {
            return pos_inf_f64();
        }
        return nan_f64();
    }
    if (is_non_positive_integer(a)) {
        return pos_inf_f64();
    }
    if (a < 0.5) {
        let sin_term = sin(PI * a);
        if (abs(sin_term) <= EPSILON) {
            return pos_inf_f64();
        }
        let gamma_one_minus = lanczos_gamma(1.0 - a);
        return PI / (sin_term * gamma_one_minus);
    }
    return lanczos_gamma(a);
}

fn gammaln_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a == 0.0 || a == pos_inf_f64()) {
        return pos_inf_f64();
    }
    if (a < 0.0) {
        return nan_f64();
    }
    if (a < SMALL_REFLECTION_CUTOFF) {
        return -log(a);
    }
    if (a < 0.5) {
        return log(PI) - log(sin(PI * a)) - lanczos_gammaln(1.0 - a);
    }
    return lanczos_gammaln(a);
}

fn factorial_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a == 0.0) {
        return 1.0;
    }
    if (is_inf64(a)) {
        if (a > 0.0) {
            return pos_inf_f64();
        }
        return nan_f64();
    }
    if (a < 0.0) {
        return nan_f64();
    }
    let rounded = round(a);
    if (abs(a - rounded) > FACTORIAL_INT_TOL) {
        return nan_f64();
    }
    if (rounded < 0.0) {
        return nan_f64();
    }
    if (rounded > f64(FACTORIAL_MAX)) {
        return pos_inf_f64();
    }
    let n = i32(rounded);
    if (n <= 1) {
        return 1.0;
    }
    let limit = u32(n);
    var acc: f64 = 1.0;
    var i: u32 = 2u;
    loop {
        if (i > limit) {
            break;
        }
        acc = acc * f64(i);
        i = i + 1u;
    }
    return acc;
}

fn sinc_real(a: f64) -> f64 {
    if (a == 0.0) {
        return 1.0;
    }
    let abs_a = abs(a);
    if (!is_nan64(a) && !is_inf64(a) && floor(abs_a) == abs_a) {
        return 0.0;
    }
    let scaled = PI * a;
    return sin(scaled) / scaled;
}

fn heaviside_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return 0.0;
    }
    return 0.5;
}

fn erf_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (is_inf64(a)) {
        if (a > 0.0) {
            return 1.0;
        }
        return -1.0;
    }
    if (a == 0.0) {
        return a;
    }
    var sign = 1.0;
    var x = a;
    if (a < 0.0) {
        sign = -1.0;
        x = -a;
    }
    if (x >= 6.0) {
        return sign;
    }
    let x2 = x * x;
    if (x < 3.5) {
        var sum = x;
        var term = x;
        var n: u32 = 1u;
        loop {
            if (n >= 120u) {
                break;
            }
            term = term * (-x2 / f64(n));
            let add = term / f64((2u * n) + 1u);
            sum = sum + add;
            if (abs(add) <= 1.0e-16 * max(1.0, abs(sum))) {
                break;
            }
            n = n + 1u;
        }
        return sign * 1.1283791670955126 * sum;
    }
    var asym_sum = 1.0;
    var asym_term = 1.0;
    var previous = 1.0e300;
    var k: u32 = 1u;
    loop {
        if (k >= 60u) {
            break;
        }
        asym_term = asym_term * (-(f64((2u * k) - 1u)) / (2.0 * x2));
        if (abs(asym_term) > previous) {
            break;
        }
        asym_sum = asym_sum + asym_term;
        previous = abs(asym_term);
        if (abs(asym_term) <= 1.0e-16 * abs(asym_sum)) {
            break;
        }
        k = k + 1u;
    }
    let erfc_tail = exp(-x2) * asym_sum / (x * 1.772453850905516);
    return sign * (1.0 - erfc_tail);
}

fn erfc_positive_tail_real(a: f64) -> f64 {
    if (a < 3.5) {
        return 1.0 - erf_real(a);
    }
    let x2 = a * a;
    var asym_sum = 1.0;
    var asym_term = 1.0;
    var previous = 1.0e300;
    var k: u32 = 1u;
    loop {
        if (k >= 60u) {
            break;
        }
        asym_term = asym_term * (-(f64((2u * k) - 1u)) / (2.0 * x2));
        if (abs(asym_term) > previous) {
            break;
        }
        asym_sum = asym_sum + asym_term;
        previous = abs(asym_term);
        if (abs(asym_term) <= 1.0e-16 * abs(asym_sum)) {
            break;
        }
        k = k + 1u;
    }
    return exp(-x2) * asym_sum / (a * 1.772453850905516);
}

fn erfcinv_positive_tail_real(target: f64) -> f64 {
    var lo = 0.0;
    var hi = 1.0;
    loop {
        if (hi >= 32.0 || erfc_positive_tail_real(hi) <= target) {
            break;
        }
        lo = hi;
        hi = hi * 2.0;
    }
    if (erfc_positive_tail_real(hi) > target) {
        return hi;
    }

    var step: u32 = 0u;
    loop {
        if (step >= 110u) {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if (erfc_positive_tail_real(mid) > target) {
            lo = mid;
        } else {
            hi = mid;
        }
        step = step + 1u;
    }
    return 0.5 * (lo + hi);
}

fn erfcinv_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a < 0.0 || a > 2.0) {
        return f64(0.0) / f64(0.0);
    }
    if (a == 0.0) {
        return f64(1.0) / f64(0.0);
    }
    if (a == 2.0) {
        return -f64(1.0) / f64(0.0);
    }
    if (a == 1.0) {
        return 0.0;
    }
    let target = 1.0 - a;
    var sign = 1.0;
    if (target < 0.0) {
        sign = -1.0;
    }
    let magnitude = abs(target);
    let log_term = log(1.0 - magnitude * magnitude);
    let first = (2.0 / (PI * 0.147)) + (0.5 * log_term);
    var estimate = sign * sqrt(sqrt(first * first - log_term / 0.147) - first);
    let derivative_scale = 1.1283791670955126;
    estimate = estimate - (erf_real(estimate) - target) / (derivative_scale * exp(-estimate * estimate));
    estimate = estimate - (erf_real(estimate) - target) / (derivative_scale * exp(-estimate * estimate));
    estimate = estimate - (erf_real(estimate) - target) / (derivative_scale * exp(-estimate * estimate));
    estimate = estimate - (erf_real(estimate) - target) / (derivative_scale * exp(-estimate * estimate));
    return estimate;
}

fn apply(a: f64) -> f64 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 19u: { return tan(a); }
        case 20u: { return asin(a); }
        case 21u: { return acos(a); }
        case 22u: { return atan(a); }
        case 2u: { return abs(a); }
        case 3u: { return exp(a); }
        case 4u: { return log(a); }
        case 5u: { return sqrt(a); }
        case 6u: {
            if (a > 0.0) {
                return 1.0;
            }
            if (a < 0.0) {
                return -1.0;
            }
            if (a == 0.0) {
                return 0.0;
            }
            return a;
        }
        case 7u: { return a; }
        case 8u: { return 0.0; }
        case 9u: { return a; }
        case 10u: { return atan2(f64(0.0), a); }
        case 11u: { return expm1_precise(a); }
        case 12u: { return log1p_precise(a); }
        case 13u: { return log(a) * 0.4342944819032518; }
        case 14u: { return log(a) * 1.4426950408889634; }
        case 15u: { return exp(a * 0.6931471805599453); }
        case 16u: { return floor(a); }
        case 17u: { return ceil(a); }
        case 18u: {
            let t = trunc(a);
            if (t == 0.0) {
                return 0.0;
            }
            return t;
        }
        case 23u: { return sinh(a); }
        case 24u: { return cosh(a); }
        case 25u: { return tanh(a); }
        case 26u: { return asinh(a); }
        case 27u: { return acosh(a); }
        case 28u: { return atanh(a); }
        case 29u: { return gamma_real(a); }
        case 30u: { return factorial_real(a); }
        case 31u: { return f64(f32(a)); }
        case 32u: {
            let aa = abs(a);
            if (aa == 0.0) {
                return 0.0;
            }
            return ceil(log2(aa));
        }
        case 33u: { return sinc_real(a); }
        case 34u: { return heaviside_real(a); }
        case 35u: { return erf_real(a); }
        case 36u: { return gammaln_real(a); }
        case 37u: {
            if (is_nan64(a) || is_inf64(a)) {
                return a;
            }
            if (a >= 0.0) {
                return floor(a + 0.5);
            }
            return ceil(a - 0.5);
        }
        case 38u: { return erfcinv_real(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    Out.data[idx] = apply(A.data[idx]);
}
"#;

pub const UNARY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
  	op: u32,
    offset: u32,
    total: u32,
};

const PI: f32 = 3.1415927;
const SQRT_TWO_PI: f32 = 2.5066283;
const LN_SQRT_TWO_PI: f32 = 0.9189385;
const LANCZOS_G: f32 = 7.0;
const EPSILON: f32 = 1.0e-5;
const SMALL_REFLECTION_CUTOFF: f32 = 1.0e-30;
const FACTORIAL_MAX_F32: u32 = 170u;
const FACTORIAL_INT_TOL_F32: f32 = 1.0e-4;
fn expm1_precise(a: f32) -> f32 {
    let abs_a = abs(a);
    if abs_a < 1.0e-4 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return (((a5 * (1.0 / 120.0)) + (a4 * (1.0 / 24.0)) + (a3 * (1.0 / 6.0))) + (a2 * 0.5)) + a;
    }
    return exp(a) - 1.0;
}

fn log1p_precise(a: f32) -> f32 {
    let abs_a = abs(a);
    if abs_a < 1.0e-4 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return ((((a5 * (1.0 / 5.0)) - (a4 * 0.25)) + (a3 * (1.0 / 3.0))) - (a2 * 0.5)) + a;
    }
    return log(1.0 + a);
}

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn lanczos_gamma(z: f32) -> f32 {
    let z_minus_one = z - 1.0;
    var sum: f32 = 0.99999994;
    sum = sum + 676.5204 / (z_minus_one + 1.0);
    sum = sum + -1259.1393 / (z_minus_one + 2.0);
    sum = sum + 771.3234 / (z_minus_one + 3.0);
    sum = sum + -176.61502 / (z_minus_one + 4.0);
    sum = sum + 12.507343 / (z_minus_one + 5.0);
    sum = sum + -0.1385711 / (z_minus_one + 6.0);
    sum = sum + 0.00000998437 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return SQRT_TWO_PI * pow(t, z_minus_one + 0.5) * exp(-t) * sum;
}

fn lanczos_gammaln(z: f32) -> f32 {
    let z_minus_one = z - 1.0;
    var sum: f32 = 0.99999994;
    sum = sum + 676.5204 / (z_minus_one + 1.0);
    sum = sum + -1259.1393 / (z_minus_one + 2.0);
    sum = sum + 771.3234 / (z_minus_one + 3.0);
    sum = sum + -176.61502 / (z_minus_one + 4.0);
    sum = sum + 12.507343 / (z_minus_one + 5.0);
    sum = sum + -0.1385711 / (z_minus_one + 6.0);
    sum = sum + 0.00000998437 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return LN_SQRT_TWO_PI + (z_minus_one + 0.5) * log(t) - t + log(sum);
}

fn is_non_positive_integer(x: f32) -> bool {
    if (x > 0.0) {
        return false;
    }
    let nearest = round(x);
    return abs(x - nearest) <= EPSILON * (1.0 + abs(x));
}

fn is_nan32(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u;
}

fn pos_inf_f32() -> f32 {
    var bits: u32 = 0x7f800000u;
    return bitcast<f32>(bits);
}

fn neg_inf_f32() -> f32 {
    var bits: u32 = 0xff800000u;
    return bitcast<f32>(bits);
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

fn is_inf32(x: f32) -> bool {
    let inf = pos_inf_f32();
    let neg_inf = neg_inf_f32();
    return x == inf || x == neg_inf;
}

fn gamma_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (is_inf32(a)) {
        if (a > 0.0) {
            return pos_inf_f32();
        }
        return nan_f32();
    }
    if (is_non_positive_integer(a)) {
        return pos_inf_f32();
    }
    if (a < 0.5) {
        let sin_term = sin(PI * a);
        if (abs(sin_term) <= EPSILON) {
            return pos_inf_f32();
        }
        let gamma_one_minus = lanczos_gamma(1.0 - a);
        return PI / (sin_term * gamma_one_minus);
    }
    return lanczos_gamma(a);
}

fn gammaln_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a == 0.0 || a == pos_inf_f32()) {
        return pos_inf_f32();
    }
    if (a < 0.0) {
        return nan_f32();
    }
    if (a < SMALL_REFLECTION_CUTOFF) {
        return -log(a);
    }
    if (a < 0.5) {
        return log(PI) - log(sin(PI * a)) - lanczos_gammaln(1.0 - a);
    }
    return lanczos_gammaln(a);
}

fn factorial_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a == 0.0) {
        return 1.0;
    }
    if (is_inf32(a)) {
        if (a > 0.0) {
            return pos_inf_f32();
        }
        return nan_f32();
    }
    if (a < 0.0) {
        return nan_f32();
    }
    let rounded = round(a);
    if (abs(a - rounded) > FACTORIAL_INT_TOL_F32) {
        return nan_f32();
    }
    if (rounded < 0.0) {
        return nan_f32();
    }
    if (rounded > f32(FACTORIAL_MAX_F32)) {
        return pos_inf_f32();
    }
    let n = i32(rounded);
    if (n <= 1) {
        return 1.0;
    }
    let limit = u32(n);
    var acc: f32 = 1.0;
    var i: u32 = 2u;
    loop {
        if (i > limit) {
            break;
        }
        acc = acc * f32(i);
        i = i + 1u;
    }
    return acc;
}

fn sinc_real(a: f32) -> f32 {
    if (a == 0.0) {
        return 1.0;
    }
    let abs_a = abs(a);
    if (!is_nan32(a) && !is_inf32(a) && floor(abs_a) == abs_a) {
        return 0.0;
    }
    let scaled = PI * a;
    return sin(scaled) / scaled;
}

fn heaviside_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return 0.0;
    }
    return 0.5;
}

fn erf_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (is_inf32(a)) {
        if (a > 0.0) {
            return 1.0;
        }
        return -1.0;
    }
    if (a == 0.0) {
        return a;
    }
    var sign: f32 = 1.0;
    var x = a;
    if (a < 0.0) {
        sign = -1.0;
        x = -a;
    }
    if (x >= 6.0) {
        return sign;
    }
    let x2 = x * x;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1.0 - polynomial * exp(-x2));
}

fn erfc_positive_tail_real(a: f32) -> f32 {
    if (a < 3.5) {
        return 1.0 - erf_real(a);
    }
    let x2 = a * a;
    var asym_sum = 1.0;
    var asym_term = 1.0;
    var previous = 1.0e30;
    var k: u32 = 1u;
    loop {
        if (k >= 30u) {
            break;
        }
        asym_term = asym_term * (-(f32((2u * k) - 1u)) / (2.0 * x2));
        if (abs(asym_term) > previous) {
            break;
        }
        asym_sum = asym_sum + asym_term;
        previous = abs(asym_term);
        if (abs(asym_term) <= 1.0e-6 * abs(asym_sum)) {
            break;
        }
        k = k + 1u;
    }
    return exp(-x2) * asym_sum / (a * 1.7724539);
}

fn erfcinv_positive_tail_real(target: f32) -> f32 {
    var lo = 0.0;
    var hi = 1.0;
    loop {
        if (hi >= 32.0 || erfc_positive_tail_real(hi) <= target) {
            break;
        }
        lo = hi;
        hi = hi * 2.0;
    }
    if (erfc_positive_tail_real(hi) > target) {
        return hi;
    }

    var step: u32 = 0u;
    loop {
        if (step >= 32u) {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if (erfc_positive_tail_real(mid) > target) {
            lo = mid;
        } else {
            hi = mid;
        }
        step = step + 1u;
    }
    return 0.5 * (lo + hi);
}

fn erfcinv_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a < 0.0 || a > 2.0) {
        return f32(0.0) / f32(0.0);
    }
    if (a == 0.0) {
        return f32(1.0) / f32(0.0);
    }
    if (a == 2.0) {
        return -f32(1.0) / f32(0.0);
    }
    if (a == 1.0) {
        return 0.0;
    }
    let target = 1.0 - a;
    var sign: f32 = 1.0;
    if (target < 0.0) {
        sign = -1.0;
    }
    let magnitude = abs(target);
    let log_term = log(1.0 - magnitude * magnitude);
    let first = (2.0 / (PI * 0.147)) + (0.5 * log_term);
    return sign * sqrt(sqrt(first * first - log_term / 0.147) - first);
}

fn apply(a: f32) -> f32 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 19u: { return tan(a); }
        case 20u: { return asin(a); }
        case 21u: { return acos(a); }
        case 22u: { return atan(a); }
        case 2u: { return abs(a); }
        case 3u: { return exp(a); }
        case 4u: { return log(a); }
        case 5u: { return sqrt(a); }
        case 6u: {
            if (a > 0.0) {
                return 1.0;
            }
            if (a < 0.0) {
                return -1.0;
            }
            if (a == 0.0) {
                return 0.0;
            }
            return a;
        }
        case 7u: { return a; }
        case 8u: { return 0.0; }
        case 9u: { return a; }
        case 10u: { return atan2(0.0, a); }
        case 11u: { return expm1_precise(a); }
        case 12u: { return log1p_precise(a); }
        case 13u: { return log(a) * 0.4342944819; }
        case 14u: { return log(a) * 1.4426950409; }
        case 15u: { return exp(a * 0.6931472); }
        case 16u: { return floor(a); }
        case 17u: { return ceil(a); }
        case 18u: {
            let t = trunc(a);
            if (t == 0.0) {
                return 0.0;
            }
            return t;
        }
        case 23u: { return sinh(a); }
        case 24u: { return cosh(a); }
        case 25u: { return tanh(a); }
        case 26u: { return asinh(a); }
        case 27u: { return acosh(a); }
        case 28u: { return atanh(a); }
        case 29u: { return gamma_real(a); }
        case 30u: { return factorial_real(a); }
        case 31u: { return a; }
        case 32u: {
            let aa = abs(a);
            if (aa == 0.0) {
                return 0.0;
            }
            return ceil(log2(aa));
        }
        case 33u: { return sinc_real(a); }
        case 34u: { return heaviside_real(a); }
        case 35u: { return erf_real(a); }
        case 36u: { return gammaln_real(a); }
        case 37u: {
            if (is_nan32(a) || is_inf32(a)) {
                return a;
            }
            if (a >= 0.0) {
                return floor(a + 0.5);
            }
            return ceil(a - 0.5);
        }
        case 38u: { return erfcinv_real(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    Out.data[idx] = apply(A.data[idx]);
}
"#;

pub const SCALAR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    scalar: f64,
    scalar_pad: f64,
    scalar_pad2: f64,
    scalar_pad3: f64,
    scalar_pad4: f64,
    scalar_pad5: f64,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let a = A.data[idx];
    let scalar = params.scalar;
    var result: f64 = a;
    switch params.op {
        case 0u: { result = a + scalar; }
        case 1u: { result = a - scalar; }
        case 2u: { result = a * scalar; }
        case 3u: { result = a / scalar; }
        case 4u: { result = scalar - a; }
        case 5u: { result = scalar / a; }
        case 6u: { result = max(a, scalar); }
        case 7u: { result = min(a, scalar); }
        default: { result = a; }
    }
    Out.data[idx] = result;
}
"#;

pub const SCALAR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    scalar: f32,
    scalar_pad: vec3<f32>,
    scalar_pad2: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    let a = A.data[idx];
    let s = params.scalar;
    var result: f32 = a;
    switch params.op {
        case 0u: { result = a + s; }
        case 1u: { result = a - s; }
        case 2u: { result = a * s; }
        case 3u: { result = a / s; }
        case 4u: { result = s - a; }
        case 5u: { result = s / a; }
        case 6u: { result = max(a, s); }
        case 7u: { result = min(a, s); }
        default: { result = a; }
    }
    Out.data[idx] = result;
}
"#;

pub(crate) fn round_digits_shader(precision: NumericPrecision, significant: bool) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let inv_ln10 = match precision {
        NumericPrecision::F64 => "0.4342944819032518",
        NumericPrecision::F32 => "0.4342944819",
    };
    let apply_body = if significant {
        format!(
            r#"
fn apply_round(a: {ty}) -> {ty} {{
    if !is_finite_value(a) {{
        return a;
    }}
    if a == {ty}(0.0) {{
        return {ty}(0.0);
    }}
    let order = floor(log(abs(a)) * {ty}({inv_ln10}));
    let scale_power = params.digits - 1 - i32(order);
    let scale = pow({ty}(10.0), {ty}(scale_power));
    if !is_finite_value(scale) || scale == {ty}(0.0) {{
        return a;
    }}
    return round_half_away(a * scale) / scale;
}}
"#
        )
    } else {
        format!(
            r#"
fn apply_round(a: {ty}) -> {ty} {{
    if !is_finite_value(a) {{
        return a;
    }}
    if params.digits == 0 {{
        return round_half_away(a);
    }}
    let scale = pow({ty}(10.0), {ty}(abs(params.digits)));
    if !is_finite_value(scale) || scale == {ty}(0.0) {{
        return a;
    }}
    if params.digits > 0 {{
        return round_half_away(a * scale) / scale;
    }}
    return round_half_away(a / scale) * scale;
}}
"#
        )
    };

    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    total: u32,
    digits: i32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_finite_value(a: {ty}) -> bool {{
    return (a == a) && (abs(a) < {ty}({max_finite}));
}}

fn round_half_away(a: {ty}) -> {ty} {{
    if !is_finite_value(a) {{
        return a;
    }}
    if a >= {ty}(0.0) {{
        return floor(a + {ty}(0.5));
    }}
    return ceil(a - {ty}(0.5));
}}

{apply_body}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.total {{
        return;
    }}
    Out.data[idx] = apply_round(A.data[idx]);
}}
"#
    )
}

//! One raw-word complex quotient for exact and broadcast elementwise routes.
//! Floating arithmetic only sees normal mantissas; input and output subnormal
//! payloads are decoded and packed as integer words.

use crate::backend::wgpu::types::NumericPrecision;

const MATH: &str = r#"
struct RunmatScaled { m: @F@, e: i32, negative: bool, }
struct RunmatPair { re: @W@, im: @W@, }

fn runmat_abs_bits(bits: @W@) -> @W@ { return bits & ~@SIGN@; }
fn runmat_negative(bits: @W@) -> bool { return (bits & @SIGN@) != @W@(0u); }
fn runmat_msb(frac: @W@) -> u32 { @MSB@ }

fn runmat_decode(bits: @W@) -> RunmatScaled {
    let magnitude = runmat_abs_bits(bits);
    let stored = u32((magnitude >> @P@u) & @EXP_MASK@);
    let fraction = magnitude & @FRAC_MASK@;
    let negative = runmat_negative(bits);
    if stored != 0u {
        let normal = (bits & @SIGN@) | (@W@(@BIAS_MINUS_ONE@u) << @P@u) | fraction;
        return RunmatScaled(bitcast<@F@>(normal), i32(stored) - @BIAS@ + 1, negative);
    }
    if fraction == @W@(0u) { return RunmatScaled(@F@(0.0), 0, negative); }
    let top = runmat_msb(fraction);
    let shifted = (fraction << (@P@u - top)) & @FRAC_MASK@;
    let normal = (bits & @SIGN@) | (@W@(@BIAS_MINUS_ONE@u) << @P@u) | shifted;
    return RunmatScaled(bitcast<@F@>(normal), i32(top) + 2 - @BIAS@ - @P@, negative);
}

fn runmat_product(x: @W@, y: @W@) -> RunmatScaled {
    let negative = runmat_negative(x) != runmat_negative(y);
    if runmat_abs_bits(x) == @W@(0u) || runmat_abs_bits(y) == @W@(0u) {
        return RunmatScaled(@F@(0.0), 0, negative);
    }
    let a = runmat_decode(x);
    let b = runmat_decode(y);
    return RunmatScaled(a.m * b.m, a.e + b.e, negative);
}

fn runmat_pow2(delta: i32) -> @F@ {
    return bitcast<@F@>(@W@(@BIAS@ + delta) << @P@u);
}

fn runmat_sum(a: RunmatScaled, b: RunmatScaled) -> RunmatScaled {
    if a.m == @F@(0.0) && b.m == @F@(0.0) {
        return RunmatScaled(@F@(0.0), 0, a.negative && b.negative);
    }
    if a.m == @F@(0.0) { return b; }
    if b.m == @F@(0.0) { return a; }
    let top = max(a.e, b.e);
    var am = @F@(0.0);
    var bm = @F@(0.0);
    if a.e - top >= -@ALIGN@ { am = a.m * runmat_pow2(a.e - top); }
    if b.e - top >= -@ALIGN@ { bm = b.m * runmat_pow2(b.e - top); }
    var m = am + bm;
    if m == @F@(0.0) { return RunmatScaled(@F@(0.0), 0, false); }
    var exponent = top;
    loop {
        if abs(m) >= @F@(0.5) { break; }
        m = m * @F@(2.0);
        exponent = exponent - 1;
    }
    if abs(m) >= @F@(1.0) { m = m * @F@(0.5); exponent = exponent + 1; }
    return RunmatScaled(m, exponent, m < @F@(0.0));
}

fn runmat_pack(n: RunmatScaled, d: RunmatScaled) -> @W@ {
    let negative = n.negative != d.negative;
    let sign = select(@W@(0u), @SIGN@, negative);
    if n.m == @F@(0.0) { return sign; }
    var mantissa = n.m / d.m;
    var exponent = n.e - d.e;
    loop {
        if abs(mantissa) < @F@(1.0) { break; }
        mantissa = mantissa * @F@(0.5);
        exponent = exponent + 1;
    }
    loop {
        if abs(mantissa) >= @F@(0.5) { break; }
        mantissa = mantissa * @F@(2.0);
        exponent = exponent - 1;
    }
    let fraction = bitcast<@W@>(abs(mantissa)) & @FRAC_MASK@;
    let stored = exponent + @BIAS@ - 1;
    if stored >= @EXP_MAX@ { return sign | (@EXP_MASK@ << @P@u); }
    if stored > 0 { return sign | (@W@(stored) << @P@u) | fraction; }

    // The quotient mantissa is normal. Round the integer significand into the
    // subnormal field without a floating subnormal intermediate or store.
    let shift = u32(1 - stored);
    if shift > @P_PLUS_ONE@u { return sign; }
    let significand = (@W@(1u) << @P@u) | fraction;
    var rounded = significand >> shift;
    let discarded = significand & ((@W@(1u) << shift) - @W@(1u));
    let halfway = @W@(1u) << (shift - 1u);
    if discarded > halfway || (discarded == halfway && (rounded & @W@(1u)) != @W@(0u)) {
        rounded = rounded + @W@(1u);
    }
    if rounded >= (@W@(1u) << @P@u) { return sign | (@W@(1u) << @P@u); }
    return sign | rounded;
}

fn runmat_ordinary(bits: @W@) -> bool {
    let magnitude = runmat_abs_bits(bits);
    return magnitude == @W@(0u) || (magnitude >= @LOW@ && magnitude <= @HIGH@);
}

fn runmat_legacy(ar_bits: @W@, ai_bits: @W@, br_bits: @W@, bi_bits: @W@) -> RunmatPair {
    let ar = bitcast<@F@>(ar_bits);
    let ai = bitcast<@F@>(ai_bits);
    let br = bitcast<@F@>(br_bits);
    let bi = bitcast<@F@>(bi_bits);
    let scale = max(abs(br), abs(bi));
    let sr = br / scale;
    let si = bi / scale;
    let denominator = scale * ((sr * sr) + (si * si));
    let re = ((ar * sr) + (ai * si)) / denominator;
    let im = ((ai * sr) - (ar * si)) / denominator;
    return RunmatPair(bitcast<@W@>(re), bitcast<@W@>(im));
}

fn runmat_divide(ar: @W@, ai: @W@, br: @W@, bi: @W@) -> RunmatPair {
    let exceptional = ((runmat_abs_bits(ar) >> @P@u) & @EXP_MASK@) == @EXP_MASK@
        || ((runmat_abs_bits(ai) >> @P@u) & @EXP_MASK@) == @EXP_MASK@
        || ((runmat_abs_bits(br) >> @P@u) & @EXP_MASK@) == @EXP_MASK@
        || ((runmat_abs_bits(bi) >> @P@u) & @EXP_MASK@) == @EXP_MASK@;
    if exceptional || (runmat_abs_bits(br) == @W@(0u) && runmat_abs_bits(bi) == @W@(0u)) {
        return runmat_legacy(ar, ai, br, bi);
    }
    if runmat_ordinary(ar) && runmat_ordinary(ai)
        && runmat_ordinary(br) && runmat_ordinary(bi) {
        let quotient = runmat_legacy(ar, ai, br, bi);
        // The ordinary exponent window excludes product underflow. A zero is
        // either a cancellation (+0) or a sum of signed zero products.
        var real = quotient.re;
        var imag = quotient.im;
        if runmat_abs_bits(real) == @W@(0u) {
            let both_zero = (runmat_abs_bits(ar) == @W@(0u) || runmat_abs_bits(br) == @W@(0u))
                && (runmat_abs_bits(ai) == @W@(0u) || runmat_abs_bits(bi) == @W@(0u));
            let negative = both_zero
                && (runmat_negative(ar) != runmat_negative(br))
                && (runmat_negative(ai) != runmat_negative(bi));
            real = select(@W@(0u), @SIGN@, negative);
        }
        if runmat_abs_bits(imag) == @W@(0u) {
            let both_zero = (runmat_abs_bits(ai) == @W@(0u) || runmat_abs_bits(br) == @W@(0u))
                && (runmat_abs_bits(ar) == @W@(0u) || runmat_abs_bits(bi) == @W@(0u));
            let negative = both_zero
                && (runmat_negative(ai) != runmat_negative(br))
                && (runmat_negative(ar) == runmat_negative(bi));
            imag = select(@W@(0u), @SIGN@, negative);
        }
        return RunmatPair(real, imag);
    }

    let denominator = runmat_sum(runmat_product(br, br), runmat_product(bi, bi));
    let real = runmat_sum(runmat_product(ar, br), runmat_product(ai, bi));
    let ad = runmat_product(ar, bi);
    let minus_ad = RunmatScaled(-ad.m, ad.e, !ad.negative);
    let imaginary = runmat_sum(runmat_product(ai, br), minus_ad);
    return RunmatPair(runmat_pack(real, denominator), runmat_pack(imaginary, denominator));
}
"#;

const EXACT: &str = r#"
struct Tensor { data: array<u32>, }
struct Params { len: u32, offset: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@MATH@
@IO@
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let elem = params.offset + local;
    let quotient = runmat_divide(@AR@, @AI@, @BR@, @BI@);
    runmat_write_pair(elem, quotient);
}
"#;

const BROADCAST: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor { data: array<u32>, }
struct Params {
    len: u32, offset: u32, rank: u32, op: u32,
    out_shape: PackedArray, a_shape: PackedArray,
    a_stride: PackedArray, b_shape: PackedArray, b_stride: PackedArray,
}
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@MATH@
@IO@
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let elem = params.offset + local;
    var coord: array<u32, MAX_RANK>;
    var tmp = elem;
    var d = 0u;
    loop {
        if d >= params.rank { break; }
        let extent = params.out_shape[d].value;
        if extent == 0u { coord[d] = 0u; }
        else { coord[d] = tmp % extent; tmp = tmp / extent; }
        d = d + 1u;
    }
    var ia = 0u;
    var ib = 0u;
    d = 0u;
    loop {
        if d >= params.rank { break; }
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        ia = ia + select(coord[d], 0u, ad == 1u) * params.a_stride[d].value;
        ib = ib + select(coord[d], 0u, bd == 1u) * params.b_stride[d].value;
        d = d + 1u;
    }
    let quotient = runmat_divide(@AR@, @AI@, @BR@, @BI@);
    runmat_write_pair(elem, quotient);
}
"#;

fn math(precision: NumericPrecision) -> String {
    let (float, word, p, bias, low, high, msb) = match precision {
        NumericPrecision::F32 => (
            "f32", "u32", 23, 127, "0x30800000u", "0x4e800000u",
            "return 31u - countLeadingZeros(u32(frac));",
        ),
        NumericPrecision::F64 => (
            "f64", "u64", 52, 1023, "u64(773u) << 52u", "u64(1273u) << 52u",
            "let high = u32(frac >> 32u); if high != 0u { return 63u - countLeadingZeros(high); } return 31u - countLeadingZeros(u32(frac));",
        ),
    };
    MATH.replace("@F@", float)
        .replace("@W@", word)
        .replace("@P_PLUS_ONE@", &(p + 1).to_string())
        .replace("@P@", &p.to_string())
        .replace("@BIAS_MINUS_ONE@", &(bias - 1).to_string())
        .replace("@BIAS@", &bias.to_string())
        .replace("@EXP_MAX@", &(2 * bias + 1).to_string())
        .replace(
            "@EXP_MASK@",
            &format!(
                "(({}(1u) << {}u) - {}(1u))",
                word,
                if p == 23 { 8 } else { 11 },
                word
            ),
        )
        .replace(
            "@FRAC_MASK@",
            &format!("(({}(1u) << {}u) - {}(1u))", word, p, word),
        )
        .replace(
            "@SIGN@",
            &format!("({}(1u) << {}u)", word, if p == 23 { 31 } else { 63 }),
        )
        .replace("@ALIGN@", &(p + 3).to_string())
        .replace("@LOW@", low)
        .replace("@HIGH@", high)
        .replace("@MSB@", msb)
}

fn io(precision: NumericPrecision) -> &'static str {
    match precision {
        NumericPrecision::F32 => {
            r#"
fn runmat_read_a(lane: u32) -> u32 { return A.data[lane]; }
fn runmat_read_b(lane: u32) -> u32 { return B.data[lane]; }
fn runmat_write_pair(elem: u32, pair: RunmatPair) {
    Out.data[elem * 2u] = pair.re;
    Out.data[elem * 2u + 1u] = pair.im;
}
"#
        }
        NumericPrecision::F64 => {
            r#"
fn runmat_read_a(lane: u32) -> u64 {
    let word = lane * 2u;
    return u64(A.data[word]) | (u64(A.data[word + 1u]) << 32u);
}
fn runmat_read_b(lane: u32) -> u64 {
    let word = lane * 2u;
    return u64(B.data[word]) | (u64(B.data[word + 1u]) << 32u);
}
fn runmat_write_pair(elem: u32, pair: RunmatPair) {
    let word = elem * 4u;
    Out.data[word] = u32(pair.re);
    Out.data[word + 1u] = u32(pair.re >> 32u);
    Out.data[word + 2u] = u32(pair.im);
    Out.data[word + 3u] = u32(pair.im >> 32u);
}
"#
        }
    }
}

pub(super) fn shader(
    precision: NumericPrecision,
    lhs_complex: bool,
    rhs_complex: bool,
    broadcast: bool,
) -> String {
    let (lhs_index, rhs_index) = if broadcast {
        ("ia", "ib")
    } else {
        ("elem", "elem")
    };
    let a_real = format!(
        "runmat_read_a({lhs_index}{})",
        if lhs_complex { " * 2u" } else { "" }
    );
    let b_real = format!(
        "runmat_read_b({rhs_index}{})",
        if rhs_complex { " * 2u" } else { "" }
    );
    let a_imag = if lhs_complex {
        format!("runmat_read_a({lhs_index} * 2u + 1u)")
    } else {
        match precision {
            NumericPrecision::F32 => "u32(0u)",
            NumericPrecision::F64 => "u64(0u)",
        }
        .to_owned()
    };
    let b_imag = if rhs_complex {
        format!("runmat_read_b({rhs_index} * 2u + 1u)")
    } else {
        match precision {
            NumericPrecision::F32 => "u32(0u)",
            NumericPrecision::F64 => "u64(0u)",
        }
        .to_owned()
    };
    let template = if broadcast { BROADCAST } else { EXACT };
    template
        .replace("@IO@", io(precision))
        .replace("@MATH@", &math(precision))
        .replace("@AR@", &a_real)
        .replace("@AI@", &a_imag)
        .replace("@BR@", &b_real)
        .replace("@BI@", &b_imag)
}

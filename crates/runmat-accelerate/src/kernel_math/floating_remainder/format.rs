use runmat_accelerate_api::ProviderPrecision;

pub(super) struct Format {
    pub ty: &'static str,
    pub fraction_bits: i32,
    pub bias: i32,
    pub implicit: &'static str,
    pub decode: &'static str,
    pub encode: &'static str,
    pub nan: &'static str,
}

impl Format {
    pub fn new(precision: ProviderPrecision) -> Self {
        match precision {
            ProviderPrecision::F32 => Self {
                ty: "f32",
                fraction_bits: 23,
                bias: 127,
                implicit: "8388608.0",
                decode: r#"
let bits = bitcast<u32>(value);
let encoded_exponent = (bits >> 23u) & 255u;
let fraction = f32(bits & 0x007fffffu);
let special = encoded_exponent == 255u;
let negative = (bits & 0x80000000u) != 0u;"#,
                encode: r#"
let sign = select(0u, 0x80000000u, negative);
return bitcast<f32>(sign | (encoded_exponent << 23u) | u32(significand));"#,
                nan: "bitcast<f32>(0x7fc00000u)",
            },
            ProviderPrecision::F64 => Self {
                ty: "f64",
                fraction_bits: 52,
                bias: 1023,
                implicit: "4503599627370496.0",
                decode: r#"
let bits = bitcast<vec2<u32>>(value);
let encoded_exponent = (bits.y >> 20u) & 2047u;
let fraction = f64(bits.y & 0x000fffffu) * f64(4294967296.0) + f64(bits.x);
let special = encoded_exponent == 2047u;
let negative = (bits.y & 0x80000000u) != 0u;"#,
                encode: r#"
let upper = u32(floor(significand / f64(4294967296.0)));
let lower = u32(significand - f64(upper) * f64(4294967296.0));
let sign = select(0u, 0x80000000u, negative);
return bitcast<f64>(vec2<u32>(lower, sign | (encoded_exponent << 20u) | upper));"#,
                nan: "bitcast<f64>(vec2<u32>(0u, 0x7ff80000u))",
            },
        }
    }
}

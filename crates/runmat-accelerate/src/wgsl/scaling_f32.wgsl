// Exact binary32 multiplication by an integer power of two, rounded to nearest
// with ties to even. Integer exponent construction also preserves normal
// results when a backend flushes a floating reciprocal scale to zero.
fn runmat_scale_f32_pow2(value: f32, power: i32) -> f32 {
  let bits = bitcast<u32>(value);
  let sign = bits & 0x80000000u;
  var exponent = i32((bits >> 23u) & 0xffu);
  var significand = bits & 0x007fffffu;
  if (exponent == 255 || (exponent == 0 && significand == 0u)) { return value; }
  if (power > 512) { return bitcast<f32>(sign | 0x7f800000u); }
  if (power < -512) { return bitcast<f32>(sign); }
  if (exponent == 0) {
    let shift = countLeadingZeros(significand) - 8u;
    significand = significand << shift;
    exponent = 1 - i32(shift);
  } else {
    significand = significand | 0x00800000u;
  }
  exponent = exponent + power;
  if (exponent >= 255) { return bitcast<f32>(sign | 0x7f800000u); }
  if (exponent > 0) { return bitcast<f32>(sign | (u32(exponent) << 23u) | (significand & 0x007fffffu)); }
  let shift = u32(1 - exponent);
  if (shift > 24u) { return bitcast<f32>(sign); }
  var rounded = significand >> shift;
  let remainder = significand & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) { rounded = rounded + 1u; }
  return bitcast<f32>(sign | rounded);
}

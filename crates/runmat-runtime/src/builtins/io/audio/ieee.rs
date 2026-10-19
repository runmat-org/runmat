pub(super) fn read_ieee_extended_80(bytes: &[u8]) -> Result<f64, String> {
    if bytes.len() != 10 {
        return Err("AIFF sample rate field must be 80 bits".to_string());
    }
    let sign = if bytes[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (((bytes[0] & 0x7F) as u16) << 8) | bytes[1] as u16;
    let mantissa = u64::from_be_bytes([
        bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
    ]);
    if exponent == 0 && mantissa == 0 {
        return Ok(0.0);
    }
    let fraction = mantissa as f64 / (1u64 << 63) as f64;
    Ok(sign * fraction * 2f64.powi(exponent as i32 - 16383))
}

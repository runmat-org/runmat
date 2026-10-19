//! Fixed-size identity wire strings use stack scratch; display/JSON spelling is unchanged.
pub(super) fn serialize<const N: usize, S: serde::Serializer>(
    prefix: &str,
    bytes: &[u8],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if bytes
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(prefix.len()))
        != Some(N)
    {
        return Err(serde::ser::Error::custom("identity wire length mismatch"));
    }
    let mut encoded = [0u8; N];
    encoded[..prefix.len()].copy_from_slice(prefix.as_bytes());
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, &byte) in bytes.iter().enumerate() {
        encoded[prefix.len() + 2 * i] = HEX[(byte >> 4) as usize];
        encoded[prefix.len() + 2 * i + 1] = HEX[(byte & 15) as usize];
    }
    let value = std::str::from_utf8(&encoded).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(value)
}

//! Packed geometry in a figure scene: base64 of little-endian `f32` positions and normals and
//! `u32` triangle indices. A CAD prototype's mesh packed this way is about half the size of the
//! same mesh as JSON number arrays, which keeps a large assembly's scene inside the import
//! limit.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackedMesh {
    /// x, y, z per vertex (`f32`).
    pub positions: String,
    /// Three vertex indices per triangle: `u32`, or `u16` when `index_bytes` is 2.
    pub triangles: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_bytes: Option<u8>,
    /// x, y, z per vertex (`f32`), when the source knows the surface normals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normals: Option<String>,
    /// The normals as signed normalized `i16` (value / 32767), in place of `normals`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normals16: Option<String>,
}

impl PackedMesh {
    pub(crate) fn decode(&self) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>), String> {
        let positions = triples(&f32s(&self.positions)?)?;
        let indices = if self.index_bytes == Some(2) {
            u16s(&self.triangles)?
        } else {
            u32s(&self.triangles)?
        };
        let triangles: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        let normals = match (&self.normals, &self.normals16) {
            (Some(n), _) => triples(&f32s(n)?)?,
            (None, Some(n)) => triples(&snorm16s(n)?)?,
            (None, None) => Vec::new(),
        };
        if !normals.is_empty() && normals.len() != positions.len() {
            return Err("packed normals don't match the positions".into());
        }
        if triangles
            .iter()
            .flatten()
            .any(|&i| i as usize >= positions.len())
        {
            return Err("packed triangle index past the positions".into());
        }
        Ok((positions, triangles, normals))
    }
}

fn triples(values: &[f32]) -> Result<Vec<[f32; 3]>, String> {
    if !values.len().is_multiple_of(3) {
        return Err("packed coordinates are not whole triples".into());
    }
    Ok(values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

pub(crate) fn f32s(text: &str) -> Result<Vec<f32>, String> {
    let bytes = base64(text)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("packed f32 data is not whole values".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

pub(crate) fn u32s(text: &str) -> Result<Vec<u32>, String> {
    let bytes = base64(text)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("packed u32 data is not whole values".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

pub(crate) fn u16s(text: &str) -> Result<Vec<u32>, String> {
    let bytes = base64(text)?;
    if !bytes.len().is_multiple_of(2) {
        return Err("packed u16 data is not whole values".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
        .collect())
}

fn snorm16s(text: &str) -> Result<Vec<f32>, String> {
    let bytes = base64(text)?;
    if !bytes.len().is_multiple_of(2) {
        return Err("packed i16 data is not whole values".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| (i16::from_le_bytes([b[0], b[1]]) as f32 / 32767.0).clamp(-1.0, 1.0))
        .collect())
}

/// Standard base64 (RFC 4648, `=` padding).
fn base64(text: &str) -> Result<Vec<u8>, String> {
    let value = |c: u8| -> Result<u32, String> {
        Ok(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 character {:?}", c as char)),
        })
    };
    let text = text.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    for chunk in text.chunks(4) {
        let mut n = 0u32;
        for (k, &c) in chunk.iter().enumerate() {
            n |= value(c)? << (18 - 6 * k);
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        match chunk.len() {
            4 => out.extend_from_slice(&bytes),
            3 => out.extend_from_slice(&bytes[..2]),
            2 => out.push(bytes[0]),
            _ => return Err("truncated base64".into()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_values_decode_as_little_endian() {
        // [1.0f32, -2.5f32] and [7u32, 65536u32].
        assert_eq!(f32s("AACAPwAAIMA=").unwrap(), vec![1.0, -2.5]);
        assert_eq!(u32s("BwAAAAAAAQA=").unwrap(), vec![7, 65536]);
        let mesh = PackedMesh {
            positions: "AACAPwAAAAAAAAAA".into(),
            triangles: "AAAAAAAAAAAAAAAA".into(),
            index_bytes: None,
            normals: None,
            normals16: None,
        };
        let (p, t, n) = mesh.decode().unwrap();
        assert_eq!(p, vec![[1.0, 0.0, 0.0]]);
        assert_eq!(t, vec![[0, 0, 0]]);
        assert!(n.is_empty());
        assert!(f32s("AAA*").is_err());
        // u16 indices [0, 0, 0] and snorm normals [32767, 0, -32767].
        let small = PackedMesh {
            positions: "AACAPwAAAAAAAAAA".into(),
            triangles: "AAAAAAAA".into(),
            index_bytes: Some(2),
            normals: None,
            normals16: Some("/38AAAGA".into()),
        };
        let (_, t, n) = small.decode().unwrap();
        assert_eq!(t, vec![[0, 0, 0]]);
        assert_eq!(n, vec![[1.0, 0.0, -1.0]]);
    }
}

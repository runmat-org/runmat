//! OPC packages (ECMA-376 Part 2) as ZIP archives: the writer stores entries uncompressed, as
//! OPC allows; the reader takes stored and deflated entries, so it reads packages other
//! applications write.

use miniz_oxide::inflate::decompress_to_vec;

fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    table
}

/// CRC-32 (ISO 3309), as ZIP records it.
pub fn crc32(data: &[u8]) -> u32 {
    let table = crc_table();
    let mut c = 0xFFFF_FFFFu32;
    for b in data {
        c = table[((c ^ *b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// A ZIP archive of stored (uncompressed) entries, in the order given.
pub fn zip_stored(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let size = data.len() as u32;
        let header = |buf: &mut Vec<u8>, sig: u32, central: bool| {
            buf.extend(sig.to_le_bytes());
            if central {
                buf.extend(20u16.to_le_bytes()); // made by
            }
            buf.extend(20u16.to_le_bytes()); // version needed
            buf.extend(0u16.to_le_bytes()); // flags
            buf.extend(0u16.to_le_bytes()); // stored
            buf.extend(0u16.to_le_bytes()); // time
            buf.extend(0x21u16.to_le_bytes()); // date: 1980-01-01
            buf.extend(crc.to_le_bytes());
            buf.extend(size.to_le_bytes());
            buf.extend(size.to_le_bytes());
            buf.extend((name.len() as u16).to_le_bytes());
            buf.extend(0u16.to_le_bytes()); // extra
            if central {
                buf.extend(0u16.to_le_bytes()); // comment
                buf.extend(0u16.to_le_bytes()); // disk
                buf.extend(0u16.to_le_bytes()); // internal attributes
                buf.extend(0u32.to_le_bytes()); // external attributes
                buf.extend(offset.to_le_bytes());
            }
            buf.extend(name.as_bytes());
        };
        header(&mut out, 0x0403_4b50, false);
        out.extend(data);
        header(&mut central, 0x0201_4b50, true);
    }
    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend(central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend(cd_size.to_le_bytes());
    out.extend(cd_offset.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, String> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "the ZIP archive is truncated".to_owned())
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "the ZIP archive is truncated".to_owned())
}

/// The entries of a ZIP archive by name, from its central directory, each checked against its
/// CRC. Stored and deflated entries are read; ZIP64 and encrypted archives are refused.
pub fn unzip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let eocd = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..].starts_with(&0x0605_4b50u32.to_le_bytes()))
        .ok_or("not a ZIP archive (no end of central directory)")?;
    let count = u16_at(bytes, eocd + 10)? as usize;
    let mut at = u32_at(bytes, eocd + 16)? as usize;
    if at == 0xFFFF_FFFF {
        return Err("ZIP64 archives are not read".into());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if u32_at(bytes, at)? != 0x0201_4b50 {
            return Err("a central directory record is damaged".into());
        }
        let flags = u16_at(bytes, at + 8)?;
        let method = u16_at(bytes, at + 10)?;
        let crc = u32_at(bytes, at + 16)?;
        let csize = u32_at(bytes, at + 20)? as usize;
        let size = u32_at(bytes, at + 24)? as usize;
        let name_len = u16_at(bytes, at + 28)? as usize;
        let extra_len = u16_at(bytes, at + 30)? as usize;
        let comment_len = u16_at(bytes, at + 32)? as usize;
        let local = u32_at(bytes, at + 42)? as usize;
        let name = bytes
            .get(at + 46..at + 46 + name_len)
            .ok_or("the ZIP archive is truncated")?;
        let name = String::from_utf8_lossy(name).into_owned();
        at += 46 + name_len + extra_len + comment_len;
        if flags & 1 != 0 {
            return Err(format!("{name} is encrypted"));
        }
        if u32_at(bytes, local)? != 0x0403_4b50 {
            return Err(format!("{name}: the local header is damaged"));
        }
        let start =
            local + 30 + u16_at(bytes, local + 26)? as usize + u16_at(bytes, local + 28)? as usize;
        let raw = bytes
            .get(start..start + csize)
            .ok_or_else(|| format!("{name} is truncated"))?;
        let data = match method {
            0 => raw.to_vec(),
            8 => decompress_to_vec(raw).map_err(|e| format!("{name}: bad deflate data ({e:?})"))?,
            m => {
                return Err(format!(
                    "{name} uses ZIP compression method {m}, which is not read"
                ))
            }
        };
        if data.len() != size || crc32(&data) != crc {
            return Err(format!("{name} fails its CRC or size check"));
        }
        out.push((name, data));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn stored_archives_read_back() {
        let z = zip_stored(&[("a.txt", b"alpha".to_vec()), ("dir/b.bin", vec![0, 1, 2])]);
        let back = unzip(&z).unwrap();
        assert_eq!(back[0], ("a.txt".into(), b"alpha".to_vec()));
        assert_eq!(back[1], ("dir/b.bin".into(), vec![0, 1, 2]));
        let mut bad = z.clone();
        bad[30 + 5] ^= 1; // a byte of a.txt's data
        assert!(unzip(&bad).unwrap_err().contains("CRC"));
    }
}

//! A gzip-compressed POSIX tar archive (`.tgz`) of text and binary files, for outputs that
//! are a folder tree (an ODB++ job). Entries are regular files with mode 0644 and mtime 0,
//! in the order given, so the archive is deterministic.

use super::opc::crc32;

fn octal(field: &mut [u8], v: u64) {
    let s = format!("{:0w$o}", v, w = field.len() - 1);
    field[..s.len()].copy_from_slice(s.as_bytes());
    field[s.len()] = 0;
}

/// One ustar header block for a file `path` of `size` bytes.
fn header(path: &str, size: u64) -> [u8; 512] {
    let mut h = [0u8; 512];
    let (prefix, name) = if path.len() > 100 {
        let cut = path[..path.len().min(155)].rfind('/').unwrap_or(0);
        (&path[..cut], &path[cut + 1..])
    } else {
        ("", path)
    };
    h[..name.len().min(100)].copy_from_slice(&name.as_bytes()[..name.len().min(100)]);
    octal(&mut h[100..108], 0o644);
    octal(&mut h[108..116], 0);
    octal(&mut h[116..124], 0);
    octal(&mut h[124..136], size);
    octal(&mut h[136..148], 0);
    h[156] = b'0';
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[345..345 + prefix.len()].copy_from_slice(prefix.as_bytes());
    // The checksum is the byte sum with its own field as spaces.
    h[148..156].copy_from_slice(b"        ");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    let s = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(s.as_bytes());
    h
}

/// The tar archive of `entries` (path → bytes).
pub fn tar(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (path, data) in entries {
        out.extend_from_slice(&header(path, data.len() as u64));
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.resize(out.len() + 1024, 0);
    out
}

/// `data` as a gzip member (deflate, no file name, mtime 0).
pub fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
    out.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(data, 6));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// The `.tgz` of `entries`.
pub fn tgz(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    gzip(&tar(entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_archive_reads_back_with_inflate_and_ustar_fields() {
        let entries = vec![
            ("job/matrix/matrix".to_string(), b"STEP {\n}\n".to_vec()),
            ("job/misc/info".to_string(), vec![b'x'; 700]),
        ];
        let z = tgz(&entries);
        assert_eq!(&z[..3], &[0x1f, 0x8b, 8]);
        let raw = miniz_oxide::inflate::decompress_to_vec(&z[10..z.len() - 8]).unwrap();
        assert_eq!(crc32(&raw).to_le_bytes(), z[z.len() - 8..z.len() - 4]);
        assert_eq!(raw.len(), 512 + 512 + 512 + 1024 + 1024);
        assert_eq!(&raw[..17], b"job/matrix/matrix");
        assert_eq!(&raw[257..262], b"ustar");
        let sum: u32 = raw[..512]
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if (148..156).contains(&i) {
                    32
                } else {
                    b as u32
                }
            })
            .sum();
        let stored = u32::from_str_radix(std::str::from_utf8(&raw[148..154]).unwrap(), 8).unwrap();
        assert_eq!(sum, stored);
        assert_eq!(&raw[512..521], b"STEP {\n}\n");
        assert_eq!(
            u64::from_str_radix(
                std::str::from_utf8(&raw[1024 + 124..1024 + 135]).unwrap(),
                8
            )
            .unwrap(),
            700
        );
    }
}

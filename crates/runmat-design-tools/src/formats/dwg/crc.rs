//! DWG checksums (specification 2.14 and 4.2).

fn crc16_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let mut c = i as u16;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xA001
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    t
}

/// The 16-bit CRC objects and the header variables carry (the "8-bit CRC" of 2.14.1), from
/// `seed` (0xC0C1 for objects).
pub fn crc16(seed: u16, data: &[u8]) -> u16 {
    let t = crc16_table();
    let mut c = seed;
    for b in data {
        c = (c >> 8) ^ t[((c ^ *b as u16) & 0xFF) as usize];
    }
    c
}

/// The 32-bit CRC of 2.14.2 (the standard CRC-32) from `seed`.
pub fn crc32(seed: u32, data: &[u8]) -> u32 {
    let mut c = !seed;
    for b in data {
        c ^= *b as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// The section page checksum of 4.2 (an Adler-style sum in chunks of 0x15B0 bytes).
pub fn page_checksum(seed: u32, data: &[u8]) -> u32 {
    let mut sum1 = seed & 0xFFFF;
    let mut sum2 = seed >> 16;
    for chunk in data.chunks(0x15B0) {
        for b in chunk {
            sum1 += *b as u32;
            sum2 += sum1;
        }
        sum1 %= 0xFFF1;
        sum2 %= 0xFFF1;
    }
    (sum2 << 16) | (sum1 & 0xFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_tables_match_the_specification() {
        // The specification's table starts 0x0000, 0xC0C1, 0xC181, 0x0140 (2.14.1).
        let t = crc16_table();
        assert_eq!(&t[..4], &[0x0000, 0xC0C1, 0xC181, 0x0140]);
        assert_eq!(crc32(0, b"123456789"), 0xCBF4_3926);
        assert_eq!(page_checksum(0, &[]), 0);
        assert_eq!(page_checksum(0, &[1, 2]), (4 << 16) | 3);
    }
}

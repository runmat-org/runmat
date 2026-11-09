//! The R2004 compression format (specification 4.7): an LZ77 variant of literal runs and
//! back-references with lengths and offsets in opcode-dependent encodings.

/// A literal length as the stream's first opcode, or after an opcode whose own literal count
/// is zero: `0x01..=0x0F` is the length minus 3; `0x00` starts a running total of 0x0F.
fn literal_length(out: &mut Vec<u8>, n: usize) {
    debug_assert!(n >= 4);
    let v = n - 3;
    if v <= 0x0F {
        out.push(v as u8);
        return;
    }
    out.push(0);
    let r = v - 0x0F;
    let zeros = (r - 1) / 0xFF;
    out.extend(std::iter::repeat_n(0u8, zeros));
    out.push((r - zeros * 0xFF) as u8);
}

/// `data` as one literal run and the terminator. Every reader decompresses it; `data` must be
/// at least 4 bytes (pages are whole page-size blocks).
pub fn compress(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() >= 4,
        "a compressed page holds at least four bytes"
    );
    let mut out = Vec::with_capacity(data.len() + 8);
    literal_length(&mut out, data.len());
    out.extend_from_slice(data);
    out.push(0x11);
    out
}

struct Input<'a> {
    s: &'a [u8],
    i: usize,
}

impl Input<'_> {
    fn byte(&mut self) -> Result<u8, String> {
        let b = *self.s.get(self.i).ok_or("the compressed data ends early")?;
        self.i += 1;
        Ok(b)
    }
    /// A literal length; `None` when the byte is the next opcode (high nibble set).
    fn literal_length(&mut self) -> Result<Option<usize>, String> {
        let b = *self.s.get(self.i).ok_or("the compressed data ends early")?;
        if b & 0xF0 != 0 {
            return Ok(None);
        }
        self.i += 1;
        if b != 0 {
            return Ok(Some(b as usize + 3));
        }
        let mut total = 0x0F;
        loop {
            let b = self.byte()?;
            if b == 0 {
                total += 0xFF;
            } else {
                return Ok(Some(total + b as usize + 3));
            }
        }
    }
    fn long_offset(&mut self) -> Result<usize, String> {
        let mut total = 0;
        loop {
            let b = self.byte()?;
            if b != 0 {
                return Ok(total + b as usize);
            }
            total += 0xFF;
        }
    }
    /// (offset, literal count) of a two-byte offset.
    fn two_byte(&mut self) -> Result<(usize, usize), String> {
        let a = self.byte()? as usize;
        let b = self.byte()? as usize;
        Ok(((a >> 2) | (b << 6), a & 3))
    }
}

/// Decompress a stream into at most `limit` bytes.
pub fn decompress(s: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut inp = Input { s, i: 0 };
    let mut out: Vec<u8> = Vec::with_capacity(limit);
    let copy_literal = |inp: &mut Input, out: &mut Vec<u8>, n: usize| -> Result<(), String> {
        let lit = inp
            .s
            .get(inp.i..inp.i + n)
            .ok_or("a literal run runs past the data")?;
        out.extend_from_slice(lit);
        inp.i += n;
        Ok(())
    };
    if let Some(n) = inp.literal_length()? {
        copy_literal(&mut inp, &mut out, n)?;
    }
    loop {
        let op = inp.byte()?;
        let (count, offset, lit) = match op {
            0x11 => break,
            0x10 => {
                let c = inp.long_offset()? + 9;
                let (o, l) = inp.two_byte()?;
                (c, o + 0x3FFF, l)
            }
            0x12..=0x1F => {
                let (o, l) = inp.two_byte()?;
                ((op & 0x0F) as usize + 2, o + 0x3FFF, l)
            }
            0x20 => {
                let c = inp.long_offset()? + 0x21;
                let (o, l) = inp.two_byte()?;
                (c, o, l)
            }
            0x21..=0x3F => {
                let (o, l) = inp.two_byte()?;
                (op as usize - 0x1E, o, l)
            }
            0x40..=0xFF => {
                let op2 = inp.byte()? as usize;
                (
                    ((op & 0xF0) >> 4) as usize - 1,
                    (op2 << 2) | ((op & 0x0C) >> 2) as usize,
                    (op & 3) as usize,
                )
            }
            _ => return Err(format!("compression opcode {op:#x} is not used")),
        };
        // Offsets count back from one before the current end.
        let from = out
            .len()
            .checked_sub(offset + 1)
            .ok_or("a back-reference points before the data")?;
        for k in 0..count {
            let b = out[from + k];
            out.push(b);
        }
        let lit = if lit == 0 {
            inp.literal_length()?.unwrap_or(0)
        } else {
            lit
        };
        copy_literal(&mut inp, &mut out, lit)?;
        if out.len() > limit {
            return Err("the data decompresses past its stated size".into());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_runs_round_trip_at_every_length_encoding() {
        for n in [
            4usize, 5, 0x11, 0x12, 0x13, 0x20, 0x10E, 0x10F, 0x110, 0x20D, 0x7400,
        ] {
            let data: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            assert_eq!(decompress(&compress(&data), n).unwrap(), data, "length {n}");
        }
    }

    #[test]
    fn back_references_decompress() {
        // "abcd" then a two-byte-offset copy of 4 bytes from offset 3 (the start), then end.
        let s = [0x01, b'a', b'b', b'c', b'd', 0x22, 0x0C, 0x00, 0x11];
        assert_eq!(decompress(&s, 64).unwrap(), b"abcdabcd");
        // Overlapping copy (length past the offset): 0x40-form, count 3, offset 0, one literal.
        let s = [0x01, b'x', b'y', b'z', b'w', 0x41, 0x00, b'!', 0x11];
        assert_eq!(decompress(&s, 64).unwrap(), b"xyzwwww!");
    }
}

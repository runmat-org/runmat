//! The R2010+ object frame (specification 20.1 and 20.2): an object's bits are three streams,
//! the data, the strings (every TU of the object, R2007+) and the handle references, framed
//! as MS size, MC handle-stream size in bits, the streams, padding and the CRC.
//!
//! The string stream sits at the end of the data: its bits, then its size in bits as a raw
//! short (with a second short for the high bits when the size needs more than 15), then a
//! flag bit saying a string stream is present. A reader finds the flag at the bit before the
//! handle stream and walks back from there.

use super::bits::{ms, read_mc, read_ms, umc, BitReader, BitWriter, Handle};
use super::crc::crc16;

type R<T> = Result<T, String>;

/// An object being written.
pub struct ObjectOut {
    pub data: BitWriter,
    pub strings: BitWriter,
    pub handles: BitWriter,
}

impl ObjectOut {
    /// Starts an object: its type and handle, no extended data.
    pub fn new(kind: u16, handle: u64) -> Self {
        let mut data = BitWriter::new();
        data.ot(kind);
        data.h(Handle {
            code: 0,
            value: handle,
        });
        data.bs(0); // no extended entity data
        ObjectOut {
            data,
            strings: BitWriter::new(),
            handles: BitWriter::new(),
        }
    }
    /// A text field, into the string stream.
    pub fn t(&mut self, s: &str) {
        self.strings.tu(s);
    }
    pub fn h(&mut self, code: u8, value: u64) {
        self.handles.h(Handle { code, value });
    }
    /// The object's bytes in its frame: MS size, MC handle bits, the streams, the CRC.
    pub fn finish(self) -> Vec<u8> {
        let mut body = self.data;
        strings_into(&mut body, &self.strings);
        let start = body.len_bits();
        body.append(&self.handles);
        body.pad();
        let handle_bits = body.len_bits() - start;
        let bytes = body.finish();
        let mut out = ms(bytes.len() as u64);
        out.extend(umc(handle_bits as u64));
        out.extend(&bytes);
        let crc = crc16(0xC0C1, &out);
        out.extend(crc.to_le_bytes());
        out
    }
}

/// Appends a string stream and its trailer (size, flag) to a data stream.
pub fn strings_into(body: &mut BitWriter, strings: &BitWriter) {
    let n = strings.len_bits();
    if n == 0 {
        body.b(false);
        return;
    }
    body.append(strings);
    if n >= 0x8000 {
        body.rs((n >> 15) as u16);
        body.rs((n & 0x7FFF) as u16 | 0x8000);
    } else {
        body.rs(n as u16);
    }
    body.b(true);
}

/// Where a stream's strings start, given the bit just past its data (the flag's end).
pub fn strings_start(bytes: &[u8], end: usize) -> R<Option<usize>> {
    if end == 0 {
        return Err("an object's data is empty".into());
    }
    if !BitReader::at_bits(bytes, end - 1).b()? {
        return Ok(None);
    }
    let mut at = end
        .checked_sub(17)
        .ok_or("a string stream runs before the object")?;
    let lo = BitReader::at_bits(bytes, at).rs()?;
    let mut size = lo as usize;
    if lo & 0x8000 != 0 {
        at = at
            .checked_sub(16)
            .ok_or("a string stream runs before the object")?;
        let hi = BitReader::at_bits(bytes, at).rs()?;
        size = (lo as usize & 0x7FFF) | (hi as usize) << 15;
    }
    at.checked_sub(size)
        .map(Some)
        .ok_or_else(|| "a string stream is longer than its object".into())
}

/// An object read back: its type, handle and the three streams.
pub struct ObjectIn {
    pub kind: u16,
    pub handle: u64,
    pub bytes: Vec<u8>,
    /// The data stream's position after the type, handle and extended data.
    pub data_at: usize,
    pub strings_at: Option<usize>,
    pub handles_at: usize,
}

impl ObjectIn {
    pub fn data(&self) -> BitReader<'_> {
        BitReader::at_bits(&self.bytes, self.data_at)
    }
    pub fn strings(&self) -> BitReader<'_> {
        BitReader::at_bits(&self.bytes, self.strings_at.unwrap_or(self.bytes.len() * 8))
    }
    pub fn handles(&self) -> BitReader<'_> {
        BitReader::at_bits(&self.bytes, self.handles_at)
    }
}

/// Read the object framed at `at` in the objects section, checking its CRC.
pub fn read_object(section: &[u8], at: usize) -> R<ObjectIn> {
    let mut k = at;
    let size = read_ms(section, &mut k)? as usize;
    let handle_bits = read_mc(section, &mut k, false)? as usize;
    let bytes = section
        .get(k..k + size)
        .ok_or("an object runs past the objects section")?
        .to_vec();
    let crc = section
        .get(k + size..k + size + 2)
        .ok_or("an object's CRC is missing")?;
    if crc16(0xC0C1, &section[at..k + size]) != u16::from_le_bytes([crc[0], crc[1]]) {
        return Err(format!("the object at {at} fails its CRC"));
    }
    let handles_at = (size * 8)
        .checked_sub(handle_bits)
        .ok_or("an object's handle stream is longer than the object")?;
    let strings_at = strings_start(&bytes, handles_at)?;
    let mut r = BitReader::new(&bytes);
    let kind = r.ot()?;
    let handle = r.h()?.value;
    loop {
        let n = r.bs()?;
        if n == 0 {
            break;
        }
        r.h()?;
        for _ in 0..n {
            r.rc()?;
        }
    }
    let data_at = r.position();
    Ok(ObjectIn {
        kind,
        handle,
        bytes,
        data_at,
        strings_at,
        handles_at,
    })
}

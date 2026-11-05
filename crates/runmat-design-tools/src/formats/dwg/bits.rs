//! DWG bit streams (specification chapter 2): values packed MSB-first across byte boundaries.

/// A DWG handle reference: the reference code (soft or hard, owner or pointer; or an offset
/// form) and the handle value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle {
    pub code: u8,
    pub value: u64,
}

/// A color (CMC, R2004+): the index, the RGB value with its method byte, and an optional name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Color {
    pub index: i16,
    /// `0xC2RRGGBB` for a true color, `0xC3000000 | index` by index, `0xC0000000` by layer.
    pub rgb: u32,
    pub name: Option<String>,
    pub book: Option<String>,
}

#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }
    /// Bits written so far.
    pub fn len_bits(&self) -> usize {
        self.bit
    }
    pub fn b(&mut self, v: bool) {
        if self.bit.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if v {
            *self.bytes.last_mut().expect("pushed") |= 0x80 >> (self.bit % 8);
        }
        self.bit += 1;
    }
    pub fn bits(&mut self, v: u64, n: usize) {
        for k in (0..n).rev() {
            self.b((v >> k) & 1 == 1);
        }
    }
    pub fn bb(&mut self, v: u8) {
        self.bits(v as u64, 2);
    }
    /// 3B: 0 → `0`, 2 → `10`, 6 → `110`, 7 → `111` (1 to 3 bits read until a zero).
    pub fn b3(&mut self, v: u8) {
        match v {
            0 => self.bits(0, 1),
            2 => self.bits(0b10, 2),
            6 => self.bits(0b110, 3),
            _ => self.bits(0b111, 3),
        }
    }
    pub fn rc(&mut self, v: u8) {
        self.bits(v as u64, 8);
    }
    pub fn rs(&mut self, v: u16) {
        for b in v.to_le_bytes() {
            self.rc(b);
        }
    }
    pub fn rl(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.rc(b);
        }
    }
    pub fn rd(&mut self, v: f64) {
        for b in v.to_le_bytes() {
            self.rc(b);
        }
    }
    pub fn bs(&mut self, v: i16) {
        match v {
            0 => self.bb(2),
            256 => self.bb(3),
            1..=255 => {
                self.bb(1);
                self.rc(v as u8);
            }
            _ => {
                self.bb(0);
                self.rs(v as u16);
            }
        }
    }
    pub fn bl(&mut self, v: i32) {
        match v {
            0 => self.bb(2),
            1..=255 => {
                self.bb(1);
                self.rc(v as u8);
            }
            _ => {
                self.bb(0);
                self.rl(v as u32);
            }
        }
    }
    /// BLL: the byte count in 3B form, then the bytes, least significant first.
    pub fn bll(&mut self, v: u64) {
        let n = (8 - v.leading_zeros() as usize / 8).min(7);
        self.bits(n as u64, 3);
        for k in 0..n {
            self.rc((v >> (8 * k)) as u8);
        }
    }
    pub fn bd(&mut self, v: f64) {
        if v == 1.0 {
            self.bb(1);
        } else if v == 0.0 && v.is_sign_positive() {
            self.bb(2);
        } else {
            self.bb(0);
            self.rd(v);
        }
    }
    /// DD: a double against a default, patching only the bytes that differ.
    pub fn dd(&mut self, v: f64, default: f64) {
        let (a, d) = (v.to_le_bytes(), default.to_le_bytes());
        if a == d {
            self.bb(0);
        } else if a[4..] == d[4..] {
            self.bb(1);
            for b in &a[..4] {
                self.rc(*b);
            }
        } else if a[6..] == d[6..] {
            self.bb(2);
            for b in [a[4], a[5], a[0], a[1], a[2], a[3]] {
                self.rc(b);
            }
        } else {
            self.bb(3);
            self.rd(v);
        }
    }
    /// BE: the extrusion, one bit when it is (0, 0, 1).
    pub fn be(&mut self, v: [f64; 3]) {
        if v == [0.0, 0.0, 1.0] {
            self.b(true);
        } else {
            self.b(false);
            for c in v {
                self.bd(c);
            }
        }
    }
    /// BT: the thickness, one bit when it is zero.
    pub fn bt(&mut self, v: f64) {
        if v == 0.0 {
            self.b(true);
        } else {
            self.b(false);
            self.bd(v);
        }
    }
    pub fn h(&mut self, h: Handle) {
        let n = (8 - h.value.leading_zeros() as usize / 8).min(8);
        self.bits(h.code as u64, 4);
        self.bits(n as u64, 4);
        for k in (0..n).rev() {
            self.rc((h.value >> (8 * k)) as u8);
        }
    }
    /// TU (R2007+): a BS length, then UTF-16 code units.
    pub fn tu(&mut self, s: &str) {
        let units: Vec<u16> = s.encode_utf16().collect();
        self.bs(units.len() as i16);
        for u in units {
            self.rs(u);
        }
    }
    pub fn cmc(&mut self, c: &Color) {
        self.bs(c.index);
        self.bl(c.rgb as i32);
        let flags = c.name.is_some() as u8 | (c.book.is_some() as u8) << 1;
        self.rc(flags);
        for s in [&c.name, &c.book].into_iter().flatten() {
            self.tu(s);
        }
    }
    /// OT (R2010+): the object type.
    pub fn ot(&mut self, t: u16) {
        if t < 0x100 {
            self.bb(0);
            self.rc(t as u8);
        } else if (0x1F0..0x2F0).contains(&t) {
            self.bb(1);
            self.rc((t - 0x1F0) as u8);
        } else {
            self.bb(2);
            self.rs(t);
        }
    }
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// MC: a signed modular char (7 bits per byte, least significant first; 0x40 of the last byte
/// is the sign).
pub fn mc(v: i64) -> Vec<u8> {
    let mut m = v.unsigned_abs();
    let mut out = Vec::new();
    loop {
        let b = (m & 0x7F) as u8;
        m >>= 7;
        if m == 0 && b & 0x40 == 0 {
            out.push(b | if v < 0 { 0x40 } else { 0 });
            return out;
        }
        out.push(b | 0x80);
        if m == 0 {
            out.push(if v < 0 { 0x40 } else { 0 });
            return out;
        }
    }
}

/// An unsigned modular char (handles and offsets in the object map, sizes).
pub fn umc(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
}

/// MS: a modular short (15 bits per little-endian word, least significant first).
pub fn ms(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let w = (v & 0x7FFF) as u16;
        v >>= 15;
        let w = if v == 0 { w } else { w | 0x8000 };
        out.extend(w.to_le_bytes());
        if v == 0 {
            return out;
        }
    }
}

pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

type R<T> = Result<T, String>;

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, bit: 0 }
    }
    pub fn at_bits(bytes: &'a [u8], bit: usize) -> Self {
        BitReader { bytes, bit }
    }
    pub fn position(&self) -> usize {
        self.bit
    }
    pub fn b(&mut self) -> R<bool> {
        let byte = *self.bytes.get(self.bit / 8).ok_or("the bit stream ends early")?;
        let v = byte & (0x80 >> (self.bit % 8)) != 0;
        self.bit += 1;
        Ok(v)
    }
    pub fn bits(&mut self, n: usize) -> R<u64> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.b()? as u64;
        }
        Ok(v)
    }
    pub fn bb(&mut self) -> R<u8> {
        Ok(self.bits(2)? as u8)
    }
    pub fn b3(&mut self) -> R<u8> {
        let mut v = 0;
        for _ in 0..3 {
            let b = self.b()?;
            v = (v << 1) | b as u8;
            if !b {
                break;
            }
        }
        Ok(v)
    }
    pub fn rc(&mut self) -> R<u8> {
        Ok(self.bits(8)? as u8)
    }
    fn raw<const N: usize>(&mut self) -> R<[u8; N]> {
        let mut a = [0u8; N];
        for b in &mut a {
            *b = self.rc()?;
        }
        Ok(a)
    }
    pub fn rs(&mut self) -> R<u16> {
        Ok(u16::from_le_bytes(self.raw()?))
    }
    pub fn rl(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.raw()?))
    }
    pub fn rd(&mut self) -> R<f64> {
        Ok(f64::from_le_bytes(self.raw()?))
    }
    pub fn bs(&mut self) -> R<i16> {
        Ok(match self.bb()? {
            0 => self.rs()? as i16,
            1 => self.rc()? as i16,
            2 => 0,
            _ => 256,
        })
    }
    pub fn bl(&mut self) -> R<i32> {
        match self.bb()? {
            0 => Ok(self.rl()? as i32),
            1 => Ok(self.rc()? as i32),
            2 => Ok(0),
            _ => Err("BL code 3 is not used".into()),
        }
    }
    pub fn bll(&mut self) -> R<u64> {
        let n = self.bits(3)? as usize;
        let mut v = 0u64;
        for k in 0..n {
            v |= (self.rc()? as u64) << (8 * k);
        }
        Ok(v)
    }
    pub fn bd(&mut self) -> R<f64> {
        match self.bb()? {
            0 => self.rd(),
            1 => Ok(1.0),
            2 => Ok(0.0),
            _ => Err("BD code 3 is not used".into()),
        }
    }
    pub fn dd(&mut self, default: f64) -> R<f64> {
        let mut d = default.to_le_bytes();
        match self.bb()? {
            0 => {}
            1 => {
                for b in &mut d[..4] {
                    *b = self.rc()?;
                }
            }
            2 => {
                d[4] = self.rc()?;
                d[5] = self.rc()?;
                for b in &mut d[..4] {
                    *b = self.rc()?;
                }
            }
            _ => return self.rd(),
        }
        Ok(f64::from_le_bytes(d))
    }
    pub fn be(&mut self) -> R<[f64; 3]> {
        if self.b()? {
            Ok([0.0, 0.0, 1.0])
        } else {
            Ok([self.bd()?, self.bd()?, self.bd()?])
        }
    }
    pub fn bt(&mut self) -> R<f64> {
        if self.b()? {
            Ok(0.0)
        } else {
            self.bd()
        }
    }
    pub fn h(&mut self) -> R<Handle> {
        let code = self.bits(4)? as u8;
        let n = self.bits(4)? as usize;
        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 8) | self.rc()? as u64;
        }
        Ok(Handle { code, value })
    }
    pub fn tu(&mut self) -> R<String> {
        let n = self.bs()?;
        let units = (0..n.max(0)).map(|_| self.rs()).collect::<R<Vec<u16>>>()?;
        Ok(String::from_utf16_lossy(&units))
    }
    pub fn cmc(&mut self) -> R<Color> {
        let index = self.bs()?;
        let rgb = self.bl()? as u32;
        let flags = self.rc()?;
        let name = if flags & 1 != 0 { Some(self.tu()?) } else { None };
        let book = if flags & 2 != 0 { Some(self.tu()?) } else { None };
        Ok(Color { index, rgb, name, book })
    }
    pub fn ot(&mut self) -> R<u16> {
        Ok(match self.bb()? {
            0 => self.rc()? as u16,
            1 => self.rc()? as u16 + 0x1F0,
            _ => self.rs()?,
        })
    }
}

/// A modular char read from bytes at `*at` (signed when `signed`).
pub fn read_mc(bytes: &[u8], at: &mut usize, signed: bool) -> R<i64> {
    let mut v: u64 = 0;
    let mut shift = 0;
    loop {
        let b = *bytes.get(*at).ok_or("a modular char runs past the data")?;
        *at += 1;
        if b & 0x80 != 0 {
            v |= ((b & 0x7F) as u64) << shift;
            shift += 7;
            continue;
        }
        if signed {
            v |= ((b & 0x3F) as u64) << shift;
            return Ok(if b & 0x40 != 0 { -(v as i64) } else { v as i64 });
        }
        v |= ((b & 0x7F) as u64) << shift;
        return Ok(v as i64);
    }
}

/// A modular short read from bytes at `*at`.
pub fn read_ms(bytes: &[u8], at: &mut usize) -> R<u64> {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let w = bytes.get(*at..*at + 2).ok_or("a modular short runs past the data")?;
        *at += 2;
        let w = u16::from_le_bytes([w[0], w[1]]);
        v |= ((w & 0x7FFF) as u64) << shift;
        shift += 15;
        if w & 0x8000 == 0 {
            return Ok(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitshorts_parse_as_the_specification_example() {
        // 2.2: "0000000001000000011011010000111110" is 257, 0, 256, 15, 0.
        let s = "0000000001000000011011010000111110";
        let mut w = BitWriter::new();
        for c in s.chars() {
            w.b(c == '1');
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let v: Vec<i16> = (0..5).map(|_| r.bs().unwrap()).collect();
        assert_eq!(v, [257, 0, 256, 15, 0]);
        let mut w = BitWriter::new();
        for x in [257, 0, 256, 15, 0] {
            w.bs(x);
        }
        assert_eq!(w.finish(), bytes);
    }

    #[test]
    fn modular_values_match_the_specification_examples() {
        let mut at = 0;
        assert_eq!(read_mc(&[0b1000_0010, 0b0010_0100], &mut at, true).unwrap(), 4610);
        at = 0;
        assert_eq!(read_mc(&[0b1110_1001, 0b1001_0111, 0b1110_0110, 0b0011_0101], &mut at, true).unwrap(), 112_823_273);
        at = 0;
        assert_eq!(read_mc(&[0b1000_0101, 0b0100_1011], &mut at, true).unwrap(), -1413);
        at = 0;
        assert_eq!(read_ms(&[0b0011_0001, 0b1111_0100, 0b1000_1101, 0], &mut at).unwrap(), 4_650_033);
        for v in [0i64, 5, 63, 64, 4610, -1413, 112_823_273, -9_000_000] {
            let mut at = 0;
            assert_eq!(read_mc(&mc(v), &mut at, true).unwrap(), v, "{v}");
        }
        for v in [0u64, 127, 128, 4_650_033] {
            let (mut a, mut b) = (0, 0);
            assert_eq!(read_mc(&umc(v), &mut a, false).unwrap() as u64, v);
            assert_eq!(read_ms(&ms(v), &mut b).unwrap(), v);
        }
    }

    #[test]
    fn every_bit_code_round_trips() {
        let mut w = BitWriter::new();
        w.b(true);
        w.bb(2);
        w.b3(6);
        w.bl(-1);
        w.bl(77);
        w.bll(0x1234_5678_9A);
        w.bd(1.0);
        w.bd(-2.5);
        w.dd(10.5, 10.0);
        w.dd(10.0, 10.0);
        w.be([0.0, 0.0, 1.0]);
        w.be([1.0, 0.0, 0.0]);
        w.bt(0.0);
        w.bt(3.0);
        w.h(Handle { code: 5, value: 0x1F });
        w.h(Handle { code: 3, value: 0 });
        w.tu("Layer ä");
        let c = Color { index: 7, rgb: 0xC2FF_8000, name: Some("orange".into()), book: None };
        w.cmc(&c);
        w.ot(0x13);
        w.ot(0x1F2);
        w.ot(0x3E8);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(r.b().unwrap());
        assert_eq!(r.bb().unwrap(), 2);
        assert_eq!(r.b3().unwrap(), 6);
        assert_eq!(r.bl().unwrap(), -1);
        assert_eq!(r.bl().unwrap(), 77);
        assert_eq!(r.bll().unwrap(), 0x1234_5678_9A);
        assert_eq!(r.bd().unwrap(), 1.0);
        assert_eq!(r.bd().unwrap(), -2.5);
        assert_eq!(r.dd(10.0).unwrap(), 10.5);
        assert_eq!(r.dd(10.0).unwrap(), 10.0);
        assert_eq!(r.be().unwrap(), [0.0, 0.0, 1.0]);
        assert_eq!(r.be().unwrap(), [1.0, 0.0, 0.0]);
        assert_eq!(r.bt().unwrap(), 0.0);
        assert_eq!(r.bt().unwrap(), 3.0);
        assert_eq!(r.h().unwrap(), Handle { code: 5, value: 0x1F });
        assert_eq!(r.h().unwrap(), Handle { code: 3, value: 0 });
        assert_eq!(r.tu().unwrap(), "Layer ä");
        assert_eq!(r.cmc().unwrap(), c);
        assert_eq!([r.ot().unwrap(), r.ot().unwrap(), r.ot().unwrap()], [0x13, 0x1F2, 0x3E8]);
    }
}

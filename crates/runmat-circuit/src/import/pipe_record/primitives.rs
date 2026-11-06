//! A pipe-record binary board's primitives: the `Pads6`, `Tracks6`, `Arcs6`, `Vias6`,
//! `Fills6`, `Texts6` and `Regions6` streams, read into the same records the ASCII board holds, so
//! [`super::board`] reads both.
//!
//! - A stream is a list of records: a kind byte, then a fixed number of blocks, each a
//!   little-endian `u32` length and its bytes (pads 6 blocks, texts 2, the others 1).
//! - Every primitive's first block starts with the layer (one byte), two flag bytes, the net
//!   (`u16`, `0xFFFF` for none) at byte 3 and the component (`u16`) at byte 7, both indexes
//!   among the `Nets6` and `Components6` records. Coordinates and sizes are `i32` in
//!   1/10000 mil; angles are `f64` degrees. A region's vertices are `f64`, in the same unit.
//! - Layer numbers: 1 top copper, 2–31 inner signal layers, 32 bottom copper, 33/34 the
//!   overlays, 35/36 paste, 37/38 solder mask, 39–54 inner planes, 55 drill guide,
//!   56 keep-out, 57–72 mechanical layers, 73 drill drawing, 74 every layer (through hole).
//!
//! The field positions were checked against three openly licensed boards: every net and
//! component index in range, pad and via sizes matching their holes, rotations at the
//! placement angles, and the slots inside their pads.

use super::Record;
use std::collections::BTreeMap;

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(i..i + 2)?.try_into().ok()?))
}
fn i32_at(b: &[u8], i: usize) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?))
}
fn f64_at(b: &[u8], i: usize) -> Option<f64> {
    Some(f64::from_le_bytes(b.get(i..i + 8)?.try_into().ok()?))
}

/// A layer number as the ASCII records name it.
pub fn layer_name(n: u8) -> String {
    match n {
        1 => "TOP".into(),
        2..=31 => format!("MID{}", n - 1),
        32 => "BOTTOM".into(),
        33 => "TOPOVERLAY".into(),
        34 => "BOTTOMOVERLAY".into(),
        35 => "TOPPASTE".into(),
        36 => "BOTTOMPASTE".into(),
        37 => "TOPSOLDER".into(),
        38 => "BOTTOMSOLDER".into(),
        39..=54 => format!("PLANE{}", n - 38),
        55 => "DRILLGUIDE".into(),
        56 => "KEEPOUT".into(),
        57..=72 => format!("MECHANICAL{}", n - 56),
        73 => "DRILLDRAWING".into(),
        74 => "MULTILAYER".into(),
        other => format!("LAYER{other}"),
    }
}

/// The layer number of an ASCII layer name, the inverse of [`layer_name`].
pub fn layer_number(name: &str) -> Option<u8> {
    (1..=74).find(|n| layer_name(*n).eq_ignore_ascii_case(name.trim()))
}

/// The stream's records: each its kind byte and its blocks.
fn split(data: &[u8], blocks: usize) -> Result<Vec<(u8, Vec<&[u8]>)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let kind = data[i];
        i += 1;
        let mut bl = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            let n = i32_at(data, i).ok_or("a primitive record ends inside its block length")? as u32 as usize;
            let b = data.get(i + 4..i + 4 + n).ok_or("a primitive record's block runs past its stream")?;
            bl.push(b);
            i += 4 + n;
        }
        out.push((kind, bl));
    }
    Ok(out)
}

/// A length field as an ASCII value (`1250.5mil`).
fn mil(b: &[u8], i: usize) -> Option<String> {
    Some(format!("{}mil", i32_at(b, i)? as f64 / 10_000.0))
}

/// A one-byte-length string.
fn short_string(b: &[u8]) -> String {
    let n = b.first().copied().unwrap_or(0) as usize;
    String::from_utf8_lossy(b.get(1..1 + n).unwrap_or(&[])).into_owned()
}

struct Fields(BTreeMap<String, String>);

impl Fields {
    /// The fields every primitive starts with: record kind, layer, net and component.
    fn new(kind: &str, b: &[u8]) -> Option<Fields> {
        let mut f = BTreeMap::new();
        f.insert("RECORD".to_owned(), kind.to_owned());
        f.insert("LAYER".to_owned(), layer_name(*b.first()?));
        for (key, at) in [("NET", 3), ("COMPONENT", 7)] {
            let v = u16_at(b, at)?;
            if v != 0xFFFF {
                f.insert(key.to_owned(), v.to_string());
            }
        }
        Some(Fields(f))
    }
    fn set(&mut self, key: &str, v: impl Into<String>) -> &mut Self {
        self.0.insert(key.to_owned(), v.into());
        self
    }
    fn lengths(&mut self, b: &[u8], keys: &[(&str, usize)]) -> Option<&mut Self> {
        for (k, at) in keys {
            self.set(k, mil(b, *at)?);
        }
        Some(self)
    }
    fn angle(&mut self, key: &str, b: &[u8], at: usize) -> Option<&mut Self> {
        Some(self.set(key, f64_at(b, at)?.to_string()))
    }
    fn done(self) -> Record {
        Record(self.0)
    }
}

fn track(b: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Track", b)?;
    f.lengths(b, &[("X1", 13), ("Y1", 17), ("X2", 21), ("Y2", 25), ("WIDTH", 29)])?;
    Some(f.done())
}

fn arc(b: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Arc", b)?;
    f.lengths(b, &[("LOCATION.X", 13), ("LOCATION.Y", 17), ("RADIUS", 21), ("WIDTH", 41)])?;
    f.angle("STARTANGLE", b, 25)?.angle("ENDANGLE", b, 33)?;
    Some(f.done())
}

fn via(b: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Via", b)?;
    f.lengths(b, &[("X", 13), ("Y", 17), ("DIAMETER", 21), ("HOLESIZE", 25)])?;
    f.set("STARTLAYER", layer_name(*b.get(29)?)).set("ENDLAYER", layer_name(*b.get(30)?));
    Some(f.done())
}

fn fill(b: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Fill", b)?;
    f.lengths(b, &[("X1", 13), ("Y1", 17), ("X2", 21), ("Y2", 25)])?.angle("ROTATION", b, 29)?;
    Some(f.done())
}

fn text(b: &[u8], string: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Text", b)?;
    f.lengths(b, &[("X", 13), ("Y", 17), ("HEIGHT", 21), ("WIDTH", 36)])?.angle("ROTATION", b, 27)?;
    f.set("MIRROR", if *b.get(35)? != 0 { "TRUE" } else { "FALSE" }).set("TEXT", short_string(string));
    Some(f.done())
}

/// A region: the polygon (`u16`) owning it at byte 5, the length (`u32`) of a `|KEY=value|`
/// text at byte 18 and the text (`KIND`, `ISBOARDCUTOUT` …), then rings to the record's
/// end, each a vertex count (`u32`) and its vertices (`f64` x and y in 1/10000 mil): the
/// outline first, then its holes.
fn region(b: &[u8]) -> Option<Record> {
    let mut f = Fields::new("Region", b)?;
    if let Some(p) = u16_at(b, 5).filter(|p| *p != 0xFFFF) {
        f.set("POLYGON", p.to_string());
    }
    let n = i32_at(b, 18)? as u32 as usize;
    let text = String::from_utf8_lossy(b.get(22..22 + n)?.split(|c| *c == 0).next().unwrap_or(&[])).into_owned();
    for (k, v) in Record::parse(&text).0 {
        f.0.entry(k).or_insert(v);
    }
    let mut at = 22 + n;
    let mut rings = 0usize;
    while at + 4 <= b.len() {
        let count = i32_at(b, at)? as u32 as usize;
        at += 4;
        for k in 0..count {
            let (x, y) = (f64_at(b, at)?, f64_at(b, at + 8)?);
            at += 16;
            if rings == 0 {
                f.set(&format!("VX{k}"), format!("{}mil", x / 10_000.0)).set(&format!("VY{k}"), format!("{}mil", y / 10_000.0));
            }
        }
        rings += 1;
    }
    f.set("HOLES", rings.saturating_sub(1).to_string());
    Some(f.done())
}

/// A pad's shape code as the ASCII records name it.
fn shape_name(code: u8) -> String {
    match code {
        1 => "ROUND".into(),
        2 => "RECTANGLE".into(),
        3 => "OCTAGONAL".into(),
        other => format!("shape {other}"),
    }
}

/// A pad: its name (block 0), its main fields (block 4) and, when present, the per-layer
/// sizes and shapes (block 5), of which the import reads the top layer's rounded-rectangle
/// corner, the hole shape and the slot.
fn pad(blocks: &[&[u8]]) -> Option<Record> {
    let (name, b) = (blocks.first()?, blocks.get(4)?);
    let mut f = Fields::new("Pad", b)?;
    let layer = *b.first()?;
    // An SMD pad on the bottom copper takes the bottom layer's size and shape.
    let (sx, sy, shape) = if layer == 32 { (37, 41, *b.get(51)?) } else { (21, 25, *b.get(49)?) };
    f.lengths(b, &[("X", 13), ("Y", 17), ("XSIZE", sx), ("YSIZE", sy), ("HOLESIZE", 45)])?;
    f.angle("ROTATION", b, 52)?;
    f.set("NAME", short_string(name)).set("PLATED", if *b.get(60)? != 0 { "TRUE" } else { "FALSE" }).set("SHAPE", shape_name(shape));
    if let Some(s) = blocks.get(5).filter(|s| s.len() >= 596) {
        // 32 alternative shapes from byte 532 (9: rounded rectangle) and their corner radii,
        // in percent of half the smaller side, from byte 564; index 0 is the top layer,
        // 31 the bottom.
        let li = if layer == 32 { 31 } else { 0 };
        if s[532 + li] == 9 {
            f.set("SHAPE", "ROUNDEDRECTANGLE").set("CORNERRADIUS", s[564 + li].to_string());
        }
        // Hole shape at byte 262 (2: slot), slot length (i32) at 263 and its angle in the
        // pad's frame (f64) at 267.
        if s[262] == 2 && i32_at(s, 263)? > 0 {
            f.set("SLOTSIZE", mil(s, 263)?).angle("SLOTROTATION", s, 267)?;
        }
    }
    Some(f.done())
}

/// The primitive storages and the number of blocks in each of their records.
pub const STORAGES: &[(&str, usize)] = &[("Pads6", 6), ("Tracks6", 1), ("Arcs6", 1), ("Vias6", 1), ("Fills6", 1), ("Texts6", 2), ("Regions6", 1)];

/// Reads storage `storage`'s `Data` stream into records.
pub fn read(storage: &str, data: &[u8]) -> Result<Vec<Record>, String> {
    let blocks = STORAGES.iter().find(|(s, _)| *s == storage).map(|(_, n)| *n).ok_or_else(|| format!("{storage} isn't a primitive storage"))?;
    let mut out = Vec::new();
    for (i, (_, bl)) in split(data, blocks)?.into_iter().enumerate() {
        let main = bl.first().copied().unwrap_or(&[]);
        let r = match storage {
            "Pads6" => pad(&bl),
            "Tracks6" => track(main),
            "Arcs6" => arc(main),
            "Vias6" => via(main),
            "Fills6" => fill(main),
            "Regions6" => region(main),
            _ => text(main, bl.get(1).copied().unwrap_or(&[])),
        };
        out.push(r.ok_or_else(|| format!("{storage} record {i} is shorter than its fields"))?);
    }
    Ok(out)
}

/// Writes primitive records the way [`read`] reads them, for tests. Each item is
/// (kind byte, blocks).
pub fn write(records: &[(u8, Vec<Vec<u8>>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (kind, blocks) in records {
        out.push(*kind);
        for b in blocks {
            out.extend((b.len() as u32).to_le_bytes());
            out.extend(b);
        }
    }
    out
}

/// A primitive's first block for tests: layer, net, component, then `rest` from byte 13.
pub fn head(layer: u8, net: Option<u16>, component: Option<u16>, rest: &[u8]) -> Vec<u8> {
    let mut b = vec![layer, 0x0C, 0x00];
    b.extend(net.unwrap_or(0xFFFF).to_le_bytes());
    b.extend([0xFF, 0xFF]);
    b.extend(component.unwrap_or(0xFFFF).to_le_bytes());
    b.extend([0xFF; 4]);
    b.extend(rest);
    b
}

/// A length in 1/10000 mil, little-endian, for tests.
pub fn le_mil(mil: f64) -> [u8; 4] {
    ((mil * 10_000.0).round() as i32).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_track_and_a_via_read_into_ascii_records() {
        let mut t = head(1, Some(3), None, &[]);
        for v in [100.0, 200.0, 300.0, 200.0, 10.0] {
            t.extend(le_mil(v));
        }
        t.extend([0; 16]);
        let r = &read("Tracks6", &write(&[(4, vec![t])])).unwrap()[0];
        assert_eq!((r.kind(), r.str("LAYER"), r.index("NET"), r.index("COMPONENT")), ("Track", "TOP", Some(3), None));
        assert_eq!(r.str("X2"), "300mil");
        assert_eq!(r.str("WIDTH"), "10mil");
        let mut v = head(74, Some(0), None, &[]);
        for x in [50.0, 60.0, 24.0, 12.0] {
            v.extend(le_mil(x));
        }
        v.extend([1, 32]);
        let r = &read("Vias6", &write(&[(3, vec![v])])).unwrap()[0];
        assert_eq!((r.str("STARTLAYER"), r.str("ENDLAYER"), r.str("HOLESIZE")), ("TOP", "BOTTOM", "12mil"));
    }

    #[test]
    fn a_short_record_is_an_error() {
        let e = read("Tracks6", &write(&[(4, vec![head(1, None, None, &[1, 2])])])).unwrap_err();
        assert!(e.contains("Tracks6 record 0"), "{e}");
    }
}

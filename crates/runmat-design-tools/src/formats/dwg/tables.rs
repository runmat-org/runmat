//! Non-entity objects (specification 20.4): the table control objects and their records
//! (BLOCK_HEADER, LAYER, STYLE, LTYPE, APPID, DIMSTYLE, VPORT), DICTIONARY and LAYOUT.

use super::bits::Color;
use super::stream::ObjectOut;

pub const DICTIONARY: u16 = 0x2A;
pub const BLOCK_CONTROL: u16 = 0x30;
pub const BLOCK_HEADER: u16 = 0x31;
pub const LAYER_CONTROL: u16 = 0x32;
pub const LAYER: u16 = 0x33;
pub const STYLE_CONTROL: u16 = 0x34;
pub const STYLE: u16 = 0x35;
pub const LTYPE_CONTROL: u16 = 0x38;
pub const LTYPE: u16 = 0x39;
pub const VIEW_CONTROL: u16 = 0x3C;
pub const UCS_CONTROL: u16 = 0x3E;
pub const VPORT_CONTROL: u16 = 0x40;
pub const VPORT: u16 = 0x41;
pub const APPID_CONTROL: u16 = 0x42;
pub const APPID: u16 = 0x43;
pub const DIMSTYLE_CONTROL: u16 = 0x44;
pub const DIMSTYLE: u16 = 0x45;

/// A non-entity object: no reactors, no extension dictionary, no data-store data; its
/// parent first in the handle stream.
pub fn object(kind: u16, handle: u64, parent: u64) -> ObjectOut {
    let mut o = ObjectOut::new(kind, handle);
    o.data.bl(0);
    o.data.b(true);
    o.data.b(false);
    o.h(4, parent);
    o
}

/// A table control object: its records (soft owner) and the hard-owned extras
/// (*MODEL_SPACE and *PAPER_SPACE for blocks, BYLAYER and BYBLOCK for line types).
pub fn control(kind: u16, handle: u64, entries: &[u64], owned: &[u64]) -> Vec<u8> {
    let mut o = object(kind, handle, 0);
    o.data.bl(entries.len() as i32);
    for e in entries {
        o.h(2, *e);
    }
    for e in owned {
        o.h(3, *e);
    }
    o.finish()
}

/// A table record's common start: name, not xref-dependent; its control and a null xref
/// block in the handle stream.
pub fn record(kind: u16, handle: u64, control: u64, name: &str) -> ObjectOut {
    let mut o = object(kind, handle, control);
    o.t(name);
    o.data.b(false);
    o.data.bs(0);
    o.data.b(false);
    o
}

pub fn layer(handle: u64, control: u64, name: &str, color: i16, linetype: u64) -> Vec<u8> {
    let mut o = object(LAYER, handle, control);
    o.t(name);
    o.data.b(false);
    o.data.bs(0);
    o.data.b(false);
    // On (2), plotted (16), line weight by default (0x1F << 5).
    o.data.bs(0x02 | 0x10 | (0x1F << 5));
    o.data.cmc(&Color { index: 0, rgb: 0xC300_0000 | color as u32, name: None, book: None });
    o.h(5, 0); // xref block
    o.h(5, 0); // plot style
    o.h(5, 0); // material
    o.h(5, linetype);
    o.h(5, 0);
    o.finish()
}

pub fn style(handle: u64, control: u64, name: &str) -> Vec<u8> {
    let mut o = record(STYLE, handle, control, name);
    let d = &mut o.data;
    d.b(false);
    d.b(false);
    d.bd(0.0);
    d.bd(1.0);
    d.bd(0.0);
    d.rc(0);
    d.bd(2.5);
    o.t("txt");
    o.t("");
    o.h(5, 0);
    o.finish()
}

pub fn ltype(handle: u64, control: u64, name: &str, description: &str, pattern: &[f64]) -> Vec<u8> {
    let mut o = record(LTYPE, handle, control, name);
    o.t(description);
    o.data.bd(pattern.iter().map(|d| d.abs()).sum());
    o.data.rc(b'A');
    o.data.rc(pattern.len() as u8);
    for dash in pattern {
        let d = &mut o.data;
        d.bd(*dash);
        d.bs(0);
        d.rd(0.0);
        d.rd(0.0);
        d.bd(1.0);
        d.bd(0.0);
        d.bs(0);
    }
    o.h(5, 0);
    for _ in pattern {
        o.h(5, 0);
    }
    o.finish()
}

pub fn appid(handle: u64, control: u64, name: &str) -> Vec<u8> {
    let mut o = record(APPID, handle, control, name);
    o.data.rc(0);
    o.h(5, 0);
    o.finish()
}

/// The ISO-25 dimension style: 2.5 mm text and arrows, two decimals.
pub fn dimstyle(handle: u64, control: u64, name: &str, text_style: u64) -> Vec<u8> {
    let mut o = record(DIMSTYLE, handle, control, name);
    o.t("");
    o.t("");
    for x in [1.0, 2.5, 0.625, 3.75, 1.25, 0.0, 0.0, 0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_4] {
        o.data.bd(x);
    }
    o.data.bs(0);
    o.data.cmc(&Color { index: 0, rgb: 0xC100_0000, name: None, book: None });
    for _ in 0..6 {
        o.data.b(false);
    }
    for n in [1, 8, 0, 0] {
        o.data.bs(n);
    }
    for x in [2.5, 2.5, 0.0, 25.4, 1.0, 0.0, 1.0, 0.625, 0.0] {
        o.data.bd(x);
    }
    let d = &mut o.data;
    d.b(false);
    d.bs(2);
    d.b(true);
    d.b(false);
    d.b(false);
    d.b(false);
    for n in [0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 46, 0, 0] {
        d.bs(n);
    }
    d.b(false);
    d.b(false);
    for n in [1, 0, 0, 0] {
        d.bs(n);
    }
    d.b(false);
    d.bs(3);
    d.b(false);
    d.b(false);
    d.bd(0.0);
    o.t("");
    o.data.bd(0.0);
    o.t("");
    o.data.bs(-2);
    o.data.bs(-2);
    o.data.b(false);
    o.h(5, 0);
    o.h(5, text_style);
    for _ in 0..7 {
        o.h(5, 0);
    }
    o.finish()
}

/// `*Active`: the model view framing the sheet.
pub fn vport(handle: u64, control: u64, center: [f64; 2], size: [f64; 2]) -> Vec<u8> {
    let mut o = record(VPORT, handle, control, "*Active");
    let d = &mut o.data;
    d.bd(size[1]);
    d.bd(size[0]);
    d.rd(center[0]);
    d.rd(center[1]);
    for c in [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 50.0, 0.0, 0.0] {
        d.bd(c);
    }
    d.bits(0b0001, 4);
    d.rc(0);
    d.b(true);
    d.rc(1);
    d.bd(0.0);
    d.bd(0.0);
    d.cmc(&Color { index: 0, rgb: 0xC300_00FA, name: None, book: None });
    for c in [0.0, 0.0, 1.0, 1.0] {
        d.rd(c);
    }
    d.b(false);
    d.bs(1000);
    d.b(true);
    d.bits(0b11, 2);
    d.b(false);
    d.rd(10.0);
    d.rd(10.0);
    d.b(false);
    d.b(false);
    d.bs(0);
    d.bd(0.0);
    for c in [0.0, 0.0, 10.0, 10.0] {
        d.rd(c);
    }
    d.b(false);
    d.b(true);
    for c in [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
        d.bd(c);
    }
    d.bs(0);
    d.bs(0);
    d.bs(5);
    o.h(5, 0);
    o.h(4, 0);
    o.h(5, 0);
    o.h(3, 0);
    o.h(5, 0);
    o.h(5, 0);
    o.finish()
}

/// A dictionary of named entries (soft owner).
pub fn dictionary(handle: u64, parent: u64, entries: &[(&str, u64)]) -> Vec<u8> {
    let mut o = object(DICTIONARY, handle, parent);
    o.data.bl(entries.len() as i32);
    o.data.bs(1);
    o.data.rc(0);
    for (name, h) in entries {
        o.t(name);
        o.h(2, *h);
    }
    o.finish()
}

/// A block header: its name, entities (hard owned), the BLOCK and ENDBLK markers and its layout.
pub struct BlockHeader<'a> {
    pub handle: u64,
    pub control: u64,
    pub name: &'a str,
    pub base: [f64; 2],
    pub block: u64,
    pub endblk: u64,
    pub entities: &'a [u64],
    pub layout: u64,
}

pub fn block_header(b: &BlockHeader) -> Vec<u8> {
    let mut o = record(BLOCK_HEADER, b.handle, b.control, b.name);
    let d = &mut o.data;
    d.b(b.name.starts_with('*') && !b.name.starts_with("*Model") && !b.name.starts_with("*Paper"));
    d.b(false);
    d.b(false);
    d.b(false);
    d.b(false);
    d.bl(b.entities.len() as i32);
    d.bd(b.base[0]);
    d.bd(b.base[1]);
    d.bd(0.0);
    o.t("");
    o.data.rc(0);
    o.t("");
    o.data.bl(0);
    o.data.bs(4);
    o.data.b(true);
    o.data.rc(0);
    o.h(5, 0);
    o.h(3, b.block);
    for e in b.entities {
        o.h(3, *e);
    }
    o.h(3, b.endblk);
    o.h(5, b.layout);
    o.finish()
}

/// A layout (and its plot settings) on one sheet size, millimetres, 1:1.
pub fn layout(handle: u64, parent: u64, name: &str, order: i32, paper: [f64; 2], space: u64) -> Vec<u8> {
    let mut o = object(super::entities::LAYOUT, handle, parent);
    o.t("");
    o.t("none_device");
    o.data.bs(0);
    for x in [0.0, 0.0, 0.0, 0.0, paper[0], paper[1]] {
        o.data.bd(x);
    }
    o.t(&format!("ISO_full_bleed_({:.2}_x_{:.2}_MM)", paper[0], paper[1]));
    let d = &mut o.data;
    d.bd(0.0);
    d.bd(0.0);
    d.bs(1);
    d.bs(0);
    d.bs(5);
    for x in [0.0, 0.0, paper[0], paper[1], 1.0, 1.0] {
        d.bd(x);
    }
    o.t("");
    let d = &mut o.data;
    d.bs(16);
    d.bd(1.0);
    d.bd(0.0);
    d.bd(0.0);
    d.bs(0);
    d.bs(2);
    d.bs(300);
    o.t(name);
    let d = &mut o.data;
    d.bl(order);
    d.bs(1);
    for x in [0.0, 0.0, 0.0] {
        d.bd(x);
    }
    for x in [0.0, 0.0, paper[0], paper[1]] {
        d.rd(x);
    }
    for x in [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
        d.bd(x);
    }
    d.bs(0);
    for x in [0.0, 0.0, 0.0, paper[0], paper[1], 0.0] {
        d.bd(x);
    }
    d.rl(0);
    o.h(5, 0);
    o.h(4, 0);
    o.h(4, space);
    o.h(4, 0);
    o.h(5, 0);
    o.h(5, 0);
    o.finish()
}

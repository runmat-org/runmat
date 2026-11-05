//! The small data sections of an R2018 file: AcDb:Classes (10), AcDb:Handles (23, the object
//! map), AcDb:AuxHeader (27), AcDb:Template (22), AcDb:SummaryInfo (13), AcDb:Preview (14),
//! AcDb:AppInfo (16), AcDb:FileDepList (17), AcDb:RevHistory (18) and AcDb:ObjFreeSpace (21).

use super::bits::{mc, read_mc, umc, BitReader, BitWriter};
use super::crc::crc16;
use super::stream::{strings_into, strings_start};

type R<T> = Result<T, String>;

const CLASSES: [u8; 16] = [0x8D, 0xA1, 0xC4, 0xB8, 0xC4, 0xA9, 0xF8, 0xC5, 0xC0, 0xDC, 0xF4, 0x5F, 0xE7, 0xCF, 0xB6, 0x8A];
const CLASSES_END: [u8; 16] = [0x72, 0x5E, 0x3B, 0x47, 0x3B, 0x56, 0x07, 0x3A, 0x3F, 0x23, 0x0B, 0xA0, 0x18, 0x30, 0x49, 0x75];

/// A class: number, proxy flags, C++ and DXF names, whether it makes entities, instance count.
#[derive(Clone, Debug, PartialEq)]
pub struct Class {
    pub number: i16,
    pub proxy: i16,
    pub cpp: String,
    pub dxf: String,
    pub entity: bool,
    pub count: i32,
}

/// The classes section: sentinel, data size, data size in bits (up to the end of the string
/// stream), the bit data, CRC, end sentinel.
pub fn classes(list: &[Class]) -> Vec<u8> {
    let (mut d, mut s) = (BitWriter::new(), BitWriter::new());
    d.bs(list.iter().map(|c| c.number).max().unwrap_or(499));
    d.rc(0);
    d.rc(0);
    d.b(true);
    for c in list {
        d.bs(c.number);
        d.bs(c.proxy);
        s.tu("ObjectDBX Classes");
        s.tu(&c.cpp);
        s.tu(&c.dxf);
        d.b(false);
        d.bs(if c.entity { 0x1F2 } else { 0x1F3 });
        d.bl(c.count);
        d.bs(33);
        d.bs(0);
        d.bl(0);
        d.bl(0);
    }
    strings_into(&mut d, &s);
    let bits = d.len_bits();
    d.pad();
    let data = d.finish();
    let mut body = Vec::new();
    body.extend((data.len() as u32).to_le_bytes());
    body.extend((bits as u32).to_le_bytes());
    body.extend(&data);
    let mut out = CLASSES.to_vec();
    out.extend(&body);
    out.extend(crc16(0xC0C1, &body).to_le_bytes());
    out.extend(CLASSES_END);
    out
}

pub fn read_classes(bytes: &[u8]) -> R<Vec<Class>> {
    if bytes.get(..16) != Some(&CLASSES[..]) {
        return Err("AcDb:Classes doesn't start with its sentinel".into());
    }
    let le = |at: usize| -> R<usize> {
        let b = bytes.get(at..at + 4).ok_or("AcDb:Classes is cut short")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let (size, bits) = (le(16)?, le(20)?);
    let data = bytes.get(24..24 + size).ok_or("AcDb:Classes is shorter than its size")?;
    let crc = bytes.get(24 + size..26 + size).ok_or("AcDb:Classes has no CRC")?;
    if crc16(0xC0C1, &bytes[16..24 + size]) != u16::from_le_bytes([crc[0], crc[1]]) {
        return Err("AcDb:Classes fails its CRC".into());
    }
    let strings_at = strings_start(data, bits)?.unwrap_or(bits);
    let mut d = BitReader::new(data);
    let mut s = BitReader::at_bits(data, strings_at);
    d.bs()?;
    d.rc()?;
    d.rc()?;
    d.b()?;
    let mut out = Vec::new();
    while d.position() + 8 < strings_at {
        let number = d.bs()?;
        let proxy = d.bs()?;
        s.tu()?;
        let (cpp, dxf) = (s.tu()?, s.tu()?);
        d.b()?;
        let entity = d.bs()? == 0x1F2;
        let count = d.bl()?;
        for _ in 0..2 {
            d.bs()?;
        }
        d.bl()?;
        d.bl()?;
        out.push(Class { number, proxy, cpp, dxf, entity, count });
    }
    Ok(out)
}

/// The object map: (handle, offset in AcDb:AcDbObjects) sorted by handle, as pages of at most
/// 2032 bytes (big-endian size, the handle and offset deltas as modular chars, big-endian
/// CRC), ending with an empty page.
pub fn object_map(entries: &[(u64, u64)]) -> Vec<u8> {
    let mut sorted = entries.to_vec();
    sorted.sort();
    let mut out = Vec::new();
    let page = |out: &mut Vec<u8>, data: &[u8]| {
        let mut p = ((data.len() + 2) as u16).to_be_bytes().to_vec();
        p.extend(data);
        let crc = crc16(0xC0C1, &p);
        out.extend(p);
        out.extend(crc.to_be_bytes());
    };
    let (mut last_h, mut last_at) = (0u64, 0i64);
    let mut data = Vec::new();
    for (h, at) in sorted {
        let mut e = umc(h - last_h);
        e.extend(mc(at as i64 - last_at));
        if data.len() + e.len() > 2030 {
            page(&mut out, &data);
            data.clear();
        }
        data.extend(e);
        last_h = h;
        last_at = at as i64;
    }
    if !data.is_empty() {
        page(&mut out, &data);
    }
    page(&mut out, &[]);
    out
}

pub fn read_object_map(bytes: &[u8]) -> R<Vec<(u64, u64)>> {
    let mut out = Vec::new();
    let (mut h, mut at) = (0u64, 0i64);
    let mut k = 0;
    loop {
        let size = u16::from_be_bytes([*bytes.get(k).ok_or("the object map ends early")?, *bytes.get(k + 1).ok_or("the object map ends early")?]) as usize;
        let page = bytes.get(k..k + size + 2).ok_or("an object map page runs past the section")?;
        if crc16(0xC0C1, &page[..size]) != u16::from_be_bytes([page[size], page[size + 1]]) {
            return Err("an object map page fails its CRC".into());
        }
        if size == 2 {
            return Ok(out);
        }
        let mut i = 2;
        while i < size {
            h += read_mc(page, &mut i, false)? as u64;
            at += read_mc(page, &mut i, true)?;
            out.push((h, at as u64));
        }
        k += size + 2;
    }
}

/// AcDb:AuxHeader for R2018 (`AC1032`, version code 33).
pub fn aux_header(handseed: u64, created: (i32, i32), updated: (i32, i32)) -> Vec<u8> {
    let mut o = vec![0xFF, 0x77, 0x01];
    let rs = |o: &mut Vec<u8>, v: u16| o.extend(v.to_le_bytes());
    let rl = |o: &mut Vec<u8>, v: u32| o.extend(v.to_le_bytes());
    rs(&mut o, 33);
    rs(&mut o, 0);
    rl(&mut o, 1);
    rl(&mut o, u32::MAX);
    rs(&mut o, 1);
    rs(&mut o, 0);
    rl(&mut o, 0);
    for v in [33, 0, 33, 0, 5, 0x893, 5, 0x893, 0, 1] {
        rs(&mut o, v);
    }
    for _ in 0..5 {
        rl(&mut o, 0);
    }
    for (day, ms) in [created, updated] {
        rl(&mut o, day as u32);
        rl(&mut o, ms as u32);
    }
    rl(&mut o, if handseed < 0x7FFF_FFFF { handseed as u32 } else { u32::MAX });
    rl(&mut o, 0);
    rs(&mut o, 0);
    rs(&mut o, 1);
    for v in [0, 0, 0, 1, 0, 0, 0, 0] {
        rl(&mut o, v);
    }
    for _ in 0..3 {
        rs(&mut o, 0);
    }
    o
}

/// AcDb:Template: no description, metric MEASUREMENT.
pub fn template() -> Vec<u8> {
    vec![0, 0, 1, 0]
}

/// AcDb:SummaryInfo: empty title, subject, author, keywords, comments, last saved by,
/// revision and hyperlink base; no editing time; the dates; no properties.
pub fn summary_info(created: (i32, i32), updated: (i32, i32)) -> Vec<u8> {
    let mut o = vec![0u8; 8 * 2];
    o.extend([0u8; 8]);
    for (day, ms) in [created, updated] {
        o.extend((day as u32).to_le_bytes());
        o.extend((ms as u32).to_le_bytes());
    }
    o.extend([0u8; 2]);
    o.extend([0u8; 8]);
    o
}

/// AcDb:Preview without an image.
pub fn preview() -> Vec<u8> {
    let mut o = vec![0x1F, 0x25, 0x6D, 0x07, 0xD4, 0x36, 0x28, 0x28, 0x9D, 0x57, 0xCA, 0x3F, 0x9D, 0x44, 0x10, 0x2B];
    o.extend(1u32.to_le_bytes());
    o.push(0);
    o.extend([0xE0, 0xDA, 0x92, 0xF8, 0x2B, 0xC9, 0xD7, 0xD7, 0x62, 0xA8, 0x35, 0xC0, 0x62, 0xBB, 0xEF, 0xD4]);
    o
}

fn utf16(o: &mut Vec<u8>, s: &str) {
    let units: Vec<u16> = s.encode_utf16().collect();
    o.extend(((units.len() + 1) as u16).to_le_bytes());
    for u in units.iter().chain([&0u16]) {
        o.extend(u.to_le_bytes());
    }
}

/// AcDb:AppInfo (the R21+ layout): RunMat as the writing application.
pub fn app_info() -> Vec<u8> {
    let mut o = 2u32.to_le_bytes().to_vec();
    utf16(&mut o, "AppInfoDataList");
    o.extend(3u32.to_le_bytes());
    for s in ["0.7", "", "<ProductInformation name =\"RunMat\" build_version=\"0.7\" registry_version=\"0.7\" install_id_string=\"RunMat\" registry_localeID=\"1033\"/>"] {
        o.extend([0u8; 16]);
        utf16(&mut o, s);
    }
    utf16(&mut o, "0.7");
    o
}

/// AcDb:FileDepList: no features, no files.
pub fn file_dep_list() -> Vec<u8> {
    vec![0; 8]
}

/// AcDb:RevHistory as R21+ writers write it.
pub fn rev_history() -> Vec<u8> {
    [0u32, 0, 1, 0].iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// AcDb:ObjFreeSpace: the handle count, the update date and the objects section's offset.
pub fn obj_free_space(handles: u32, updated: (i32, i32), objects_offset: u32) -> Vec<u8> {
    let mut o = 0u32.to_le_bytes().to_vec();
    o.extend(handles.to_le_bytes());
    o.extend((updated.0 as u32).to_le_bytes());
    o.extend((updated.1 as u32).to_le_bytes());
    o.extend(objects_offset.to_le_bytes());
    o.push(4);
    for v in [0x32u32, 0, 0x64, 0, 0x200, 0, 0xFFFF_FFFF, 0] {
        o.extend(v.to_le_bytes());
    }
    o
}

//! The R2004+ file container (specification chapter 4), as R2010, R2013 and R2018 use it: a
//! 0x100-byte file header whose header data is XOR-masked and CRC-checked, data sections cut
//! into masked, checksummed pages, and the two system pages (the section map and the page
//! map) that index them.

use super::crc::{crc32, page_checksum};
use super::lz77;

/// The masking sequence of 4.1 (the file header data and page padding).
pub fn magic(n: usize) -> Vec<u8> {
    let mut seed: u32 = 1;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(0x343FD).wrapping_add(0x269EC3);
            (seed >> 16) as u8
        })
        .collect()
}

/// One data section: its name (`AcDb:Header`, `AcDb:AcDbObjects`, …), its logical bytes, its
/// page size, and whether its pages are compressed.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub name: String,
    pub data: Vec<u8>,
    pub page_size: usize,
    pub compressed: bool,
}

/// A whole file: the version string (`AC1032` for R2018) and the sections in stream order.
#[derive(Clone, Debug, PartialEq)]
pub struct Container {
    pub version: String,
    pub maintenance: u8,
    pub codepage: u16,
    pub sections: Vec<Section>,
}

const PAGE_MAP: u32 = 0x4163_0E3B;
const SECTION_MAP: u32 = 0x4163_003B;
const DATA_PAGE: u32 = 0x4163_043B;

fn le32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

fn pad_to(out: &mut Vec<u8>, align: usize) {
    let n = (align - out.len() % align) % align;
    out.extend_from_slice(&magic(n));
}

/// A system page: header, compressed data, padding to 0x20.
fn system_page(kind: u32, data: &[u8]) -> Vec<u8> {
    let mut padded = data.to_vec();
    while padded.len() < 4 {
        padded.push(0);
    }
    let comp = lz77::compress(&padded);
    let mut head = Vec::with_capacity(20);
    for v in [kind, padded.len() as u32, comp.len() as u32, 2, 0] {
        head.extend(le32(v));
    }
    let seed = page_checksum(0, &head);
    head[16..20].copy_from_slice(&le32(page_checksum(seed, &comp)));
    let mut page = head;
    page.extend(comp);
    page
}

struct PageOut {
    number: u32,
    size: usize,
}

impl Container {
    /// The file's bytes.
    pub fn write(&self) -> Vec<u8> {
        let mut file = vec![0u8; 0x100];
        let mut pages: Vec<PageOut> = Vec::new();
        // Section ids: the stream's first section is 1 and so on (0 is the empty section).
        let mut descriptions = Vec::new();
        for (si, sec) in self.sections.iter().enumerate() {
            let id = si as u32 + 1;
            let mut local = Vec::new();
            let chunks: Vec<&[u8]> = if sec.data.is_empty() { Vec::new() } else { sec.data.chunks(sec.page_size).collect() };
            for (k, chunk) in chunks.iter().enumerate() {
                let mut whole = chunk.to_vec();
                whole.resize(sec.page_size, 0);
                let payload = if sec.compressed { lz77::compress(&whole) } else { whole };
                let address = file.len();
                let size = (32 + payload.len()).div_ceil(0x20) * 0x20;
                let number = pages.len() as u32 + 1;
                let data_sum = page_checksum(0, &payload);
                let mut head = Vec::with_capacity(32);
                for v in [DATA_PAGE, id, payload.len() as u32, size as u32] {
                    head.extend(le32(v));
                }
                head.extend(((k * sec.page_size) as u64).to_le_bytes());
                head.extend(le32(0));
                head.extend(le32(data_sum));
                let head_sum = page_checksum(data_sum, &head);
                head[24..28].copy_from_slice(&le32(head_sum));
                let mask = 0x4164_536B ^ address as u32;
                for w in head.chunks_mut(4) {
                    let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) ^ mask;
                    w.copy_from_slice(&le32(v));
                }
                file.extend(head);
                file.extend(&payload);
                pad_to(&mut file, 0x20);
                local.push((number, payload.len() as u32, (k * sec.page_size) as u64));
                pages.push(PageOut { number, size: file.len() - address });
            }
            descriptions.push((sec, id, local));
        }
        // The section map.
        let mut map = Vec::new();
        for v in [descriptions.len() as u32, 2, 0x7400, 0, descriptions.len() as u32] {
            map.extend(le32(v));
        }
        for (sec, id, local) in &descriptions {
            map.extend((sec.data.len() as u64).to_le_bytes());
            for v in [local.len() as u32, sec.page_size as u32, 1, if sec.compressed { 2 } else { 1 }, *id, 0] {
                map.extend(le32(v));
            }
            let mut name = sec.name.as_bytes().to_vec();
            name.resize(64, 0);
            map.extend(name);
            for (n, size, start) in local {
                map.extend(le32(*n));
                map.extend(le32(*size));
                map.extend(start.to_le_bytes());
            }
        }
        // System pages follow the data pages after a gap of one page number.
        let section_map_number = pages.len() as u32 + 2;
        let page_map_number = section_map_number + 1;
        let address = file.len();
        file.extend(system_page(SECTION_MAP, &map));
        pad_to(&mut file, 0x20);
        pages.push(PageOut { number: section_map_number, size: file.len() - address });
        // The page map lists every page, its own included; its size is known before it is
        // written only up to padding, so its entry is computed from the written length.
        let page_map_address = file.len();
        let entries = |own: usize| {
            let mut m = Vec::new();
            for p in &pages {
                m.extend(le32(p.number));
                m.extend(le32(p.size as u32));
            }
            m.extend(le32(page_map_number));
            m.extend(le32(own as u32));
            m
        };
        let mut own = system_page(PAGE_MAP, &entries(0)).len().div_ceil(0x20) * 0x20;
        let mut page = system_page(PAGE_MAP, &entries(own));
        if page.len().div_ceil(0x20) * 0x20 != own {
            own = page.len().div_ceil(0x20) * 0x20;
            page = system_page(PAGE_MAP, &entries(own));
        }
        file.extend(page);
        pad_to(&mut file, 0x20);
        let end = file.len();
        let page_count = pages.len() as u32 + 1;

        // The file header data, masked at 0x80 and repeated at the end.
        let mut h = Vec::with_capacity(0x6C);
        h.extend(b"AcFssFcAJMB\0");
        for v in [0u32, 0x6C, 0x04, 0, 0, 0, 1, page_map_number] {
            h.extend(le32(v));
        }
        h.extend(((end - 0x100) as u64).to_le_bytes());
        h.extend((end as u64).to_le_bytes());
        for v in [0u32, page_count, 0x20, 0x80, 0x40, page_map_number] {
            h.extend(le32(v));
        }
        h.extend(((page_map_address - 0x100) as u64).to_le_bytes());
        for v in [section_map_number, page_count, 0, 0] {
            h.extend(le32(v));
        }
        let crc = crc32(0, &h);
        h[0x68..0x6C].copy_from_slice(&le32(crc));
        let mask = magic(0x6C);
        let masked: Vec<u8> = h.iter().zip(&mask).map(|(a, b)| a ^ b).collect();
        file.extend(&masked);

        let v = self.version.as_bytes();
        file[..6].copy_from_slice(&v[..6]);
        file[0x0B] = self.maintenance;
        file[0x0C] = 3;
        file[0x11] = 33; // application version byte, as R2018 writers set it
        file[0x13..0x15].copy_from_slice(&self.codepage.to_le_bytes());
        file[0x28..0x2C].copy_from_slice(&le32(0x80));
        file[0x80..0xEC].copy_from_slice(&masked);
        file[0xEC..0x100].copy_from_slice(&magic(0x100)[0xEC..0x100]);
        file
    }

    /// Read a file written in the R2004+ container.
    pub fn read(bytes: &[u8]) -> Result<Container, String> {
        if bytes.len() < 0x100 {
            return Err("the file is shorter than a DWG header".into());
        }
        let version = String::from_utf8_lossy(&bytes[..6]).into_owned();
        if !["AC1018", "AC1024", "AC1027", "AC1032"].contains(&version.as_str()) {
            return Err(format!("{version} is not an R2004+ DWG version"));
        }
        let h: Vec<u8> = bytes[0x80..0xEC].iter().zip(magic(0x6C)).map(|(a, b)| a ^ b).collect();
        if &h[..11] != b"AcFssFcAJMB" {
            return Err("the file header data does not decode".into());
        }
        let u32_at = |b: &[u8], at: usize| -> Result<u32, String> {
            b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| "the data is truncated".to_owned())
        };
        let u64_at = |b: &[u8], at: usize| -> Result<u64, String> {
            b.get(at..at + 8).map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes"))).ok_or_else(|| "the data is truncated".to_owned())
        };
        let mut zeroed = h.clone();
        zeroed[0x68..0x6C].fill(0);
        if crc32(0, &zeroed) != u32_at(&h, 0x68)? {
            return Err("the file header data fails its CRC".into());
        }
        let page_map_at = u64_at(&h, 0x54)? as usize + 0x100;
        let section_map_id = u32_at(&h, 0x5C)?;
        let system = |at: usize, kind: u32| -> Result<Vec<u8>, String> {
            let head = bytes.get(at..at + 20).ok_or("a system page is past the end of the file")?;
            if u32_at(head, 0)? != kind {
                return Err(format!("the page at {at:#x} is not the expected system page"));
            }
            let (size, csize) = (u32_at(head, 4)? as usize, u32_at(head, 8)? as usize);
            let comp = bytes.get(at + 20..at + 20 + csize).ok_or("a system page is truncated")?;
            let mut zero = head.to_vec();
            zero[16..20].fill(0);
            if page_checksum(page_checksum(0, &zero), comp) != u32_at(head, 16)? {
                return Err(format!("the system page at {at:#x} fails its checksum"));
            }
            lz77::decompress(comp, size)
        };
        let page_map = system(page_map_at, PAGE_MAP)?;
        let mut address = 0x100usize;
        let mut addresses = std::collections::BTreeMap::new();
        let mut at = 0;
        while at + 8 <= page_map.len() {
            let number = u32_at(&page_map, at)? as i32;
            let size = u32_at(&page_map, at + 4)? as usize;
            at += 8;
            if number < 0 {
                at += 16;
            } else {
                addresses.insert(number as u32, address);
            }
            address += size;
        }
        let map_at = *addresses.get(&section_map_id).ok_or("the page map does not place the section map")?;
        let map = system(map_at, SECTION_MAP)?;
        let count = u32_at(&map, 0)? as usize;
        let mut at = 20;
        let mut sections = Vec::with_capacity(count);
        for _ in 0..count {
            let size = u64_at(&map, at)? as usize;
            let pages = u32_at(&map, at + 8)? as usize;
            let page_size = u32_at(&map, at + 12)? as usize;
            let compressed = u32_at(&map, at + 20)? == 2;
            let name_bytes = map.get(at + 32..at + 96).ok_or("the section map is truncated")?;
            let name = String::from_utf8_lossy(name_bytes).trim_end_matches('\0').to_owned();
            at += 96;
            let mut data = vec![0u8; size];
            for _ in 0..pages {
                let number = u32_at(&map, at)?;
                let start = u64_at(&map, at + 8)? as usize;
                at += 16;
                let pa = *addresses.get(&number).ok_or_else(|| format!("{name}: page {number} is not in the page map"))?;
                let mask = 0x4164_536B ^ pa as u32;
                let raw = bytes.get(pa..pa + 32).ok_or("a data page is past the end of the file")?;
                let mut head: Vec<u8> = Vec::with_capacity(32);
                for w in raw.chunks(4) {
                    head.extend(le32(u32::from_le_bytes([w[0], w[1], w[2], w[3]]) ^ mask));
                }
                if u32_at(&head, 0)? != DATA_PAGE {
                    return Err(format!("{name}: the page at {pa:#x} is not a data page"));
                }
                let dsize = u32_at(&head, 8)? as usize;
                let payload = bytes.get(pa + 32..pa + 32 + dsize).ok_or("a data page is truncated")?;
                let data_sum = page_checksum(0, payload);
                let mut zero = head.clone();
                zero[24..28].fill(0);
                if data_sum != u32_at(&head, 28)? || page_checksum(data_sum, &zero) != u32_at(&head, 24)? {
                    return Err(format!("{name}: the page at {pa:#x} fails its checksum"));
                }
                let whole = if compressed { lz77::decompress(payload, page_size)? } else { payload.to_vec() };
                let n = page_size.min(size.saturating_sub(start)).min(whole.len());
                data.get_mut(start..start + n).ok_or("a page starts past its section")?.copy_from_slice(&whole[..n]);
            }
            sections.push(Section { name, data, page_size, compressed });
        }
        Ok(Container { version, maintenance: bytes[0x0B], codepage: u16::from_le_bytes([bytes[0x13], bytes[0x14]]), sections })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_magic_sequence_matches_the_specification() {
        // The 92 bytes 4.1 lists.
        let listed = "29 23 BE 84 E1 6C D6 AE 52 90 49 F1 F1 BB E9 EB B3 A6 DB 3C 87 0C 3E 99 24 5E 0D 1C 06 B7 47 DE \
                      B3 12 4D C8 43 BB 8B A6 1F 03 5A 7D 09 38 25 1F 5D D4 CB FC 96 F5 45 3B 13 0D 89 0A 1C DB AE 32 \
                      20 9A 50 EE 40 78 36 FD 12 49 32 F6 9E 7D 49 DC AD 4F 14 F2 44 40 66 D0 6B C4 30 B7";
        let want: Vec<u8> = listed.split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap()).collect();
        assert_eq!(want.len(), 92);
        assert_eq!(magic(92), want);
    }

    #[test]
    fn sections_round_trip_through_pages_and_maps() {
        let big: Vec<u8> = (0..0x7400 * 2 + 77).map(|i| (i % 251) as u8).collect();
        let c = Container {
            version: "AC1032".into(),
            maintenance: 0,
            codepage: 30,
            sections: vec![
                Section { name: "AcDb:SummaryInfo".into(), data: vec![1, 2, 3, 4, 5], page_size: 0x100, compressed: false },
                Section { name: "AcDb:AcDbObjects".into(), data: big.clone(), page_size: 0x7400, compressed: true },
                Section { name: "AcDb:Header".into(), data: b"header variables".to_vec(), page_size: 0x7400, compressed: true },
            ],
        };
        let bytes = c.write();
        assert_eq!(&bytes[..6], b"AC1032");
        assert_eq!(bytes.len() % 0x20, 0x6C % 0x20, "pages are 0x20-aligned; the header copy ends the file");
        let back = Container::read(&bytes).expect("reads");
        assert_eq!(back, c);
        let mut bad = bytes.clone();
        bad[0x100 + 40] ^= 0xFF; // inside the first data page's payload
        assert!(Container::read(&bad).unwrap_err().contains("checksum"));
    }
}

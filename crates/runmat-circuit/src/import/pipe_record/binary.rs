//! Pipe-record binary files: a compound-file container (the public structured-storage
//! format: 512-byte header, sector allocation table, directory of storages and streams) whose
//! streams hold length-prefixed records.
//!
//! - Each record is a 4-byte little-endian header, its low 24 bits the payload length and its
//!   high byte the payload kind (0: `|KEY=value|` text, ending in a NUL; others: a binary
//!   structure), then the payload.
//! - A schematic keeps its records in the `FileHeader` stream, the header record first.
//! - A board keeps each record kind in its own storage's `Data` stream. The text-record
//!   streams read here are `Board6` (the outline and settings), `Nets6`, `Classes6`, `Rules6`,
//!   `Components6` and `Polygons6`; a stream's records get their `RECORD` from the storage.
//! - The copper primitives (pads, tracks, arcs, vias, fills, texts) are binary structures,
//!   which [`super::primitives`] reads into the same records, with the regions. 3D bodies
//!   are reported as not carried.

use super::{import_records, Record, FORMAT_BINARY};
use crate::import::Imported;
use std::collections::BTreeMap;

const SIGNATURE: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
const END_OF_CHAIN: u32 = 0xFFFF_FFFE;
const FREE: u32 = 0xFFFF_FFFF;

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(i..i + 2)?.try_into().ok()?))
}
fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

/// The streams of a compound file, by path (`Board6/Data`), and its storages.
pub struct Compound {
    pub streams: BTreeMap<String, Vec<u8>>,
}

struct Entry {
    name: String,
    kind: u8,
    left: u32,
    right: u32,
    child: u32,
    start: u32,
    size: u64,
}

impl Compound {
    /// Reads a compound file.
    pub fn read(bytes: &[u8]) -> Result<Compound, String> {
        let bad = || "the compound file is cut short or damaged".to_owned();
        if bytes.get(..8) != Some(&SIGNATURE[..]) {
            return Err("not a compound file (no structured-storage signature)".into());
        }
        let shift = u16_at(bytes, 0x1E).ok_or_else(bad)?;
        let mini_shift = u16_at(bytes, 0x20).ok_or_else(bad)?;
        if !(7..=16).contains(&shift) || mini_shift > shift {
            return Err("the compound file's sector size is out of range".into());
        }
        let size = 1usize << shift;
        let mini = 1usize << mini_shift;
        let fat_count = u32_at(bytes, 0x2C).ok_or_else(bad)? as usize;
        let dir_start = u32_at(bytes, 0x30).ok_or_else(bad)?;
        let cutoff = u32_at(bytes, 0x38).ok_or_else(bad)? as u64;
        let minifat_start = u32_at(bytes, 0x3C).ok_or_else(bad)?;
        let mut difat_next = u32_at(bytes, 0x44).ok_or_else(bad)?;
        let sector = |n: u32| -> Option<&[u8]> {
            let at = (n as usize + 1).checked_mul(size)?;
            bytes.get(at..at + size)
        };
        // The allocation table's sectors: 109 in the header, the rest in DIFAT sectors.
        let mut fat_sectors: Vec<u32> = (0..109)
            .filter_map(|i| u32_at(bytes, 0x4C + 4 * i))
            .filter(|s| *s != FREE)
            .collect();
        let mut guard = 0;
        while difat_next != END_OF_CHAIN && difat_next != FREE && guard < 1 << 16 {
            let s = sector(difat_next).ok_or_else(bad)?;
            let n = size / 4 - 1;
            fat_sectors.extend(
                (0..n)
                    .filter_map(|i| u32_at(s, 4 * i))
                    .filter(|v| *v != FREE),
            );
            difat_next = u32_at(s, 4 * n).ok_or_else(bad)?;
            guard += 1;
        }
        if fat_count > 0 {
            fat_sectors.truncate(fat_count);
        }
        let mut fat: Vec<u32> = Vec::new();
        for s in &fat_sectors {
            let d = sector(*s).ok_or_else(bad)?;
            fat.extend((0..size / 4).filter_map(|i| u32_at(d, 4 * i)));
        }
        let chain = |start: u32, table: &[u32]| -> Vec<u32> {
            let mut out = Vec::new();
            let mut s = start;
            while s != END_OF_CHAIN
                && s != FREE
                && (s as usize) < table.len()
                && out.len() <= table.len()
            {
                out.push(s);
                s = table[s as usize];
            }
            out
        };
        let read_chain = |start: u32| -> Vec<u8> {
            chain(start, &fat)
                .into_iter()
                .filter_map(sector)
                .flatten()
                .copied()
                .collect()
        };
        let dir = read_chain(dir_start);
        let entries: Vec<Entry> = dir
            .chunks_exact(128)
            .map(|e| {
                let len = (u16_at(e, 64).unwrap_or(0) as usize).min(64);
                let units: Vec<u16> = (0..len.saturating_sub(2) / 2)
                    .filter_map(|i| u16_at(e, 2 * i))
                    .collect();
                Entry {
                    name: String::from_utf16_lossy(&units),
                    kind: e[66],
                    left: u32_at(e, 68).unwrap_or(FREE),
                    right: u32_at(e, 72).unwrap_or(FREE),
                    child: u32_at(e, 76).unwrap_or(FREE),
                    start: u32_at(e, 116).unwrap_or(END_OF_CHAIN),
                    size: u32_at(e, 120).unwrap_or(0) as u64
                        | if shift > 9 {
                            (u32_at(e, 124).unwrap_or(0) as u64) << 32
                        } else {
                            0
                        },
                }
            })
            .collect();
        let root = entries
            .first()
            .filter(|e| e.kind == 5)
            .ok_or("the compound file has no root entry")?;
        let mini_stream = read_chain(root.start);
        let minifat: Vec<u32> = read_chain(minifat_start)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut streams = BTreeMap::new();
        // Walk the directory tree: each storage's children are a binary tree of siblings.
        let mut stack: Vec<(u32, String)> = vec![(root.child, String::new())];
        let mut seen = vec![false; entries.len()];
        while let Some((i, prefix)) = stack.pop() {
            let Some(e) = entries.get(i as usize) else {
                continue;
            };
            if seen[i as usize] {
                continue;
            }
            seen[i as usize] = true;
            stack.push((e.left, prefix.clone()));
            stack.push((e.right, prefix.clone()));
            let path = format!("{prefix}{}", e.name);
            match e.kind {
                1 => stack.push((e.child, format!("{path}/"))),
                2 => {
                    let n = e.size as usize;
                    let mut data: Vec<u8> = if e.size < cutoff {
                        chain(e.start, &minifat)
                            .into_iter()
                            .filter_map(|s| {
                                mini_stream.get(s as usize * mini..s as usize * mini + mini)
                            })
                            .flatten()
                            .copied()
                            .collect()
                    } else {
                        read_chain(e.start)
                    };
                    data.truncate(n);
                    streams.insert(path, data);
                }
                _ => {}
            }
        }
        Ok(Compound { streams })
    }
}

/// The text records of a record stream, and how many binary records it holds.
pub fn stream_records(data: &[u8]) -> (Vec<Record>, usize) {
    let (mut out, mut binary) = (Vec::new(), 0);
    let mut i = 0;
    while let Some(h) = u32_at(data, i) {
        let len = (h & 0x00FF_FFFF) as usize;
        let kind = h >> 24;
        let Some(payload) = data.get(i + 4..i + 4 + len) else {
            break;
        };
        if kind == 0 {
            let text = String::from_utf8_lossy(payload.split(|b| *b == 0).next().unwrap_or(&[]));
            out.push(Record::parse(&text));
        } else {
            binary += 1;
        }
        i += 4 + len;
    }
    (out, binary)
}

/// The board storages whose streams hold text records, with the record kind they hold.
const BOARD_TEXT: &[(&str, &str)] = &[
    ("Board6", "Board"),
    ("Nets6", "Net"),
    ("Classes6", "Class"),
    ("Rules6", "Rule"),
    ("Components6", "Component"),
    ("Polygons6", "Polygon"),
];
/// The board storages of binary structures the import doesn't read, as the report names them.
const BOARD_BINARY: &[(&str, &str)] = &[("ComponentBodies6", "3D bodies of components")];

/// A binary file's records: a schematic's `FileHeader` stream, or a board's text streams.
/// Returns the records and what the binary primitives left out (by kind, count).
pub fn file_records(bytes: &[u8]) -> Result<(Vec<Record>, Vec<(String, usize)>), String> {
    let c = Compound::read(bytes)?;
    // A board has a `FileHeader` stream too (its version text), so the board storages decide.
    if let Some(data) = c
        .streams
        .get("FileHeader")
        .filter(|_| !c.streams.contains_key("Board6/Data"))
    {
        let (records, binary) = stream_records(data);
        let skipped = if binary > 0 {
            vec![(
                "binary-encoded schematic records (pins written in binary form)".to_owned(),
                binary,
            )]
        } else {
            vec![]
        };
        return Ok((records, skipped));
    }
    let mut records = Vec::new();
    for (storage, kind) in BOARD_TEXT {
        let Some(data) = c.streams.get(&format!("{storage}/Data")) else {
            continue;
        };
        for mut r in stream_records(data).0 {
            r.0.insert("RECORD".into(), (*kind).into());
            records.push(r);
        }
    }
    if records.is_empty() {
        return Err(
            "the compound file has no FileHeader stream and no board record streams".into(),
        );
    }
    for (storage, _) in super::primitives::STORAGES {
        if let Some(data) = c.streams.get(&format!("{storage}/Data")) {
            records.extend(super::primitives::read(storage, data)?);
        }
    }
    let mut skipped = Vec::new();
    for (storage, what) in BOARD_BINARY {
        if let Some(data) = c
            .streams
            .get(&format!("{storage}/Data"))
            .filter(|d| !d.is_empty())
        {
            skipped.push(((*what).to_owned(), data.len()));
        }
    }
    Ok((records, skipped))
}

/// Imports binary pipe-record documents (file name → bytes).
pub fn import(files: &BTreeMap<String, Vec<u8>>, target_stem: &str) -> Result<Imported, String> {
    let mut given = BTreeMap::new();
    let mut skipped = Vec::new();
    for (name, bytes) in files {
        let (records, s) = file_records(bytes).map_err(|e| format!("{name}: {e}"))?;
        given.insert(name.clone(), records);
        skipped.extend(s);
    }
    let mut out = import_records(&given, target_stem, FORMAT_BINARY)?;
    for (what, _) in skipped {
        out.report
            .not_carried(format!("{what} (binary structures)"));
    }
    Ok(out)
}

/// Writes a minimal compound file holding `streams` (path → bytes, one storage level), for
/// tests: version 3, 512-byte sectors, every stream in regular sectors (cutoff 0).
pub fn write_compound(streams: &[(&str, &[u8])]) -> Vec<u8> {
    const S: usize = 512;
    // Directory: root, storages, streams.
    struct Dir {
        name: String,
        kind: u8,
        child: u32,
        right: u32,
        start: u32,
        size: u32,
    }
    let mut dirs = vec![Dir {
        name: "Root Entry".into(),
        kind: 5,
        child: FREE,
        right: FREE,
        start: END_OF_CHAIN,
        size: 0,
    }];
    let mut data_sectors: Vec<Vec<u8>> = Vec::new();
    let mut fat: Vec<u32> = Vec::new();
    // Sectors are laid out: streams' data first, then the directory, then the FAT.
    let place = |bytes: &[u8], fat: &mut Vec<u32>, data_sectors: &mut Vec<Vec<u8>>| -> u32 {
        if bytes.is_empty() {
            return END_OF_CHAIN;
        }
        let start = data_sectors.len() as u32;
        for (k, chunk) in bytes.chunks(S).enumerate() {
            let mut c = chunk.to_vec();
            c.resize(S, 0);
            data_sectors.push(c);
            let last = (k + 1) * S >= bytes.len();
            fat.push(if last {
                END_OF_CHAIN
            } else {
                start + k as u32 + 1
            });
        }
        start
    };
    let mut top: Vec<u32> = Vec::new();
    let mut storages: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (path, bytes) in streams {
        let start = place(bytes, &mut fat, &mut data_sectors);
        let (storage, name) = path
            .split_once('/')
            .map(|(a, b)| (Some(a), b))
            .unwrap_or((None, path));
        dirs.push(Dir {
            name: name.into(),
            kind: 2,
            child: FREE,
            right: FREE,
            start,
            size: bytes.len() as u32,
        });
        let id = dirs.len() as u32 - 1;
        match storage {
            Some(s) => storages.entry(s.to_owned()).or_default().push(id),
            None => top.push(id),
        }
    }
    for (s, kids) in storages {
        dirs.push(Dir {
            name: s,
            kind: 1,
            child: FREE,
            right: FREE,
            start: 0,
            size: 0,
        });
        let id = dirs.len() as u32 - 1;
        for w in kids.windows(2) {
            dirs[w[0] as usize].right = w[1];
        }
        dirs[id as usize].child = kids[0];
        top.push(id);
    }
    for w in top.windows(2) {
        dirs[w[0] as usize].right = w[1];
    }
    dirs[0].child = top.first().copied().unwrap_or(FREE);
    let mut dir_bytes = Vec::new();
    for d in &dirs {
        let mut e = vec![0u8; 128];
        let units: Vec<u16> = d.name.encode_utf16().collect();
        for (i, u) in units.iter().enumerate().take(31) {
            e[2 * i..2 * i + 2].copy_from_slice(&u.to_le_bytes());
        }
        e[64..66].copy_from_slice(&((units.len().min(31) as u16 + 1) * 2).to_le_bytes());
        e[66] = d.kind;
        e[68..72].copy_from_slice(&FREE.to_le_bytes());
        e[72..76].copy_from_slice(&d.right.to_le_bytes());
        e[76..80].copy_from_slice(&d.child.to_le_bytes());
        e[116..120].copy_from_slice(&d.start.to_le_bytes());
        e[120..124].copy_from_slice(&d.size.to_le_bytes());
        dir_bytes.extend(e);
    }
    let dir_start = place(&dir_bytes, &mut fat, &mut data_sectors);
    // The FAT itself: one sector holds 128 entries, which these test files stay within.
    let fat_sector = data_sectors.len() as u32;
    fat.push(0xFFFF_FFFD);
    let mut fat_bytes: Vec<u8> = fat.iter().flat_map(|v| v.to_le_bytes()).collect();
    fat_bytes.resize(S, 0xFF);
    data_sectors.push(fat_bytes);
    let mut header = vec![0u8; S];
    header[..8].copy_from_slice(&SIGNATURE);
    header[0x18..0x1A].copy_from_slice(&0x3Eu16.to_le_bytes());
    header[0x1A..0x1C].copy_from_slice(&3u16.to_le_bytes());
    header[0x1C..0x1E].copy_from_slice(&0xFFFEu16.to_le_bytes());
    header[0x1E..0x20].copy_from_slice(&9u16.to_le_bytes());
    header[0x20..0x22].copy_from_slice(&6u16.to_le_bytes());
    header[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
    header[0x30..0x34].copy_from_slice(&dir_start.to_le_bytes());
    header[0x38..0x3C].copy_from_slice(&0u32.to_le_bytes());
    header[0x3C..0x40].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
    header[0x44..0x48].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
    for i in 0..109 {
        header[0x4C + 4 * i..0x50 + 4 * i].copy_from_slice(&FREE.to_le_bytes());
    }
    header[0x4C..0x50].copy_from_slice(&fat_sector.to_le_bytes());
    let mut out = header;
    for s in data_sectors {
        out.extend(s);
    }
    out
}

/// A record stream holding `records` as text records, for tests.
pub fn write_records(records: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        let mut payload = r.as_bytes().to_vec();
        payload.push(0);
        out.extend((payload.len() as u32).to_le_bytes());
        out.extend(payload);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compound_file_gives_back_its_streams_and_records() {
        let header = write_records(&[
            "|HEADER=Schematic binary file|",
            "|RECORD=1|LIBREFERENCE=Res|",
        ]);
        let nets = write_records(&["|NAME=GND|", "|NAME=VCC|"]);
        let big = vec![7u8; 1500];
        let file = write_compound(&[
            ("FileHeader", &header),
            ("Nets6/Data", &nets),
            ("Pads6/Data", &big),
        ]);
        let c = Compound::read(&file).unwrap();
        assert_eq!(c.streams["FileHeader"], header);
        assert_eq!(c.streams["Nets6/Data"], nets);
        assert_eq!(c.streams["Pads6/Data"], big);
        let (recs, binary) = stream_records(&header);
        assert_eq!((recs.len(), binary), (2, 0));
        assert_eq!(recs[1].str("LIBREFERENCE"), "Res");
        assert!(Compound::read(b"not a compound file").is_err());
    }
}

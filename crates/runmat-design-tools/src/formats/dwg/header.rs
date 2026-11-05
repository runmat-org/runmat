//! AcDb:Header (specification chapter 9): the header variables of an R2018 drawing, in file
//! order, with the R2007+ split into data, string and handle streams.
//!
//! One table lists every variable R2018 stores, its bit code and its value; the writer writes
//! it and the reader walks the same table, so the two can't disagree about the order. The
//! values are AutoCAD's metric defaults, apart from the handles of the drawing's database,
//! the extents, the dates, INSUNITS (millimetres) and the handle seed.

use super::bits::{BitReader, BitWriter, Color, Handle};
use super::crc::crc16;
use super::stream::{strings_into, strings_start};
use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;

pub const SENTINEL: [u8; 16] = [0xCF, 0x7B, 0x1F, 0x23, 0xFD, 0xDE, 0x38, 0xA9, 0x5F, 0x7C, 0x68, 0xB8, 0x4E, 0x6D, 0x33, 0x5F];
pub const END_SENTINEL: [u8; 16] = [0x30, 0x84, 0xE0, 0xDC, 0x02, 0x21, 0xC7, 0x56, 0xA0, 0x83, 0x97, 0x47, 0xB1, 0x92, 0xCC, 0xA0];

/// The database objects the header points at, by name (`CLAYER`, `BLOCK_CONTROL`, …).
pub type Handles = BTreeMap<&'static str, u64>;

/// What a drawing's header carries beyond the defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Header {
    pub handles: Handles,
    pub handseed: u64,
    pub extmin: [f64; 3],
    pub extmax: [f64; 3],
    pub limmax: [f64; 2],
    /// Julian day and milliseconds of creation and update.
    pub created: (i32, i32),
    pub updated: (i32, i32),
    pub insunits: i16,
}

#[derive(Clone, Debug, PartialEq)]
enum V {
    B(bool),
    Bs(i16),
    Bl(i32),
    Bll(u64),
    Bd(f64),
    Rc(u8),
    Tv(String),
    /// A handle reference in the handle stream (code, key in [`Handles`]).
    H(u8, &'static str),
    /// HANDSEED: a handle in the data stream.
    Seed(u64),
    Cmc(u32),
    P3([f64; 3]),
    P2([f64; 2]),
}
use V::*;

const BYLAYER: u32 = 0xC000_0000;
const BYBLOCK: u32 = 0xC100_0000;

fn space(h: &Header, v: &mut Vec<(&'static str, V)>, paper: bool) {
    let (ext_lo, ext_hi, lim) = if paper { ([0.0; 3], [0.0; 3], [420.0, 297.0]) } else { (h.extmin, h.extmax, h.limmax) };
    v.extend([
        ("INSBASE", P3([0.0; 3])),
        ("EXTMIN", P3(ext_lo)),
        ("EXTMAX", P3(ext_hi)),
        ("LIMMIN", P2([0.0, 0.0])),
        ("LIMMAX", P2(lim)),
        ("ELEVATION", Bd(0.0)),
        ("UCSORG", P3([0.0; 3])),
        ("UCSXDIR", P3([1.0, 0.0, 0.0])),
        ("UCSYDIR", P3([0.0, 1.0, 0.0])),
        ("UCSNAME", H(5, "NULL")),
        ("UCSORTHOREF", H(5, "NULL")),
        ("UCSORTHOVIEW", Bs(0)),
        ("UCSBASE", H(5, "NULL")),
    ]);
    for _ in 0..6 {
        v.push(("UCSORG_ORTHO", P3([0.0; 3])));
    }
}

/// The R2018 header variables in file order.
fn table(h: &Header) -> Vec<(&'static str, V)> {
    let mut v: Vec<(&'static str, V)> = vec![
        ("REQUIREDVERSIONS", Bll(0)),
        ("", Bd(412148564080.0)),
        ("", Bd(1.0)),
        ("", Bd(1.0)),
        ("", Bd(1.0)),
        ("", Tv(String::new())),
        ("", Tv(String::new())),
        ("", Tv(String::new())),
        ("", Tv(String::new())),
        ("", Bl(24)),
        ("", Bl(0)),
    ];
    for (name, on) in [
        ("DIMASO", true),
        ("DIMSHO", true),
        ("PLINEGEN", false),
        ("ORTHOMODE", false),
        ("REGENMODE", true),
        ("FILLMODE", true),
        ("QTEXTMODE", false),
        ("PSLTSCALE", true),
        ("LIMCHECK", false),
        ("", false),
        ("USRTIMER", true),
        ("SKPOLY", false),
        ("ANGDIR", false),
        ("SPLFRAME", false),
        ("MIRRTEXT", false),
        ("WORLDVIEW", true),
        ("TILEMODE", true),
        ("PLIMCHECK", false),
        ("VISRETAIN", true),
        ("DISPSILH", false),
        ("PELLIPSE", false),
    ] {
        v.push((name, B(on)));
    }
    for (name, n) in [("PROXYGRAPHICS", 1), ("TREEDEPTH", 3020), ("LUNITS", 2), ("LUPREC", 4), ("AUNITS", 0), ("AUPREC", 0), ("ATTMODE", 1), ("PDMODE", 0)] {
        v.push((name, Bs(n)));
    }
    v.extend([("", Bl(0)), ("", Bl(0)), ("", Bl(0))]);
    for (name, n) in [
        ("USERI1", 0),
        ("USERI2", 0),
        ("USERI3", 0),
        ("USERI4", 0),
        ("USERI5", 0),
        ("SPLINESEGS", 8),
        ("SURFU", 6),
        ("SURFV", 6),
        ("SURFTYPE", 6),
        ("SURFTAB1", 6),
        ("SURFTAB2", 6),
        ("SPLINETYPE", 6),
        ("SHADEDGE", 3),
        ("SHADEDIF", 70),
        ("UNITMODE", 0),
        ("MAXACTVP", 64),
        ("ISOLINES", 4),
        ("CMLJUST", 0),
        ("TEXTQLTY", 50),
    ] {
        v.push((name, Bs(n)));
    }
    for (name, x) in [
        ("LTSCALE", 1.0),
        ("TEXTSIZE", 2.5),
        ("TRACEWID", 1.0),
        ("SKETCHINC", 1.0),
        ("FILLETRAD", 0.0),
        ("THICKNESS", 0.0),
        ("ANGBASE", 0.0),
        ("PDSIZE", 0.0),
        ("PLINEWID", 0.0),
        ("USERR1", 0.0),
        ("USERR2", 0.0),
        ("USERR3", 0.0),
        ("USERR4", 0.0),
        ("USERR5", 0.0),
        ("CHAMFERA", 0.0),
        ("CHAMFERB", 0.0),
        ("CHAMFERC", 0.0),
        ("CHAMFERD", 0.0),
        ("FACETRES", 0.5),
        ("CMLSCALE", 1.0),
        ("CELTSCALE", 1.0),
    ] {
        v.push((name, Bd(x)));
    }
    v.extend([
        ("TDCREATE", Bl(h.created.0)),
        ("TDCREATE_MS", Bl(h.created.1)),
        ("TDUPDATE", Bl(h.updated.0)),
        ("TDUPDATE_MS", Bl(h.updated.1)),
        ("", Bl(0)),
        ("", Bl(0)),
        ("", Bl(0)),
        ("TDINDWG", Bl(0)),
        ("TDINDWG_MS", Bl(0)),
        ("TDUSRTIMER", Bl(0)),
        ("TDUSRTIMER_MS", Bl(0)),
        ("CECOLOR", Cmc(BYLAYER)),
        ("HANDSEED", Seed(h.handseed)),
        ("CLAYER", H(5, "LAYER_0")),
        ("TEXTSTYLE", H(5, "STYLE_STANDARD")),
        ("CELTYPE", H(5, "LTYPE_BYLAYER")),
        ("CMATERIAL", H(5, "NULL")),
        ("DIMSTYLE", H(5, "DIMSTYLE_STANDARD")),
        ("CMLSTYLE", H(5, "NULL")),
        ("PSVPSCALE", Bd(0.0)),
    ]);
    space(h, &mut v, true);
    space(h, &mut v, false);
    v.extend([("DIMPOST", Tv(String::new())), ("DIMAPOST", Tv(String::new()))]);
    for (name, x) in [("DIMSCALE", 1.0), ("DIMASZ", 2.5), ("DIMEXO", 0.625), ("DIMDLI", 3.75), ("DIMEXE", 1.25), ("DIMRND", 0.0), ("DIMDLE", 0.0), ("DIMTP", 0.0), ("DIMTM", 0.0), ("DIMFXL", 1.0), ("DIMJOGANG", FRAC_PI_2 / 2.0)] {
        v.push((name, Bd(x)));
    }
    v.extend([("DIMTFILL", Bs(0)), ("DIMTFILLCLR", Cmc(BYBLOCK))]);
    for name in ["DIMTOL", "DIMLIM", "DIMTIH", "DIMTOH", "DIMSE1", "DIMSE2"] {
        v.push((name, B(false)));
    }
    v.extend([("DIMTAD", Bs(1)), ("DIMZIN", Bs(8)), ("DIMAZIN", Bs(0)), ("DIMARCSYM", Bs(0))]);
    for (name, x) in [("DIMTXT", 2.5), ("DIMCEN", 2.5), ("DIMTSZ", 0.0), ("DIMALTF", 25.4), ("DIMLFAC", 1.0), ("DIMTVP", 0.0), ("DIMTFAC", 1.0), ("DIMGAP", 0.625), ("DIMALTRND", 0.0)] {
        v.push((name, Bd(x)));
    }
    v.extend([
        ("DIMALT", B(false)),
        ("DIMALTD", Bs(2)),
        ("DIMTOFL", B(true)),
        ("DIMSAH", B(false)),
        ("DIMTIX", B(false)),
        ("DIMSOXD", B(false)),
        ("DIMCLRD", Cmc(BYBLOCK)),
        ("DIMCLRE", Cmc(BYBLOCK)),
        ("DIMCLRT", Cmc(BYBLOCK)),
    ]);
    for (name, n) in [("DIMADEC", 0), ("DIMDEC", 2), ("DIMTDEC", 2), ("DIMALTU", 2), ("DIMALTTD", 2), ("DIMAUNIT", 0), ("DIMFRAC", 0), ("DIMLUNIT", 2), ("DIMDSEP", 46), ("DIMTMOVE", 0), ("DIMJUST", 0)] {
        v.push((name, Bs(n)));
    }
    v.extend([
        ("DIMSD1", B(false)),
        ("DIMSD2", B(false)),
        ("DIMTOLJ", Bs(1)),
        ("DIMTZIN", Bs(0)),
        ("DIMALTZ", Bs(0)),
        ("DIMALTTZ", Bs(0)),
        ("DIMUPT", B(false)),
        ("DIMATFIT", Bs(3)),
        ("DIMFXLON", B(false)),
        ("DIMTXTDIRECTION", B(false)),
        ("DIMALTMZF", Bd(0.0)),
        ("DIMALTMZS", Tv(String::new())),
        ("DIMMZF", Bd(0.0)),
        ("DIMMZS", Tv(String::new())),
        ("DIMTXSTY", H(5, "STYLE_STANDARD")),
        ("DIMLDRBLK", H(5, "NULL")),
        ("DIMBLK", H(5, "NULL")),
        ("DIMBLK1", H(5, "NULL")),
        ("DIMBLK2", H(5, "NULL")),
        ("DIMLTYPE", H(5, "NULL")),
        ("DIMLTEX1", H(5, "NULL")),
        ("DIMLTEX2", H(5, "NULL")),
        ("DIMLWD", Bs(-2)),
        ("DIMLWE", Bs(-2)),
    ]);
    for name in ["BLOCK_CONTROL", "LAYER_CONTROL", "STYLE_CONTROL", "LTYPE_CONTROL", "VIEW_CONTROL", "UCS_CONTROL", "VPORT_CONTROL", "APPID_CONTROL", "DIMSTYLE_CONTROL"] {
        v.push((name, H(3, name)));
    }
    v.extend([
        ("DICT_ACAD_GROUP", H(5, "DICT_ACAD_GROUP")),
        ("DICT_ACAD_MLINESTYLE", H(5, "DICT_ACAD_MLINESTYLE")),
        ("DICT_NAMED_OBJECTS", H(3, "DICT_NAMED_OBJECTS")),
        ("TSTACKALIGN", Bs(1)),
        ("TSTACKSIZE", Bs(70)),
        ("HYPERLINKBASE", Tv(String::new())),
        ("STYLESHEET", Tv(String::new())),
        ("DICT_LAYOUTS", H(5, "DICT_LAYOUTS")),
        ("DICT_PLOTSETTINGS", H(5, "DICT_PLOTSETTINGS")),
        ("DICT_PLOTSTYLES", H(5, "NULL")),
        ("DICT_MATERIALS", H(5, "NULL")),
        ("DICT_COLORS", H(5, "NULL")),
        ("DICT_VISUALSTYLE", H(5, "NULL")),
        ("", H(5, "NULL")),
        // CELWEIGHT by layer (0x1F), line weights not displayed (0x200), extended names (0x800).
        ("FLAGS", Bl(0x1F | 0x200 | 0x800)),
        ("INSUNITS", Bs(h.insunits)),
        ("CEPSNTYPE", Bs(0)),
        ("FINGERPRINTGUID", Tv("{8B5F1E3A-6C2D-4E7F-9A10-2B3C4D5E6F70}".into())),
        ("VERSIONGUID", Tv("{1A2B3C4D-5E6F-4071-8293-A4B5C6D7E8F9}".into())),
        ("SORTENTS", Rc(127)),
        ("INDEXCTL", Rc(0)),
        ("HIDETEXT", Rc(1)),
        ("XCLIPFRAME", Rc(2)),
        ("DIMASSOC", Rc(2)),
        ("HALOGAP", Rc(0)),
        ("OBSCUREDCOLOR", Bs(257)),
        ("INTERSECTIONCOLOR", Bs(257)),
        ("OBSCUREDLTYPE", Rc(0)),
        ("INTERSECTIONDISPLAY", Rc(0)),
        ("PROJECTNAME", Tv(String::new())),
        ("PAPER_SPACE", H(5, "PAPER_SPACE")),
        ("MODEL_SPACE", H(5, "MODEL_SPACE")),
        ("LTYPE_BYLAYER", H(5, "LTYPE_BYLAYER")),
        ("LTYPE_BYBLOCK", H(5, "LTYPE_BYBLOCK")),
        ("LTYPE_CONTINUOUS", H(5, "LTYPE_CONTINUOUS")),
        ("CAMERADISPLAY", B(false)),
        ("", Bl(0)),
        ("", Bl(0)),
        ("", Bd(0.0)),
        ("STEPSPERSEC", Bd(2.0)),
        ("STEPSIZE", Bd(6.0)),
        ("3DDWFPREC", Bd(2.0)),
        ("LENSLENGTH", Bd(50.0)),
        ("CAMERAHEIGHT", Bd(0.0)),
        ("SOLIDHIST", Rc(0)),
        ("SHOWHIST", Rc(1)),
        ("PSOLWIDTH", Bd(0.25)),
        ("PSOLHEIGHT", Bd(4.0)),
        ("LOFTANG1", Bd(FRAC_PI_2)),
        ("LOFTANG2", Bd(FRAC_PI_2)),
        ("LOFTMAG1", Bd(0.0)),
        ("LOFTMAG2", Bd(0.0)),
        ("LOFTPARAM", Bs(7)),
        ("LOFTNORMALS", Rc(1)),
        ("LATITUDE", Bd(37.795)),
        ("LONGITUDE", Bd(-122.394)),
        ("NORTHDIRECTION", Bd(0.0)),
        ("TIMEZONE", Bl(-8000)),
        ("LIGHTGLYPHDISPLAY", Rc(1)),
        ("TILEMODELIGHTSYNCH", Rc(1)),
        ("DWFFRAME", Rc(2)),
        ("DGNFRAME", Rc(0)),
        ("", B(false)),
        ("INTERFERECOLOR", Cmc(0xC300_0001)),
        ("INTERFEREOBJVS", H(5, "NULL")),
        ("INTERFEREVPVS", H(5, "NULL")),
        ("DRAGVS", H(5, "NULL")),
        ("CSHADOW", Rc(0)),
        ("", Bd(0.0)),
    ]);
    v
}

fn color(rgb: u32) -> Color {
    Color { index: 0, rgb, name: None, book: None }
}

/// The section's bytes: sentinel, size, the R2018 unknown long, the data's size in bits (up
/// to the handle stream), the three streams, the CRC and the end sentinel.
pub fn write(h: &Header) -> Vec<u8> {
    let (mut d, mut s, mut hs) = (BitWriter::new(), BitWriter::new(), BitWriter::new());
    for (_, value) in table(h) {
        match value {
            B(x) => d.b(x),
            Bs(x) => d.bs(x),
            Bl(x) => d.bl(x),
            Bll(x) => d.bll(x),
            Bd(x) => d.bd(x),
            Rc(x) => d.rc(x),
            Tv(x) => s.tu(&x),
            H(code, key) => hs.h(Handle { code, value: h.handles.get(key).copied().unwrap_or(0) }),
            Seed(x) => d.h(Handle { code: 0, value: x }),
            Cmc(x) => d.cmc(&color(x)),
            P3(p) => p.iter().for_each(|c| d.bd(*c)),
            P2(p) => p.iter().for_each(|c| d.rd(*c)),
        }
    }
    strings_into(&mut d, &s);
    let bits = d.len_bits();
    d.append(&hs);
    d.pad();
    let data = d.finish();
    let mut body = Vec::new();
    body.extend((data.len() as u32 + 4).to_le_bytes());
    body.extend(0u32.to_le_bytes());
    body.extend((bits as u32).to_le_bytes());
    body.extend(&data);
    let crc = crc16(0xC0C1, &body);
    let mut out = SENTINEL.to_vec();
    out.extend(body);
    out.extend(crc.to_le_bytes());
    out.extend(END_SENTINEL);
    out
}

/// Read a header this writer wrote: the database handles, seed, extents, dates and units.
pub fn read(bytes: &[u8]) -> Result<Header, String> {
    if bytes.get(..16) != Some(&SENTINEL[..]) {
        return Err("AcDb:Header doesn't start with its sentinel".into());
    }
    let le = |at: usize| -> Result<u32, String> {
        let b = bytes.get(at..at + 4).ok_or("AcDb:Header is cut short")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let size = le(16)? as usize;
    let bits = le(24)? as usize;
    let data = bytes.get(28..24 + size).ok_or("AcDb:Header is shorter than its size")?;
    let crc = bytes.get(24 + size..26 + size).ok_or("AcDb:Header has no CRC")?;
    if crc16(0xC0C1, &bytes[16..24 + size]) != u16::from_le_bytes([crc[0], crc[1]]) {
        return Err("AcDb:Header fails its CRC".into());
    }
    if bytes.get(26 + size..42 + size) != Some(&END_SENTINEL[..]) {
        return Err("AcDb:Header doesn't end with its sentinel".into());
    }
    let mut d = BitReader::new(data);
    let mut s = BitReader::at_bits(data, strings_start(data, bits)?.unwrap_or(bits));
    let mut hs = BitReader::at_bits(data, bits);
    let mut out = Header::default();
    let mut points: Vec<(&str, [f64; 3])> = Vec::new();
    let mut longs: BTreeMap<&str, i32> = BTreeMap::new();
    for (name, value) in table(&Header::default()) {
        match value {
            B(_) => { let _ = d.b()?; }
            Bs(_) => {
                let x = d.bs()?;
                if name == "INSUNITS" {
                    out.insunits = x;
                }
            }
            Bl(_) => {
                longs.insert(name, d.bl()?);
            }
            Bll(_) => { let _ = d.bll()?; }
            Bd(_) => { let _ = d.bd()?; }
            Rc(_) => { let _ = d.rc()?; }
            Tv(_) => { let _ = s.tu()?; }
            H(_, key) => {
                let v = hs.h()?.value;
                if key != "NULL" {
                    out.handles.insert(key, v);
                }
            }
            Seed(_) => out.handseed = d.h()?.value,
            Cmc(_) => { let _ = d.cmc()?; }
            P3(_) => points.push((name, [d.bd()?, d.bd()?, d.bd()?])),
            P2(_) => {
                let p = [d.rd()?, d.rd()?];
                if name == "LIMMAX" {
                    out.limmax = p;
                }
            }
        }
    }
    // The model space values come second.
    out.extmin = points.iter().rev().find(|p| p.0 == "EXTMIN").map(|p| p.1).unwrap_or_default();
    out.extmax = points.iter().rev().find(|p| p.0 == "EXTMAX").map(|p| p.1).unwrap_or_default();
    let l = |k: &str| longs.get(k).copied().unwrap_or(0);
    out.created = (l("TDCREATE"), l("TDCREATE_MS"));
    out.updated = (l("TDUPDATE"), l("TDUPDATE_MS"));
    Ok(out)
}

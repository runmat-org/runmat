//! A [`Sheet`] as an R2018 DWG: the database (tables, dictionaries, layouts, model and paper
//! space, blocks) and the sheet's entities in model space at 1:1 millimetres, in the R2004+
//! container with every section an R2018 file carries.

use super::container::{Container, Section};
use super::entities::{self, Refs, BLOCK, ENDBLK, HATCH, LAYOUT, LWPOLYLINE};
use super::header::{self, Header};
use super::sections::{self, Class};
use super::tables::{self, BlockHeader};
use crate::formats::sheet::{Shape, Sheet};
use std::collections::BTreeMap;

struct Db {
    next: u64,
    objects: Vec<(u64, Vec<u8>)>,
}

impl Db {
    fn handle(&mut self) -> u64 {
        self.next += 1;
        self.next - 1
    }
}

/// Fixed handles of the database's own objects.
mod h {
    pub const BLOCK_CONTROL: u64 = 1;
    pub const LAYER_CONTROL: u64 = 2;
    pub const STYLE_CONTROL: u64 = 3;
    pub const LTYPE_CONTROL: u64 = 5;
    pub const VIEW_CONTROL: u64 = 6;
    pub const UCS_CONTROL: u64 = 7;
    pub const VPORT_CONTROL: u64 = 8;
    pub const APPID_CONTROL: u64 = 9;
    pub const DIMSTYLE_CONTROL: u64 = 0xA;
    pub const NAMED_OBJECTS: u64 = 0xC;
    pub const ACAD_GROUP: u64 = 0xD;
    pub const ACAD_MLINESTYLE: u64 = 0xE;
    pub const LAYER_0: u64 = 0x10;
    pub const STYLE_STANDARD: u64 = 0x11;
    pub const APPID_ACAD: u64 = 0x12;
    pub const LTYPE_BYBLOCK: u64 = 0x14;
    pub const LTYPE_BYLAYER: u64 = 0x15;
    pub const LTYPE_CONTINUOUS: u64 = 0x16;
    pub const DIMSTYLE_STANDARD: u64 = 0x1D;
    pub const VPORT_ACTIVE: u64 = 0x29;
    pub const ACAD_LAYOUT: u64 = 0x1A;
    pub const ACAD_PLOTSETTINGS: u64 = 0x19;
    pub const MODEL_SPACE: u64 = 0x1F;
    pub const PAPER_SPACE: u64 = 0x1B;
    pub const FIRST_FREE: u64 = 0x30;
}

/// Julian day and milliseconds of 2026-01-01 00:00 UTC; written as both creation and
/// update so the same sheet always gives the same bytes.
const DATE: (i32, i32) = (2_461_042, 0);

/// The DWG bytes of a sheet.
pub fn write(sheet: &Sheet) -> Vec<u8> {
    let mut db = Db { next: h::FIRST_FREE, objects: Vec::new() };
    // Tables.
    let mut layers: BTreeMap<String, u64> = BTreeMap::new();
    layers.insert("0".into(), h::LAYER_0);
    let mut lt: BTreeMap<String, u64> = BTreeMap::new();
    lt.insert("CONTINUOUS".into(), h::LTYPE_CONTINUOUS);
    let mut ltypes = vec![h::LTYPE_CONTINUOUS];
    for t in &sheet.linetypes {
        let hd = db.handle();
        lt.insert(t.name.clone(), hd);
        ltypes.push(hd);
        db.objects.push((hd, tables::ltype(hd, h::LTYPE_CONTROL, &t.name, &t.description, &t.pattern)));
    }
    let mut layer_list = vec![h::LAYER_0];
    for l in &sheet.layers {
        let hd = db.handle();
        layers.insert(l.name.clone(), hd);
        layer_list.push(hd);
        let ltype = lt.get(&l.linetype).copied().unwrap_or(h::LTYPE_CONTINUOUS);
        db.objects.push((hd, tables::layer(hd, h::LAYER_CONTROL, &l.name, l.color, ltype)));
    }
    let o = &mut db.objects;
    o.push((h::LAYER_0, tables::layer(h::LAYER_0, h::LAYER_CONTROL, "0", 7, h::LTYPE_CONTINUOUS)));
    o.push((h::LTYPE_BYBLOCK, tables::ltype(h::LTYPE_BYBLOCK, h::LTYPE_CONTROL, "ByBlock", "", &[])));
    o.push((h::LTYPE_BYLAYER, tables::ltype(h::LTYPE_BYLAYER, h::LTYPE_CONTROL, "ByLayer", "", &[])));
    o.push((h::LTYPE_CONTINUOUS, tables::ltype(h::LTYPE_CONTINUOUS, h::LTYPE_CONTROL, "Continuous", "Solid line", &[])));
    o.push((h::STYLE_STANDARD, tables::style(h::STYLE_STANDARD, h::STYLE_CONTROL, "Standard")));
    o.push((h::APPID_ACAD, tables::appid(h::APPID_ACAD, h::APPID_CONTROL, "ACAD")));
    o.push((h::DIMSTYLE_STANDARD, tables::dimstyle(h::DIMSTYLE_STANDARD, h::DIMSTYLE_CONTROL, "Standard", h::STYLE_STANDARD)));
    let (lo, hi) = sheet.extents();
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let size = [(hi[0] - lo[0]).max(1.0) * 1.1, (hi[1] - lo[1]).max(1.0) * 1.1];
    o.push((h::VPORT_ACTIVE, tables::vport(h::VPORT_ACTIVE, h::VPORT_CONTROL, centre, size)));
    // Blocks: user blocks first (so inserts and dimensions can name them), then the spaces.
    let mut block_handles: BTreeMap<String, u64> = BTreeMap::new();
    let mut headers: Vec<(u64, &str, [f64; 2], &[crate::formats::sheet::Entity])> = Vec::new();
    for b in &sheet.blocks {
        let hd = db.handle();
        block_handles.insert(b.name.clone(), hd);
        headers.push((hd, b.name.as_str(), b.base, &b.entities));
    }
    headers.push((h::MODEL_SPACE, "*Model_Space", [0.0, 0.0], &sheet.entities));
    headers.push((h::PAPER_SPACE, "*Paper_Space", [0.0, 0.0], &[]));
    let (model_layout, paper_layout) = (db.handle(), db.handle());
    let mut counts = (0, 0);
    for (bh, name, base, list) in &headers {
        let owner = if *bh == h::MODEL_SPACE || *bh == h::PAPER_SPACE { None } else { Some(*bh) };
        let lookup = |n: &str| block_handles.get(n).copied().unwrap_or(0);
        let marker_refs = Refs { owner: Some(*bh), layer: h::LAYER_0, style: h::STYLE_STANDARD, dimstyle: h::DIMSTYLE_STANDARD, appid: h::APPID_ACAD, block: &lookup };
        let (begin, end) = (db.handle(), db.handle());
        db.objects.push((begin, entities::block_marker(BLOCK, begin, Some(name), &marker_refs)));
        db.objects.push((end, entities::block_marker(ENDBLK, end, None, &marker_refs)));
        let mut owned = Vec::new();
        for e in list.iter() {
            let hd = db.handle();
            let refs = Refs {
                owner,
                layer: layers.get(&e.layer).copied().unwrap_or(h::LAYER_0),
                style: h::STYLE_STANDARD,
                dimstyle: h::DIMSTYLE_STANDARD,
                appid: h::APPID_ACAD,
                block: &lookup,
            };
            match e.shape {
                Shape::Polyline { .. } => counts.0 += 1,
                Shape::Hatch { .. } => counts.1 += 1,
                _ => {}
            }
            db.objects.push((hd, entities::write(hd, &e.shape, &refs)));
            owned.push(hd);
        }
        let layout = match *bh {
            h::MODEL_SPACE => model_layout,
            h::PAPER_SPACE => paper_layout,
            _ => 0,
        };
        let header = BlockHeader { handle: *bh, control: h::BLOCK_CONTROL, name, base: *base, block: begin, endblk: end, entities: &owned, layout };
        db.objects.push((*bh, tables::block_header(&header)));
    }
    let paper = if sheet.paper[0] > 0.0 { sheet.paper } else { [420.0, 297.0] };
    let user_blocks: Vec<u64> = sheet.blocks.iter().map(|b| block_handles[&b.name]).collect();
    let o = &mut db.objects;
    o.push((model_layout, tables::layout(model_layout, h::ACAD_LAYOUT, "Model", 0, paper, h::MODEL_SPACE)));
    o.push((paper_layout, tables::layout(paper_layout, h::ACAD_LAYOUT, "Layout1", 1, paper, h::PAPER_SPACE)));
    // Controls and dictionaries.
    o.push((h::BLOCK_CONTROL, tables::control(tables::BLOCK_CONTROL, h::BLOCK_CONTROL, &user_blocks, &[h::MODEL_SPACE, h::PAPER_SPACE])));
    o.push((h::LAYER_CONTROL, tables::control(tables::LAYER_CONTROL, h::LAYER_CONTROL, &layer_list, &[])));
    o.push((h::STYLE_CONTROL, tables::control(tables::STYLE_CONTROL, h::STYLE_CONTROL, &[h::STYLE_STANDARD], &[])));
    o.push((h::LTYPE_CONTROL, tables::control(tables::LTYPE_CONTROL, h::LTYPE_CONTROL, &ltypes, &[h::LTYPE_BYLAYER, h::LTYPE_BYBLOCK])));
    o.push((h::VIEW_CONTROL, tables::control(tables::VIEW_CONTROL, h::VIEW_CONTROL, &[], &[])));
    o.push((h::UCS_CONTROL, tables::control(tables::UCS_CONTROL, h::UCS_CONTROL, &[], &[])));
    o.push((h::VPORT_CONTROL, tables::control(tables::VPORT_CONTROL, h::VPORT_CONTROL, &[h::VPORT_ACTIVE], &[])));
    o.push((h::APPID_CONTROL, tables::control(tables::APPID_CONTROL, h::APPID_CONTROL, &[h::APPID_ACAD], &[])));
    o.push((h::DIMSTYLE_CONTROL, tables::control(tables::DIMSTYLE_CONTROL, h::DIMSTYLE_CONTROL, &[h::DIMSTYLE_STANDARD], &[])));
    o.push((
        h::NAMED_OBJECTS,
        tables::dictionary(
            h::NAMED_OBJECTS,
            0,
            &[("ACAD_GROUP", h::ACAD_GROUP), ("ACAD_LAYOUT", h::ACAD_LAYOUT), ("ACAD_MLINESTYLE", h::ACAD_MLINESTYLE), ("ACAD_PLOTSETTINGS", h::ACAD_PLOTSETTINGS)],
        ),
    ));
    o.push((h::ACAD_GROUP, tables::dictionary(h::ACAD_GROUP, h::NAMED_OBJECTS, &[])));
    o.push((h::ACAD_MLINESTYLE, tables::dictionary(h::ACAD_MLINESTYLE, h::NAMED_OBJECTS, &[])));
    o.push((h::ACAD_PLOTSETTINGS, tables::dictionary(h::ACAD_PLOTSETTINGS, h::NAMED_OBJECTS, &[])));
    o.push((h::ACAD_LAYOUT, tables::dictionary(h::ACAD_LAYOUT, h::NAMED_OBJECTS, &[("Layout1", paper_layout), ("Model", model_layout)])));

    // AcDb:AcDbObjects: the 0x0DCA long, then each object; the map records their offsets.
    db.objects.sort_by_key(|(hd, _)| *hd);
    let mut objects = 0x0DCAu32.to_le_bytes().to_vec();
    let mut map = Vec::new();
    for (hd, bytes) in &db.objects {
        map.push((*hd, objects.len() as u64));
        objects.extend(bytes);
    }
    let mut handles = header::Handles::new();
    for (k, v) in [
        ("LAYER_0", h::LAYER_0),
        ("STYLE_STANDARD", h::STYLE_STANDARD),
        ("LTYPE_BYLAYER", h::LTYPE_BYLAYER),
        ("LTYPE_BYBLOCK", h::LTYPE_BYBLOCK),
        ("LTYPE_CONTINUOUS", h::LTYPE_CONTINUOUS),
        ("DIMSTYLE_STANDARD", h::DIMSTYLE_STANDARD),
        ("BLOCK_CONTROL", h::BLOCK_CONTROL),
        ("LAYER_CONTROL", h::LAYER_CONTROL),
        ("STYLE_CONTROL", h::STYLE_CONTROL),
        ("LTYPE_CONTROL", h::LTYPE_CONTROL),
        ("VIEW_CONTROL", h::VIEW_CONTROL),
        ("UCS_CONTROL", h::UCS_CONTROL),
        ("VPORT_CONTROL", h::VPORT_CONTROL),
        ("APPID_CONTROL", h::APPID_CONTROL),
        ("DIMSTYLE_CONTROL", h::DIMSTYLE_CONTROL),
        ("DICT_ACAD_GROUP", h::ACAD_GROUP),
        ("DICT_ACAD_MLINESTYLE", h::ACAD_MLINESTYLE),
        ("DICT_NAMED_OBJECTS", h::NAMED_OBJECTS),
        ("DICT_LAYOUTS", h::ACAD_LAYOUT),
        ("DICT_PLOTSETTINGS", h::ACAD_PLOTSETTINGS),
        ("PAPER_SPACE", h::PAPER_SPACE),
        ("MODEL_SPACE", h::MODEL_SPACE),
    ] {
        handles.insert(k, v);
    }
    let head = Header {
        handles,
        handseed: db.next,
        extmin: [lo[0], lo[1], 0.0],
        extmax: [hi[0], hi[1], 0.0],
        limmax: paper,
        created: DATE,
        updated: DATE,
        insunits: 4,
    };
    let classes = [
        Class { number: LWPOLYLINE as i16, proxy: 0, cpp: "AcDbPolyline".into(), dxf: "LWPOLYLINE".into(), entity: true, count: counts.0 },
        Class { number: HATCH as i16, proxy: 0x480, cpp: "AcDbHatch".into(), dxf: "HATCH".into(), entity: true, count: counts.1 },
        Class { number: LAYOUT as i16, proxy: 0, cpp: "AcDbLayout".into(), dxf: "LAYOUT".into(), entity: false, count: 2 },
    ];
    let section = |name: &str, data: Vec<u8>, page_size: usize, compressed: bool| Section { name: name.into(), data, page_size, compressed };
    let count = db.objects.len() as u32;
    Container {
        version: "AC1032".into(),
        maintenance: 0,
        codepage: 30,
        sections: vec![
            section("AcDb:SummaryInfo", sections::summary_info(DATE, DATE), 0x100, false),
            section("AcDb:Preview", sections::preview(), 0x400, false),
            section("AcDb:AppInfo", sections::app_info(), 0x300, false),
            section("AcDb:FileDepList", sections::file_dep_list(), 0x80, false),
            section("AcDb:RevHistory", sections::rev_history(), 0x7400, true),
            section("AcDb:AcDbObjects", objects, 0x7400, true),
            section("AcDb:ObjFreeSpace", sections::obj_free_space(count, DATE, 0), 0x7400, true),
            section("AcDb:Template", sections::template(), 0x7400, true),
            section("AcDb:Handles", sections::object_map(&map), 0x7400, true),
            section("AcDb:Classes", sections::classes(&classes), 0x7400, true),
            section("AcDb:AuxHeader", sections::aux_header(db.next, DATE, DATE), 0x7400, true),
            section("AcDb:Header", header::write(&head), 0x7400, true),
        ],
    }
    .write()
}

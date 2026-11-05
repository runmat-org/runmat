//! Reading an R2018 DWG back into a [`Sheet`]: the container, the header's database handles,
//! the object map, then the tables and every block's entities by following the block
//! headers. Objects of kinds this crate doesn't write are skipped.

use super::container::Container;
use super::entities::{self, BLOCK, ENDBLK};
use super::header;
use super::sections;
use super::stream::{read_object, ObjectIn};
use super::tables::{BLOCK_HEADER, LAYER, LTYPE};
use crate::formats::sheet::{Block, Entity, Layer, Linetype, Shape, Sheet};
use std::collections::BTreeMap;

type R<T> = Result<T, String>;

/// What the reader checks and finds besides the sheet.
#[derive(Debug, Default)]
pub struct Report {
    pub version: String,
    pub objects: usize,
    pub classes: Vec<String>,
    pub handseed: u64,
    pub sections: Vec<String>,
}

fn section<'a>(c: &'a Container, name: &str) -> R<&'a [u8]> {
    c.sections.iter().find(|s| s.name == name).map(|s| &s.data[..]).ok_or_else(|| format!("no {name} section"))
}

fn record_name(o: &ObjectIn) -> R<String> {
    let mut d = o.data();
    d.bl()?;
    d.b()?;
    d.b()?;
    o.strings().tu()
}

/// Read a DWG this crate wrote: the sheet and what the reader checked.
pub fn read(bytes: &[u8]) -> R<(Sheet, Report)> {
    let c = Container::read(bytes)?;
    let head = header::read(section(&c, "AcDb:Header")?)?;
    let classes = sections::read_classes(section(&c, "AcDb:Classes")?)?;
    let map = sections::read_object_map(section(&c, "AcDb:Handles")?)?;
    let data = section(&c, "AcDb:AcDbObjects")?;
    let mut objects: BTreeMap<u64, ObjectIn> = BTreeMap::new();
    for (h, at) in &map {
        let o = read_object(data, *at as usize)?;
        if o.handle != *h {
            return Err(format!("the object map puts handle {h:#X} at {at}, which holds {:#X}", o.handle));
        }
        objects.insert(*h, o);
    }
    if head.handseed <= map.last().map(|e| e.0).unwrap_or(0) {
        return Err("HANDSEED isn't past every handle".into());
    }
    let mut sheet = Sheet { paper: head.limmax, ..Sheet::default() };
    // Line types by handle (the database's own ByBlock, ByLayer and Continuous stay out).
    let mut ltypes: BTreeMap<u64, String> = BTreeMap::new();
    for (h, o) in objects.iter().filter(|(_, o)| o.kind == LTYPE) {
        let name = record_name(o)?;
        ltypes.insert(*h, name.clone());
        if ["ByBlock", "ByLayer", "Continuous"].contains(&name.as_str()) {
            continue;
        }
        let mut d = o.data();
        d.bl()?;
        d.b()?;
        d.b()?;
        d.b()?;
        d.bs()?;
        d.b()?;
        let mut s = o.strings();
        s.tu()?;
        let description = s.tu()?;
        d.bd()?;
        d.rc()?;
        let n = d.rc()?;
        let mut pattern = Vec::new();
        for _ in 0..n {
            pattern.push(d.bd()?);
            d.bs()?;
            d.rd()?;
            d.rd()?;
            d.bd()?;
            d.bd()?;
            d.bs()?;
        }
        sheet.linetypes.push(Linetype { name, description, pattern });
    }
    let mut layers: BTreeMap<u64, String> = BTreeMap::new();
    for (h, o) in objects.iter().filter(|(_, o)| o.kind == LAYER) {
        let name = record_name(o)?;
        layers.insert(*h, name.clone());
        if name == "0" {
            continue;
        }
        let mut d = o.data();
        d.bl()?;
        d.b()?;
        d.b()?;
        d.b()?;
        d.bs()?;
        d.b()?;
        d.bs()?;
        let color = d.cmc()?;
        let mut hs = o.handles();
        for _ in 0..4 {
            hs.h()?;
        }
        let lt = hs.h()?.value;
        let linetype = match ltypes.get(&lt).map(String::as_str) {
            Some("Continuous") | None => "CONTINUOUS".to_owned(),
            Some(n) => n.to_owned(),
        };
        sheet.layers.push(Layer { name, color: (color.rgb & 0xFF) as i16, linetype });
    }
    // Block headers: names, bases and owned entities.
    let mut block_names: BTreeMap<u64, String> = BTreeMap::new();
    let mut headers = Vec::new();
    for (h, o) in objects.iter().filter(|(_, o)| o.kind == BLOCK_HEADER) {
        let name = record_name(o)?;
        let mut d = o.data();
        d.bl()?;
        d.b()?;
        d.b()?;
        d.b()?;
        d.bs()?;
        d.b()?;
        for _ in 0..5 {
            d.b()?;
        }
        let owned = d.bl()?.max(0) as usize;
        let base = [d.bd()?, d.bd()?];
        let mut hs = o.handles();
        hs.h()?;
        hs.h()?;
        hs.h()?;
        let entities: Vec<u64> = (0..owned).map(|_| hs.h().map(|h| h.value)).collect::<R<_>>()?;
        block_names.insert(*h, name.clone());
        headers.push((name, base, entities));
    }
    for (name, base, list) in headers {
        let mut out = Vec::new();
        for h in list {
            let o = objects.get(&h).ok_or_else(|| format!("block {name} owns {h:#X}, which isn't in the file"))?;
            if o.kind == BLOCK || o.kind == ENDBLK {
                continue;
            }
            let e = entities::read(o)?;
            let Some(mut shape) = e.shape else { continue };
            let named = e.block.and_then(|b| block_names.get(&b).cloned()).unwrap_or_default();
            match &mut shape {
                Shape::Insert { block, .. } => *block = named,
                Shape::Dimension(d) => d.block = named,
                _ => {}
            }
            out.push(Entity { layer: layers.get(&e.layer).cloned().unwrap_or_else(|| "0".into()), shape });
        }
        match name.as_str() {
            "*Model_Space" => sheet.entities = out,
            "*Paper_Space" => {}
            _ => sheet.blocks.push(Block { name, base, entities: out }),
        }
    }
    let report = Report {
        version: c.version.clone(),
        objects: objects.len(),
        classes: classes.iter().map(|k| k.dxf.clone()).collect(),
        handseed: head.handseed,
        sections: c.sections.iter().map(|s| s.name.clone()).collect(),
    };
    Ok((sheet, report))
}

//! EAGLE XML schematics (`.sch`), from the format's public DTD:
//!
//! - each placed part with a package becomes a component, its device a `part` declaration
//!   (one per part number) and its package a footprint file;
//! - parts without a package (supply symbols, frames) aren't components; a supply symbol's
//!   pin makes its net global;
//! - nets join by name across sheets; each `pinref` becomes the part's pin;
//! - net classes carry their track width and clearance, assembly variants their unfitted parts.

use super::library::{device_part, package, param_key};
use crate::import::{file_safe, xml::Element, Report};
use crate::ir::{Connection, ComponentDecl, Design, Endpoint, Fitting, NetClass, NetDecl, Param, PartRef, Tolerance, VariantDef};
use crate::library::Package;
use std::collections::{BTreeMap, BTreeSet};

/// What a schematic import makes.
pub struct Schematic {
    pub design: Design,
    /// Footprint key → package (written as `<key>.kicad_mod`).
    pub packages: BTreeMap<String, Package>,
    /// Designator → footprint key, for the board.
    pub footprints: BTreeMap<String, String>,
}

/// A library's key: its name and, for files that carry one, its URN.
fn library_key(name: &str, urn: Option<&str>) -> String {
    format!("{name}|{}", urn.unwrap_or(""))
}

/// The libraries a schematic or board embeds, by [`library_key`].
pub fn libraries(drawing: &Element) -> BTreeMap<String, &Element> {
    drawing.list("libraries", "library").map(|l| (library_key(l.attr("name").unwrap_or(""), l.attr("urn")), l)).collect()
}

/// The library a placed part or element names.
pub fn library_of<'a>(libs: &BTreeMap<String, &'a Element>, item: &Element) -> Option<&'a Element> {
    let name = item.attr("library").unwrap_or("");
    libs.get(&library_key(name, item.attr("library_urn")))
        .or_else(|| libs.iter().find(|(k, _)| k.split('|').next() == Some(name)).map(|(_, l)| l))
        .copied()
}

/// The footprint key of `package` in library `lib` (`resistors_0603`).
pub fn footprint_key(lib: &str, package: &str) -> String {
    file_safe(&format!("{lib}_{package}"))
}

fn by_name<'a>(parent: &'a Element, outer: &str, inner: &'a str, name: &str) -> Option<&'a Element> {
    parent.list(outer, inner).find(|e| e.attr("name").unwrap_or("") == name)
}

/// EAGLE replaces `*` in a device set's name with the technology and `?` with the device.
fn device_name(deviceset: &str, device: &str, technology: &str) -> String {
    let mut s = deviceset.replace('*', technology);
    if s.contains('?') {
        s = s.replace('?', device);
    } else {
        s.push_str(device);
    }
    s
}

/// A length param in mm (as the file writes it).
fn length(mm: f64) -> Param {
    Param::quantity(runmat_rules::Quantity::length_m(mm * 1e-3), Tolerance::Exact)
}

/// Reads the `<schematic>` of `drawing`. `footprint_dir` is the folder the footprint files go
/// in, relative to the `.circuit` (the parts' `package` paths start with it).
pub fn read(drawing: &Element, name: &str, footprint_dir: &str, report: &mut Report) -> Result<Schematic, String> {
    let sch = drawing.child("schematic").ok_or("the file has no <schematic>")?;
    let libs = libraries(sch);
    let mut design = Design::new(name);
    let mut packages: BTreeMap<String, Package> = BTreeMap::new();
    let mut footprints = BTreeMap::new();

    // Net classes by number.
    let mut classes: BTreeMap<String, String> = BTreeMap::new();
    for c in sch.list("classes", "class") {
        let number = c.attr("number").unwrap_or("0").to_owned();
        let class = param_key(c.attr("name").unwrap_or("default"));
        let mut params = BTreeMap::new();
        if let Some(w) = c.num("width").filter(|w| *w > 0.0) {
            params.insert("track_width".to_owned(), length(w));
        }
        if let Some(v) = c.all("clearance").filter_map(|k| k.num("value")).find(|v| *v > 0.0) {
            params.insert("clearance".to_owned(), length(v));
        }
        if c.num("drill").is_some_and(|d| d > 0.0) {
            report.not_carried(format!("net class {class}: its minimum via drill"));
        }
        if !params.is_empty() {
            design.net_classes.insert(class.clone(), NetClass { name: class.clone(), params, source: Default::default() });
            classes.insert(number, class);
        }
    }

    // Parts: components with a package; supply symbols (no package) make their nets global.
    let mut pin_names: BTreeMap<String, BTreeMap<(String, String), String>> = BTreeMap::new();
    let mut supplies: BTreeSet<String> = BTreeSet::new();
    let mut components = Vec::new();
    let mut variants: BTreeMap<String, VariantDef> = BTreeMap::new();
    for v in sch.list("variantdefs", "variantdef") {
        let vname = v.attr("name").unwrap_or("").to_owned();
        variants.insert(vname.clone(), VariantDef { name: vname, base: None, overrides: BTreeMap::new() });
    }
    for part in sch.list("parts", "part") {
        let designator = part.attr("name").unwrap_or("").to_owned();
        let Some(lib) = library_of(&libs, part) else {
            report.not_carried(format!("part {designator}: its library isn't in the file"));
            continue;
        };
        let lib_name = lib.attr("name").unwrap_or("");
        let set_name = part.attr("deviceset").unwrap_or("");
        let Some(deviceset) = by_name(lib, "devicesets", "deviceset", set_name) else {
            report.not_carried(format!("part {designator}: device set {set_name} isn't in its library"));
            continue;
        };
        let dev_name = part.attr("device").unwrap_or("");
        let Some(device) = by_name(deviceset, "devices", "device", dev_name) else {
            report.not_carried(format!("part {designator}: device {dev_name} isn't in {set_name}"));
            continue;
        };
        let Some(pkg_name) = device.attr("package") else {
            supplies.insert(designator);
            continue;
        };
        let key = footprint_key(lib_name, pkg_name);
        if !packages.contains_key(&key) {
            let Some(pkg) = by_name(lib, "packages", "package", pkg_name) else {
                report.not_carried(format!("part {designator}: package {pkg_name} isn't in its library"));
                continue;
            };
            packages.insert(key.clone(), package(&key, pkg, report));
        }
        let technology = part.attr("technology").unwrap_or("");
        let mut attrs: BTreeMap<String, String> = BTreeMap::new();
        if let Some(t) = device.list("technologies", "technology").find(|t| t.attr("name").unwrap_or("") == technology) {
            for a in t.all("attribute") {
                attrs.insert(a.attr("name").unwrap_or("").to_owned(), a.attr("value").unwrap_or("").to_owned());
            }
        }
        for a in part.all("attribute") {
            if let Some(v) = a.attr("value") {
                attrs.insert(a.attr("name").unwrap_or("").to_owned(), v.to_owned());
            }
        }
        let value = part.attr("value").unwrap_or("").trim().to_owned();
        let base = device_name(set_name, dev_name, technology);
        let fallback = if value.is_empty() || value == base { base.clone() } else { format!("{base}-{value}") };
        let symbols: BTreeMap<String, &Element> = lib.list("symbols", "symbol").map(|s| (s.attr("name").unwrap_or("").to_owned(), s)).collect();
        let made = device_part(lib_name, deviceset, device, &symbols, &format!("{footprint_dir}/{key}.kicad_mod"), &attrs, &fallback);
        if made.part.mpn == fallback {
            report.approximated("parts without a part number are named from their device and value");
        }
        let mpn = made.part.mpn.clone();
        design.parts.entry(mpn.clone()).or_insert(made.part);
        pin_names.insert(designator.clone(), made.pins);
        footprints.insert(designator.clone(), key);
        let mut c = ComponentDecl::new(&designator, PartRef::Part(mpn));
        if !value.is_empty() {
            c.properties.insert("Value".into(), value);
        }
        for v in part.all("variant") {
            if v.attr("populate") == Some("no") {
                if let Some(def) = variants.get_mut(v.attr("name").unwrap_or("")) {
                    def.overrides.insert(designator.clone(), Fitting::NotFitted);
                }
            }
            if v.attr("value").is_some() || v.attr("technology").is_some() {
                report.not_carried(format!("variant {}: part {designator}'s changed value or technology", v.attr("name").unwrap_or("")));
            }
        }
        components.push(c);
    }

    // Nets, by name across every sheet.
    let mut nets: BTreeMap<String, (Option<String>, bool, Vec<Endpoint>)> = BTreeMap::new();
    let mut sheets = 0;
    for sheet in sch.list("sheets", "sheet") {
        sheets += 1;
        if sheet.child("busses").is_some_and(|b| !b.children.is_empty()) {
            report.approximated("busses are carried as their member nets");
        }
        for net in sheet.list("nets", "net") {
            let net_name = net.attr("name").unwrap_or("").to_owned();
            let entry = nets.entry(net_name.clone()).or_insert((None, false, Vec::new()));
            if let Some(class) = net.attr("class").and_then(|n| classes.get(n)) {
                entry.0 = Some(class.clone());
            }
            for pin in net.all("segment").flat_map(|s| s.all("pinref")) {
                let part = pin.attr("part").unwrap_or("");
                if supplies.contains(part) {
                    entry.1 = true;
                    continue;
                }
                let (gate, pin_name) = (pin.attr("gate").unwrap_or("").to_owned(), pin.attr("pin").unwrap_or("").to_owned());
                match pin_names.get(part).and_then(|m| m.get(&(gate, pin_name.clone()))) {
                    Some(p) => {
                        let end = Endpoint::pin(part, p);
                        if !entry.2.contains(&end) {
                            entry.2.push(end);
                        }
                    }
                    None if pin_names.contains_key(part) => report.not_carried(format!("part {part}: pin {pin_name} has no pad")),
                    None => {}
                }
            }
        }
    }
    if sch.child("modules").is_some_and(|m| !m.children.is_empty()) {
        report.not_carried("design blocks (modules) and their instances");
    }

    let main = design.sheet_mut("main");
    main.components = components;
    let mut net_count = 0;
    for (net_name, (class, global, ends)) in nets {
        if ends.is_empty() {
            continue;
        }
        net_count += 1;
        if class.is_some() || global {
            main.nets.push(NetDecl { name: net_name.clone(), class, global, params: Default::default(), renamed_from: None, source: Default::default() });
        }
        let mut all = vec![Endpoint::Net(net_name)];
        all.extend(ends);
        main.connections.push(Connection { ends: all, source: Default::default() });
    }
    design.variants = variants;
    report.count("components", footprints.len());
    report.count("parts", design.parts.len());
    report.count("footprints", packages.len());
    report.count("nets", net_count);
    report.count("net classes", design.net_classes.len());
    if sheets > 1 {
        report.approximated(format!("{sheets} sheets are joined into one sheet (their nets join by name, as in the file)"));
    }
    if !design.variants.is_empty() {
        report.count("variants", design.variants.len());
    }
    Ok(Schematic { design, packages, footprints })
}

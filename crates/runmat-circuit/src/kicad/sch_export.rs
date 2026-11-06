//! Fresh `.kicad_sch` export of an elaborated design: the schematic the Desktop draws.
//!
//! The pages, symbols, units, wires, labels, junctions and no-connect markers all come from
//! [`crate::schematic::layout`], so the KiCad project and the schematic canvas show the same
//! drawing; nothing is placed here.
//!
//! - **Files.** One `.kicad_sch` per sheet instance: the root page is `{project}.kicad_sch`,
//!   each instance `/a/b` is `{project}-a_b.kicad_sch`, drawn on its parent as a sheet with a
//!   sheet pin per net crossing it. Page numbers follow the hierarchy's order.
//! - **Symbols.** One library symbol per entity and unit layout, one KiCad unit per schematic
//!   unit ([`crate::schematic::units`]: a large part's banks and power units included, with
//!   their names); pins numbered by pad (a pin on several pads becomes stacked pins).
//! - **Labels.** Local labels, global labels (global nets) and hierarchical labels, where and
//!   as the layout draws them; label and sheet pin text is the net's name.
//! - **Power rails.** A power port is a KiCad power symbol (`power:GND`, `power:+5V`): a ground
//!   bar under a ground or negative rail, an arrow over a positive one, its value the net's
//!   name, its reference `#PWR…` (hidden, not in the BOM), one library symbol per rail.
//! - **Coordinates.** The layout's page millimetres (y up, origin bottom left) flip to KiCad's
//!   (y down, origin top left).
//!
//! Reading the files back with [`super::sch::read_hierarchy`] gives the same connectivity.

use super::sexpr::{self, SExpr};
use crate::elab::Netlist;
use crate::geom::Coord;
use crate::library::Pool;
use crate::schematic::{
    layout, merge_pages, units, NetLabel, PinSide, SchematicSheet, UnitLayout, NAME_OFFSET,
};
use std::collections::{BTreeMap, BTreeSet};

mod symbols;
use symbols::{lib_symbol, power_lib_symbol};

#[derive(Clone, Debug, PartialEq)]
pub struct SchExportOptions {
    /// Project name (file prefix and instance paths).
    pub project: String,
}

impl Default for SchExportOptions {
    fn default() -> Self {
        SchExportOptions {
            project: "design".into(),
        }
    }
}

fn uuid(seed: &str) -> String {
    crate::ids::Uuid(crate::ids::fnv128(seed.as_bytes()))
        .with_version(8)
        .to_string()
}

fn mm(v: Coord) -> SExpr {
    SExpr::num(v as f64 * 1e-6)
}

/// A length in millimetres, on the 0.1 µm grid.
fn len(v: f64) -> SExpr {
    SExpr::num((v * 1.0e4).round() / 1.0e4)
}

fn effects(justify: Option<&str>) -> SExpr {
    let mut v = vec![SExpr::list(
        "font",
        vec![SExpr::list(
            "size",
            vec![
                mm(crate::schematic::PIN_TEXT),
                mm(crate::schematic::PIN_TEXT),
            ],
        )],
    )];
    if let Some(j) = justify {
        v.push(SExpr::list(
            "justify",
            j.split(' ').map(SExpr::sym).collect(),
        ));
    }
    SExpr::list("effects", v)
}

fn hidden_effects() -> SExpr {
    SExpr::list(
        "effects",
        vec![
            SExpr::list(
                "font",
                vec![SExpr::list(
                    "size",
                    vec![
                        mm(crate::schematic::PIN_TEXT),
                        mm(crate::schematic::PIN_TEXT),
                    ],
                )],
            ),
            SExpr::sym("hide"),
        ],
    )
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || "-_.+".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn uuid_node(seed: &str) -> SExpr {
    SExpr::list("uuid", vec![SExpr::str(&uuid(seed))])
}

/// Page millimetres (y up) → KiCad's `(at x y angle)` (y down).
struct Page {
    height: f64,
}

impl Page {
    fn xy(&self, x: f64, y: f64) -> (SExpr, SExpr) {
        (len(x), len(self.height - y))
    }
    fn at(&self, x: f64, y: f64, angle: f64) -> SExpr {
        let (x, y) = self.xy(x, y);
        SExpr::list("at", vec![x, y, SExpr::num(angle)])
    }
    fn pt(&self, x: f64, y: f64) -> SExpr {
        let (x, y) = self.xy(x, y);
        SExpr::list("xy", vec![x, y])
    }
}

/// `(paper …)` for a layout frame: the ISO size, or a user size when A0 grew taller.
fn paper(sheet: &SchematicSheet) -> SExpr {
    let standard = match sheet.frame.as_str() {
        "A4" => Some(210.0),
        "A3" => Some(297.0),
        "A2" => Some(420.0),
        "A1" => Some(594.0),
        "A0" => Some(841.0),
        _ => None,
    };
    match standard {
        Some(h) if (h - sheet.height_mm).abs() < 1e-6 => {
            SExpr::list("paper", vec![SExpr::str(&sheet.frame)])
        }
        _ => SExpr::list(
            "paper",
            vec![
                SExpr::str("User"),
                len(sheet.width_mm),
                len(sheet.height_mm),
            ],
        ),
    }
}

fn file_of(project: &str, path: &str) -> String {
    if path == "/" {
        format!("{project}.kicad_sch")
    } else {
        format!(
            "{project}-{}.kicad_sch",
            sanitize(path.trim_start_matches('/').replace('/', "_").as_str())
        )
    }
}

/// The instance path KiCad writes for symbols on page `path`: the root's UUID, then each
/// sheet instance's UUID down to the page.
fn instance_path(project: &str, path: &str) -> String {
    let mut out = format!("/{}", uuid(&format!("{project}|root")));
    let mut prefix = String::new();
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        prefix.push('/');
        prefix.push_str(seg);
        out.push('/');
        out.push_str(&uuid(&format!("{project}|sheet|{prefix}")));
    }
    out
}

fn stroke() -> SExpr {
    SExpr::list(
        "stroke",
        vec![
            SExpr::list("width", vec![SExpr::num(0.0)]),
            SExpr::list("type", vec![SExpr::sym("default")]),
        ],
    )
}

/// Whether a rail is drawn as a ground bar (grounds and negative rails) or an arrow.
fn is_ground_rail(net: &str) -> bool {
    let upper = net.to_ascii_uppercase();
    upper.contains("GND")
        || upper.starts_with("VSS")
        || upper.starts_with("VEE")
        || upper.starts_with("VNEG")
        || upper.starts_with('-')
}

/// The library name of a rail's power symbol (`power:+5V`).
fn power_lib(net: &str) -> String {
    format!("power:{}", sanitize(net))
}


/// A power port placed as its rail's power symbol, value text on the far side of the shape.
fn power_symbol_node(page: &Page, label: &NetLabel, reference: &str, here: &str, project: &str, seed: &str) -> SExpr {
    let ground = is_ground_rail(&label.net);
    // KiCad's power symbols point up (supplies) and down (grounds) at 0°; a port drawn
    // another way turns by the quarter that takes it there.
    let angle = match (ground, label.side.as_str()) {
        (true, "up") | (false, "down") => 180.0,
        (true, "right") | (false, "left") => 90.0,
        (true, "left") | (false, "right") => 270.0,
        _ => 0.0,
    };
    let value_y = if ground { label.y_mm - 3.81 } else { label.y_mm + 3.556 };
    SExpr::list(
        "symbol",
        vec![
            SExpr::list("lib_id", vec![SExpr::str(&power_lib(&label.net))]),
            page.at(label.x_mm, label.y_mm, angle),
            SExpr::list("unit", vec![SExpr::num(1.0)]),
            SExpr::list("in_bom", vec![SExpr::sym("no")]),
            SExpr::list("on_board", vec![SExpr::sym("yes")]),
            uuid_node(seed),
            SExpr::list(
                "property",
                vec![
                    SExpr::str("Reference"),
                    SExpr::str(reference),
                    page.at(label.x_mm, label.y_mm, 0.0),
                    hidden_effects(),
                ],
            ),
            SExpr::list(
                "property",
                vec![
                    SExpr::str("Value"),
                    SExpr::str(&label.net),
                    page.at(label.x_mm, value_y, 0.0),
                    effects(None),
                ],
            ),
            SExpr::list("pin", vec![SExpr::str("1"), uuid_node(&format!("{seed}|pin"))]),
            SExpr::list(
                "instances",
                vec![SExpr::list(
                    "project",
                    vec![
                        SExpr::str(project),
                        SExpr::list(
                            "path",
                            vec![
                                SExpr::str(here),
                                SExpr::list("reference", vec![SExpr::str(reference)]),
                                SExpr::list("unit", vec![SExpr::num(1.0)]),
                            ],
                        ),
                    ],
                )],
            ),
        ],
    )
}

fn label_node(page: &Page, label: &NetLabel, seed: &str) -> SExpr {
    let left = label.side == "left";
    let (head, shaped) = match label.kind.as_str() {
        "local" => ("label", false),
        "hier" => ("hierarchical_label", true),
        _ => ("global_label", true),
    };
    let mut v = vec![SExpr::str(&label.net)];
    if shaped {
        v.push(SExpr::list("shape", vec![SExpr::sym("passive")]));
    }
    v.push(page.at(label.x_mm, label.y_mm, if left { 180.0 } else { 0.0 }));
    v.push(effects(Some(if left { "right" } else { "left" })));
    v.push(uuid_node(seed));
    SExpr::list(head, v)
}

/// Exports the netlist as one file per sheet instance. Returns file name → text; the root
/// file is `{project}.kicad_sch`.
pub fn export(nl: &Netlist, pool: &Pool, opts: &SchExportOptions) -> BTreeMap<String, String> {
    let project = sanitize(&opts.project);
    // KiCad keeps one file per sheet instance: an instance's second page joins its first.
    let drawn = merge_pages(&layout(nl));
    // Library symbols: one per entity and unit layouts; each component's units by name.
    let mut libs: BTreeMap<String, Vec<(UnitLayout, Option<String>)>> = BTreeMap::new();
    let mut lib_of: Vec<String> = Vec::new();
    let mut unit_index: Vec<BTreeMap<String, usize>> = Vec::new();
    for c in &nl.components {
        let ename = c
            .entity
            .and_then(|u| pool.entities.get(&u))
            .map(|e| e.name.clone())
            .unwrap_or_else(|| c.designator.trim_end_matches(char::is_numeric).to_string());
        let us = units(c);
        let layouts: Vec<(UnitLayout, Option<String>)> = us
            .iter()
            .map(|u| (UnitLayout::of(c, u, us.len()), u.label.clone()))
            .collect();
        let mut lib = format!("RunMat:{}", sanitize(&ename));
        let mut k = 1;
        while let Some(existing) = libs.get(&lib) {
            let same = existing.len() == layouts.len()
                && existing.iter().zip(&layouts).all(|(a, b)| a == b);
            if same {
                break;
            }
            k += 1;
            lib = format!("RunMat:{}_{}", sanitize(&ename), k);
        }
        libs.entry(lib.clone()).or_insert(layouts);
        lib_of.push(lib);
        unit_index.push(
            us.iter()
                .enumerate()
                .map(|(k, u)| (u.name.clone(), k))
                .collect(),
        );
    }
    let comp_of: BTreeMap<String, usize> = nl
        .components
        .iter()
        .enumerate()
        .map(|(k, c)| (c.path.to_string(), k))
        .collect();
    let page_no: BTreeMap<&str, usize> = drawn
        .sheets
        .iter()
        .enumerate()
        .map(|(k, s)| (s.path.as_str(), k + 1))
        .collect();
    let mut files = BTreeMap::new();
    // `#PWR` references number across the whole project, page by page.
    let mut power_refs = 0usize;
    for sheet in &drawn.sheets {
        let page = Page {
            height: sheet.height_mm,
        };
        let here = instance_path(&project, &sheet.path);
        let seed = |what: &str| format!("{project}|{}|{what}", sheet.path);
        let mut used: BTreeSet<&str> = BTreeSet::new();
        let mut items: Vec<SExpr> = Vec::new();
        for symbol in &sheet.symbols {
            let Some(&ci) = comp_of.get(&symbol.path) else {
                continue;
            };
            let c = &nl.components[ci];
            let lib = &lib_of[ci];
            used.insert(lib);
            let u = symbol
                .unit
                .as_ref()
                .and_then(|n| unit_index[ci].get(n).copied())
                .unwrap_or(0);
            let lay = &libs[lib][u].0;
            let sseed = format!("{project}|{}|{u}", c.designator);
            let package = c
                .package
                .and_then(|k| pool.packages.get(&k))
                .map(|pk| format!("RunMat:{}", pk.name))
                .or_else(|| c.unresolved_package.clone())
                .unwrap_or_default();
            let value = c
                .properties
                .get("Value")
                .cloned()
                .or_else(|| {
                    c.part
                        .and_then(|k| pool.parts.get(&k))
                        .map(|pt| pt.mpn.clone())
                })
                .unwrap_or_else(|| c.path.leaf().to_string());
            let mut sym = vec![
                SExpr::list("lib_id", vec![SExpr::str(lib)]),
                page.at(symbol.x_mm, symbol.y_mm, symbol.rotation),
                SExpr::list("unit", vec![SExpr::num((u + 1) as f64)]),
                SExpr::list("in_bom", vec![SExpr::sym("yes")]),
                SExpr::list("on_board", vec![SExpr::sym("yes")]),
                uuid_node(&sseed),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Reference"),
                        SExpr::str(&c.designator),
                        page.at(symbol.designator_at[0], symbol.designator_at[1], 0.0),
                        effects(None),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Value"),
                        SExpr::str(&value),
                        page.at(symbol.value_at[0], symbol.value_at[1], 0.0),
                        effects(None),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Footprint"),
                        SExpr::str(&package),
                        page.at(symbol.x_mm, symbol.y_mm, 0.0),
                        hidden_effects(),
                    ],
                ),
            ];
            for (k, v) in &c.properties {
                if k != "Value" {
                    sym.push(SExpr::list(
                        "property",
                        vec![
                            SExpr::str(k),
                            SExpr::str(v),
                            page.at(symbol.x_mm, symbol.y_mm, 0.0),
                            hidden_effects(),
                        ],
                    ));
                }
            }
            if symbol.mirror {
                sym.insert(2, SExpr::list("mirror", vec![SExpr::sym("y")]));
            }
            for lp in &lay.pins {
                for pad in &lp.pads {
                    sym.push(SExpr::list(
                        "pin",
                        vec![SExpr::str(pad), uuid_node(&format!("{sseed}|pin|{pad}"))],
                    ));
                }
            }
            sym.push(SExpr::list(
                "instances",
                vec![SExpr::list(
                    "project",
                    vec![
                        SExpr::str(&project),
                        SExpr::list(
                            "path",
                            vec![
                                SExpr::str(&here),
                                SExpr::list("reference", vec![SExpr::str(&c.designator)]),
                                SExpr::list("unit", vec![SExpr::num((u + 1) as f64)]),
                            ],
                        ),
                    ],
                )],
            ));
            items.push(SExpr::list("symbol", sym));
        }
        for child in &sheet.sheet_symbols {
            let leaf = child.path.rsplit('/').next().unwrap_or(&child.path);
            let file = file_of(&project, &child.path);
            let top = child.y_mm + child.h_mm;
            let mut v = vec![
                page.at(child.x_mm, top, 0.0),
                SExpr::list("size", vec![len(child.w_mm), len(child.h_mm)]),
                SExpr::list(
                    "stroke",
                    vec![
                        SExpr::list("width", vec![SExpr::num(0.1524)]),
                        SExpr::list("type", vec![SExpr::sym("solid")]),
                    ],
                ),
                SExpr::list(
                    "fill",
                    vec![SExpr::list(
                        "color",
                        vec![
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                        ],
                    )],
                ),
                uuid_node(&format!("{project}|sheet|{}", child.path)),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Sheetname"),
                        SExpr::str(leaf),
                        page.at(child.x_mm, top + 1.27, 0.0),
                        effects(Some("left bottom")),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Sheetfile"),
                        SExpr::str(&file),
                        page.at(child.x_mm, child.y_mm - 1.27, 0.0),
                        effects(Some("left top")),
                    ],
                ),
            ];
            for pin in &child.pins {
                let left = pin.side == "left";
                v.push(SExpr::list(
                    "pin",
                    vec![
                        SExpr::str(&pin.net),
                        SExpr::sym("passive"),
                        page.at(pin.x_mm, pin.y_mm, if left { 180.0 } else { 0.0 }),
                        effects(Some(if left { "left" } else { "right" })),
                        uuid_node(&format!("{project}|sheetpin|{}|{}", child.path, pin.net)),
                    ],
                ));
            }
            v.push(SExpr::list(
                "instances",
                vec![SExpr::list(
                    "project",
                    vec![
                        SExpr::str(&project),
                        SExpr::list(
                            "path",
                            vec![
                                SExpr::str(&here),
                                SExpr::list(
                                    "page",
                                    vec![SExpr::str(
                                        &page_no
                                            .get(child.path.as_str())
                                            .copied()
                                            .unwrap_or(0)
                                            .to_string(),
                                    )],
                                ),
                            ],
                        ),
                    ],
                )],
            ));
            items.push(SExpr::list("sheet", v));
        }
        for (w, wire) in sheet.wires.iter().enumerate() {
            for (k, seg) in wire.points.windows(2).enumerate() {
                items.push(SExpr::list(
                    "wire",
                    vec![
                        SExpr::list(
                            "pts",
                            vec![page.pt(seg[0][0], seg[0][1]), page.pt(seg[1][0], seg[1][1])],
                        ),
                        stroke(),
                        uuid_node(&seed(&format!("w{w}.{k}"))),
                    ],
                ));
            }
        }
        for (k, j) in sheet.junctions.iter().enumerate() {
            items.push(SExpr::list(
                "junction",
                vec![
                    {
                        let (x, y) = page.xy(j[0], j[1]);
                        SExpr::list("at", vec![x, y])
                    },
                    SExpr::list("diameter", vec![SExpr::num(0.0)]),
                    SExpr::list(
                        "color",
                        vec![
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                            SExpr::num(0.0),
                        ],
                    ),
                    uuid_node(&seed(&format!("j{k}"))),
                ],
            ));
        }
        for (k, nc) in sheet.no_connects.iter().enumerate() {
            let (x, y) = page.xy(nc[0], nc[1]);
            items.push(SExpr::list(
                "no_connect",
                vec![
                    SExpr::list("at", vec![x, y]),
                    uuid_node(&seed(&format!("nc{k}"))),
                ],
            ));
        }
        let mut rails: BTreeSet<&str> = BTreeSet::new();
        for (k, label) in sheet.labels.iter().enumerate() {
            if label.kind == "power" {
                power_refs += 1;
                rails.insert(&label.net);
                items.push(power_symbol_node(
                    &page,
                    label,
                    &format!("#PWR{power_refs:02}"),
                    &here,
                    &project,
                    &seed(&format!("p{k}")),
                ));
            } else {
                items.push(label_node(&page, label, &seed(&format!("l{k}"))));
            }
        }
        let mut lib_nodes: Vec<SExpr> = used
            .iter()
            .map(|lib| lib_symbol(lib, &libs[*lib]))
            .collect();
        lib_nodes.extend(rails.iter().map(|net| power_lib_symbol(net)));
        let root = sheet.path == "/";
        let mut page_items = vec![
            SExpr::list("version", vec![SExpr::sym("20231120")]),
            SExpr::list("generator", vec![SExpr::str("runmat")]),
            SExpr::list(
                "uuid",
                vec![SExpr::str(&uuid(&if root {
                    format!("{project}|root")
                } else {
                    format!("{project}|page|{}", sheet.path)
                }))],
            ),
            paper(sheet),
            SExpr::list("lib_symbols", lib_nodes),
        ];
        page_items.extend(items);
        if root {
            page_items.push(SExpr::list(
                "sheet_instances",
                vec![SExpr::list(
                    "path",
                    vec![SExpr::str("/"), SExpr::list("page", vec![SExpr::str("1")])],
                )],
            ));
        }
        files.insert(
            file_of(&project, &sheet.path),
            sexpr::write(&SExpr::list("kicad_sch", page_items)),
        );
    }
    files
}

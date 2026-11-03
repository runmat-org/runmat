//! Fresh `.kicad_sch` export of an elaborated design, with automatic placement.
//!
//! Layout rules, chosen for a readable sheet rather than a wiring diagram:
//!
//! - **Pages.** Components are grouped by their top-level hierarchy instance (`ch[3]`,
//!   `power`, …; top-level components form one group), each group starting a new page; a
//!   group too large for one page continues on the next. The root sheet holds one
//!   hierarchical sheet per page.
//! - **Order.** Inside a group, components follow connectivity: a breadth-first walk over
//!   shared signal nets (rails and nets with more than eight pins are ignored for ordering),
//!   starting from the most connected component, so connected parts sit side by side.
//! - **Placement.** Shelf packing in rows on the 1.27 mm grid; each cell reserves room for
//!   the symbol, its pins and its labels.
//! - **Connections are labels, not wires.** Every connected pin gets a 2.54 mm wire stub and a
//!   label naming its net: a local label for a net that stays on its page, a global label for
//!   global nets and nets spanning pages. Unconnected pins get a no-connect marker.
//! - **Symbols.** One library symbol per entity, generated as a box: one KiCad unit per gate;
//!   inputs, power inputs and the first half of passive pins on the left, the rest on the
//!   right; pins numbered by pad (a pin on several pads becomes stacked pins).
//!
//! Reading the files back with [`super::sch::read_hierarchy`] gives the same connectivity.

use super::sexpr::{self, SExpr};
use crate::elab::Netlist;
use crate::geom::{Coord, Point};
use crate::library::{PinType, Pool};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const GRID: Coord = 1_270_000;
const PIN_PITCH: Coord = 2_540_000;
const PIN_LEN: Coord = 2_540_000;
const STUB: Coord = 2_540_000;
/// Usable A3 area (mm): x 25…395, y 25…270.
const PAGE: (Coord, Coord, Coord, Coord) = (25_400_000, 25_400_000, 393_700_000, 269_240_000);

#[derive(Clone, Debug, PartialEq)]
pub struct SchExportOptions {
    /// Project name (file prefix and instance paths).
    pub project: String,
    /// Nets with more pins than this are not used to order components.
    pub ordering_fanout: usize,
}

impl Default for SchExportOptions {
    fn default() -> Self {
        SchExportOptions {
            project: "design".into(),
            ordering_fanout: 8,
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

/// Schematic point (y down) from page coordinates (y down, nm).
fn at(x: Coord, y: Coord, angle: f64) -> SExpr {
    SExpr::list("at", vec![mm(x), mm(y), SExpr::num(angle)])
}

fn effects(justify: Option<&str>) -> SExpr {
    let mut v = vec![SExpr::list(
        "font",
        vec![SExpr::list(
            "size",
            vec![SExpr::num(1.27), SExpr::num(1.27)],
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
                    vec![SExpr::num(1.27), SExpr::num(1.27)],
                )],
            ),
            SExpr::sym("hide"),
        ],
    )
}

fn pin_type(k: PinType) -> &'static str {
    match k {
        PinType::Input => "input",
        PinType::Output => "output",
        PinType::Bidirectional => "bidirectional",
        PinType::TriState => "tri_state",
        PinType::Passive => "passive",
        PinType::PowerIn => "power_in",
        PinType::PowerOut => "power_out",
        PinType::OpenCollector => "open_collector",
        PinType::OpenEmitter => "open_emitter",
        PinType::NotConnected => "no_connect",
        PinType::Free => "free",
        _ => "unspecified",
    }
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

/// A gate's pins `(name, pads, type)` and each pin's net.
type UnitPins = (
    String,
    Vec<(String, Vec<String>, PinType)>,
    Vec<Option<usize>>,
);

/// One generated unit: pins on the left and right, in library coordinates (y up).
#[derive(Clone, Debug)]
struct UnitLayout {
    /// (pin name, pad numbers, type, left side, library y of the pin row).
    pins: Vec<(String, Vec<String>, PinType, bool, Coord)>,
    half_w: Coord,
    half_h: Coord,
}

impl UnitLayout {
    fn new(pins: &[(String, Vec<String>, PinType)]) -> UnitLayout {
        let passive: Vec<usize> = (0..pins.len())
            .filter(|&i| pins[i].2 == PinType::Passive)
            .collect();
        let left_of = |i: usize| match pins[i].2 {
            PinType::Input | PinType::PowerIn => true,
            PinType::Passive => {
                passive.iter().position(|&p| p == i).unwrap() < passive.len().div_ceil(2)
            }
            _ => false,
        };
        let (mut left, mut right): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for i in 0..pins.len() {
            if left_of(i) {
                left.push(i)
            } else {
                right.push(i)
            }
        }
        if right.is_empty() && left.len() > 1 {
            right = left.split_off(left.len().div_ceil(2));
        }
        let rows = left.len().max(right.len()).max(1) as Coord;
        let half_h = (rows * PIN_PITCH) / 2 + GRID;
        let name_len = pins.iter().map(|p| p.0.chars().count()).max().unwrap_or(1) as Coord;
        let half_w =
            ((((name_len * GRID + 5 * GRID) / 2) + GRID - 1) / GRID * GRID).max(2 * PIN_PITCH);
        let mut out = Vec::new();
        for (side, list) in [(true, &left), (false, &right)] {
            let top = (list.len() as Coord - 1) * PIN_PITCH / 2;
            for (k, &i) in list.iter().enumerate() {
                // Rows on the 2.54 mm grid from the top, centred.
                let y = (top - k as Coord * PIN_PITCH).div_euclid(GRID) * GRID;
                out.push((pins[i].0.clone(), pins[i].1.clone(), pins[i].2, side, y));
            }
        }
        UnitLayout {
            pins: out,
            half_w,
            half_h,
        }
    }
    /// Pin tip in library coordinates.
    fn tip(&self, left: bool, y: Coord) -> Point {
        let x = self.half_w + PIN_LEN;
        Point::new(if left { -x } else { x }, y)
    }
}

/// Exports the netlist as a root sheet plus one file per page. Returns file name → text;
/// the root file is `{project}.kicad_sch`.
pub fn export(nl: &Netlist, pool: &Pool, opts: &SchExportOptions) -> BTreeMap<String, String> {
    let project = sanitize(&opts.project);
    // Units per component: gate → pins (name, pads, type).
    struct Comp {
        idx: usize,
        lib: String,
        units: Vec<UnitPins>,
    }
    let mut libs: BTreeMap<String, Vec<UnitLayout>> = BTreeMap::new();
    let mut comps: Vec<Comp> = Vec::new();
    for (idx, c) in nl.components.iter().enumerate() {
        let ename = c
            .entity
            .and_then(|u| pool.entities.get(&u))
            .map(|e| e.name.clone())
            .unwrap_or_else(|| c.designator.trim_end_matches(char::is_numeric).to_string());
        let mut gates: Vec<String> = Vec::new();
        for p in &c.pins {
            if !gates.contains(&p.gate) {
                gates.push(p.gate.clone());
            }
        }
        let units: Vec<_> = gates
            .iter()
            .map(|g| {
                let pins: Vec<_> = c
                    .pins
                    .iter()
                    .filter(|p| &p.gate == g)
                    .map(|p| {
                        let pads = if p.pads.is_empty() {
                            vec![p.pin.clone()]
                        } else {
                            p.pads.clone()
                        };
                        (p.pin.clone(), pads, p.kind)
                    })
                    .collect();
                let nets = c
                    .pins
                    .iter()
                    .filter(|p| &p.gate == g)
                    .map(|p| p.net)
                    .collect();
                (g.clone(), pins, nets)
            })
            .collect();
        // One library symbol per entity and pin signature.
        let sig: String = units
            .iter()
            .map(|u| {
                u.1.iter()
                    .map(|p| format!("{}={}", p.0, p.1.join("+")))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("|");
        let mut lib = format!("RunMat:{}", sanitize(&ename));
        let layouts: Vec<UnitLayout> = units.iter().map(|u| UnitLayout::new(&u.1)).collect();
        let mut k = 1;
        while let Some(existing) = libs.get(&lib) {
            let same = existing.len() == layouts.len()
                && existing.iter().zip(&layouts).all(|(a, b)| {
                    a.pins
                        .iter()
                        .map(|p| (&p.0, &p.1))
                        .eq(b.pins.iter().map(|p| (&p.0, &p.1)))
                });
            if same {
                break;
            }
            k += 1;
            lib = format!("RunMat:{}_{}", sanitize(&ename), k);
        }
        let _ = sig;
        libs.entry(lib.clone()).or_insert(layouts);
        comps.push(Comp { idx, lib, units });
    }
    // Groups by top-level instance, in netlist order.
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (k, c) in comps.iter().enumerate() {
        let path = &nl.components[c.idx].path;
        let key = if path.depth() > 1 {
            path.0[0].clone()
        } else {
            String::new()
        };
        groups.entry(key).or_default().push(k);
    }
    // Ordering nets: signal nets with few pins.
    let ordering: Vec<Vec<usize>> = nl
        .nets
        .iter()
        .map(|n| {
            if n.global || n.pins.len() > opts.ordering_fanout {
                return vec![];
            }
            let mut v: Vec<usize> = n.pins.iter().map(|p| p.0).collect();
            v.dedup();
            v
        })
        .collect();
    let mut comp_k: BTreeMap<usize, usize> = BTreeMap::new();
    for (k, c) in comps.iter().enumerate() {
        comp_k.insert(c.idx, k);
    }
    let mut neighbours: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); comps.len()];
    for comps_on in &ordering {
        for &a in comps_on {
            for &b in comps_on {
                if a != b {
                    neighbours[comp_k[&a]].insert(comp_k[&b]);
                }
            }
        }
    }
    // Pages: (group name, ordered components with their cell positions).
    struct Placed {
        k: usize,
        unit: usize,
        origin: Point,
    }
    let mut pages: Vec<(String, Vec<Placed>)> = Vec::new();
    for (gname, members) in &groups {
        let set: BTreeSet<usize> = members.iter().copied().collect();
        let mut order = Vec::new();
        let mut seen = BTreeSet::new();
        let mut rest: Vec<usize> = members.clone();
        rest.sort_by_key(|k| {
            (
                std::cmp::Reverse(neighbours[*k].intersection(&set).count()),
                *k,
            )
        });
        for &start in &rest {
            if !seen.insert(start) {
                continue;
            }
            let mut q = VecDeque::from([start]);
            while let Some(k) = q.pop_front() {
                order.push(k);
                for &n in &neighbours[k] {
                    if set.contains(&n) && seen.insert(n) {
                        q.push_back(n);
                    }
                }
            }
        }
        // Shelf packing of every unit.
        let label_room = |k: usize| -> Coord {
            let longest = comps[k]
                .units
                .iter()
                .flat_map(|u| u.2.iter())
                .filter_map(|n| n.map(|n| nl.nets[n].name.chars().count()))
                .max()
                .unwrap_or(4) as Coord;
            STUB + (longest + 4) * GRID
        };
        let mut page: Vec<Placed> = Vec::new();
        let (mut x, mut y, mut row_h) = (PAGE.0, PAGE.1, 0);
        let mut part = 0;
        for k in order {
            for (u, _) in comps[k].units.iter().enumerate() {
                let lay = &libs[&comps[k].lib][u];
                let room = label_room(k);
                let w = 2 * (lay.half_w + PIN_LEN + room);
                let h = 2 * lay.half_h + 4 * GRID;
                if x + w > PAGE.2 && x > PAGE.0 {
                    x = PAGE.0;
                    y += row_h;
                    row_h = 0;
                }
                if y + h > PAGE.3 && !page.is_empty() {
                    let name = if part == 0 {
                        gname.clone()
                    } else {
                        format!("{gname}~{part}")
                    };
                    pages.push((name, std::mem::take(&mut page)));
                    part += 1;
                    x = PAGE.0;
                    y = PAGE.1;
                    row_h = 0;
                }
                let origin = Point::new(
                    (x + w / 2).div_euclid(GRID) * GRID,
                    (y + h / 2).div_euclid(GRID) * GRID,
                );
                page.push(Placed { k, unit: u, origin });
                x += w;
                row_h = row_h.max(h);
            }
        }
        if !page.is_empty() {
            let name = if part == 0 {
                gname.clone()
            } else {
                format!("{gname}~{part}")
            };
            pages.push((name, page));
        }
    }
    // Which pages each net appears on.
    let mut net_pages: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nl.nets.len()];
    for (pi, (_, placed)) in pages.iter().enumerate() {
        for p in placed {
            for n in comps[p.k].units[p.unit].2.iter().flatten() {
                net_pages[*n].insert(pi);
            }
        }
    }
    let root_uuid = uuid(&format!("{project}|root"));
    let mut files = BTreeMap::new();
    let mut root_items = vec![
        SExpr::list("version", vec![SExpr::sym("20231120")]),
        SExpr::list("generator", vec![SExpr::str("runmat")]),
        SExpr::list("uuid", vec![SExpr::str(&root_uuid)]),
        SExpr::list("paper", vec![SExpr::str("A3")]),
        SExpr::list("lib_symbols", vec![]),
    ];
    for (pi, (gname, placed)) in pages.iter().enumerate() {
        let title = if gname.is_empty() {
            "top".to_string()
        } else {
            gname.clone()
        };
        let file = format!("{project}-{}.kicad_sch", sanitize(&title));
        let sheet_uuid = uuid(&format!("{project}|sheet|{title}"));
        let path = format!("/{root_uuid}/{sheet_uuid}");
        // The sheet box on the root page.
        let (sx, sy) = (
            PAGE.0 + (pi as Coord % 6) * 60 * GRID,
            PAGE.1 + (pi as Coord / 6) * 24 * GRID,
        );
        root_items.push(SExpr::list(
            "sheet",
            vec![
                at(sx, sy, 0.0),
                SExpr::list("size", vec![mm(50 * GRID), mm(16 * GRID)]),
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
                SExpr::list("uuid", vec![SExpr::str(&sheet_uuid)]),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Sheetname"),
                        SExpr::str(&title),
                        at(sx, sy - GRID, 0.0),
                        effects(Some("left bottom")),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Sheetfile"),
                        SExpr::str(&file),
                        at(sx, sy + 17 * GRID, 0.0),
                        effects(Some("left top")),
                    ],
                ),
                SExpr::list(
                    "instances",
                    vec![SExpr::list(
                        "project",
                        vec![
                            SExpr::str(&project),
                            SExpr::list(
                                "path",
                                vec![
                                    SExpr::str(&format!("/{root_uuid}")),
                                    SExpr::list("page", vec![SExpr::str(&(pi + 2).to_string())]),
                                ],
                            ),
                        ],
                    )],
                ),
            ],
        ));
        // The page itself.
        let mut used_libs: BTreeSet<String> = BTreeSet::new();
        let mut items: Vec<SExpr> = Vec::new();
        for p in placed {
            let c = &comps[p.k];
            let ci = &nl.components[c.idx];
            used_libs.insert(c.lib.clone());
            let lay = &libs[&c.lib][p.unit];
            let seed = format!("{project}|{}|{}", ci.designator, p.unit);
            let package = ci
                .package
                .and_then(|u| pool.packages.get(&u))
                .map(|pk| format!("RunMat:{}", pk.name))
                .unwrap_or_default();
            let value = ci
                .properties
                .get("Value")
                .cloned()
                .or_else(|| {
                    ci.part
                        .and_then(|u| pool.parts.get(&u))
                        .map(|pt| pt.mpn.clone())
                })
                .unwrap_or_else(|| ci.path.leaf().to_string());
            let o = p.origin;
            let mut sym = vec![
                SExpr::list("lib_id", vec![SExpr::str(&c.lib)]),
                at(o.x, o.y, 0.0),
                SExpr::list("unit", vec![SExpr::num((p.unit + 1) as f64)]),
                SExpr::list("in_bom", vec![SExpr::sym("yes")]),
                SExpr::list("on_board", vec![SExpr::sym("yes")]),
                SExpr::list("uuid", vec![SExpr::str(&uuid(&seed))]),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Reference"),
                        SExpr::str(&ci.designator),
                        at(o.x, o.y - lay.half_h - GRID, 0.0),
                        effects(None),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Value"),
                        SExpr::str(&value),
                        at(o.x, o.y + lay.half_h + GRID, 0.0),
                        effects(None),
                    ],
                ),
                SExpr::list(
                    "property",
                    vec![
                        SExpr::str("Footprint"),
                        SExpr::str(&package),
                        at(o.x, o.y, 0.0),
                        hidden_effects(),
                    ],
                ),
            ];
            for (k, v) in &ci.properties {
                if k != "Value" {
                    sym.push(SExpr::list(
                        "property",
                        vec![
                            SExpr::str(k),
                            SExpr::str(v),
                            at(o.x, o.y, 0.0),
                            hidden_effects(),
                        ],
                    ));
                }
            }
            for (_, pads, _, _, _) in &lay.pins {
                for pad in pads {
                    sym.push(SExpr::list(
                        "pin",
                        vec![
                            SExpr::str(pad),
                            SExpr::list(
                                "uuid",
                                vec![SExpr::str(&uuid(&format!("{seed}|pin|{pad}")))],
                            ),
                        ],
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
                                SExpr::str(&path),
                                SExpr::list("reference", vec![SExpr::str(&ci.designator)]),
                                SExpr::list("unit", vec![SExpr::num((p.unit + 1) as f64)]),
                            ],
                        ),
                    ],
                )],
            ));
            items.push(SExpr::list("symbol", sym));
            // Stubs and labels, matched to pins by name.
            let unit = &c.units[p.unit];
            for (pi_, (name, _, _, left, ly)) in lay.pins.iter().enumerate() {
                let Some(ui) = unit.1.iter().position(|q| &q.0 == name) else {
                    continue;
                };
                let tip = lay.tip(*left, *ly);
                // Library y up → page y down.
                let (tx, ty) = (o.x + tip.x, o.y - tip.y);
                let pseed = format!("{seed}|{pi_}");
                match unit.2[ui] {
                    None => items.push(SExpr::list(
                        "no_connect",
                        vec![
                            SExpr::list("at", vec![mm(tx), mm(ty)]),
                            SExpr::list("uuid", vec![SExpr::str(&uuid(&format!("{pseed}|nc")))]),
                        ],
                    )),
                    Some(n) => {
                        let net = &nl.nets[n];
                        let ex = if *left { tx - STUB } else { tx + STUB };
                        items.push(SExpr::list(
                            "wire",
                            vec![
                                SExpr::list(
                                    "pts",
                                    vec![
                                        SExpr::list("xy", vec![mm(tx), mm(ty)]),
                                        SExpr::list("xy", vec![mm(ex), mm(ty)]),
                                    ],
                                ),
                                SExpr::list(
                                    "stroke",
                                    vec![
                                        SExpr::list("width", vec![SExpr::num(0.0)]),
                                        SExpr::list("type", vec![SExpr::sym("default")]),
                                    ],
                                ),
                                SExpr::list("uuid", vec![SExpr::str(&uuid(&format!("{pseed}|w")))]),
                            ],
                        ));
                        let global = net.global || net_pages[n].len() > 1;
                        let angle = if *left { 180.0 } else { 0.0 };
                        let justify = if *left { "right" } else { "left" };
                        let mut l = vec![SExpr::str(&net.name)];
                        if global {
                            let shape = match unit.1[ui].2 {
                                PinType::Input => "output",
                                PinType::Output | PinType::PowerOut => "input",
                                PinType::Bidirectional | PinType::TriState => "bidirectional",
                                _ => "passive",
                            };
                            l.push(SExpr::list("shape", vec![SExpr::sym(shape)]));
                        }
                        l.push(at(ex, ty, angle));
                        l.push(effects(Some(justify)));
                        l.push(SExpr::list(
                            "uuid",
                            vec![SExpr::str(&uuid(&format!("{pseed}|l")))],
                        ));
                        items.push(SExpr::list(
                            if global { "global_label" } else { "label" },
                            l,
                        ));
                    }
                }
            }
        }
        let mut lib_nodes = Vec::new();
        for lib in &used_libs {
            lib_nodes.push(lib_symbol(lib, &libs[lib]));
        }
        let mut page_items = vec![
            SExpr::list("version", vec![SExpr::sym("20231120")]),
            SExpr::list("generator", vec![SExpr::str("runmat")]),
            SExpr::list(
                "uuid",
                vec![SExpr::str(&uuid(&format!("{project}|page|{title}")))],
            ),
            SExpr::list("paper", vec![SExpr::str("A3")]),
            SExpr::list("lib_symbols", lib_nodes),
        ];
        page_items.extend(items);
        files.insert(file, sexpr::write(&SExpr::list("kicad_sch", page_items)));
    }
    root_items.push(SExpr::list(
        "sheet_instances",
        vec![SExpr::list(
            "path",
            vec![SExpr::str("/"), SExpr::list("page", vec![SExpr::str("1")])],
        )],
    ));
    files.insert(
        format!("{project}.kicad_sch"),
        sexpr::write(&SExpr::list("kicad_sch", root_items)),
    );
    files
}

fn lib_symbol(lib: &str, units: &[UnitLayout]) -> SExpr {
    let short = lib.split_once(':').map(|x| x.1).unwrap_or(lib);
    let mut v = vec![
        SExpr::str(lib),
        SExpr::list("in_bom", vec![SExpr::sym("yes")]),
        SExpr::list("on_board", vec![SExpr::sym("yes")]),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Reference"),
                SExpr::str("U"),
                SExpr::list(
                    "at",
                    vec![SExpr::num(0.0), SExpr::num(0.0), SExpr::num(0.0)],
                ),
                effects(None),
            ],
        ),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Value"),
                SExpr::str(short),
                SExpr::list(
                    "at",
                    vec![SExpr::num(0.0), SExpr::num(0.0), SExpr::num(0.0)],
                ),
                effects(None),
            ],
        ),
    ];
    for (u, lay) in units.iter().enumerate() {
        let mut body = vec![
            SExpr::str(&format!("{short}_{}_1", u + 1)),
            SExpr::list(
                "rectangle",
                vec![
                    SExpr::list("start", vec![mm(-lay.half_w), mm(lay.half_h)]),
                    SExpr::list("end", vec![mm(lay.half_w), mm(-lay.half_h)]),
                    SExpr::list(
                        "stroke",
                        vec![
                            SExpr::list("width", vec![SExpr::num(0.254)]),
                            SExpr::list("type", vec![SExpr::sym("default")]),
                        ],
                    ),
                    SExpr::list(
                        "fill",
                        vec![SExpr::list("type", vec![SExpr::sym("background")])],
                    ),
                ],
            ),
        ];
        for (name, pads, kind, left, y) in &lay.pins {
            let tip = lay.tip(*left, *y);
            for pad in pads {
                body.push(SExpr::list(
                    "pin",
                    vec![
                        SExpr::sym(pin_type(*kind)),
                        SExpr::sym("line"),
                        SExpr::list(
                            "at",
                            vec![
                                mm(tip.x),
                                mm(tip.y),
                                SExpr::num(if *left { 0.0 } else { 180.0 }),
                            ],
                        ),
                        SExpr::list("length", vec![mm(PIN_LEN)]),
                        SExpr::list("name", vec![SExpr::str(name), effects(None)]),
                        SExpr::list("number", vec![SExpr::str(pad), effects(None)]),
                    ],
                ));
            }
        }
        v.push(SExpr::list("symbol", body));
    }
    SExpr::list("symbol", v)
}

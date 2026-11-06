//! Board design rules through `runmat-rules`.
//!
//! [`BoardGraph`] presents a [`Board`] as a selector item graph:
//!
//! | Kind | Matches also | Name (`#…`) | Classes (`.…`) | Attributes | Parent |
//! |---|---|---|---|---|---|
//! | `net` | | net name | its net class | `class`, `name` | — |
//! | `footprint` | `component` | designator | `top`/`bottom`, `smd`/`tht` | `ref`, `value`, `package`, `side`, `locked`, properties | — |
//! | `pad` | `copper` | `U1.3` | `smd`, `tht`, `npth`, `connector`, `thermal` | `name`, `net`, `width`, `height`, `drill` | footprint |
//! | `track`, `arc` | `copper` (`arc` is also `track`) | — | — | `width`, `layer`, `net`, `length` | — |
//! | `via` | `copper` | — | `through`/`blind`/`buried`/`micro` | `diameter`, `drill`, `from`, `to`, `net` | — |
//! | `zone` | `copper` | zone name | — | `priority`, `net`, `layer` | — |
//! | `edge` | | `edge` | — | — | — (the board outline, for edge clearance) |
//!
//! Pseudo-classes: `:net-class(a, b…)`, `:net(name…)` (a trailing `*` is a prefix match),
//! `:layer(outer|inner|top|bottom|<name>…)`, `:side(top|bottom)`, `:diff-pair`,
//! `:in-room(name)`.
//!
//! Single-item properties (see [`props`]) and pair properties (clearances) are evaluated
//! with the shared precedence of `runmat-rules`; [`BoardRules::new`] seeds every property
//! with the board's [`FabLimits`] as the default, so every lookup has a value and an
//! explanation.

use crate::geom::{Coord, Point};
use crate::layout::{Board, FabLimits, ItemId, Layer, NetId, ViaKind};
use crate::library::PadKind;
use runmat_rules::{
    AttrValue, BuiltinUnits, Cascade, Dim, ItemGraph, PairCascade, PairRule, PseudoArg, Quantity,
    Rule, Schema, Selector, SelectorError, SourceKey, SourceOrder,
};
use std::collections::{BTreeMap, BTreeSet};

/// Item key in the board graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BItem {
    Edge,
    Net(u32),
    Footprint(ItemId),
    Pad(ItemId, u32),
    Track(ItemId),
    Via(ItemId),
    Zone(ItemId),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RuleValue {
    Len(Coord),
    Num(f64),
    Int(i64),
    Bool(bool),
    Text(String),
    ViaKinds(Vec<ViaKind>),
}

impl RuleValue {
    pub fn as_len(&self) -> Option<Coord> {
        match self {
            RuleValue::Len(v) => Some(*v),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            RuleValue::Num(v) => Some(*v),
            RuleValue::Int(v) => Some(*v as f64),
            RuleValue::Len(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn int(&self) -> Option<i64> {
        match self {
            RuleValue::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn text(&self) -> Option<&str> {
        match self {
            RuleValue::Text(v) => Some(v),
            _ => None,
        }
    }
}

/// Property names.
pub mod props {
    pub const TRACK_WIDTH: &str = "track_width";
    pub const TRACK_WIDTH_MIN: &str = "track_width_min";
    pub const TRACK_WIDTH_MAX: &str = "track_width_max";
    pub const VIA_DIAMETER: &str = "via_diameter";
    pub const VIA_DRILL: &str = "via_drill";
    pub const ANNULAR_MIN: &str = "annular_min";
    pub const HOLE_MIN: &str = "hole_min";
    pub const VIA_TYPES: &str = "via_types";
    pub const DIFF_GAP: &str = "diff_gap";
    pub const DIFF_WIDTH: &str = "diff_width";
    pub const DIFF_UNCOUPLED_MAX: &str = "diff_uncoupled_max";
    pub const LENGTH_MIN: &str = "length_min";
    pub const LENGTH_MAX: &str = "length_max";
    pub const SKEW_MAX: &str = "skew_max";
    pub const MATCH_GROUP: &str = "match_group";
    pub const IMPEDANCE: &str = "impedance";
    pub const THERMAL_GAP: &str = "thermal_gap";
    pub const SPOKE_WIDTH: &str = "spoke_width";
    pub const SPOKE_MIN: &str = "spoke_min";
    pub const ZONE_CONNECT: &str = "zone_connect";
    // Pair properties.
    pub const CLEARANCE: &str = "clearance";
    pub const HOLE_TO_HOLE: &str = "hole_to_hole";
    pub const SILK_CLEARANCE: &str = "silk_clearance";
    pub const COURTYARD_CLEARANCE: &str = "courtyard_clearance";
    pub const EDGE_CLEARANCE: &str = "edge_clearance";
}

/// The id of the hypothetical track a router is about to place (see [`BoardGraph::probe`]).
pub const PROBE_TRACK: ItemId = ItemId(u64::MAX);

/// A track not on the board yet: net, layer and width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackProbe {
    pub net: Option<NetId>,
    pub layer: u8,
    pub width: Coord,
}

/// The board as an item graph, plus derived membership the pseudo-classes need.
pub struct BoardGraph<'a> {
    pub board: &'a Board,
    pub diff_nets: BTreeSet<u32>,
    /// Answers for `BItem::Track(PROBE_TRACK)` without adding it to the board.
    pub probe: Option<TrackProbe>,
    items: Vec<BItem>,
    names: BTreeMap<BItem, String>,
    room_of: BTreeMap<ItemId, Vec<String>>,
}

impl<'a> BoardGraph<'a> {
    pub fn new(board: &'a Board, diff_pairs: &[(NetId, NetId)]) -> Self {
        let mut items = vec![BItem::Edge];
        let mut names = BTreeMap::new();
        names.insert(BItem::Edge, "edge".to_string());
        for (i, n) in board.nets.iter().enumerate() {
            items.push(BItem::Net(i as u32));
            names.insert(BItem::Net(i as u32), n.name.clone());
        }
        for f in board.footprints.values() {
            items.push(BItem::Footprint(f.id));
            names.insert(BItem::Footprint(f.id), f.reference.clone());
            for (k, p) in f.package.pads.iter().enumerate() {
                let it = BItem::Pad(f.id, k as u32);
                items.push(it);
                names.insert(it, format!("{}.{}", f.reference, p.name));
            }
        }
        items.extend(board.tracks.keys().map(|k| BItem::Track(*k)));
        items.extend(board.vias.keys().map(|k| BItem::Via(*k)));
        for z in board.zones.values() {
            items.push(BItem::Zone(z.id));
            names.insert(BItem::Zone(z.id), z.name.clone());
        }
        let mut room_of: BTreeMap<ItemId, Vec<String>> = BTreeMap::new();
        for r in &board.rooms {
            for f in board.footprints.values() {
                let inside = r
                    .members
                    .iter()
                    .any(|m| m == &f.reference || Some(m) == f.component.as_ref())
                    || r.outline.contains(f.at) == Some(true);
                if inside {
                    room_of.entry(f.id).or_default().push(r.name.clone());
                }
            }
        }
        let mut diff_nets = BTreeSet::new();
        for (p, n) in diff_pairs {
            diff_nets.insert(p.0);
            diff_nets.insert(n.0);
        }
        BoardGraph {
            board,
            diff_nets,
            probe: None,
            items,
            names,
            room_of,
        }
    }

    /// The graph answering also for a probe track (not listed in `items`).
    pub fn probe(mut self, p: TrackProbe) -> Self {
        self.probe = Some(p);
        self
    }

    fn probe_of(&self, it: BItem) -> Option<TrackProbe> {
        match it {
            BItem::Track(PROBE_TRACK) => self.probe,
            _ => None,
        }
    }

    pub fn net_of(&self, it: BItem) -> Option<NetId> {
        if let Some(p) = self.probe_of(it) {
            return p.net;
        }
        let b = self.board;
        match it {
            BItem::Net(n) => Some(NetId(n)),
            BItem::Pad(f, k) => {
                let fp = &b.footprints[&f];
                fp.pad_nets.get(&fp.package.pads[k as usize].name).copied()
            }
            BItem::Track(t) => b.tracks[&t].net,
            BItem::Via(v) => b.vias[&v].net,
            BItem::Zone(z) => b.zones[&z].net,
            _ => None,
        }
    }

    fn layers_of(&self, it: BItem) -> Vec<u8> {
        if let Some(p) = self.probe_of(it) {
            return vec![p.layer];
        }
        let b = self.board;
        let n = b.copper_count();
        match it {
            BItem::Track(t) => vec![b.tracks[&t].layer],
            BItem::Via(v) => b.vias[&v].layers().collect(),
            BItem::Zone(z) => b.zones[&z].layers.clone(),
            BItem::Pad(f, k) => {
                let fp = &b.footprints[&f];
                fp.pad(k as usize, n)
                    .map(|p| p.layers.into_iter().collect())
                    .unwrap_or_default()
            }
            BItem::Footprint(f) => {
                vec![if b.footprints[&f].side == crate::layout::Side::Top {
                    0
                } else {
                    n.saturating_sub(1)
                }]
            }
            _ => vec![],
        }
    }

    pub fn copper_items(&self) -> impl Iterator<Item = BItem> + '_ {
        self.items.iter().copied().filter(|i| {
            matches!(
                i,
                BItem::Pad(..) | BItem::Track(_) | BItem::Via(_) | BItem::Zone(_)
            )
        })
    }
}

fn len_attr(v: Coord) -> AttrValue {
    AttrValue::Num(Quantity::length_m(v as f64 * 1e-9))
}

impl ItemGraph for BoardGraph<'_> {
    type Id = BItem;
    fn items(&self) -> Vec<BItem> {
        self.items.clone()
    }
    fn kind(&self, id: BItem) -> &str {
        match id {
            BItem::Edge => "edge",
            BItem::Net(_) => "net",
            BItem::Footprint(_) => "footprint",
            BItem::Pad(..) => "pad",
            BItem::Track(PROBE_TRACK) if self.probe.is_some() => "track",
            BItem::Track(t) => match self.board.tracks[&t].geom {
                crate::layout::TrackGeom::Segment { .. } => "track",
                crate::layout::TrackGeom::Arc { .. } => "arc",
            },
            BItem::Via(_) => "via",
            BItem::Zone(_) => "zone",
        }
    }
    fn is_kind(&self, id: BItem, kind: &str) -> bool {
        let k = self.kind(id);
        k == kind
            || (kind == "copper"
                && matches!(
                    id,
                    BItem::Pad(..) | BItem::Track(_) | BItem::Via(_) | BItem::Zone(_)
                ))
            || (kind == "track" && k == "arc")
            || (kind == "component" && k == "footprint")
    }
    fn name(&self, id: BItem) -> Option<&str> {
        self.names.get(&id).map(|s| s.as_str())
    }
    fn has_class(&self, id: BItem, c: &str) -> bool {
        let b = self.board;
        match id {
            BItem::Net(n) => b.nets[n as usize].class.as_deref() == Some(c),
            BItem::Footprint(f) => {
                let fp = &b.footprints[&f];
                match c {
                    "top" => fp.side == crate::layout::Side::Top,
                    "bottom" => fp.side == crate::layout::Side::Bottom,
                    "smd" => fp.package.smd,
                    "tht" => !fp.package.smd,
                    _ => false,
                }
            }
            BItem::Pad(f, k) => {
                let p = &b.footprints[&f].package.pads[k as usize];
                match c {
                    "smd" => p.kind == PadKind::Smd,
                    "tht" => p.kind == PadKind::Tht,
                    "npth" => p.kind == PadKind::Npth,
                    "connector" => p.kind == PadKind::Connector,
                    "thermal" => p.thermal_pad,
                    _ => false,
                }
            }
            BItem::Via(v) => {
                let k = b.vias[&v].kind(b.copper_count());
                matches!(
                    (k, c),
                    (ViaKind::Through, "through")
                        | (ViaKind::Blind, "blind")
                        | (ViaKind::Buried, "buried")
                        | (ViaKind::Micro, "micro")
                )
            }
            _ => false,
        }
    }
    fn attr(&self, id: BItem, name: &str) -> Option<AttrValue> {
        let b = self.board;
        let net_name = || self.net_of(id).map(|n| AttrValue::str(b.net_name(n)));
        match (id, name) {
            (_, "net") => net_name(),
            (BItem::Net(n), "class") => b.nets[n as usize].class.clone().map(AttrValue::Str),
            (BItem::Net(n), "name") => Some(AttrValue::str(&b.nets[n as usize].name)),
            (BItem::Footprint(f), _) => {
                let fp = &b.footprints[&f];
                match name {
                    "ref" => Some(AttrValue::str(&fp.reference)),
                    "value" => Some(AttrValue::str(&fp.value)),
                    "package" => Some(AttrValue::str(&fp.package.name)),
                    "side" => Some(AttrValue::str(if fp.side == crate::layout::Side::Top {
                        "top"
                    } else {
                        "bottom"
                    })),
                    "locked" => Some(AttrValue::Bool(fp.locked)),
                    other => fp.properties.get(other).map(AttrValue::str),
                }
            }
            (BItem::Pad(f, k), _) => {
                let p = &b.footprints[&f].package.pads[k as usize];
                let (w, h) = p.shape.size();
                match name {
                    "name" => Some(AttrValue::str(&p.name)),
                    "width" => Some(len_attr(w)),
                    "height" => Some(len_attr(h)),
                    "drill" => p.drill.as_ref().map(|d| len_attr(d.diameter)),
                    _ => None,
                }
            }
            (BItem::Track(PROBE_TRACK), _) if self.probe.is_some() => {
                let p = self.probe.unwrap();
                match name {
                    "width" => Some(len_attr(p.width)),
                    "layer" => Some(AttrValue::str(crate::layout::copper_name(
                        p.layer,
                        b.copper_count(),
                    ))),
                    _ => None,
                }
            }
            (BItem::Track(t), _) => {
                let tr = &b.tracks[&t];
                match name {
                    "width" => Some(len_attr(tr.width)),
                    "layer" => Some(AttrValue::str(crate::layout::copper_name(
                        tr.layer,
                        b.copper_count(),
                    ))),
                    "length" => Some(len_attr(tr.length() as Coord)),
                    _ => None,
                }
            }
            (BItem::Via(v), _) => {
                let via = &b.vias[&v];
                match name {
                    "diameter" => Some(len_attr(via.diameter)),
                    "drill" => Some(len_attr(via.drill)),
                    "from" => Some(AttrValue::Num(Quantity::scalar(via.from as f64))),
                    "to" => Some(AttrValue::Num(Quantity::scalar(via.to as f64))),
                    _ => None,
                }
            }
            (BItem::Zone(z), _) => {
                let zz = &b.zones[&z];
                match name {
                    "priority" => Some(AttrValue::Num(Quantity::scalar(zz.priority as f64))),
                    "name" => Some(AttrValue::str(&zz.name)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn parent(&self, id: BItem) -> Option<BItem> {
        match id {
            BItem::Pad(f, _) => Some(BItem::Footprint(f)),
            _ => None,
        }
    }
    fn children(&self, id: BItem) -> Vec<BItem> {
        match id {
            BItem::Footprint(f) => (0..self.board.footprints[&f].package.pads.len() as u32)
                .map(|k| BItem::Pad(f, k))
                .collect(),
            _ => vec![],
        }
    }
    fn pseudo(&self, id: BItem, name: &str, args: &[PseudoArg]) -> Option<bool> {
        let b = self.board;
        let n = b.copper_count();
        match name {
            "net-class" => {
                let class = self
                    .net_of(id)
                    .and_then(|n| b.nets[n.0 as usize].class.clone());
                Some(args.iter().any(|a| Some(&a.text) == class.as_ref()))
            }
            "net" => {
                let Some(net) = self.net_of(id) else {
                    return Some(false);
                };
                let nm = b.net_name(net);
                Some(args.iter().any(|a| match a.text.strip_suffix('*') {
                    Some(p) => nm.starts_with(p),
                    None => a.text == nm,
                }))
            }
            "layer" => {
                let ls = self.layers_of(id);
                Some(args.iter().any(|a| {
                    ls.iter().any(|&l| match a.text.as_str() {
                        "outer" => l == 0 || l + 1 == n,
                        "inner" => l > 0 && l + 1 < n,
                        "top" => l == 0,
                        "bottom" => l + 1 == n,
                        t => crate::layout::copper_name(l, n) == t,
                    })
                }))
            }
            "side" => {
                let side = match id {
                    BItem::Footprint(f) | BItem::Pad(f, _) => b.footprints[&f].side,
                    _ => return Some(false),
                };
                Some(args.iter().any(|a| {
                    (a.text == "top" && side == crate::layout::Side::Top)
                        || (a.text == "bottom" && side == crate::layout::Side::Bottom)
                }))
            }
            "diff-pair" => Some(
                self.net_of(id)
                    .is_some_and(|n| self.diff_nets.contains(&n.0)),
            ),
            "in-room" => {
                let f = match id {
                    BItem::Footprint(f) | BItem::Pad(f, _) => f,
                    _ => return Some(false),
                };
                let rooms = self.room_of.get(&f);
                Some(
                    args.iter()
                        .any(|a| rooms.is_some_and(|r| r.contains(&a.text))),
                )
            }
            _ => None,
        }
    }
}

/// The schema rule selectors are validated against.
pub fn schema() -> Schema {
    let len = runmat_rules::AttrType::Num(Dim::LENGTH);
    Schema::new()
        .kinds(&[
            "edge",
            "net",
            "footprint",
            "component",
            "pad",
            "track",
            "arc",
            "via",
            "zone",
            "copper",
        ])
        .attr("net", runmat_rules::AttrType::Str)
        .attr("class", runmat_rules::AttrType::Str)
        .attr("name", runmat_rules::AttrType::Str)
        .attr("ref", runmat_rules::AttrType::Str)
        .attr("value", runmat_rules::AttrType::Str)
        .attr("package", runmat_rules::AttrType::Str)
        .attr("side", runmat_rules::AttrType::Str)
        .attr("locked", runmat_rules::AttrType::Bool)
        .attr("width", len)
        .attr("height", len)
        .attr("drill", len)
        .attr("diameter", len)
        .attr("length", len)
        .attr("layer", runmat_rules::AttrType::Str)
        .attr("from", runmat_rules::AttrType::Num(Dim::NONE))
        .attr("to", runmat_rules::AttrType::Num(Dim::NONE))
        .attr("priority", runmat_rules::AttrType::Num(Dim::NONE))
        .pseudo("net-class", 1, 32)
        .pseudo("net", 1, 64)
        .pseudo("layer", 1, 64)
        .pseudo("side", 1, 1)
        .pseudo("diff-pair", 0, 0)
        .pseudo("in-room", 1, 8)
}

/// A rule as authored (front ends build these).
#[derive(Clone, Debug, PartialEq)]
pub struct RuleSpec {
    pub id: String,
    pub selector: String,
    /// Second selector for pair rules.
    pub other: Option<String>,
    pub priority: i32,
    pub source: SourceKey,
    pub order: SourceOrder,
    pub decls: Vec<(String, RuleValue)>,
}

impl RuleSpec {
    pub fn single(id: &str, selector: &str) -> Self {
        RuleSpec {
            id: id.into(),
            selector: selector.into(),
            other: None,
            priority: 0,
            source: SourceKey::default(),
            order: SourceOrder::default(),
            decls: vec![],
        }
    }
    pub fn pair(id: &str, a: &str, b: &str) -> Self {
        RuleSpec {
            other: Some(b.into()),
            ..RuleSpec::single(id, a)
        }
    }
    pub fn set(mut self, p: &str, v: RuleValue) -> Self {
        self.decls.push((p.into(), v));
        self
    }
    pub fn at(mut self, source: SourceKey, order: SourceOrder) -> Self {
        self.source = source;
        self.order = order;
        self
    }
    pub fn priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }
}

/// Compiled board rules: single-item cascade plus pair cascade.
pub struct BoardRules {
    pub single: Cascade<RuleValue, BItem>,
    pub pair: PairCascade<RuleValue, BItem>,
    pub specs: Vec<RuleSpec>,
    /// The rule a property falls back to when no rule sets it: the layout's `limits` value
    /// (`limits.clearance` at its text) or, when the layout doesn't write it, RunMat's default
    /// rule set (`default.clearance`, no source).
    pub defaults: BTreeMap<&'static str, (String, SourceKey)>,
}

/// The `limits` key behind each property's default, if a limit sets it.
pub const LIMIT_OF: [(&str, &str); 7] = [
    (props::CLEARANCE, "clearance"),
    (props::TRACK_WIDTH_MIN, "track"),
    (props::HOLE_MIN, "drill"),
    (props::ANNULAR_MIN, "annular"),
    (props::HOLE_TO_HOLE, "hole_to_hole"),
    (props::EDGE_CLEARANCE, "copper_to_edge"),
    (props::SILK_CLEARANCE, "silk_to_pad"),
];

/// The name of RunMat's default rule set, the origin of every value a layout doesn't write.
pub const DEFAULT_RULE_SET: &str = "RunMat default limits";

fn default_origins(limits: &FabLimits) -> BTreeMap<&'static str, (String, SourceKey)> {
    let defaults = [
        props::CLEARANCE,
        props::TRACK_WIDTH_MIN,
        props::TRACK_WIDTH,
        props::TRACK_WIDTH_MAX,
        props::VIA_DIAMETER,
        props::VIA_DRILL,
        props::ANNULAR_MIN,
        props::HOLE_MIN,
        props::VIA_TYPES,
        props::SPOKE_MIN,
        props::HOLE_TO_HOLE,
        props::SILK_CLEARANCE,
        props::COURTYARD_CLEARANCE,
        props::EDGE_CLEARANCE,
    ];
    defaults
        .into_iter()
        .map(|prop| {
            let written = LIMIT_OF
                .iter()
                .find(|(p, _)| *p == prop)
                .and_then(|(_, key)| Some((*key, limits.sources.get(*key)?)));
            let origin = match written {
                Some((key, src)) => (format!("limits.{key}"), src.clone()),
                None => (format!("default.{prop}"), SourceKey::default()),
            };
            (prop, origin)
        })
        .collect()
}

impl BoardRules {
    /// Compiles rules; selector errors (syntax or schema) are returned per rule id.
    pub fn new(
        limits: &FabLimits,
        specs: &[RuleSpec],
    ) -> (BoardRules, Vec<(String, SelectorError)>) {
        let u = BuiltinUnits;
        let schema = schema();
        let mut single: Cascade<RuleValue, BItem> = Cascade::new();
        let mut pair: PairCascade<RuleValue, BItem> = PairCascade::new();
        let l = |v| Some(RuleValue::Len(v));
        single.register_property(props::TRACK_WIDTH_MIN, false, l(limits.min_track));
        single.register_property(
            props::TRACK_WIDTH,
            false,
            l(limits.min_track.max(crate::geom::mm(0.2))),
        );
        single.register_property(props::TRACK_WIDTH_MAX, false, None);
        single.register_property(
            props::VIA_DIAMETER,
            false,
            l(limits.min_via_diameter.max(crate::geom::mm(0.6))),
        );
        single.register_property(
            props::VIA_DRILL,
            false,
            l(limits.min_drill.max(crate::geom::mm(0.3))),
        );
        single.register_property(props::ANNULAR_MIN, false, l(limits.min_annular));
        single.register_property(props::HOLE_MIN, false, l(limits.min_drill));
        single.register_property(
            props::VIA_TYPES,
            false,
            Some(RuleValue::ViaKinds(vec![
                ViaKind::Through,
                ViaKind::Blind,
                ViaKind::Buried,
                ViaKind::Micro,
            ])),
        );
        single.register_property(props::DIFF_GAP, false, None);
        single.register_property(props::DIFF_WIDTH, false, None);
        single.register_property(props::DIFF_UNCOUPLED_MAX, false, None);
        single.register_property(props::LENGTH_MIN, false, None);
        single.register_property(props::LENGTH_MAX, false, None);
        single.register_property(props::SKEW_MAX, false, None);
        single.register_property(props::MATCH_GROUP, false, None);
        single.register_property(props::IMPEDANCE, false, None);
        single.register_property(props::THERMAL_GAP, true, None);
        single.register_property(props::SPOKE_WIDTH, true, None);
        single.register_property(props::SPOKE_MIN, true, Some(RuleValue::Int(2)));
        single.register_property(props::ZONE_CONNECT, true, None);
        pair.set_default(props::CLEARANCE, RuleValue::Len(limits.min_clearance));
        pair.set_default(props::HOLE_TO_HOLE, RuleValue::Len(limits.min_hole_to_hole));
        pair.set_default(
            props::SILK_CLEARANCE,
            RuleValue::Len(limits.min_silk_to_pad),
        );
        pair.set_default(props::COURTYARD_CLEARANCE, RuleValue::Len(0));
        pair.set_default(
            props::EDGE_CLEARANCE,
            RuleValue::Len(limits.min_copper_to_edge),
        );
        let mut errs = Vec::new();
        for s in specs {
            let parse =
                |t: &str, errs: &mut Vec<(String, SelectorError)>| match Selector::parse(t, &u) {
                    Ok(sel) => {
                        let v = sel.validate(&schema);
                        if v.is_empty() {
                            Some(sel)
                        } else {
                            errs.extend(v.into_iter().map(|e| (s.id.clone(), e)));
                            None
                        }
                    }
                    Err(e) => {
                        errs.push((s.id.clone(), e));
                        None
                    }
                };
            let Some(a) = parse(&s.selector, &mut errs) else {
                continue;
            };
            match &s.other {
                None => {
                    let mut r = Rule::new(&s.id, a)
                        .priority(s.priority)
                        .at(s.source.clone(), s.order);
                    for (p, v) in &s.decls {
                        r = r.set(p.clone(), v.clone());
                    }
                    single.push_rule(r);
                }
                Some(o) => {
                    let Some(b) = parse(o, &mut errs) else {
                        continue;
                    };
                    let mut r = PairRule::new(&s.id, a, b)
                        .priority(s.priority)
                        .at(s.source.clone(), s.order);
                    for (p, v) in &s.decls {
                        r = r.set(p.clone(), v.clone());
                    }
                    pair.push_rule(r);
                }
            }
        }
        (
            BoardRules {
                single,
                pair,
                specs: specs.to_vec(),
                defaults: default_origins(limits),
            },
            errs,
        )
    }

    pub fn rebuild(&mut self, g: &BoardGraph) {
        self.single.rebuild(g);
        self.pair.rebuild(g);
    }

    pub fn len(&self, g: &BoardGraph, it: BItem, prop: &str) -> Option<Coord> {
        self.single
            .resolve(g, it, prop)
            .and_then(|r| r.value.as_len())
    }
    pub fn value(&self, g: &BoardGraph, it: BItem, prop: &str) -> Option<RuleValue> {
        self.single.resolve(g, it, prop).map(|r| r.value)
    }
    pub fn pair_len(&self, a: BItem, b: BItem, prop: &str) -> Coord {
        self.pair_len_id(a, b, self.pair.prop_id(prop))
    }
    /// [`BoardRules::pair_len`] by interned property id: a memo lookup, no allocation.
    pub fn pair_len_id(&self, a: BItem, b: BItem, prop: runmat_rules::PropId) -> Coord {
        self.pair.with_resolved(a, b, prop, |r| {
            r.and_then(|r| r.value.as_len()).unwrap_or(0)
        })
    }
    /// The rule matches of a track about to be placed (for routers).
    pub fn probe(&self, board: &Board, p: TrackProbe) -> runmat_rules::Probe {
        self.pair.probe(
            &BoardGraph::new(board, &[]).probe(p),
            BItem::Track(PROBE_TRACK),
        )
    }
    /// A pair length between a probed track and an item.
    pub fn probe_len(&self, p: &runmat_rules::Probe, other: BItem, prop: &str) -> Coord {
        self.pair
            .resolve_probe(p, other, prop)
            .and_then(|r| r.value.as_len())
            .unwrap_or(0)
    }
    /// Largest clearance any rule can require: the spatial query radius.
    pub fn max_clearance(&self) -> Coord {
        self.pair
            .max_value_by(props::CLEARANCE, |v| v.as_len().unwrap_or(0))
            .unwrap_or(0)
    }
    /// The rule id and source that set a pair value, for findings.
    pub fn pair_origin(&self, a: BItem, b: BItem, prop: &str) -> (String, SourceKey) {
        match self.pair.resolve(a, b, prop).and_then(|r| r.rule) {
            Some(w) => (w.rule_id, w.source),
            None => self.default_origin(prop),
        }
    }
    pub fn single_origin(&self, g: &BoardGraph, it: BItem, prop: &str) -> (String, SourceKey) {
        match self.single.resolve(g, it, prop).map(|r| r.origin) {
            Some(runmat_rules::Origin::Rule {
                rule_id, source, ..
            }) => (rule_id, source),
            Some(runmat_rules::Origin::Inherited { origin, .. }) => match *origin {
                runmat_rules::Origin::Rule {
                    rule_id, source, ..
                } => (rule_id, source),
                _ => self.default_origin(prop),
            },
            _ => self.default_origin(prop),
        }
    }
    /// The origin of `prop`'s value when no rule sets it (see [`BoardRules::defaults`]).
    pub fn default_origin(&self, prop: &str) -> (String, SourceKey) {
        self.defaults
            .get(prop)
            .cloned()
            .unwrap_or_else(|| (format!("default.{prop}"), SourceKey::default()))
    }
    /// Whether `rule` is a fallback value (a `limits` entry or RunMat's default), not a rule.
    pub fn is_fallback(rule: &str) -> bool {
        rule.starts_with("limits.") || rule.starts_with("default.")
    }
}

/// Layer of a technical item, for silk checks.
pub fn silk_layer(side: crate::layout::Side) -> Layer {
    Layer::Silk(side)
}

/// Helper so callers can name points in rules tests.
pub fn pt(x: f64, y: f64) -> Point {
    Point::mm(x, y)
}

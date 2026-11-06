//! Printing a board as `.circuit.layout` text.

use super::*;
use super::super::layout::{print_sync, SyncSection};
use crate::geom::poly::Polygon;
use crate::ids::Uuid;
use crate::library::{
    Package,
};
use runmat_design_syntax::document::UseItem;
use runmat_design_syntax::json::INDENT;
use runmat_rules::{SourceKey, SourceOrder};
use std::collections::BTreeMap;

/* ------------------------------------------------------------------------------------------------
 * Printing
 * --------------------------------------------------------------------------------------------- */

pub(super) fn net_head(board: &Board, word: &str, net: Option<NetId>) -> String {
    match net {
        Some(n) => format!("{word} {} =", name_text(board.net_name(n))),
        None => format!("{word} ="),
    }
}

pub(super) fn polygon_fields(r: &mut Rec, p: &Polygon) {
    r.put(words::OUTLINE, ring_text(&p.outer));
    if !p.holes.is_empty() {
        r.put("holes", rings_text(&p.holes));
    }
}

/// The package key each footprint prints: its package name, or `<name>#<n>` when another
/// footprint uses a different package of the same name.
pub(super) fn package_keys(board: &Board) -> (BTreeMap<ItemId, String>, Vec<(String, Package)>) {
    let mut keys = BTreeMap::new();
    let mut defined: Vec<(String, Package)> = Vec::new();
    for f in board.footprints.values() {
        let p = &f.package;
        let key = match defined.iter().find(|(_, x)| x.name == p.name && x == p) {
            Some((k, _)) => k.clone(),
            None => {
                let taken = |k: &str| defined.iter().any(|(x, _)| x == k);
                let mut key = p.name.clone();
                let mut n = 1;
                while taken(&key) {
                    n += 1;
                    key = format!("{}#{n}", p.name);
                }
                defined.push((key.clone(), p.clone()));
                key
            }
        };
        keys.insert(f.id, key);
    }
    (keys, defined)
}

pub(super) fn stackup_text(s: &Stackup) -> String {
    let inner = INDENT.repeat(2);
    let mut out = format!("{} {} {{\n", words::STACKUP, name_text(&s.name));
    if !s.finish.is_empty() {
        out.push_str(&format!(
            "{inner}{} = {}\n",
            words::FINISH,
            name_text(&s.finish)
        ));
    }
    for l in &s.layers {
        let mut r = Rec::new();
        match l.kind {
            StackKind::Copper { weight_oz } => {
                r.put("kind", "copper");
                r.len("thickness", l.thickness);
                r.put("weight", f64_text(weight_oz));
            }
            StackKind::Dielectric { core } => {
                r.put("kind", if core { "core" } else { "prepreg" });
                r.len("thickness", l.thickness);
            }
            StackKind::SolderMask => {
                r.put("kind", "mask").len("thickness", l.thickness);
            }
            StackKind::Silkscreen => {
                r.put("kind", "silk").len("thickness", l.thickness);
            }
            StackKind::Paste => {
                r.put("kind", "paste").len("thickness", l.thickness);
            }
        }
        r.put("material", material_text(&l.material));
        out.push_str(&format!(
            "{inner}{} {} = {}\n",
            words::LAYER,
            name_text(&l.name),
            r.text()
        ));
    }
    out.push_str(INDENT);
    out.push('}');
    out
}

pub(super) fn package_text(key: &str, p: &Package) -> String {
    let inner = INDENT.repeat(2);
    let mut out = format!("{} {} {{\n", words::PACKAGE, name_text(key));
    let mut line = |head: String, value: String| {
        out.push_str(&inner);
        out.push_str(&head);
        out.push_str(" = ");
        out.push_str(&value);
        out.push('\n');
    };
    if p.name != key {
        line(words::NAME.into(), name_text(&p.name));
    }
    if p.uuid != Uuid::NIL {
        line(
            words::UUID.into(),
            crate::syntax::values::name_text(&p.uuid.to_string()),
        );
    }
    if !p.description.is_empty() {
        line(words::DESCRIPTION.into(), name_text(&p.description));
    }
    if p.smd {
        line(words::SMD.into(), "true".into());
    }
    for pad in &p.pads {
        line(
            format!("{} {}", words::PAD, name_text(&pad.name)),
            pad_text(pad),
        );
    }
    for ring in &p.courtyard {
        line(words::COURTYARD.into(), ring_text(ring));
    }
    for (word, graphics) in [(words::SILK, &p.silk), (words::FAB, &p.fab)] {
        for g in graphics {
            let (kind, text) = graphic_text(g, Rec::new());
            line(format!("{word} {kind}"), text);
        }
    }
    if let Some(m) = &p.model {
        let mut r = Rec::new();
        r.put("path", name_text(&m.path));
        if m.offset != (0.0, 0.0, 0.0) {
            r.put("offset", floats3_text(m.offset));
        }
        if m.rotation != (0.0, 0.0, 0.0) {
            r.put("rotation", floats3_text(m.rotation));
        }
        if m.scale != (1.0, 1.0, 1.0) {
            r.put("scale", floats3_text(m.scale));
        }
        if m.origin != "footprint" {
            r.put("origin", name_text(&m.origin));
        }
        line(words::MODEL.into(), r.text());
    }
    if let Some(b) = &p.body {
        let mut r = Rec::new();
        r.len("width", b.width)
            .len("length", b.length)
            .len("height", b.height);
        line(words::BODY.into(), r.text());
    }
    out.push_str(INDENT);
    out.push('}');
    out
}

/// Every statement of a board, in print order, with its site.
pub fn statements(board: &Board) -> Vec<Printed> {
    statements_where(board, &|_| true)
}

/// [`statements`], with the text of only the items `want` names: the others carry their
/// group, site and item with empty text.
pub fn statements_where(board: &Board, want: &dyn Fn(ItemId) -> bool) -> Vec<Printed> {
    let mut out: Vec<Printed> = Vec::new();
    let mut sites = Sites::default();
    let layers = Layers::of(&board.stackup);
    let mut push = |group: Group, site: String, text: String, item: Option<ItemId>| {
        out.push(Printed {
            group,
            site,
            text,
            item,
        })
    };
    for s in std::iter::once(&board.stackup).chain(board.stackups.values()) {
        push(
            Group::Stackup,
            sites.named(words::STACKUP, &s.name),
            stackup_text(s),
            None,
        );
    }
    for r in &board.regions {
        let mut rec = Rec::new();
        rec.put(words::STACKUP, name_text(&r.stackup));
        polygon_fields(&mut rec, &r.outline);
        if !r.bends.is_empty() {
            let bends: Vec<String> = r
                .bends
                .iter()
                .map(|b| {
                    let mut x = Rec::new();
                    x.pt("from", b.a)
                        .pt("to", b.b)
                        .len("radius", b.radius)
                        .put("angle", f64_text(b.angle));
                    if b.keepout != 0 {
                        x.len("keepout", b.keepout);
                    }
                    x.text()
                })
                .collect();
            rec.put("bends", format!("[{}]", bends.join(", ")));
        }
        push(
            Group::Region,
            sites.named(words::REGION, &r.name),
            format!("{} {} = {}", words::REGION, name_text(&r.name), rec.text()),
            None,
        );
    }
    if !board.outline.outer.is_empty() {
        push(
            Group::Outline,
            "outline".into(),
            format!("{} = {}", words::OUTLINE, ring_text(&board.outline.outer)),
            None,
        );
    }
    for hole in &board.outline.holes {
        push(
            Group::Outline,
            sites.indexed(words::CUTOUT),
            format!("{} = {}", words::CUTOUT, ring_text(hole)),
            None,
        );
    }
    {
        // A capability set is written by name, with only the keys that differ from it.
        let mut l = board.limits.clone();
        let mut set = l.capability.map(|c| c.limits());
        let mut r = Rec::new();
        if let Some(c) = l.capability {
            r.put(CAPABILITY, c.word());
        }
        let base: Vec<Option<crate::geom::Coord>> = match set.as_mut() {
            Some(set) => limit_fields(set).into_iter().map(|v| Some(*v)).collect(),
            None => vec![None; LIMIT_KEYS.len()],
        };
        for ((key, field), base) in LIMIT_KEYS.iter().zip(limit_fields(&mut l)).zip(base) {
            if base != Some(*field) {
                r.len(key, *field);
            }
        }
        push(
            Group::Limits,
            "limits".into(),
            format!("{} = {}", words::LIMITS, r.text()),
            None,
        );
    }
    if let Some(settings) = &board.fabrication {
        push(
            Group::Limits,
            words::FABRICATION.into(),
            format!("{} = {}", words::FABRICATION, fabrication_text(settings)),
            None,
        );
    }
    for (class, c) in &board.net_class_rules {
        let mut r = Rec::new();
        for (k, v) in &c.decls {
            r.put(k, rule_value_text(v));
        }
        push(
            Group::NetClass,
            sites.named(words::NET_CLASS, class),
            format!("{} {} = {}", words::NET_CLASS, name_text(class), r.text()),
            None,
        );
    }
    for rule in &board.rules {
        let mut r = Rec::new();
        match &rule.other {
            None => r.put("select", name_text(&rule.selector)),
            Some(o) => r.put(
                "between",
                format!("[{}, {}]", name_text(&rule.selector), name_text(o)),
            ),
        };
        if rule.priority != 0 {
            r.put("priority", rule.priority.to_string());
        }
        for (k, v) in &rule.decls {
            r.put(k, rule_value_text(v));
        }
        push(
            Group::Rule,
            sites.named(words::RULE, &rule.id),
            format!("{} {} = {}", words::RULE, name_text(&rule.id), r.text()),
            None,
        );
    }
    for n in &board.nets {
        let text = match &n.class {
            Some(c) => format!("{} {} : {}", words::NET, name_text(&n.name), name_text(c)),
            None => format!("{} {}", words::NET, name_text(&n.name)),
        };
        push(Group::Net, sites.named(words::NET, &n.name), text, None);
    }
    for room in &board.rooms {
        let mut r = Rec::new();
        polygon_fields(&mut r, &room.outline);
        if !room.members.is_empty() {
            let m: Vec<String> = room.members.iter().map(|m| name_text(m)).collect();
            r.put("members", format!("[{}]", m.join(", ")));
        }
        push(
            Group::Room,
            sites.named(words::ROOM, &room.name),
            format!("{} {} = {}", words::ROOM, name_text(&room.name), r.text()),
            None,
        );
    }
    let (keys, packages) = package_keys(board);
    for f in board.footprints.values() {
        if !want(f.id) {
            let site = sites.named(words::FOOTPRINT, &f.reference);
            push(Group::Footprint, site, String::new(), Some(f.id));
            continue;
        }
        let mut r = Rec::new();
        r.put(words::PACKAGE, name_text(&keys[&f.id]))
            .pt("at", f.at);
        if f.rotation != 0.0 {
            r.put("rotation", f64_text(f.rotation));
        }
        if f.side != Side::Top {
            r.put("side", side_name(f.side));
        }
        r.flag("locked", f.locked);
        if !f.value.is_empty() {
            r.put("value", name_text(&f.value));
        }
        if !f.fitted {
            r.put("fitted", "false");
        }
        if !f.pad_nets.is_empty() {
            let mut pads = Rec::new();
            for (pad, net) in &f.pad_nets {
                pads.put(pad, name_text(board.net_name(*net)));
            }
            r.put("pads", pads.text());
        }
        if !f.properties.is_empty() {
            let mut props = Rec::new();
            for (k, v) in &f.properties {
                props.put(k, name_text(v));
            }
            r.put("properties", props.text());
        }
        push(
            Group::Footprint,
            sites.named(words::FOOTPRINT, &f.reference),
            format!(
                "{} {} = {}",
                words::FOOTPRINT,
                name_text(&f.reference),
                r.text()
            ),
            Some(f.id),
        );
    }
    for fid in board.fiducials.values() {
        if !want(fid.id) {
            let site = sites.named(words::FIDUCIAL, &fid.name);
            push(Group::Fiducial, site, String::new(), Some(fid.id));
            continue;
        }
        let mut r = Rec::new();
        r.pt("at", fid.at);
        if fid.side != Side::Top {
            r.put("side", side_name(fid.side));
        }
        r.len("diameter", fid.diameter)
            .len("opening", fid.opening)
            .flag("locked", fid.locked);
        push(
            Group::Fiducial,
            sites.named(words::FIDUCIAL, &fid.name),
            format!(
                "{} {} = {}",
                words::FIDUCIAL,
                name_text(&fid.name),
                r.text()
            ),
            Some(fid.id),
        );
    }
    for t in board.tracks.values() {
        if !want(t.id) {
            let word = match t.geom {
                TrackGeom::Segment { .. } => words::TRACK,
                TrackGeom::Arc { .. } => words::ARC,
            };
            push(
                Group::Copper,
                sites.indexed(word),
                String::new(),
                Some(t.id),
            );
            continue;
        }
        let mut r = Rec::new();
        r.put("layer", layers.copper_text(t.layer))
            .len("width", t.width);
        let word = match t.geom {
            TrackGeom::Segment { a, b } => {
                r.pt("from", a).pt("to", b);
                words::TRACK
            }
            TrackGeom::Arc { start, mid, end } => {
                r.pt("from", start).pt("mid", mid).pt("to", end);
                words::ARC
            }
        };
        r.flag("locked", t.locked);
        push(
            Group::Copper,
            sites.indexed(word),
            format!("{} {}", net_head(board, word, t.net), r.text()),
            Some(t.id),
        );
    }
    let n = board.copper_count();
    for v in board.vias.values() {
        if !want(v.id) {
            push(
                Group::Via,
                sites.indexed(words::VIA),
                String::new(),
                Some(v.id),
            );
            continue;
        }
        let mut r = Rec::new();
        r.pt("at", v.at)
            .len("size", v.diameter)
            .len("drill", v.drill);
        if v.from != 0 || v.to + 1 != n {
            r.put(
                "layers",
                format!(
                    "[{}, {}]",
                    layers.copper_text(v.from),
                    layers.copper_text(v.to)
                ),
            );
        }
        r.flag("micro", v.micro);
        if !v.tented {
            r.put("tented", "false");
        }
        r.flag("locked", v.locked);
        push(
            Group::Via,
            sites.indexed(words::VIA),
            format!("{} {}", net_head(board, words::VIA, v.net), r.text()),
            Some(v.id),
        );
    }
    for z in board.zones.values() {
        if !want(z.id) {
            push(
                Group::Pour,
                sites.indexed(words::POUR),
                String::new(),
                Some(z.id),
            );
            continue;
        }
        let d = Zone::new(z.id, "", None, Vec::new(), Polygon::default());
        let mut r = Rec::new();
        if !z.name.is_empty() {
            r.put("name", name_text(&z.name));
        }
        let l: Vec<String> = z.layers.iter().map(|l| layers.copper_text(*l)).collect();
        r.put("layers", format!("[{}]", l.join(", ")));
        polygon_fields(&mut r, &z.outline);
        if z.priority != 0 {
            r.put("priority", z.priority.to_string());
        }
        if let Some(c) = z.clearance {
            r.len("clearance", c);
        }
        if z.min_width != d.min_width {
            r.len("min_width", z.min_width);
        }
        if z.connect != d.connect {
            r.put("connect", connect_name(z.connect));
        }
        if z.thermal_gap != d.thermal_gap {
            r.len("thermal_gap", z.thermal_gap);
        }
        if z.spoke_width != d.spoke_width {
            r.len("spoke_width", z.spoke_width);
        }
        if z.islands != d.islands {
            r.put("islands", islands_text(z.islands));
        }
        r.flag("locked", z.locked);
        push(
            Group::Pour,
            sites.indexed(words::POUR),
            format!("{} {}", net_head(board, words::POUR, z.net), r.text()),
            Some(z.id),
        );
    }
    for k in board.keepouts.values() {
        if !want(k.id) {
            push(
                Group::Keepout,
                sites.indexed(words::KEEPOUT),
                String::new(),
                Some(k.id),
            );
            continue;
        }
        let mut r = Rec::new();
        polygon_fields(&mut r, &k.outline);
        if !k.layers.is_empty() {
            let l: Vec<String> = k.layers.iter().map(|l| layers.copper_text(*l)).collect();
            r.put("layers", format!("[{}]", l.join(", ")));
        }
        r.flag("tracks", k.rules.tracks)
            .flag("vias", k.rules.vias)
            .flag("pads", k.rules.pads)
            .flag("pours", k.rules.pours)
            .flag("footprints", k.rules.footprints);
        if let Some(h) = k.rules.max_height {
            r.len("max_height", h);
        }
        let head = if k.name.is_empty() {
            format!("{} =", words::KEEPOUT)
        } else {
            format!("{} {} =", words::KEEPOUT, name_text(&k.name))
        };
        push(
            Group::Keepout,
            sites.indexed(words::KEEPOUT),
            format!("{head} {}", r.text()),
            Some(k.id),
        );
    }
    for t in board.texts.values() {
        if !want(t.id) {
            push(
                Group::Text,
                sites.indexed(words::TEXT),
                String::new(),
                Some(t.id),
            );
            continue;
        }
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(t.layer))
            .pt("at", t.at)
            .len("size", t.size)
            .len("thickness", t.thickness);
        if t.angle != 0.0 {
            r.put("angle", f64_text(t.angle));
        }
        r.flag("mirrored", t.mirrored);
        push(
            Group::Text,
            sites.indexed(words::TEXT),
            format!(
                "{} {} = {}",
                words::TEXT,
                runmat_design_syntax::json::quote(&t.text),
                r.text()
            ),
            Some(t.id),
        );
    }
    for g in board.graphics.values() {
        if !want(g.id) {
            push(
                Group::Graphic,
                sites.indexed(words::GRAPHIC),
                String::new(),
                Some(g.id),
            );
            continue;
        }
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(g.layer));
        let (kind, text) = graphic_text(&g.graphic, r);
        push(
            Group::Graphic,
            sites.indexed(words::GRAPHIC),
            format!("{} {kind} = {text}", words::GRAPHIC),
            Some(g.id),
        );
    }
    for d in board.dimensions.values() {
        if !want(d.id) {
            push(
                Group::Dimension,
                sites.indexed(words::DIMENSION),
                String::new(),
                Some(d.id),
            );
            continue;
        }
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(d.layer))
            .pt("from", d.a)
            .pt("to", d.b);
        if d.offset != 0 {
            r.len("offset", d.offset);
        }
        if d.precision != 2 {
            r.put("precision", d.precision.to_string());
        }
        push(
            Group::Dimension,
            sites.indexed(words::DIMENSION),
            format!("{} = {}", words::DIMENSION, r.text()),
            Some(d.id),
        );
    }
    for (key, p) in &packages {
        push(
            Group::Package,
            sites.named(words::PACKAGE, key),
            package_text(key, p),
            None,
        );
    }
    let section = SyncSection::of(board);
    if section != SyncSection::default() {
        push(Group::Sync, "sync".into(), print_sync(&section, 1), None);
    }
    out
}

/// The file's head: the version header and the `use` lines.
pub fn print_head(uses: &[UseItem]) -> String {
    let mut out = String::from("runmat 0.7\nstd 1\n\n");
    for u in uses {
        out.push_str(&u.print());
        out.push('\n');
    }
    if !uses.is_empty() {
        out.push('\n');
    }
    out
}

/// A board as a whole `.circuit.layout` file.
pub fn print_board(board: &Board, uses: &[UseItem]) -> String {
    let printed = statements(board);
    let size: usize = printed.iter().map(|p| p.text.len() + 6).sum();
    let mut out = String::with_capacity(size + 64);
    out.push_str(&print_head(uses));
    out.push_str(&format!(
        "{} {} {{\n",
        words::LAYOUT,
        name_text(&board.name)
    ));
    for p in &printed {
        out.push_str(INDENT);
        out.push_str(&p.text);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

/// `board` with its items renumbered in print order, derived data (pour fills) dropped and
/// rule source positions cleared: the board its printed text reads back as.
pub fn canonical(board: &Board) -> Board {
    let mut b = board.clone();
    let mut next = 1u64;
    let mut renumber = |id: &mut ItemId| {
        *id = ItemId(next);
        next += 1;
    };
    macro_rules! renumber_map {
        ($field:ident) => {{
            let items = std::mem::take(&mut b.$field);
            b.$field = items
                .into_values()
                .map(|mut item| {
                    renumber(&mut item.id);
                    (item.id, item)
                })
                .collect();
        }};
    }
    renumber_map!(footprints);
    renumber_map!(fiducials);
    renumber_map!(tracks);
    renumber_map!(vias);
    renumber_map!(zones);
    renumber_map!(keepouts);
    renumber_map!(texts);
    renumber_map!(graphics);
    renumber_map!(dimensions);
    b.next_id = next;
    for z in b.zones.values_mut() {
        z.fill.clear();
    }
    for r in &mut b.rules {
        r.source = SourceKey::default();
        r.order = SourceOrder::default();
    }
    for c in b.net_class_rules.values_mut() {
        c.source = SourceKey::default();
        c.order = SourceOrder::default();
    }
    b
}

/// Whether two boards are the same model: equal once both are [`canonical`].
pub fn same_board(a: &Board, b: &Board) -> bool {
    canonical(a) == canonical(b)
}

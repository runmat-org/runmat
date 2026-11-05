//! Reading a `.circuit.layout` document into a board: the layout block, stack-up, rules,
//! footprints, copper, zones and the other board items, with their source sites.

use super::*;
use super::super::layout::{read_sync_section, SyncSection};
use crate::board_rules::{RuleSpec, RuleValue};
use crate::geom::poly::Polygon;
use crate::geom::{mm, Coord};
use crate::library::{
    Package,
};
use runmat_design_syntax::ast::{self, Expr, Headed, Item, Statement};
use runmat_design_syntax::document::UseItem;
use runmat_design_syntax::{parse_source, Dialect};
use runmat_rules::{SourceKey, SourceOrder};
use runmat_types::Span;
use std::collections::{BTreeMap, HashMap};

/* ------------------------------------------------------------------------------------------------
 * Reading
 * --------------------------------------------------------------------------------------------- */

pub(super) struct Reader<'a> {
    pub(super) path: &'a str,
    pub(super) out: ParsedBoard,
    pub(super) sites: Sites,
}

/// A statement's head: its word, name and value.
pub(super) struct Head {
    pub(super) word: String,
    pub(super) name: Option<(String, Span)>,
    pub(super) type_text: Option<String>,
    pub(super) value: Option<Val>,
    pub(super) span: Span,
}

pub(super) fn head(statement: &Statement) -> Read<Option<Head>> {
    let span = statement.span();
    let value = |e: Option<Expr>| e.as_ref().map(val).transpose();
    Ok(Some(match statement {
        Statement::Value(v) => Head {
            word: v.keyword(),
            name: v.name().map(|n| (n.text(), n.span())),
            type_text: v.type_text(),
            value: value(v.value())?,
            span,
        },
        Statement::Assignment(a) => {
            let target = a.target();
            if target.len() != 1 {
                return Err(diagnostic(
                    a.target_span(),
                    "expected a statement word such as `outline` or `track`",
                ));
            }
            Head {
                word: target[0].clone(),
                name: None,
                type_text: None,
                value: value(a.value())?,
                span,
            }
        }
        Statement::Command(c) => {
            // `net GND` / `net "+3V3"`: a name and nothing else.
            let name = match c.argument() {
                None => None,
                Some(Expr::Reference(r)) => Some((r.segments().join("."), r.span())),
                Some(Expr::String(s)) => Some((s.value(), s.span())),
                Some(other) => {
                    return Err(diagnostic(
                        other.span(),
                        format!("expected `{} <name> = <value>`", c.word()),
                    ))
                }
            };
            Head {
                word: c.word(),
                name,
                type_text: None,
                value: None,
                span,
            }
        }
        _ => return Ok(None),
    }))
}

pub(super) fn unquote_type(t: &str) -> String {
    match t.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        Some(inner) => inner.replace("\\\"", "\"").replace("\\\\", "\\"),
        None => t.to_owned(),
    }
}

pub(super) fn need_value(h: &Head) -> Read<&Val> {
    h.value
        .as_ref()
        .ok_or_else(|| diagnostic(h.span, format!("`{}` needs `= <value>`", h.word)))
}

pub(super) fn need_name(h: &Head) -> Read<String> {
    h.name
        .as_ref()
        .map(|(n, _)| n.clone())
        .ok_or_else(|| diagnostic(h.span, format!("`{}` needs a name", h.word)))
}

pub(super) fn polygon(f: &mut Fields) -> Read<Polygon> {
    let outer = ring(f.need(words::OUTLINE)?)?;
    let holes = f.or("holes", Vec::new(), rings)?;
    Ok(Polygon { outer, holes })
}

/// Parses a `.circuit.layout` source.
pub fn parse_board(path: &str, source: &str) -> ParsedBoard {
    let syntax = parse_source(Dialect::CircuitLayout, source);
    let mut reader = Reader {
        path,
        out: ParsedBoard {
            board: None,
            name: None,
            uses: Vec::new(),
            statements: Vec::new(),
            spans: BTreeMap::new(),
            sites: BTreeMap::new(),
            block: None,
            diagnostics: syntax.diagnostics.clone(),
            packages: HashMap::new(),
        },
        sites: Sites::default(),
    };
    let mut board = None;
    for item in syntax.file().items() {
        match item {
            Item::Use(u) => match use_item(&u) {
                Ok(item) => reader.out.uses.push(item),
                Err(e) => reader.out.diagnostics.push(e),
            },
            Item::Import(i) => reader.out.diagnostics.push(diagnostic(
                i.span(),
                "a layout reads other files with `use`",
            )),
            Item::Declaration(d) if d.keyword() == words::LAYOUT => {
                if board.is_some() {
                    reader
                        .out
                        .diagnostics
                        .push(diagnostic(d.keyword_span(), "a file holds one `layout`"));
                    continue;
                }
                board = Some(reader.layout(&d));
            }
            Item::Declaration(_) | Item::Error(_) => {}
        }
    }
    if board.is_none() {
        reader.out.diagnostics.push(diagnostic(
            Span::default(),
            "expected a `layout <name> { … }` declaration",
        ));
    }
    let mut out = reader.out;
    for d in &mut out.diagnostics {
        d.span = trim_span(source, d.span);
    }
    if out.diagnostics.is_empty() {
        out.board = board;
    }
    out
}

pub(crate) fn use_item(u: &ast::UseDeclaration) -> Read<UseItem> {
    let from = match u.source() {
        Some(ast::UseSource::Path(path)) => path,
        Some(ast::UseSource::Package(segments)) => segments.join("."),
        None => return Err(diagnostic(u.span(), "`use` needs a source")),
    };
    if u.overrides().is_some() {
        return Err(diagnostic(
            u.span(),
            "a `use` here takes no `{ … }` overrides",
        ));
    }
    Ok(UseItem {
        symbol: u.symbol().map(|(kind, name)| (kind.text(), name.text())),
        from,
        alias: u.alias().map(|a| a.text()),
        revision: u.pin(),
        span: u.span(),
    })
}

impl Reader<'_> {
    pub(super) fn note<T>(&mut self, r: Read<T>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.out.diagnostics.push(e);
                None
            }
        }
    }

    pub(super) fn record(&mut self, group: Group, site: String, span: Span, item: Option<ItemId>) {
        if let Some(id) = item {
            self.out.sites.insert(id, site.clone());
        }
        self.out.spans.insert(site.clone(), span);
        self.out.statements.push(BoardStatement {
            group,
            site,
            span,
            item,
        });
    }

    pub(super) fn layout(&mut self, d: &ast::Declaration) -> Board {
        let name = d.name().map(|n| n.text()).unwrap_or_default();
        let stem = Dialect::CircuitLayout.stem(self.path);
        if !self.path.is_empty() && name != stem {
            self.out.diagnostics.push(diagnostic(
                d.name().map_or(d.span(), |n| n.span()),
                format!("this layout is named `{name}`; its file is `{stem}`"),
            ));
        }
        self.out.name = Some(name.clone());
        let statements = match d.block() {
            Some(block) => {
                self.out.block = Some(block.span());
                block.statements()
            }
            None => Vec::new(),
        };
        // Pass 1: what the rest refers to — stackups, packages and nets.
        let mut stackups: Vec<Stackup> = Vec::new();
        let mut packages: HashMap<String, Package> = HashMap::new();
        let mut nets: Vec<BoardNet> = Vec::new();
        let mut net_ids: HashMap<String, NetId> = HashMap::new();
        for statement in &statements {
            match statement {
                Statement::Block(b) if b.keyword() == words::STACKUP => {
                    let name = b.name().map(|n| n.text()).unwrap_or_default();
                    let site = self.sites.named(words::STACKUP, &name);
                    self.record(Group::Stackup, site, statement.span(), None);
                    if let Some(s) = self.note(read_stackup(b, name)) {
                        if stackups.iter().any(|x| x.name == s.name) {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                format!("stackup `{}` is given twice", s.name),
                            ));
                        }
                        stackups.push(s);
                    }
                }
                Statement::Block(b) if b.keyword() == words::PACKAGE => {
                    let key = b.name().map(|n| n.text()).unwrap_or_default();
                    let site = self.sites.named(words::PACKAGE, &key);
                    self.record(Group::Package, site, statement.span(), None);
                    if let Some(p) = self.note(read_package(b, &key)) {
                        if packages.insert(key.clone(), p).is_some() {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                format!("package `{key}` is given twice"),
                            ));
                        }
                    }
                }
                _ => {
                    let is_net = match statement {
                        Statement::Value(v) => v.keyword() == words::NET,
                        Statement::Command(c) => c.word() == words::NET,
                        _ => false,
                    };
                    if !is_net {
                        continue;
                    }
                    let Ok(Some(h)) = head(statement) else {
                        continue;
                    };
                    if h.word != words::NET {
                        continue;
                    }
                    let Some(name) = self.note(need_name(&h)) else {
                        continue;
                    };
                    let site = self.sites.named(words::NET, &name);
                    self.record(Group::Net, site, h.span, None);
                    if h.value.is_some() {
                        self.out.diagnostics.push(diagnostic(
                            h.span,
                            "a net is `net <name>` or `net <name> : <class>`",
                        ));
                    }
                    if net_ids.contains_key(&name) {
                        self.out
                            .diagnostics
                            .push(diagnostic(h.span, format!("net `{name}` is given twice")));
                        continue;
                    }
                    net_ids.insert(name.clone(), NetId(nets.len() as u32));
                    nets.push(BoardNet {
                        name,
                        class: h.type_text.as_deref().map(unquote_type),
                    });
                }
            }
        }
        self.out.packages = packages.clone();
        let mut stackups = stackups.into_iter();
        let main = stackups
            .next()
            .unwrap_or_else(|| Stackup::standard(2, mm(1.6), 1.0, 0.5));
        let mut board = Board::new(&name, Polygon::default(), main);
        board.stackups = stackups.map(|s| (s.name.clone(), s)).collect();
        board.nets = nets;
        let layers = Layers::of(&board.stackup);
        let ctx = Ctx {
            layers: &layers,
            nets: &net_ids,
            packages: &packages,
        };
        // Pass 2: everything else, in source order.
        let mut sync: Option<(SyncSection, Span)> = None;
        let mut outline_seen = false;
        let mut limits_seen = false;
        for statement in &statements {
            let at = statement.span();
            if let Statement::Block(b) = statement {
                match b.keyword().as_str() {
                    words::STACKUP | words::PACKAGE => {}
                    words::SYNC => {
                        self.record(Group::Sync, "sync".into(), at, None);
                        if sync.is_some() {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                "a layout holds one `sync` section",
                            ));
                            continue;
                        }
                        if let Some(name) = b.name() {
                            self.out
                                .diagnostics
                                .push(diagnostic(name.span(), "the `sync` section takes no name"));
                        }
                        let inner = b.block().map(|b| b.statements()).unwrap_or_default();
                        let (section, errors) = read_sync_section(&inner, at);
                        self.out.diagnostics.extend(errors);
                        if let Some(section) = section {
                            sync = Some((section, at));
                        }
                    }
                    other => self.out.diagnostics.push(diagnostic(
                        b.keyword_span(),
                        format!("`{other} {{ … }}` is not a layout section"),
                    )),
                }
                continue;
            }
            let h = match head(statement) {
                Ok(Some(h)) => h,
                Ok(None) => {
                    self.out.diagnostics.push(diagnostic(
                        at,
                        "expected a board statement such as `footprint`, `track` or `via`",
                    ));
                    continue;
                }
                Err(e) => {
                    self.out.diagnostics.push(e);
                    continue;
                }
            };
            match h.word.as_str() {
                words::NET => {}
                words::OUTLINE => {
                    self.record(Group::Outline, "outline".into(), at, None);
                    if outline_seen {
                        self.out
                            .diagnostics
                            .push(diagnostic(at, "`outline` is given twice"));
                    }
                    outline_seen = true;
                    if let Some(r) = self.note(need_value(&h).and_then(ring)) {
                        board.outline.outer = r;
                    }
                }
                words::CUTOUT => {
                    let site = self.sites.indexed(words::CUTOUT);
                    self.record(Group::Outline, site, at, None);
                    if let Some(r) = self.note(need_value(&h).and_then(ring)) {
                        board.outline.holes.push(r);
                    }
                }
                words::LIMITS => {
                    self.record(Group::Limits, "limits".into(), at, None);
                    if limits_seen {
                        self.out
                            .diagnostics
                            .push(diagnostic(at, "`limits` is given twice"));
                    }
                    limits_seen = true;
                    if let Some(l) = self.note(need_value(&h).and_then(read_limits)) {
                        board.limits = l;
                    }
                }
                words::REGION => {
                    let r = need_name(&h).and_then(|name| {
                        let mut f = Fields::of(need_value(&h)?, "region")?;
                        let stackup = text(f.need(words::STACKUP)?)?;
                        let outline = polygon(&mut f)?;
                        let bends = f.or("bends", Vec::new(), |v| {
                            list(v)?.iter().map(read_bend).collect()
                        })?;
                        f.finish()?;
                        Ok(StackRegion {
                            name,
                            outline,
                            stackup,
                            bends,
                        })
                    });
                    let site = self
                        .sites
                        .named(words::REGION, &h.name.clone().unwrap_or_default().0);
                    self.record(Group::Region, site, at, None);
                    if let Some(r) = self.note(r) {
                        board.regions.push(r);
                    }
                }
                words::NET_CLASS => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::NET_CLASS, &name);
                    self.record(Group::NetClass, site, at, None);
                    let r = need_name(&h).and_then(|_| {
                        let mut f = Fields::of(need_value(&h)?, "net class")?;
                        f.rest()
                            .into_iter()
                            .map(|(k, v)| Ok((k.to_owned(), rule_value(v)?)))
                            .collect::<Read<Vec<_>>>()
                    });
                    if let Some(decls) = self.note(r) {
                        if board.net_class_rules.insert(name.clone(), decls).is_some() {
                            self.out
                                .diagnostics
                                .push(diagnostic(at, format!("net class `{name}` is given twice")));
                        }
                    }
                }
                words::RULE => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::RULE, &name);
                    self.record(Group::Rule, site, at, None);
                    let order = SourceOrder {
                        file_rank: 0,
                        offset: at.start as u32,
                    };
                    let key = SourceKey::new(self.path, at.start as u32, at.end as u32);
                    let r = need_name(&h)
                        .and_then(|id| read_rule(&id, need_value(&h)?))
                        .map(|r| r.at(key, order));
                    if let Some(r) = self.note(r) {
                        if board.rules.iter().any(|x| x.id == r.id) {
                            self.out
                                .diagnostics
                                .push(diagnostic(at, format!("rule `{}` is given twice", r.id)));
                        }
                        board.rules.push(r);
                    }
                }
                words::ROOM => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::ROOM, &name);
                    self.record(Group::Room, site, at, None);
                    let r = need_name(&h).and_then(|name| {
                        let mut f = Fields::of(need_value(&h)?, "room")?;
                        let outline = polygon(&mut f)?;
                        let members = f.or("members", Vec::new(), texts)?;
                        f.finish()?;
                        Ok(Room {
                            name,
                            outline,
                            members,
                        })
                    });
                    if let Some(r) = self.note(r) {
                        board.rooms.push(r);
                    }
                }
                w if ITEM_WORDS.contains(&w) => {
                    self.read_item(&mut board, &ctx, &h, at, None);
                }
                other => self.out.diagnostics.push(diagnostic(
                    at,
                    format!(
                        "`{other}` does not start a board statement; expected stackup, region, \
                         outline, cutout, limits, net_class, rule, net, room, footprint, \
                         fiducial, track, arc, via, pour, keepout, text, graphic, dimension, \
                         package or sync"
                    ),
                )),
            }
        }
        self.out.statements.sort_by_key(|s| s.span.start);
        if let Some((section, at)) = sync {
            board.sync = section.sync;
            let by_ref: HashMap<&str, ItemId> = board
                .footprints
                .values()
                .map(|f| (f.reference.as_str(), f.id))
                .collect();
            let mut links = Vec::new();
            for (reference, path) in &section.links {
                match by_ref.get(reference.as_str()) {
                    Some(id) => links.push((*id, path.clone())),
                    None => self.out.diagnostics.push(diagnostic(
                        at,
                        format!("the sync section links footprint `{reference}`, which the board does not have"),
                    )),
                }
            }
            for (id, path) in links {
                if let Some(f) = board.footprints.get_mut(&id) {
                    f.component = Some(path);
                }
            }
        }
        board
    }

    /// Reads one item statement (footprint, fiducial, copper, via, pour, keepout, text,
    /// graphic or dimension) into `board`, under `reuse` or a new id. The item's id, whether
    /// or not it read.
    pub(super) fn read_item(
        &mut self,
        board: &mut Board,
        ctx: &Ctx,
        h: &Head,
        at: Span,
        reuse: Option<ItemId>,
    ) -> Option<ItemId> {
        let layers = ctx.layers;
        match h.word.as_str() {
            words::FOOTPRINT => {
                let name = h.name.clone().unwrap_or_default().0;
                let site = self.sites.named(words::FOOTPRINT, &name);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Footprint, site, at, Some(id));
                let r = need_name(h).and_then(|r| ctx.footprint(id, r, need_value(h)?));
                if let Some(f) = self.note(r) {
                    board.footprints.insert(id, f);
                }
                Some(id)
            }
            words::FIDUCIAL => {
                let name = h.name.clone().unwrap_or_default().0;
                let site = self.sites.named(words::FIDUCIAL, &name);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Fiducial, site, at, Some(id));
                let r = need_name(h).and_then(|name| {
                    let mut f = Fields::of(need_value(h)?, "fiducial")?;
                    let fid = Fiducial {
                        id,
                        name,
                        at: point(f.need("at")?)?,
                        side: f.or("side", Side::Top, side_of)?,
                        diameter: len(f.need("diameter")?)?,
                        opening: len(f.need("opening")?)?,
                        locked: f.or("locked", false, boolean)?,
                    };
                    f.finish()?;
                    Ok(fid)
                });
                if let Some(f) = self.note(r) {
                    board.fiducials.insert(id, f);
                }
                Some(id)
            }
            words::TRACK | words::ARC => {
                let word = if h.word == words::TRACK {
                    words::TRACK
                } else {
                    words::ARC
                };
                let site = self.sites.indexed(word);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Copper, site, at, Some(id));
                let r = ctx.track(id, word, h);
                if let Some(t) = self.note(r) {
                    board.tracks.insert(id, t);
                }
                Some(id)
            }
            words::VIA => {
                let site = self.sites.indexed(words::VIA);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Via, site, at, Some(id));
                let n = board.copper_count();
                let r = ctx.via(id, h, n);
                if let Some(v) = self.note(r) {
                    board.vias.insert(id, v);
                }
                Some(id)
            }
            words::POUR => {
                let site = self.sites.indexed(words::POUR);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Pour, site, at, Some(id));
                let r = ctx.pour(id, h);
                if let Some(z) = self.note(r) {
                    board.zones.insert(id, z);
                }
                Some(id)
            }
            words::KEEPOUT => {
                let site = self.sites.indexed(words::KEEPOUT);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Keepout, site, at, Some(id));
                let r = ctx.keepout(id, h);
                if let Some(k) = self.note(r) {
                    board.keepouts.insert(id, k);
                }
                Some(id)
            }
            words::TEXT => {
                let site = self.sites.indexed(words::TEXT);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Text, site, at, Some(id));
                let r = need_name(h).and_then(|text| {
                    let mut f = Fields::of(need_value(h)?, "text")?;
                    let t = BoardText {
                        id,
                        layer: layers.layer(f.need("layer")?)?,
                        at: point(f.need("at")?)?,
                        text,
                        size: len(f.need("size")?)?,
                        thickness: len(f.need("thickness")?)?,
                        angle: f.or("angle", 0.0, angle)?,
                        mirrored: f.or("mirrored", false, boolean)?,
                    };
                    f.finish()?;
                    Ok(t)
                });
                if let Some(t) = self.note(r) {
                    board.texts.insert(id, t);
                }
                Some(id)
            }
            words::GRAPHIC => {
                let site = self.sites.indexed(words::GRAPHIC);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Graphic, site, at, Some(id));
                let r = need_name(h).and_then(|kind| {
                    let mut f = Fields::of(need_value(h)?, "graphic")?;
                    let layer = layers.layer(f.need("layer")?)?;
                    let graphic = read_graphic(&kind, &mut f, h.span)?;
                    f.finish()?;
                    Ok(BoardGraphic { id, layer, graphic })
                });
                if let Some(g) = self.note(r) {
                    board.graphics.insert(id, g);
                }
                Some(id)
            }
            words::DIMENSION => {
                let site = self.sites.indexed(words::DIMENSION);
                let id = reuse.unwrap_or_else(|| board.alloc());
                self.record(Group::Dimension, site, at, Some(id));
                let r = need_value(h).and_then(|v| {
                    let mut f = Fields::of(v, "dimension")?;
                    let d = Dimension {
                        id,
                        layer: layers.layer(f.need("layer")?)?,
                        a: point(f.need("from")?)?,
                        b: point(f.need("to")?)?,
                        offset: f.or("offset", 0, len)?,
                        precision: f.or("precision", 2, |v| {
                            u8::try_from(whole(v)?)
                                .map_err(|_| diagnostic(v.span, "precision is 0 to 255"))
                        })?,
                    };
                    f.finish()?;
                    Ok(d)
                });
                if let Some(d) = self.note(r) {
                    board.dimensions.insert(id, d);
                }
                Some(id)
            }
            _ => None,
        }
    }
}

pub(super) struct Ctx<'a> {
    pub(super) layers: &'a Layers,
    pub(super) nets: &'a HashMap<String, NetId>,
    pub(super) packages: &'a HashMap<String, Package>,
}

impl Ctx<'_> {
    pub(super) fn net(&self, name: &str, span: Span) -> Read<NetId> {
        self.nets.get(name).copied().ok_or_else(|| {
            diagnostic(
                span,
                format!("`{name}` is not a net of this board; declare it with `net {name}`"),
            )
        })
    }

    pub(super) fn head_net(&self, h: &Head) -> Read<Option<NetId>> {
        h.name
            .as_ref()
            .map(|(n, span)| self.net(n, *span))
            .transpose()
    }

    pub(super) fn footprint(&self, id: ItemId, reference: String, v: &Val) -> Read<Footprint> {
        let mut f = Fields::of(v, "footprint")?;
        let pv = f.need(words::PACKAGE)?;
        let key = text(pv)?;
        let package = self.packages.get(&key).cloned().ok_or_else(|| {
            diagnostic(
                pv.span,
                format!("package `{key}` is not defined in this layout"),
            )
        })?;
        let mut pad_nets = BTreeMap::new();
        if let Some(pads) = f.get("pads") {
            let V::Rec(entries) = &pads.v else {
                return Err(diagnostic(pads.span, "pads are `{ \"1\" = <net>, … }`"));
            };
            for (pad, net) in entries {
                pad_nets.insert(pad.clone(), self.net(&text(net)?, net.span)?);
            }
        }
        let properties = match f.get("properties") {
            Some(p) => {
                let V::Rec(entries) = &p.v else {
                    return Err(diagnostic(p.span, "properties are `{ <name> = \"…\", … }`"));
                };
                entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), text(v)?)))
                    .collect::<Read<_>>()?
            }
            None => BTreeMap::new(),
        };
        let fp = Footprint {
            id,
            reference,
            component: None,
            value: f.or("value", String::new(), text)?,
            package,
            at: point(f.need("at")?)?,
            rotation: f.or("rotation", 0.0, angle)?,
            side: f.or("side", Side::Top, side_of)?,
            locked: f.or("locked", false, boolean)?,
            pad_nets,
            fitted: f.or("fitted", true, boolean)?,
            properties,
        };
        f.finish()?;
        Ok(fp)
    }

    pub(super) fn track(&self, id: ItemId, word: &'static str, h: &Head) -> Read<Track> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, word)?;
        let layer = self.layers.copper(f.need("layer")?)?;
        let width = len(f.need("width")?)?;
        let geom = if word == words::ARC {
            TrackGeom::Arc {
                start: point(f.need("from")?)?,
                mid: point(f.need("mid")?)?,
                end: point(f.need("to")?)?,
            }
        } else {
            TrackGeom::Segment {
                a: point(f.need("from")?)?,
                b: point(f.need("to")?)?,
            }
        };
        let t = Track {
            id,
            net,
            layer,
            width,
            geom,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(t)
    }

    pub(super) fn via(&self, id: ItemId, h: &Head, copper: u8) -> Read<Via> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "via")?;
        let (from, to) = match f.get("layers") {
            None => (0, copper.saturating_sub(1)),
            Some(v) => match list(v)? {
                [a, b] => {
                    let (a, b) = (self.layers.copper(a)?, self.layers.copper(b)?);
                    (a.min(b), a.max(b))
                }
                _ => {
                    return Err(diagnostic(
                        v.span,
                        "a via's layers are its first and last copper layer: `[F.Cu, In2.Cu]`",
                    ))
                }
            },
        };
        let v = Via {
            id,
            net,
            at: point(f.need("at")?)?,
            diameter: len(f.need("size")?)?,
            drill: len(f.need("drill")?)?,
            from,
            to,
            micro: f.or("micro", false, boolean)?,
            tented: f.or("tented", true, boolean)?,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(v)
    }

    pub(super) fn copper_list(&self, v: &Val) -> Read<Vec<u8>> {
        list(v)?.iter().map(|l| self.layers.copper(l)).collect()
    }

    pub(super) fn pour(&self, id: ItemId, h: &Head) -> Read<Zone> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "pour")?;
        let name = f.or("name", String::new(), text)?;
        let layers = self.copper_list(f.need("layers")?)?;
        let outline = polygon(&mut f)?;
        let mut z = Zone::new(id, &name, net, layers, outline);
        z.priority = f.or("priority", 0, whole)? as i32;
        z.clearance = f.opt("clearance", len)?;
        z.min_width = f.or("min_width", z.min_width, len)?;
        z.connect = f.or("connect", z.connect, connect_of)?;
        z.thermal_gap = f.or("thermal_gap", z.thermal_gap, len)?;
        z.spoke_width = f.or("spoke_width", z.spoke_width, len)?;
        z.islands = f.or("islands", z.islands, islands_of)?;
        z.locked = f.or("locked", false, boolean)?;
        f.finish()?;
        Ok(z)
    }

    pub(super) fn keepout(&self, id: ItemId, h: &Head) -> Read<Keepout> {
        let name = h.name.clone().map(|n| n.0).unwrap_or_default();
        let mut f = Fields::of(need_value(h)?, "keepout")?;
        let outline = polygon(&mut f)?;
        let layers = f.or("layers", Vec::new(), |v| self.copper_list(v))?;
        let rules = KeepoutRules {
            tracks: f.or("tracks", false, boolean)?,
            vias: f.or("vias", false, boolean)?,
            pads: f.or("pads", false, boolean)?,
            pours: f.or("pours", false, boolean)?,
            footprints: f.or("footprints", false, boolean)?,
            max_height: f.opt("max_height", len)?,
        };
        f.finish()?;
        Ok(Keepout {
            id,
            name,
            outline,
            layers,
            rules,
        })
    }
}

pub(super) fn read_bend(v: &Val) -> Read<Bend> {
    let mut f = Fields::of(v, "bend")?;
    let b = Bend {
        a: point(f.need("from")?)?,
        b: point(f.need("to")?)?,
        radius: len(f.need("radius")?)?,
        angle: angle(f.need("angle")?)?,
        keepout: f.or("keepout", 0, len)?,
    };
    f.finish()?;
    Ok(b)
}

pub(super) const LIMIT_KEYS: [&str; 9] = [
    "clearance",
    "track",
    "via_diameter",
    "drill",
    "annular",
    "hole_to_hole",
    "copper_to_edge",
    "silk_to_pad",
    "microvia_drill",
];

pub(super) fn limit_fields(l: &mut FabLimits) -> [&mut Coord; 9] {
    [
        &mut l.min_clearance,
        &mut l.min_track,
        &mut l.min_via_diameter,
        &mut l.min_drill,
        &mut l.min_annular,
        &mut l.min_hole_to_hole,
        &mut l.min_copper_to_edge,
        &mut l.min_silk_to_pad,
        &mut l.min_microvia_drill,
    ]
}

pub(super) fn read_limits(v: &Val) -> Read<FabLimits> {
    let mut f = Fields::of(v, "limits")?;
    let mut l = FabLimits::default();
    for (key, field) in LIMIT_KEYS.iter().zip(limit_fields(&mut l)) {
        if let Some(v) = f.opt(key, len)? {
            *field = v;
        }
    }
    f.finish()?;
    Ok(l)
}

pub(super) const VIA_KINDS: [(ViaKind, &str); 4] = [
    (ViaKind::Through, "through"),
    (ViaKind::Blind, "blind"),
    (ViaKind::Buried, "buried"),
    (ViaKind::Micro, "micro"),
];

/// A rule value: `0.2 mm` is a length, `2` a whole number, `2.5` a number, `true` a flag,
/// `[through, micro]` via kinds, anything else text.
pub(super) fn rule_value(v: &Val) -> Read<RuleValue> {
    Ok(match &v.v {
        V::Qty(..) => RuleValue::Len(len(v)?),
        V::Num(t) if t.contains(['.', 'e', 'E']) => RuleValue::Num(num(v)?),
        V::Num(_) => RuleValue::Int(whole(v)?),
        V::Word(w) if w == "true" || w == "false" => RuleValue::Bool(boolean(v)?),
        V::Word(w) | V::Str(w) => RuleValue::Text(w.clone()),
        V::List(items) => RuleValue::ViaKinds(
            items
                .iter()
                .map(|i| {
                    let t = text(i)?;
                    VIA_KINDS
                        .iter()
                        .find(|(_, n)| *n == t)
                        .map(|(k, _)| *k)
                        .ok_or_else(|| {
                            diagnostic(
                                i.span,
                                format!("`{t}` is not a via kind; expected through, blind, buried or micro"),
                            )
                        })
                })
                .collect::<Read<_>>()?,
        ),
        V::Rec(_) => return Err(diagnostic(v.span, "a rule value is a length, number, flag or text")),
    })
}

pub(super) fn rule_value_text(v: &RuleValue) -> String {
    match v {
        RuleValue::Len(l) => format!("{} mm", mm_text(*l)),
        RuleValue::Num(n) => {
            let t = f64_text(*n);
            if t.contains(['.', 'e', 'E', 'N', 'i']) {
                t
            } else {
                format!("{t}.0")
            }
        }
        RuleValue::Int(i) => i.to_string(),
        RuleValue::Bool(b) => b.to_string(),
        RuleValue::Text(t) => name_text(t),
        RuleValue::ViaKinds(kinds) => {
            let names: Vec<&str> = kinds
                .iter()
                .map(|k| VIA_KINDS.iter().find(|(x, _)| x == k).unwrap().1)
                .collect();
            format!("[{}]", names.join(", "))
        }
    }
}

pub(super) fn read_rule(id: &str, v: &Val) -> Read<RuleSpec> {
    let mut f = Fields::of(v, "rule")?;
    let mut spec = match (f.get("select"), f.get("between")) {
        (Some(s), None) => RuleSpec::single(id, &text(s)?),
        (None, Some(b)) => match list(b)? {
            [a, c] => RuleSpec::pair(id, &text(a)?, &text(c)?),
            _ => return Err(diagnostic(b.span, "`between` takes two selectors")),
        },
        _ => {
            return Err(diagnostic(
                v.span,
                "a rule has `select = \"<selector>\"` or `between = [\"<a>\", \"<b>\"]`",
            ))
        }
    };
    spec.priority = f.or("priority", 0, whole)? as i32;
    for (k, v) in f.rest() {
        spec.decls.push((k.to_owned(), rule_value(v)?));
    }
    Ok(spec)
}

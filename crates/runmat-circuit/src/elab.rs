//! Elaboration: flattens the sheet hierarchy into component instances and nets.
//!
//! - Sheet instances are walked from the root; a repeated instance `adc` with `repeat = 4`
//!   produces channels `adc[0]`…`adc[3]`, each a full copy with its own paths.
//! - Connectivity is a union-find over *nodes*: local nets per sheet instance, global nets by
//!   name, ports per instance (bus ports and harness ports per member), bus and harness
//!   members, and component pins. Connections, port bindings (shared, bus-indexed or
//!   net-indexed per channel) and bus-to-bus connections union nodes member by member.
//! - Each resulting net is named by priority: a global name, then the shallowest declared
//!   local net or bus member, then the shallowest port, then an automatic
//!   `Net-(<designator>-Pad<n>)` name. Local names below the root carry their sheet path
//!   (`/power/VOUT`).
//! - Designators: explicit unique names are kept; a designator recorded in the lock section
//!   for the same path is kept (stable re-annotation); everything else (`R?`, repeated
//!   channels, collisions) is numbered per prefix in path order.
//! - Variants: the selected variant's overrides (by path or designator), then its bases, then
//!   the component's own per-variant fitting; alternates swap the part and merge parameters.
//!
//! Every problem is a [`Diagnostic`] with the item's path and the declaration's source key.

use crate::ids::{LockItem, LockSection, NamePath, Uuid};
use crate::ir::*;
use crate::library::{PinType, Pool, ResolvedPin};
use runmat_rules::SourceKey;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A finding from elaboration, ERC, DRC or a review suite. `item` is the path (or board
/// item key) it is about; `related` lists other items involved.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub item: String,
    pub related: Vec<String>,
    pub source: SourceKey,
}

impl Diagnostic {
    pub fn new(sev: Severity, code: &str, item: impl Into<String>, msg: impl Into<String>) -> Self {
        Diagnostic {
            severity: sev,
            code: code.to_string(),
            message: msg.into(),
            item: item.into(),
            related: Vec::new(),
            source: SourceKey::default(),
        }
    }
    pub fn at(mut self, s: &SourceKey) -> Self {
        self.source = s.clone();
        self
    }
    pub fn with(mut self, r: impl Into<String>) -> Self {
        self.related.push(r.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FitState {
    Fitted,
    NotFitted,
    /// Fitted with an alternate part or parameters.
    Alternate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PinInst {
    pub gate: String,
    pub pin: String,
    pub kind: PinType,
    pub pads: Vec<String>,
    pub net: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompInst {
    pub path: NamePath,
    pub designator: String,
    pub sheet: String,
    pub channel: Option<u32>,
    pub part: Option<Uuid>,
    pub entity: Option<Uuid>,
    pub package: Option<Uuid>,
    pub params: BTreeMap<String, Param>,
    pub properties: BTreeMap<String, String>,
    pub classes: Vec<String>,
    pub power: Option<runmat_rules::Quantity>,
    pub fitting: FitState,
    pub pins: Vec<PinInst>,
    pub renamed_from: Option<NamePath>,
    pub source: SourceKey,
}

impl CompInst {
    pub fn pin(&self, name: &str) -> Option<&PinInst> {
        self.pins
            .iter()
            .find(|p| p.pin == name || format!("{}.{}", p.gate, p.pin) == name)
    }
    pub fn fitted(&self) -> bool {
        self.fitting != FitState::NotFitted
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetInst {
    pub name: String,
    pub path: NamePath,
    pub class: Option<String>,
    pub global: bool,
    /// Every other name this net carries (ports, lower-level nets, bus members).
    pub aliases: Vec<String>,
    /// `(component index, pin index)` pairs, sorted.
    pub pins: Vec<(usize, usize)>,
    pub renamed_from: Option<NamePath>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffPairInst {
    pub name: String,
    pub p: usize,
    pub n: usize,
    pub class: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Netlist {
    pub design: String,
    pub variant: Option<String>,
    pub components: Vec<CompInst>,
    pub nets: Vec<NetInst>,
    pub diff_pairs: Vec<DiffPairInst>,
    /// Sheet-instance paths (for the hierarchy tree), sorted.
    pub sheets: Vec<(NamePath, String)>,
    pub no_erc: BTreeSet<(usize, usize)>,
    /// New sheet-instance path → previous path, for renamed instances and their children.
    pub sheet_renames: BTreeMap<NamePath, NamePath>,
    pub diagnostics: Vec<Diagnostic>,
    /// Schematic places recorded in the lock (see [`crate::ids::SchPlace`]).
    pub sch_places: BTreeMap<String, crate::ids::SchPlace>,
    /// Hand-drawn schematic wires recorded in the lock (see [`crate::ids::SchWire`]).
    pub sch_wires: BTreeMap<crate::ids::WireKey, crate::ids::SchWire>,
    /// Each sheet definition's notes (`text` statements): name and words, in source order.
    pub sheet_texts: BTreeMap<String, Vec<(String, String)>>,
}

impl Netlist {
    pub fn component(&self, designator: &str) -> Option<(usize, &CompInst)> {
        self.components
            .iter()
            .enumerate()
            .find(|(_, c)| c.designator == designator)
    }
    pub fn net(&self, name: &str) -> Option<(usize, &NetInst)> {
        self.nets
            .iter()
            .enumerate()
            .find(|(_, n)| n.name == name || n.aliases.iter().any(|a| a == name))
    }
    /// Net of a pin by designator and pin name.
    pub fn net_of(&self, designator: &str, pin: &str) -> Option<&NetInst> {
        let (_, c) = self.component(designator)?;
        c.pin(pin).and_then(|p| p.net).map(|n| &self.nets[n])
    }
    /// Items for [`LockSection::reconcile`]: components, nets and sheet instances with
    /// fingerprints (part plus connected nets for components; member pins for nets).
    pub fn lock_items(&self) -> Vec<LockItem> {
        let mut out = Vec::new();
        for c in &self.components {
            let mut nets: Vec<&str> = c
                .pins
                .iter()
                .filter_map(|p| p.net.map(|n| self.nets[n].name.as_str()))
                .collect();
            nets.sort();
            nets.dedup();
            let mut it = LockItem::new(c.path.clone(), "component");
            it.fingerprint = Some(format!(
                "{}|{}",
                c.part.map(|u| u.to_string()).unwrap_or_default(),
                nets.join(",")
            ));
            it.designator = Some(c.designator.clone());
            it.renamed_from = c.renamed_from.clone();
            out.push(it);
        }
        for n in &self.nets {
            let mut pins: Vec<String> = n
                .pins
                .iter()
                .map(|(ci, pi)| {
                    format!(
                        "{}.{}",
                        self.components[*ci].path, self.components[*ci].pins[*pi].pin
                    )
                })
                .collect();
            pins.sort();
            let mut it = LockItem::new(n.path.clone(), "net");
            it.fingerprint = (!pins.is_empty()).then(|| pins.join(","));
            it.renamed_from = n.renamed_from.clone();
            out.push(it);
        }
        for (p, sheet) in &self.sheets {
            if p.depth() > 0 {
                let mut it = LockItem::new(p.clone(), "sheet");
                it.fingerprint = Some(sheet.clone());
                it.renamed_from = self.sheet_renames.get(p).cloned();
                out.push(it);
            }
        }
        out
    }
}

#[derive(Clone, Debug, Default)]
pub struct ElabOptions {
    pub variant: Option<String>,
}

struct Uf {
    parent: Vec<usize>,
}

impl Uf {
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.parent[hi] = lo;
        }
    }
}

#[derive(Clone, Debug)]
enum NodeKind {
    Global(String),
    /// Declared (or implicitly referenced) local net or bus member: name, sheet depth.
    Local {
        name: String,
        inst: NamePath,
        declared: bool,
        class: Option<String>,
        renamed_from: Option<String>,
    },
    Port {
        name: String,
        inst: NamePath,
    },
    Pin {
        comp: usize,
        pin: usize,
    },
}

struct Ctx<'a> {
    design: &'a Design,
    pool: &'a Pool,
    opts: &'a ElabOptions,
    lock: Option<&'a LockSection>,
    keys: BTreeMap<String, usize>,
    nodes: Vec<NodeKind>,
    uf: Uf,
    comps: Vec<CompInst>,
    comp_index: BTreeMap<NamePath, usize>,
    globals: BTreeSet<String>,
    diags: Vec<Diagnostic>,
    sheets: Vec<(NamePath, String)>,
    diff: Vec<(NamePath, DiffPairDecl)>,
    no_erc: Vec<(usize, usize)>,
    explicit_designator: BTreeMap<usize, String>,
    sheet_renames: BTreeMap<NamePath, NamePath>,
}

impl<'a> Ctx<'a> {
    fn node(&mut self, key: String, kind: NodeKind) -> usize {
        if let Some(i) = self.keys.get(&key) {
            // Upgrade an implicit local net to a declared one.
            if let (
                NodeKind::Local {
                    declared: d0,
                    class: c0,
                    renamed_from: r0,
                    ..
                },
                NodeKind::Local {
                    declared: true,
                    class,
                    renamed_from,
                    ..
                },
            ) = (&mut self.nodes[*i], &kind)
            {
                *d0 = true;
                if class.is_some() {
                    *c0 = class.clone();
                }
                if renamed_from.is_some() {
                    *r0 = renamed_from.clone();
                }
            }
            return *i;
        }
        let i = self.nodes.len();
        self.nodes.push(kind);
        self.uf.parent.push(i);
        self.keys.insert(key, i);
        i
    }

    fn net_node(&mut self, inst: &NamePath, name: &str) -> usize {
        if self.globals.contains(name) {
            return self.node(format!("global:{name}"), NodeKind::Global(name.to_string()));
        }
        self.node(
            format!("net:{inst}|{name}"),
            NodeKind::Local {
                name: name.to_string(),
                inst: inst.clone(),
                declared: false,
                class: None,
                renamed_from: None,
            },
        )
    }

    fn port_node(&mut self, inst: &NamePath, port: &str) -> usize {
        self.node(
            format!("port:{inst}|{port}"),
            NodeKind::Port {
                name: port.to_string(),
                inst: inst.clone(),
            },
        )
    }

    /// Member names of a bus-like endpoint in a sheet, or `None` for scalar endpoints.
    fn members(&self, sheet: &SheetDef, ep: &Endpoint) -> Option<Vec<String>> {
        match ep {
            Endpoint::Net(n) => sheet
                .buses
                .iter()
                .find(|b| &b.name == n)
                .map(|b| b.members.clone())
                .or_else(|| {
                    let e = expand_bus(n);
                    (e.len() > 1).then_some(e)
                }),
            Endpoint::Port(p) => {
                sheet
                    .ports
                    .iter()
                    .find(|x| &x.name == p)
                    .and_then(|x| match &x.kind {
                        PortKind::Bus(m) => Some(m.clone()),
                        PortKind::Interface(i) => {
                            self.design.interfaces.get(i).map(|d| d.members.clone())
                        }
                        PortKind::Signal => None,
                    })
            }
            Endpoint::HarnessMember { .. } | Endpoint::BusMember { .. } | Endpoint::Pin { .. } => {
                None
            }
            Endpoint::InstancePort { instance, port, .. } => {
                let si = sheet.instances.iter().find(|i| &i.name == instance)?;
                let child = self.design.sheets.get(&si.sheet)?;
                child
                    .ports
                    .iter()
                    .find(|x| &x.name == port)
                    .and_then(|x| match &x.kind {
                        PortKind::Bus(m) => Some(m.clone()),
                        PortKind::Interface(i) => {
                            self.design.interfaces.get(i).map(|d| d.members.clone())
                        }
                        PortKind::Signal => None,
                    })
            }
        }
    }

    /// Nodes for an endpoint; `member` selects one member of a bus-like endpoint.
    fn resolve(
        &mut self,
        sheet: &SheetDef,
        inst: &NamePath,
        ep: &Endpoint,
        member: Option<&str>,
        src: &SourceKey,
    ) -> Vec<usize> {
        match ep {
            Endpoint::Net(n) => {
                let is_bus = self.members(sheet, ep).is_some();
                match (is_bus, member) {
                    (true, Some(m)) => vec![self.net_node(inst, m)],
                    (true, None) => {
                        let ms = self.members(sheet, ep).unwrap();
                        ms.iter().map(|m| self.net_node(inst, m)).collect()
                    }
                    (false, _) => vec![self.net_node(inst, n)],
                }
            }
            Endpoint::BusMember { member: m, .. } => vec![self.net_node(inst, m)],
            Endpoint::HarnessMember { harness, member: m } => {
                vec![self.net_node(inst, &format!("{harness}.{m}"))]
            }
            Endpoint::Port(p) => {
                if sheet.ports.iter().all(|x| &x.name != p) {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "unknown-port",
                            inst.child(p).to_string(),
                            format!("sheet `{}` has no port `{p}`", sheet.name),
                        )
                        .at(src),
                    );
                }
                match (self.members(sheet, ep), member) {
                    (Some(_), Some(m)) => vec![self.port_node(inst, &format!("{p}.{m}"))],
                    (Some(ms), None) => ms
                        .iter()
                        .map(|m| self.port_node(inst, &format!("{p}.{m}")))
                        .collect(),
                    (None, _) => vec![self.port_node(inst, p)],
                }
            }
            Endpoint::InstancePort {
                instance,
                port,
                channel,
            } => {
                let Some(si) = sheet.instances.iter().find(|i| &i.name == instance) else {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "unknown-instance",
                            inst.child(instance).to_string(),
                            format!("sheet `{}` has no instance `{instance}`", sheet.name),
                        )
                        .at(src),
                    );
                    return vec![];
                };
                let paths: Vec<NamePath> = match (si.repeat, channel) {
                    (Some(_), Some(c)) => vec![inst.child(format!("{instance}[{c}]"))],
                    (Some(n), None) => (0..n)
                        .map(|c| inst.child(format!("{instance}[{c}]")))
                        .collect(),
                    (None, _) => vec![inst.child(instance.clone())],
                };
                let ms = self.members(sheet, ep);
                let mut out = Vec::new();
                for p in paths {
                    match (&ms, member) {
                        (Some(_), Some(m)) => out.push(self.port_node(&p, &format!("{port}.{m}"))),
                        (Some(ms), None) => {
                            for m in ms {
                                out.push(self.port_node(&p, &format!("{port}.{m}")));
                            }
                        }
                        (None, _) => out.push(self.port_node(&p, port)),
                    }
                }
                out
            }
            Endpoint::Pin { component, pin } => {
                let path = inst.child(component.clone());
                let Some(&ci) = self.comp_index.get(&path) else {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "unknown-component",
                            path.to_string(),
                            format!("sheet `{}` has no component `{component}`", sheet.name),
                        )
                        .at(src),
                    );
                    return vec![];
                };
                let pins = find_pins(&self.comps[ci].pins, pin);
                match pins {
                    Ok(ps) => ps
                        .into_iter()
                        .map(|pi| {
                            self.node(
                                format!("pin:{ci}|{pi}"),
                                NodeKind::Pin { comp: ci, pin: pi },
                            )
                        })
                        .collect(),
                    Err(msg) => {
                        self.diags.push(
                            Diagnostic::new(Severity::Error, "unknown-pin", path.to_string(), msg)
                                .at(src),
                        );
                        vec![]
                    }
                }
            }
        }
    }

    fn connect_all(
        &mut self,
        sheet: &SheetDef,
        inst: &NamePath,
        ends: &[Endpoint],
        src: &SourceKey,
    ) {
        // Bus-like endpoints connect member by member with each other; scalars join a
        // single node.
        let widths: Vec<Option<Vec<String>>> =
            ends.iter().map(|e| self.members(sheet, e)).collect();
        let bus_eps: Vec<usize> = (0..ends.len()).filter(|i| widths[*i].is_some()).collect();
        if bus_eps.len() >= 2 {
            let w0 = widths[bus_eps[0]].as_ref().unwrap().clone();
            for &b in &bus_eps[1..] {
                let wb = widths[b].as_ref().unwrap();
                if wb.len() != w0.len() {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "bus-width-mismatch",
                            inst.to_string(),
                            format!("bus widths differ: {} vs {}", w0.len(), wb.len()),
                        )
                        .at(src),
                    );
                    return;
                }
            }
            for k in 0..w0.len() {
                let mut nodes = Vec::new();
                for &b in &bus_eps {
                    let m = widths[b].as_ref().unwrap()[k].clone();
                    nodes.extend(self.resolve(sheet, inst, &ends[b], Some(&m), src));
                }
                for w in nodes.windows(2) {
                    self.uf.union(w[0], w[1]);
                }
            }
            if bus_eps.len() == ends.len() {
                return;
            }
        }
        let mut nodes = Vec::new();
        for (i, e) in ends.iter().enumerate() {
            if bus_eps.len() >= 2 && bus_eps.contains(&i) {
                continue;
            }
            nodes.extend(self.resolve(sheet, inst, e, None, src));
        }
        for w in nodes.windows(2) {
            self.uf.union(w[0], w[1]);
        }
    }

    fn fitting(&self, path: &NamePath, decl: &ComponentDecl) -> Fitting {
        let mut v = self.opts.variant.clone();
        let mut seen = BTreeSet::new();
        while let Some(name) = v {
            if !seen.insert(name.clone()) {
                break;
            }
            if let Some(vd) = self.design.variants.get(&name) {
                let key_path = path.to_string();
                if let Some(f) = vd
                    .overrides
                    .get(&key_path)
                    .or_else(|| vd.overrides.get(&decl.name))
                {
                    return f.clone();
                }
            }
            if let Some(f) = decl.variants.get(&name) {
                return f.clone();
            }
            v = self
                .design
                .variants
                .get(&name)
                .and_then(|vd| vd.base.clone());
        }
        Fitting::Fitted
    }

    /// `old` is the instance's previous path when it or an ancestor was renamed.
    fn walk(
        &mut self,
        sheet_name: &str,
        inst: NamePath,
        channel: Option<u32>,
        depth: usize,
        old: Option<NamePath>,
    ) {
        if depth > 64 {
            self.diags.push(Diagnostic::new(
                Severity::Error,
                "recursive-hierarchy",
                inst.to_string(),
                "sheet hierarchy is recursive or deeper than 64 levels",
            ));
            return;
        }
        let Some(sheet) = self.design.sheets.get(sheet_name) else {
            self.diags.push(Diagnostic::new(
                Severity::Error,
                "unknown-sheet",
                inst.to_string(),
                format!("no sheet definition `{sheet_name}`"),
            ));
            return;
        };
        let sheet: &'a SheetDef = sheet;
        self.sheets.push((inst.clone(), sheet_name.to_string()));
        if let Some(o) = &old {
            self.sheet_renames.insert(inst.clone(), o.clone());
        }
        // Declared nets.
        for n in &sheet.nets {
            if n.global {
                continue;
            }
            for m in expand_bus(&n.name) {
                self.node(
                    format!("net:{inst}|{m}"),
                    NodeKind::Local {
                        name: m.clone(),
                        inst: inst.clone(),
                        declared: true,
                        class: n.class.clone(),
                        renamed_from: n
                            .renamed_from
                            .clone()
                            .or_else(|| old.as_ref().map(|_| m.clone())),
                    },
                );
            }
        }
        for b in &sheet.buses {
            for m in &b.members {
                self.node(
                    format!("net:{inst}|{m}"),
                    NodeKind::Local {
                        name: m.clone(),
                        inst: inst.clone(),
                        declared: true,
                        class: None,
                        renamed_from: None,
                    },
                );
            }
        }
        for h in &sheet.harnesses {
            match self.design.interfaces.get(&h.interface) {
                Some(i) => {
                    for m in &i.members {
                        let name = format!("{}.{m}", h.name);
                        self.node(
                            format!("net:{inst}|{name}"),
                            NodeKind::Local {
                                name,
                                inst: inst.clone(),
                                declared: true,
                                class: None,
                                renamed_from: None,
                            },
                        );
                    }
                    for (p, n) in &i.diff_pairs {
                        self.diff.push((
                            inst.clone(),
                            DiffPairDecl {
                                name: format!("{}.{}", h.name, p.trim_end_matches(['P', 'p', '+'])),
                                p: format!("{}.{p}", h.name),
                                n: format!("{}.{n}", h.name),
                                class: None,
                                source: h.source.clone(),
                            },
                        ));
                    }
                }
                None => self.diags.push(
                    Diagnostic::new(
                        Severity::Error,
                        "unknown-interface",
                        inst.child(h.name.clone()).to_string(),
                        format!("no interface type `{}`", h.interface),
                    )
                    .at(&h.source),
                ),
            }
        }
        // Components.
        for decl in &sheet.components {
            let path = inst.child(decl.name.clone());
            let fit = self.fitting(&path, decl);
            let (part_ref, mut params, state) = match &fit {
                Fitting::Fitted => (decl.part.clone(), decl.params.clone(), FitState::Fitted),
                Fitting::NotFitted => (decl.part.clone(), decl.params.clone(), FitState::NotFitted),
                Fitting::Alternate { part, params } => {
                    let mut p = decl.params.clone();
                    for (k, v) in params {
                        p.insert(k.clone(), v.clone());
                    }
                    (
                        part.clone().unwrap_or_else(|| decl.part.clone()),
                        p,
                        FitState::Alternate,
                    )
                }
            };
            let (part, entity, package, pins) = self.resolve_part(&part_ref, &path, &decl.source);
            primary_value(&mut params);
            if let Some(p) = part.and_then(|u| self.pool.parts.get(&u)) {
                for (k, v) in &self.pool.effective_part(p).params {
                    params.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            let ci = self.comps.len();
            self.comps.push(CompInst {
                path: path.clone(),
                designator: String::new(),
                sheet: sheet_name.to_string(),
                channel,
                part,
                entity,
                package,
                params,
                properties: decl.properties.clone(),
                classes: decl.classes.clone(),
                power: decl.power,
                fitting: state,
                pins: pins
                    .into_iter()
                    .map(|r| PinInst {
                        gate: r.gate,
                        pin: r.pin,
                        kind: r.kind,
                        pads: r.pads,
                        net: None,
                    })
                    .collect(),
                renamed_from: match (&decl.renamed_from, &old) {
                    (Some(r), o) => Some(o.as_ref().unwrap_or(&inst).child(r.clone())),
                    (None, Some(o)) => Some(o.child(decl.name.clone())),
                    (None, None) => None,
                },
                source: decl.source.clone(),
            });
            self.comp_index.insert(path, ci);
            let repeated = channel.is_some() || inst.0.iter().any(|s| s.ends_with(']'));
            if !decl.name.contains('?') && !repeated {
                self.explicit_designator.insert(ci, decl.name.clone());
            }
        }
        // Sub-sheets first so their ports exist; bindings after.
        for si in &sheet.instances {
            let n = si.repeat.unwrap_or(1);
            for ch in 0..n {
                let (child, chan) = match si.repeat {
                    Some(_) => (inst.child(format!("{}[{ch}]", si.name)), Some(ch)),
                    None => (inst.child(si.name.clone()), channel),
                };
                let old_child = match (&si.renamed_from, &old) {
                    (None, None) => None,
                    (r, o) => {
                        let base = o.clone().unwrap_or_else(|| inst.clone());
                        let nm = r.clone().unwrap_or_else(|| si.name.clone());
                        Some(match si.repeat {
                            Some(_) => base.child(format!("{nm}[{ch}]")),
                            None => base.child(nm),
                        })
                    }
                };
                self.walk(&si.sheet, child.clone(), chan, depth + 1, old_child);
                let child_def: Option<&'a SheetDef> = self.design.sheets.get(&si.sheet);
                let Some(child_def) = child_def else { continue };
                for (port, binding) in &si.bindings {
                    let Some(pd) = child_def.ports.iter().find(|p| &p.name == port) else {
                        self.diags.push(
                            Diagnostic::new(
                                Severity::Error,
                                "unknown-port",
                                child.child(port.clone()).to_string(),
                                format!("sheet `{}` has no port `{port}`", si.sheet),
                            )
                            .at(&si.source),
                        );
                        continue;
                    };
                    let members: Option<Vec<String>> = match &pd.kind {
                        PortKind::Signal => None,
                        PortKind::Bus(m) => Some(m.clone()),
                        PortKind::Interface(i) => {
                            self.design.interfaces.get(i).map(|d| d.members.clone())
                        }
                    };
                    let child_nodes: Vec<usize> = match &members {
                        None => vec![self.port_node(&child, port)],
                        Some(ms) => ms
                            .iter()
                            .map(|m| self.port_node(&child, &format!("{port}.{m}")))
                            .collect(),
                    };
                    let parent_ep = match binding {
                        Binding::Shared(ep) => ep.clone(),
                        Binding::BusIndexed { bus, offset } => {
                            let ms = sheet
                                .buses
                                .iter()
                                .find(|b| &b.name == bus)
                                .map(|b| b.members.clone())
                                .unwrap_or_else(|| expand_bus(bus));
                            match ms.get((offset + ch) as usize) {
                                Some(m) => Endpoint::BusMember {
                                    bus: bus.clone(),
                                    member: m.clone(),
                                },
                                None => {
                                    self.diags.push(
                                        Diagnostic::new(
                                            Severity::Error,
                                            "bus-index",
                                            child.to_string(),
                                            format!("bus `{bus}` has no member {}", offset + ch),
                                        )
                                        .at(&si.source),
                                    );
                                    continue;
                                }
                            }
                        }
                        Binding::NetIndexed { prefix, offset } => {
                            Endpoint::Net(format!("{prefix}{}", offset + ch))
                        }
                    };
                    let parent_nodes: Vec<usize> = match &members {
                        None => self.resolve(sheet, &inst, &parent_ep, None, &si.source),
                        Some(ms) => {
                            let pw = self.members(sheet, &parent_ep);
                            match pw {
                                Some(pm) if pm.len() == ms.len() => pm
                                    .iter()
                                    .flat_map(|m| {
                                        self.resolve(sheet, &inst, &parent_ep, Some(m), &si.source)
                                    })
                                    .collect(),
                                Some(pm) => {
                                    self.diags.push(
                                        Diagnostic::new(
                                            Severity::Error,
                                            "bus-width-mismatch",
                                            child.child(port.clone()).to_string(),
                                            format!(
                                                "port has {} members, binding has {}",
                                                ms.len(),
                                                pm.len()
                                            ),
                                        )
                                        .at(&si.source),
                                    );
                                    continue;
                                }
                                None => {
                                    self.diags.push(
                                        Diagnostic::new(
                                            Severity::Error,
                                            "bus-binding",
                                            child.child(port.clone()).to_string(),
                                            "a bus or harness port must bind to a bus or harness",
                                        )
                                        .at(&si.source),
                                    );
                                    continue;
                                }
                            }
                        }
                    };
                    for (a, b) in child_nodes.iter().zip(parent_nodes.iter()) {
                        self.uf.union(*a, *b);
                    }
                }
            }
        }
        for c in &sheet.connections {
            self.connect_all(sheet, &inst, &c.ends, &c.source);
        }
        for dp in &sheet.diff_pairs {
            self.diff.push((inst.clone(), dp.clone()));
        }
        for ep in &sheet.no_erc {
            if let Endpoint::Pin { component, pin } = ep {
                if let Some(&ci) = self.comp_index.get(&inst.child(component.clone())) {
                    if let Ok(ps) = find_pins(&self.comps[ci].pins, pin) {
                        for pi in ps {
                            self.no_erc.push((ci, pi));
                        }
                    }
                }
            }
        }
    }

    fn resolve_part(
        &mut self,
        r: &PartRef,
        path: &NamePath,
        src: &SourceKey,
    ) -> (Option<Uuid>, Option<Uuid>, Option<Uuid>, Vec<ResolvedPin>) {
        match r {
            PartRef::Part(name) => match self.pool.part(name) {
                Some(p) => {
                    let p = self.pool.effective_part(p);
                    let pins = self
                        .pool
                        .entities
                        .get(&p.entity)
                        .map(|e| self.pool.pins(e, Some(&p)))
                        .unwrap_or_default();
                    (Some(p.uuid), Some(p.entity), Some(p.package), pins)
                }
                None => {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "unknown-part",
                            path.to_string(),
                            format!("no part `{name}` in the pool"),
                        )
                        .at(src),
                    );
                    (None, None, None, vec![])
                }
            },
            PartRef::Entity { entity, package } => {
                let Some(e) = self.pool.entity(entity) else {
                    self.diags.push(
                        Diagnostic::new(
                            Severity::Error,
                            "unknown-entity",
                            path.to_string(),
                            format!("no entity `{entity}` in the pool"),
                        )
                        .at(src),
                    );
                    return (None, None, None, vec![]);
                };
                let pkg = package.as_ref().and_then(|p| self.pool.package(p));
                let mut pins = self.pool.pins(e, None);
                if let Some(pkg) = pkg {
                    // Without a part, pins map to pads of the same name.
                    for p in &mut pins {
                        if pkg.pad(&p.pin).is_some() {
                            p.pads = vec![p.pin.clone()];
                        }
                    }
                }
                (None, Some(e.uuid), pkg.map(|p| p.uuid), pins)
            }
        }
    }
}

/// Resolves a pin reference: gate-qualified name, unique pin name (power pins repeated across
/// gates all match), or pad number.
pub fn find_pins(pins: &[PinInst], pin: &str) -> Result<Vec<usize>, String> {
    if let Some((g, p)) = pin.split_once('.') {
        if let Some(i) = pins.iter().position(|x| x.gate == g && x.pin == p) {
            return Ok(vec![i]);
        }
    }
    let by_name: Vec<usize> = (0..pins.len()).filter(|i| pins[*i].pin == pin).collect();
    if by_name.len() == 1 {
        return Ok(by_name);
    }
    if by_name.len() > 1 {
        let all_power = by_name.iter().all(|i| {
            matches!(
                pins[*i].kind,
                PinType::PowerIn | PinType::PowerOut | PinType::Passive
            )
        });
        if all_power {
            return Ok(by_name);
        }
        return Err(format!(
            "pin `{pin}` is ambiguous across gates {}; qualify it as `<gate>.{pin}`",
            by_name
                .iter()
                .map(|i| pins[*i].gate.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let by_pad: Vec<usize> = (0..pins.len())
        .filter(|i| pins[*i].pads.iter().any(|p| p == pin))
        .collect();
    if !by_pad.is_empty() {
        return Ok(by_pad);
    }
    Err(format!("no pin `{pin}`"))
}

fn prefix_of(name: &str) -> String {
    let p: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if p.is_empty() {
        "U".to_string()
    } else {
        p
    }
}

/// Flattens `design` against `pool`.
pub fn elaborate(
    design: &Design,
    pool: &Pool,
    opts: &ElabOptions,
    lock: Option<&LockSection>,
) -> Netlist {
    let mut cx = Ctx {
        design,
        pool,
        opts,
        lock,
        keys: BTreeMap::new(),
        nodes: Vec::new(),
        uf: Uf { parent: Vec::new() },
        comps: Vec::new(),
        comp_index: BTreeMap::new(),
        globals: BTreeSet::new(),
        diags: Vec::new(),
        sheets: Vec::new(),
        diff: Vec::new(),
        no_erc: Vec::new(),
        explicit_designator: BTreeMap::new(),
        sheet_renames: BTreeMap::new(),
    };
    let mut global_class: BTreeMap<String, Option<String>> = BTreeMap::new();
    for s in design.sheets.values() {
        for n in &s.nets {
            if n.global {
                cx.globals.insert(n.name.clone());
                let e = global_class.entry(n.name.clone()).or_insert(None);
                if n.class.is_some() {
                    *e = n.class.clone();
                }
            }
        }
    }
    if let Some(v) = &opts.variant {
        if !design.variants.contains_key(v) {
            cx.diags.push(Diagnostic::new(
                Severity::Error,
                "unknown-variant",
                v.clone(),
                format!("no variant `{v}`"),
            ));
        }
    }
    cx.walk(&design.root.clone(), NamePath::root(), None, 0, None);

    // Every pin gets a node so unconnected pins become single-pin nets only if connected.
    // Group nodes by root.
    let n_nodes = cx.nodes.len();
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n_nodes {
        let r = cx.uf.find(i);
        groups.entry(r).or_default().push(i);
    }
    let mut nets: Vec<NetInst> = Vec::new();
    let mut taken: BTreeMap<String, usize> = BTreeMap::new();
    let designators = annotate(&mut cx);
    for (_, members) in groups {
        let mut pins = Vec::new();
        let mut cands: Vec<NameCandidate> = Vec::new();
        for &m in &members {
            match &cx.nodes[m] {
                NodeKind::Pin { comp, pin } => pins.push((*comp, *pin)),
                NodeKind::Global(n) => cands.push((
                    (0, 0, n.clone()),
                    n.clone(),
                    NamePath::parse(n),
                    global_class.get(n).cloned().flatten(),
                    true,
                    None,
                )),
                NodeKind::Local {
                    name,
                    inst,
                    declared,
                    class,
                    renamed_from,
                } => {
                    let full = if inst.depth() == 0 {
                        name.clone()
                    } else {
                        format!("{inst}/{name}")
                    };
                    cands.push((
                        (if *declared { 1 } else { 2 }, inst.depth(), full.clone()),
                        full,
                        inst.child(name.clone()),
                        class.clone(),
                        false,
                        renamed_from.clone().map(|r| {
                            cx.sheet_renames
                                .get(inst)
                                .unwrap_or(inst)
                                .child(r)
                                .to_string()
                        }),
                    ))
                }
                NodeKind::Port { name, inst } => {
                    let full = format!("{inst}/{name}");
                    cands.push((
                        (3, inst.depth(), full.clone()),
                        full,
                        inst.child(name.clone()),
                        None,
                        false,
                        None,
                    ))
                }
            }
        }
        if pins.is_empty() {
            // Nets with no pins exist only when declared at the root or global (rails with no
            // loads are reported by ERC as unused); drop port-only fragments.
            if !cands.iter().any(|c| c.0 .0 <= 1) {
                continue;
            }
        }
        pins.sort();
        cands.sort_by(|a, b| a.0.cmp(&b.0));
        let (name, path, global, renamed) = match cands.first() {
            Some(c) => (c.1.clone(), c.2.clone(), c.4, c.5.clone()),
            None => {
                let (ci, pi) = pins[0];
                let pad = cx.comps[ci].pins[pi]
                    .pads
                    .first()
                    .cloned()
                    .unwrap_or_else(|| cx.comps[ci].pins[pi].pin.clone());
                let n = format!("Net-({}-Pad{pad})", designators[ci]);
                (n.clone(), NamePath::parse(&n), false, None)
            }
        };
        let mut classes: Vec<(u8, usize, String)> = cands
            .iter()
            .filter_map(|c| c.3.clone().map(|cl| (c.0 .0, c.0 .1, cl)))
            .collect();
        classes.sort();
        classes.dedup_by(|a, b| a.2 == b.2);
        if classes.len() > 1 {
            cx.diags.push(Diagnostic::new(
                Severity::Warning,
                "net-class-conflict",
                name.clone(),
                format!(
                    "net carries classes {}; using `{}`",
                    classes
                        .iter()
                        .map(|c| c.2.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                    classes[0].2
                ),
            ));
        }
        let mut aliases: Vec<String> = cands.iter().skip(1).map(|c| c.1.clone()).collect();
        aliases.sort();
        aliases.dedup();
        aliases.retain(|a| a != &name);
        let mut final_name = name.clone();
        if let Some(prev) = taken.get(&name) {
            cx.diags.push(
                Diagnostic::new(
                    Severity::Warning,
                    "net-name-collision",
                    name.clone(),
                    "two distinct nets have the same name; the second is suffixed",
                )
                .with(nets[*prev].name.clone()),
            );
            final_name = format!("{name}#{}", nets.len());
        }
        taken.insert(final_name.clone(), nets.len());
        nets.push(NetInst {
            name: final_name,
            path,
            class: classes.first().map(|c| c.2.clone()),
            global,
            aliases,
            pins,
            renamed_from: renamed.map(|r| NamePath::parse(&r)),
        });
    }
    nets.sort_by(|a, b| a.name.cmp(&b.name));
    for (ni, n) in nets.iter().enumerate() {
        for &(ci, pi) in &n.pins {
            cx.comps[ci].pins[pi].net = Some(ni);
        }
    }
    for (ci, d) in designators.into_iter().enumerate() {
        cx.comps[ci].designator = d;
    }
    // Net classes must exist.
    for n in &nets {
        if let Some(c) = &n.class {
            if !design.net_classes.contains_key(c) && c != "default" {
                cx.diags.push(Diagnostic::new(
                    Severity::Warning,
                    "unknown-net-class",
                    n.name.clone(),
                    format!("net class `{c}` is not defined"),
                ));
            }
        }
    }
    // Differential pairs.
    let mut diff_pairs = Vec::new();
    let mut by_alias: BTreeMap<String, usize> = BTreeMap::new();
    for (i, n) in nets.iter().enumerate() {
        by_alias.insert(n.name.clone(), i);
        for a in &n.aliases {
            by_alias.entry(a.clone()).or_insert(i);
        }
    }
    let local_name = |inst: &NamePath, n: &str| {
        if inst.depth() == 0 || cx.globals.contains(n) {
            n.to_string()
        } else {
            format!("{inst}/{n}")
        }
    };
    for (inst, dp) in &cx.diff {
        let p = by_alias.get(&local_name(inst, &dp.p));
        let n = by_alias.get(&local_name(inst, &dp.n));
        match (p, n) {
            (Some(&p), Some(&n)) if p != n => diff_pairs.push(DiffPairInst {
                name: local_name(inst, &dp.name),
                p,
                n,
                class: dp.class.clone(),
            }),
            (Some(&p), Some(&n)) if p == n => cx.diags.push(
                Diagnostic::new(
                    Severity::Error,
                    "diff-pair-shorted",
                    local_name(inst, &dp.name),
                    "both sides of the differential pair are the same net",
                )
                .at(&dp.source),
            ),
            _ => cx.diags.push(
                Diagnostic::new(
                    Severity::Error,
                    "diff-pair-net",
                    local_name(inst, &dp.name),
                    format!("differential pair nets `{}`/`{}` not found", dp.p, dp.n),
                )
                .at(&dp.source),
            ),
        }
    }
    diff_pairs.sort_by(|a, b| a.name.cmp(&b.name));
    let mut no_erc: BTreeSet<(usize, usize)> = cx.no_erc.iter().copied().collect();
    no_erc.retain(|(c, p)| *c < cx.comps.len() && *p < cx.comps[*c].pins.len());
    let mut diags = cx.diags;
    diags.sort();
    let mut sheets = cx.sheets;
    sheets.sort();
    Netlist {
        design: design.name.clone(),
        variant: opts.variant.clone(),
        components: cx.comps,
        nets,
        diff_pairs,
        sheets,
        no_erc,
        sheet_renames: cx.sheet_renames,
        diagnostics: diags,
        sch_places: lock.map(|l| l.places.clone()).unwrap_or_default(),
        sch_wires: lock.map(|l| l.wires.clone()).unwrap_or_default(),
        sheet_texts: design
            .sheets
            .iter()
            .filter(|(_, s)| !s.texts.is_empty())
            .map(|(n, s)| (n.clone(), s.texts.iter().map(|t| (t.name.clone(), t.text.clone())).collect()))
            .collect(),
    }
}

/// Net-name candidate: (priority key, full name, path, class, global, renamed-from).
type NameCandidate = (
    (u8, usize, String),
    String,
    NamePath,
    Option<String>,
    bool,
    Option<String>,
);

fn annotate(cx: &mut Ctx) -> Vec<String> {
    let n = cx.comps.len();
    let mut out = vec![String::new(); n];
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| cx.comps[*a].path.cmp(&cx.comps[*b].path));
    // Pass 1: explicit unique names.
    let mut count: BTreeMap<String, usize> = BTreeMap::new();
    for d in cx.explicit_designator.values() {
        *count.entry(d.clone()).or_default() += 1;
    }
    for &i in &order {
        if let Some(d) = cx.explicit_designator.get(&i) {
            if count[d] == 1 && used.insert(d.clone()) {
                out[i] = d.clone();
            } else {
                cx.diags.push(Diagnostic::new(
                    Severity::Warning,
                    "duplicate-designator",
                    cx.comps[i].path.to_string(),
                    format!("designator `{d}` is used more than once; renumbering"),
                ));
            }
        }
    }
    // Pass 2: lock-recorded designators.
    if let Some(lock) = cx.lock {
        for &i in &order {
            if !out[i].is_empty() {
                continue;
            }
            if let Some(d) = lock
                .entries
                .get(&cx.comps[i].path)
                .and_then(|e| e.designator.clone())
            {
                if used.insert(d.clone()) {
                    out[i] = d;
                }
            }
        }
    }
    // Pass 3: number the rest per prefix.
    let mut next: BTreeMap<String, u32> = BTreeMap::new();
    for &i in &order {
        if !out[i].is_empty() {
            continue;
        }
        let local = cx.comps[i].path.leaf().to_string();
        // The entity's prefix (R, C, U) when known; else the letters of the local name.
        let prefix = cx.comps[i]
            .entity
            .and_then(|e| cx.pool.entities.get(&e))
            .map(|e| e.prefix.clone())
            .unwrap_or_else(|| prefix_of(&local));
        let k = next.entry(prefix.clone()).or_insert(1);
        loop {
            let d = format!("{prefix}{k}");
            *k += 1;
            if used.insert(d.clone()) {
                out[i] = d;
                break;
            }
        }
    }
    out
}

/// A component's written `value` (`value = 9.95 kohm`, or one an `.m` function computed) is its
/// primary parameter: a resistance, capacitance or inductance by its dimension. It goes in under
/// that name ahead of the part's own, so the simulation, the BOM and the checks read the
/// component's value, not the part's nominal one.
fn primary_value(params: &mut BTreeMap<String, Param>) {
    use runmat_rules::Dim;
    let Some(value) = params.get("value").cloned() else { return };
    let Some(q) = value.as_quantity() else { return };
    let key = match q.dim {
        d if d == Dim::RESISTANCE => "resistance",
        d if d == Dim::CAPACITANCE => "capacitance",
        d if d == Dim::INDUCTANCE => "inductance",
        _ => return,
    };
    // The written `value` stays the one that records how it was computed (the index's
    // `computed` rows, the hover); the primary parameter is its number.
    params.insert(key.to_owned(), Param { call: None, ..value });
}

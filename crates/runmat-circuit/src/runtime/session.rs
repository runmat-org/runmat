//! A document runtime that keeps what it compiled.
//!
//! [`Session`] answers the same calls as the stateless [`compile`](super::compile) and
//! [`edit`](super::edit), with the same results, and keeps the last circuit's state between
//! calls: the schematic's netlist and projection, and the board's reading, rules, fill tiles,
//! DRC and ECO plans. A call whose schematic files are unchanged reuses the schematic; a
//! layout whose text changed in a few item statements is re-read there only
//! ([`reparse_board`]), and the board's rules, fills (the tiles the change reaches), DRC (the
//! changed items, their neighbours and nets) and ECO plans update from that change.
//!
//! The Desktop's worker holds one session, so a small edit to a 50k-item board recompiles in
//! a fraction of a second where a fresh compile takes seconds.

use crate::layer_set::CopperIndex;
use super::*;
use crate::board_rules::BItem;
use crate::syntax::board::{reparse_board_keeping, BoardDelta, BoardItem};

mod reuse;
mod since;
pub use since::Since;
use crate::syntax::board_edit::board_command_parsed;
use crate::zone::{fill_zones_cached, FillCache, FillDirty, FillReport};
use std::collections::HashSet;
use std::time::Duration;

/// Where the last call spent its time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionTimings {
    pub schematic: Duration,
    pub read: Duration,
    pub rules: Duration,
    pub fill: Duration,
    pub drc: Duration,
    pub eco: Duration,
    pub index: Duration,
    /// The board was compiled from scratch (first call, or a change the session can't patch).
    pub whole: bool,
}

struct SchematicState {
    circuit_path: String,
    /// The schematic's inputs: every file but the layout, and the variant.
    inputs: Files,
    variant: Option<String>,
    /// The `.m` value results it was built with.
    values: super::values::Values,
    project: Project,
    index: CircuitIndex,
}

struct BoardState {
    path: String,
    text: String,
    parsed: ParsedBoard,
    rules: BoardRules,
    rule_errors: Vec<Diagnostic>,
    fills: FillCache,
    reports: Vec<FillReport>,
    drc: Drc,
    eco: Option<EcoSettings>,
    plans: (EcoPlan, EcoPlan),
    index: CircuitIndex,
}

/// The runtime with the last circuit's compiled state.
#[derive(Default)]
pub struct Session {
    schematic: Option<SchematicState>,
    board: Option<BoardState>,
    pub timings: SessionTimings,
    /// The projection's number: it goes up by one each time the projection may have changed.
    generation: u64,
    /// The change from projection `generation - 1` to `generation`, when the session made one.
    last_delta: Option<serde_json::Value>,
    /// Set while a sync runs: `Some` when the projection may have changed, with its delta when known.
    change: Option<Option<serde_json::Value>>,
    /// The projection a whole recompile replaces, for its delta.
    prev: Option<CircuitIndex>,
}

/// A stopwatch for [`SessionTimings`]; wasm32 has no clock in `std`, so it reads zero there.
struct Clock(#[cfg(not(target_arch = "wasm32"))] std::time::Instant);

impl Clock {
    fn elapsed(&self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        return self.0.elapsed();
        #[cfg(target_arch = "wasm32")]
        Duration::ZERO
    }
}

fn now() -> Clock {
    Clock(
        #[cfg(not(target_arch = "wasm32"))]
        std::time::Instant::now(),
    )
}

/// The copper layers an item is on (all of them for footprints, whose pads may be through).
fn item_layers(board: &Board, item: &BoardItem) -> Vec<CopperIndex> {
    let n = board.copper_count();
    match item {
        BoardItem::Track(t) => vec![t.layer],
        BoardItem::Via(v) => (v.from..=v.to).collect(),
        BoardItem::Footprint(_) | BoardItem::Zone(_) | BoardItem::Keepout(_) => (0..n).collect(),
        _ => Vec::new(),
    }
}

fn item_bbox(board: &Board, item: &BoardItem) -> crate::geom::Rect {
    use crate::geom::Rect;
    match item {
        BoardItem::Track(t) => t.prims().iter().fold(Rect::EMPTY, |r, p| r.union(p.bbox())),
        BoardItem::Via(v) => Rect::new(v.at, v.at).inflate(v.diameter / 2),
        BoardItem::Footprint(f) => {
            let n = board.copper_count();
            f.pads(n)
                .iter()
                .fold(f.bbox(), |r, p| r.union(p.prim.bbox()))
        }
        BoardItem::Keepout(k) => k.outline.bbox(),
        BoardItem::Zone(z) => z.outline.bbox(),
        _ => Rect::EMPTY,
    }
}

/// The DRC and rules items of a board item: a footprint with its pads.
fn bitems(board: &Board, item: &BoardItem, out: &mut Vec<BItem>) {
    match item {
        BoardItem::Footprint(f) => {
            out.push(BItem::Footprint(f.id));
            let n = board.copper_count();
            out.extend(f.pads(n).iter().map(|p| BItem::Pad(f.id, p.index as u32)));
        }
        BoardItem::Track(t) => out.push(BItem::Track(t.id)),
        BoardItem::Via(v) => out.push(BItem::Via(v.id)),
        BoardItem::Zone(z) => out.push(BItem::Zone(z.id)),
        _ => {}
    }
}

impl Session {
    pub fn new() -> Self {
        Session::default()
    }

    /// The schematic for `circuit_path`, reused when its inputs are the ones it was built from;
    /// `true` when it was built anew.
    fn schematic(
        &mut self,
        path: &str,
        text: &str,
        circuit_path: &str,
        layout_path: &str,
        files: &Files,
        options: &CompileOptions,
    ) -> Result<bool, Vec<Diagnostic>> {
        let mut inputs: Files = files
            .iter()
            .filter(|(p, _)| p.as_str() != layout_path)
            .map(|(p, t)| (p.clone(), t.clone()))
            .collect();
        if !is_layout(path) {
            inputs.insert(path.to_owned(), text.to_owned());
        }
        if let Some(s) = &self.schematic {
            if s.circuit_path == circuit_path
                && s.inputs == inputs
                && s.variant == options.variant
                && s.values == options.values
            {
                return Ok(false);
            }
        }
        self.schematic = None;
        let circuit_text = inputs.get(circuit_path).cloned().ok_or_else(|| {
            vec![Diagnostic::error(
                path,
                Some(text),
                0,
                0,
                format!("{circuit_path} can't be read; the layout's `use circuit` line names it"),
            )]
        })?;
        let project = load_with(circuit_path, &circuit_text, &inputs, options, false)?;
        let mut erc = project.netlist.diagnostics.clone();
        erc.extend(erc::run(
            &project.netlist,
            &ErcOptions::for_design(&project.design, &project.netlist),
        ));
        let findings = review::run_all(&project.netlist, &project.pool, &ReviewOptions::default());
        let mut index = build_index(
            &project.design,
            &project.netlist,
            &project.pool,
            &erc,
            &findings,
        );
        super::values::note_files(&mut index, &inputs);
        self.schematic = Some(SchematicState {
            circuit_path: circuit_path.to_owned(),
            inputs,
            variant: options.variant.clone(),
            values: options.values.clone(),
            project,
            index,
        });
        Ok(true)
    }

    /// Brings the session to `path`/`text` and `files`; `None` diagnostics when it compiles.
    /// Numbers the projection: a sync that may have changed it takes the next generation.
    fn sync(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        options: &CompileOptions,
    ) -> Result<(), Vec<Diagnostic>> {
        self.change = None;
        let out = self.sync_inner(path, text, files, options);
        self.prev = None;
        if out.is_err() {
            self.change = Some(None);
        }
        if let Some(delta) = self.change.take() {
            self.generation += 1;
            self.last_delta = delta;
        }
        out
    }

    fn sync_inner(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        options: &CompileOptions,
    ) -> Result<(), Vec<Diagnostic>> {
        self.timings = SessionTimings::default();
        let t = now();
        let (circuit_path, layout_path) = if is_layout(path) {
            let circuit = layout_circuit(path, &head_uses(Dialect::CircuitLayout, text), files);
            let Some(Ok(circuit)) = circuit else {
                // The whole reader reports a missing or unresolved `use circuit` line.
                return load(path, text, files, options).map(|_| ());
            };
            (circuit, path.to_owned())
        } else {
            (path.to_owned(), layout_path_for(path))
        };
        let layout_text = if is_layout(path) {
            Some(text)
        } else {
            files.get(&layout_path).map(String::as_str)
        };
        let rebuilt = self.schematic(path, text, &circuit_path, &layout_path, files, options)?;
        self.timings.schematic = t.elapsed();
        let Some(layout_text) = layout_text else {
            if rebuilt || self.board.is_some() {
                self.change = Some(None);
            }
            self.board = None;
            return Ok(());
        };
        // Read the layout: re-read the change when the session holds this layout.
        let t = now();
        let held = self.board.take().filter(|b| b.path == layout_path);
        let mut kept: Option<(String, ParsedBoard)> = None;
        let (state, delta) = match held {
            Some(mut b) if b.text == layout_text && !rebuilt && b.eco == options.eco => {
                self.timings.read = t.elapsed();
                b.text = layout_text.to_owned();
                self.board = Some(b);
                return Ok(());
            }
            Some(b) if !rebuilt => {
                let BoardState {
                    text: old_text,
                    parsed,
                    rules,
                    rule_errors,
                    fills,
                    reports,
                    drc,
                    eco,
                    plans,
                    index,
                    ..
                } = b;
                let (parsed, delta, old_parsed) =
                    reparse_board_keeping(&layout_path, &old_text, parsed, layout_text);
                // A whole re-read keeps the old reading, to keep the fill tiles it didn't reach.
                kept = old_parsed.map(|p| (old_text, p));
                (
                    Some((
                        parsed,
                        rules,
                        rule_errors,
                        fills,
                        reports,
                        drc,
                        eco,
                        plans,
                        index,
                    )),
                    Some(delta),
                )
            }
            other => {
                // A whole recompile: the delta is taken against the projection it replaces.
                self.prev = other.map(|b| b.index);
                (None, None)
            }
        };
        self.timings.read = t.elapsed();
        match (state, delta) {
            (Some(s), Some(delta)) if !delta.whole && s.0.diagnostics.is_empty() => {
                self.patch(layout_path, layout_text, s, delta, options);
            }
            (Some(s), _) if !s.0.diagnostics.is_empty() => {
                let diags =
                    s.0.diagnostics
                        .iter()
                        .map(|d| Diagnostic::syntax(&layout_path, layout_text, d))
                        .collect();
                return Err(diags);
            }
            (Some(s), Some(_)) => {
                // A whole re-read: it is the parse a fresh compile does, and the fills keep
                // the tiles the change didn't reach.
                let reuse = kept.map(|(text, old)| (text, old, s.3));
                self.prev = Some(s.8);
                self.whole(layout_path, layout_text, Some(s.0), reuse, options)?;
            }
            _ => self.whole(layout_path, layout_text, None, None, options)?,
        }
        Ok(())
    }

    fn whole(
        &mut self,
        path: String,
        text: &str,
        parsed: Option<ParsedBoard>,
        reuse: Option<(String, ParsedBoard, FillCache)>,
        options: &CompileOptions,
    ) -> Result<(), Vec<Diagnostic>> {
        self.timings.whole = true;
        let t = now();
        let mut parsed = parsed.unwrap_or_else(|| parse_board(&path, text));
        if !parsed.diagnostics.is_empty() {
            return Err(parsed
                .diagnostics
                .iter()
                .map(|d| Diagnostic::syntax(&path, text, d))
                .collect());
        }
        self.timings.read += t.elapsed();
        let t = now();
        // The circuit's declared pairs: DRC checks their gap, coupling and skew.
        let pairs = match (&self.schematic, parsed.board.as_ref()) {
            (Some(s), Some(board)) => super::coupled::pair_nets(&s.project, board),
            _ => Vec::new(),
        };
        let (rules, rule_errors) = {
            let board = parsed.board.as_ref().expect("a layout without errors has a board");
            let (mut rules, errors) = BoardRules::for_board(board);
            let rule_errors: Vec<Diagnostic> = errors
                .into_iter()
                .map(|e| Diagnostic::error(&path, Some(text), 0, 0, format!("{e:?}")))
                .collect();
            rules.rebuild(&BoardGraph::new(board, &pairs));
            (rules, rule_errors)
        };
        self.timings.rules = t.elapsed();
        let t = now();
        let (mut fills, dirty) = match reuse {
            Some((old_text, old, fills)) => {
                let reach = rules.max_clearance() + crate::geom::mm(1.0);
                (fills, reuse::whole_dirty(&old_text, &old, text, &parsed, reach))
            }
            None => (FillCache::default(), FillDirty::everywhere()),
        };
        let board = parsed
            .board
            .as_mut()
            .expect("a layout without errors has a board");
        if !dirty.everywhere {
            // The re-read numbered the zones afresh: find each cached fill's zone by content.
            fills.rekey(&board.zones);
        }
        let (reports, _) = fill_zones_cached(board, &rules, &pairs, &mut fills, &dirty);
        self.timings.fill = t.elapsed();
        let t = now();
        let drc = Drc::run(board, &rules, &pairs, &reports);
        self.timings.drc = t.elapsed();
        let s = self.schematic.as_ref().expect("synced");
        let t = now();
        let plans = plans(&s.project, board, options.eco.as_ref());
        self.timings.eco = t.elapsed();
        let t = now();
        let mut index = s.index.clone();
        add_board(&mut index, board, &rules, &drc);
        add_eco(&mut index, &plans.0, &plans.1, board);
        self.timings.index = t.elapsed();
        self.change = Some(self.prev.take().map(|prev| super::delta::index_delta(&prev, &index)));
        self.board = Some(BoardState {
            path,
            text: text.to_owned(),
            parsed,
            rules,
            rule_errors,
            fills,
            reports,
            drc,
            eco: options.eco.clone(),
            plans,
            index,
        });
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn patch(
        &mut self,
        path: String,
        text: &str,
        (mut parsed, mut rules, rule_errors, mut fills, _, mut drc, _, plans, mut index): (
            ParsedBoard,
            BoardRules,
            Vec<Diagnostic>,
            FillCache,
            Vec<FillReport>,
            Drc,
            Option<EcoSettings>,
            (EcoPlan, EcoPlan),
            CircuitIndex,
        ),
        delta: BoardDelta,
        options: &CompileOptions,
    ) {
        let board = parsed.board.as_mut().expect("read");
        // The items the change touched, old and new.
        let mut changed: Vec<BItem> = Vec::new();
        let mut removed: Vec<BItem> = Vec::new();
        let mut dirty = FillDirty::default();
        let reach = rules.max_clearance() + crate::geom::mm(1.0);
        // ECO compares footprints' identity, values and nets, and which room holds each.
        let rooms_of = |b: &Board, f: &crate::layout::Footprint| -> Vec<bool> {
            b.rooms
                .iter()
                .map(|r| r.outline.contains(f.at) == Some(true))
                .collect()
        };
        let eco_view = |b: &Board, f: &crate::layout::Footprint| {
            (
                f.reference.clone(),
                f.component.clone(),
                f.value.clone(),
                f.pad_nets.clone(),
                f.fitted,
                f.properties.clone(),
                f.package.name.clone(),
                rooms_of(b, f),
            )
        };
        let mut footprints_changed = false;
        let mut nets: BTreeSet<crate::layout::NetId> = BTreeSet::new();
        let item_nets = |item: &BoardItem, nets: &mut BTreeSet<crate::layout::NetId>| match item {
            BoardItem::Track(t) => nets.extend(t.net),
            BoardItem::Via(v) => nets.extend(v.net),
            BoardItem::Zone(z) => nets.extend(z.net),
            BoardItem::Footprint(f) => nets.extend(f.pad_nets.values().copied()),
            _ => {}
        };
        for old in &delta.replaced {
            item_nets(old, &mut nets);
            if let BoardItem::Footprint(f) = old {
                let same = board
                    .footprints
                    .get(&f.id)
                    .is_some_and(|now| eco_view(board, now) == eco_view(board, f));
                footprints_changed |= !same;
            }
        }
        for old in &delta.replaced {
            let r = item_bbox(board, old).inflate(reach);
            for l in item_layers(board, old) {
                dirty.add(l, r);
            }
            if delta.removed.contains(&old.id()) {
                bitems(board, old, &mut removed);
            }
        }
        for id in &delta.changed {
            if let Some(item) = crate::syntax::board::item_of(board, *id) {
                let r = item_bbox(board, &item).inflate(reach);
                for l in item_layers(board, &item) {
                    dirty.add(l, r);
                }
                if let BoardItem::Footprint(f) = &item {
                    footprints_changed |= !delta.replaced.iter().any(|o| o.id() == f.id);
                }
                item_nets(&item, &mut nets);
                bitems(board, &item, &mut changed);
            }
        }
        // A footprint whose pads went away: its old pads are removed items.
        for old in &delta.replaced {
            if let BoardItem::Footprint(f) = old {
                if let Some(now) = board.footprints.get(&f.id) {
                    let n = board.copper_count();
                    let have = now.pads(n).len() as u32;
                    for p in f.pads(n) {
                        if p.index as u32 >= have {
                            removed.push(BItem::Pad(f.id, p.index as u32));
                        }
                    }
                }
            }
        }
        let t = now();
        // The pairs the whole run took: a board edit doesn't change the circuit's pairs.
        let pairs = drc.diff_pairs().to_vec();
        let g = BoardGraph::new(board, &pairs);
        rules.single.items_changed(&g, &changed, &removed);
        rules.pair.items_changed(&g, &changed, &removed);
        drop(g);
        self.timings.rules = t.elapsed();
        let t = now();
        let (reports, fill_change) = fill_zones_cached(board, &rules, &pairs, &mut fills, &dirty);
        self.timings.fill = t.elapsed();
        let t = now();
        let mut drc_changed = changed.clone();
        for (z, _) in &fill_change.merged {
            drc_changed.push(BItem::Zone(*z));
        }
        drc_changed.sort();
        drc_changed.dedup();
        drc.update(board, &rules, &drc_changed, &removed, Some(&reports));
        self.timings.drc = t.elapsed();
        let s = self.schematic.as_ref().expect("synced");
        let t = now();
        let plans = if footprints_changed
            || options.eco != self.board.as_ref().and_then(|b| b.eco.clone())
        {
            super::plans(&s.project, board, options.eco.as_ref())
        } else {
            plans
        };
        self.timings.eco = t.elapsed();
        let t = now();
        let mut items: BTreeSet<ItemId> = delta.changed.iter().copied().collect();
        items.extend(delta.removed.iter().copied());
        for (z, _) in &fill_change.merged {
            items.insert(*z);
            nets.extend(board.zones.get(z).and_then(|z| z.net));
        }
        // What the delta compares: the old rows of the items rebuilt below, and the fields
        // rebuilt whole. Every other row moves over unchanged.
        let refs: HashSet<&str> = items.iter().filter_map(|id| board.footprints.get(id)).map(|f| f.reference.as_str()).collect();
        let base = super::delta::PatchBase::take(&mut index, |id| items.contains(&ItemId(id)), |r| refs.contains(r));
        drop(refs);
        let reuse = index.board.take().map(|old| crate::index::Reuse {
            old,
            items: &items,
            nets: &nets,
        });
        crate::index::add_board_reusing(&mut index, board, &rules, &drc, reuse);
        add_eco(&mut index, &plans.0, &plans.1, board);
        self.change = Some(base.map(|b| super::delta::patched_delta(&b, &index)));
        self.timings.index = t.elapsed();
        self.board = Some(BoardState {
            path,
            text: text.to_owned(),
            parsed,
            rules,
            rule_errors,
            fills,
            reports,
            drc,
            eco: options.eco.clone(),
            plans,
            index,
        });
    }

    /// [`compile`](super::compile), from the session's state where it can.
    pub fn compile(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        options: &CompileOptions,
    ) -> CompileOutput {
        match self.sync(path, text, files, options) {
            Err(diagnostics) => {
                self.board = None;
                CompileOutput {
                    index: None,
                    diagnostics,
                }
            }
            Ok(()) => match &self.board {
                Some(b) => CompileOutput {
                    index: Some(b.index.clone()),
                    diagnostics: b.rule_errors.clone(),
                },
                None => CompileOutput {
                    index: self.schematic.as_ref().map(|s| s.index.clone()),
                    diagnostics: Vec::new(),
                },
            },
        }
    }

    /// [`compile`](Self::compile) written as JSON (`{ index?, diagnostics }`) without copying
    /// the projection.
    pub fn compile_json(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        options: &CompileOptions,
    ) -> String {
        #[derive(Serialize)]
        struct Out<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            index: Option<&'a CircuitIndex>,
            diagnostics: Vec<Diagnostic>,
            #[serde(skip_serializing_if = "Option::is_none")]
            pending_values: Option<serde_json::Value>,
        }
        let diagnostics = match self.sync(path, text, files, options) {
            Err(d) => {
                self.board = None;
                let pending = super::values::pending_json(&d, path, text, files, &options.values)
                    .and_then(|j| serde_json::from_str(&j).ok());
                return serde_json::to_string(&Out {
                    index: None,
                    diagnostics: d,
                    pending_values: pending,
                })
                .unwrap_or_default();
            }
            Ok(()) => self
                .board
                .as_ref()
                .map(|b| b.rule_errors.clone())
                .unwrap_or_default(),
        };
        serde_json::to_string(&Out {
            index: self.index(),
            diagnostics,
            pending_values: None,
        })
        .unwrap_or_default()
    }

    /// The projection of the last successful compile, without a copy.
    pub fn index(&self) -> Option<&CircuitIndex> {
        match &self.board {
            Some(b) => Some(&b.index),
            None => self.schematic.as_ref().map(|s| &s.index),
        }
    }

    /// The last compiled board in 3D ([`crate::board3d`]), for the 3D tab.
    pub fn board_model(&self) -> Option<crate::board3d::BoardModel> {
        let state = self.board.as_ref()?;
        state.parsed.board.as_ref().map(crate::board3d::board_model)
    }

    /// The last compiled board as an exact solid's input ([`crate::board3d::board_solid`]).
    pub fn board_solid(&self) -> Option<crate::board3d::BoardSolid> {
        let state = self.board.as_ref()?;
        state.parsed.board.as_ref().map(crate::board3d::board_solid)
    }

    /// The board's mass properties ([`crate::mass::board_mass`]), with the library parts' masses
    /// when the schematic compiled.
    pub fn board_mass(&self) -> Option<crate::mass::BoardMass> {
        let board = self.board.as_ref()?.parsed.board.as_ref()?;
        let project = self.schematic.as_ref().map(|s| &s.project);
        Some(crate::mass::board_mass(board, project.map(|p| &p.netlist), project.map(|p| &p.pool)))
    }

    /// [`edit`](super::edit): a board catalog command edits the session's reading of the
    /// layout; ECO and room commands run through the stateless runtime.
    pub fn edit(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        request: &EditRequest,
    ) -> Result<EditOutput, EditError> {
        // Schematic, library and simulation commands edit the circuit's files, and the board
        // commands that need the project work across both; the rest are board edits on the
        // kept board.
        if super::edit::needs_project(&request.command) || !request.command.starts_with("board.") {
            return super::edit(path, text, files, request);
        }
        self.sync(path, text, files, &request.options)
            .map_err(|diagnostics| EditError {
                message: format!(
                    "the circuit has {} error(s), so the edit wasn't made",
                    diagnostics.len()
                ),
                diagnostics,
            })?;
        let Some(state) = self.board.as_mut() else {
            let s = self.schematic.as_ref().expect("synced");
            return Err(EditError::new(format!(
                "{} has no board yet (no {} next to it)",
                s.circuit_path,
                layout_path_for(&s.circuit_path)
            )));
        };
        // Fills are derived: the edit works on the board without them.
        let board = state.parsed.board.as_mut().expect("read");
        let fills: Vec<(ItemId, BTreeMap<CopperIndex, crate::geom::poly::PolySet>)> = board
            .zones
            .values_mut()
            .map(|z| (z.id, std::mem::take(&mut z.fill)))
            .collect();
        let result = (|| {
            let board = state.parsed.board.as_ref().expect("read");
            let edit = board_edit(board, &state.parsed.sites, &request.command, request)?;
            let change = board_command_parsed(&state.path, &state.text, &state.parsed, &edit)
                .map_err(failure)?;
            Ok(EditOutput {
                summary: format!("{} on {}", edit.command(), state.path),
                files: vec![file_change(&state.path, &state.text, change)],
            })
        })();
        let board = state.parsed.board.as_mut().expect("read");
        for (id, f) in fills {
            if let Some(z) = board.zones.get_mut(&id) {
                z.fill = f;
            }
        }
        result
    }
}

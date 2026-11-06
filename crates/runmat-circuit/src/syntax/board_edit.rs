//! Structured edits on `.circuit.layout` sources.
//!
//! An edit changes the [`Board`]; [`diff_board`] then finds the smallest text change that makes
//! the source describe the new board:
//!
//! - a statement whose printed form is unchanged keeps its text, comments and layout, however
//!   it was written;
//! - a changed statement is replaced in place, keeping the comments around it;
//! - a removed statement takes its own lines and the comment lines directly above it;
//! - an added statement goes next to its neighbours of the same kind, in print order.
//!
//! Statements are compared by kind (footprints with footprints, copper with copper), matched
//! by printed text and then by site, and kept in order by a longest increasing subsequence, so
//! a 10k-track board that gains one track changes by one line.
//!
//! The result is checked: the edited source must read back as the new board ([`same_board`]).
//! If it would not, the whole file is reprinted and [`SourceChange::reprinted`] says so.
//!
//! [`edit_board`] runs any model edit this way. The ECO apply ([`apply_eco`]), Copy Room
//! Formats ([`copy_room_formats`]) and the board catalog's model edits ([`BoardEdit`],
//! [`apply_board_edit`]) are built on it.

use super::board::{
    canonical, parse_board, print_board, same_board, statements, statements_where, Group,
    ParsedBoard, Printed,
};
use crate::eco::{ApplyError, ApplyOptions, ApplyReport, EcoPlan};
use crate::geom::poly::Polygon;
use crate::geom::{Coord, Point};
use crate::layout::*;
use crate::library::Pool;
use crate::room_formats::{RoomFormatError, RoomFormatOptions, RoomFormatReport};
use runmat_design_syntax::edit::{apply_edits, EditError, TextEdit};
use runmat_design_syntax::json::INDENT;
use runmat_design_syntax::FeaSyntaxDiagnostic;
use runmat_types::Span;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// The text change for a board edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChange {
    pub edits: Vec<TextEdit>,
    /// The source after the edits.
    pub source: String,
    /// The minimal edits did not reproduce the board, so the file was reprinted whole.
    pub reprinted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EditFailure {
    /// The source has errors; edits need a file that reads.
    Invalid(Vec<FeaSyntaxDiagnostic>),
    Apply(EditError),
    /// The edit names something the board does not have, or cannot be made.
    Model(String),
}

impl std::fmt::Display for EditFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditFailure::Invalid(d) => write!(
                f,
                "the layout has {} error(s); the first: {}",
                d.len(),
                d.first().map_or("", |d| d.message.as_str())
            ),
            EditFailure::Apply(e) => write!(f, "{e}"),
            EditFailure::Model(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for EditFailure {}

/// The range a statement owns: its lines when it is alone on them, with the comment lines
/// directly above it and a trailing comment.
fn own_range(source: &str, span: Span) -> Span {
    let bytes = source.as_bytes();
    let mut start = span.start;
    while start > 0 && matches!(bytes[start - 1], b' ' | b'\t') {
        start -= 1;
    }
    let alone_before = start == 0 || bytes[start - 1] == b'\n';
    let mut end = span.end;
    while end < bytes.len() && matches!(bytes[end], b' ' | b'\t') {
        end += 1;
    }
    if source[end..].starts_with("//") {
        end += source[end..].find('\n').unwrap_or(source.len() - end);
    }
    let alone_after = end == bytes.len() || bytes[end] == b'\n';
    if !(alone_before && alone_after) {
        return span;
    }
    if end < bytes.len() {
        end += 1;
    }
    // Comment lines directly above.
    loop {
        if start == 0 {
            break;
        }
        let prev_end = start - 1;
        let prev_start = source[..prev_end].rfind('\n').map_or(0, |i| i + 1);
        if source[prev_start..prev_end].trim_start().starts_with("//") {
            start = prev_start;
        } else {
            break;
        }
    }
    Span { start, end }
}

/// The offset just after a statement's line (where a statement after it goes).
fn after_line(source: &str, span: Span) -> Option<usize> {
    let r = own_range(source, span);
    (r != span || r.end == source.len()).then_some(r.end)
}

/// Longest increasing subsequence of `values` (indices into it).
fn stable_indices(values: &[usize]) -> BTreeSet<usize> {
    let mut tails: Vec<usize> = Vec::new();
    let mut previous = vec![usize::MAX; values.len()];
    for (index, value) in values.iter().enumerate() {
        let position = tails.partition_point(|tail| values[*tail] < *value);
        if position > 0 {
            previous[index] = tails[position - 1];
        }
        if position == tails.len() {
            tails.push(index);
        } else {
            tails[position] = index;
        }
    }
    let mut out = BTreeSet::new();
    let mut cursor = tails.last().copied();
    while let Some(index) = cursor {
        out.insert(index);
        cursor = (previous[index] != usize::MAX).then(|| previous[index]);
    }
    out
}

/// The minimal edits turning `source` (read as `parsed`) into a source for `new`.
pub fn diff_board(
    source: &str,
    parsed: &ParsedBoard,
    new: &Board,
) -> Result<SourceChange, EditFailure> {
    let Some(old) = parsed.board.as_ref() else {
        return Err(EditFailure::Invalid(parsed.diagnostics.clone()));
    };
    let reprint = || {
        let text = print_board(new, &parsed.uses);
        Ok(SourceChange {
            edits: vec![TextEdit::replace(
                Span {
                    start: 0,
                    end: source.len(),
                },
                text.clone(),
            )],
            source: text,
            reprinted: true,
        })
    };
    let Some(block) = parsed.block else {
        return reprint();
    };
    let old_text: HashMap<String, String> = statements(old)
        .into_iter()
        .map(|p| (p.site, p.text))
        .collect();
    let printed: Vec<Printed> = statements(new);
    // Per group: old statements in source order, new ones in print order.
    let mut old_by_group: BTreeMap<Group, Vec<usize>> = BTreeMap::new();
    for (i, s) in parsed.statements.iter().enumerate() {
        old_by_group.entry(s.group).or_default().push(i);
    }
    let mut new_by_group: BTreeMap<Group, Vec<usize>> = BTreeMap::new();
    for (j, p) in printed.iter().enumerate() {
        new_by_group.entry(p.group).or_default().push(j);
    }
    // new index → (old index, exact)
    let mut kept: BTreeMap<usize, (usize, bool)> = BTreeMap::new();
    let mut groups: BTreeSet<Group> = old_by_group.keys().copied().collect();
    groups.extend(new_by_group.keys().copied());
    for g in &groups {
        let olds = old_by_group.get(g).map(Vec::as_slice).unwrap_or_default();
        let news = new_by_group.get(g).map(Vec::as_slice).unwrap_or_default();
        let mut by_text: HashMap<&str, VecDeque<usize>> = HashMap::new();
        for &j in news {
            by_text
                .entry(printed[j].text.as_str())
                .or_default()
                .push_back(j);
        }
        let mut pairs: Vec<(usize, usize, bool)> = Vec::new();
        let mut taken_new: BTreeSet<usize> = BTreeSet::new();
        let mut unmatched_old: Vec<usize> = Vec::new();
        for &i in olds {
            let text = old_text.get(&parsed.statements[i].site);
            match text
                .and_then(|t| by_text.get_mut(t.as_str()))
                .and_then(|q| q.pop_front())
            {
                Some(j) => {
                    pairs.push((i, j, true));
                    taken_new.insert(j);
                }
                None => unmatched_old.push(i),
            }
        }
        // A changed statement keeps its place: pair it with the new statement of its site.
        let by_site: HashMap<&str, usize> = news
            .iter()
            .filter(|j| !taken_new.contains(j))
            .map(|&j| (printed[j].site.as_str(), j))
            .collect();
        for i in unmatched_old {
            if let Some(&j) = by_site.get(parsed.statements[i].site.as_str()) {
                if taken_new.insert(j) {
                    pairs.push((i, j, false));
                }
            }
        }
        pairs.sort_unstable();
        let order: Vec<usize> = pairs.iter().map(|p| p.1).collect();
        for k in stable_indices(&order) {
            let (i, j, exact) = pairs[k];
            kept.insert(j, (i, exact));
        }
    }
    let kept_old: BTreeSet<usize> = kept.values().map(|(i, _)| *i).collect();
    let mut edits: Vec<TextEdit> = Vec::new();
    for (i, s) in parsed.statements.iter().enumerate() {
        if !kept_old.contains(&i) {
            edits.push(TextEdit::delete(own_range(source, s.span)));
        }
    }
    for (&j, &(i, exact)) in &kept {
        if !exact {
            edits.push(TextEdit::replace(
                parsed.statements[i].span,
                printed[j].text.clone(),
            ));
        }
    }
    // Insertions, each after the kept statement before it in its group, else before the
    // kept statement after it, else after the last kept statement before it in print order.
    let open_line = {
        let after_brace = block.start + 1;
        match source[after_brace..].find('\n') {
            Some(n) if source[after_brace..after_brace + n].trim().is_empty() => {
                after_brace + n + 1
            }
            _ => return reprint(),
        }
    };
    let mut inserts: BTreeMap<usize, String> = BTreeMap::new();
    let mut insert_order: Vec<usize> = Vec::new();
    for (j, p) in printed.iter().enumerate() {
        if kept.contains_key(&j) {
            continue;
        }
        let same: &[usize] = &new_by_group[&p.group];
        let prev_same = same
            .iter()
            .rev()
            .filter(|&&x| x < j)
            .find_map(|x| kept.get(x));
        let next_same = same.iter().filter(|&&x| x > j).find_map(|x| kept.get(x));
        let at = if let Some((i, _)) = prev_same {
            after_line(source, parsed.statements[*i].span)
        } else if let Some((i, _)) = next_same {
            Some(own_range(source, parsed.statements[*i].span).start)
                .filter(|s| *s == 0 || source.as_bytes()[s - 1] == b'\n')
        } else {
            match kept.range(..j).next_back() {
                Some((_, (i, _))) => after_line(source, parsed.statements[*i].span),
                None => Some(open_line),
            }
        };
        let Some(at) = at else {
            return reprint();
        };
        let text = inserts.entry(at).or_insert_with(|| {
            insert_order.push(at);
            String::new()
        });
        text.push_str(INDENT);
        text.push_str(&p.text);
        text.push('\n');
    }
    for at in insert_order {
        edits.push(TextEdit::insert(
            at,
            inserts.remove(&at).unwrap_or_default(),
        ));
    }
    let edited = match apply_edits(source, &edits) {
        Ok(text) => text,
        Err(_) => return reprint(),
    };
    Ok(SourceChange {
        edits,
        source: edited,
        reprinted: false,
    })
}

/// Runs `edit` on the board `source` describes and returns the minimal text change, checked
/// to read back as the edited board (else the file is reprinted).
pub fn edit_board<T>(
    path: &str,
    source: &str,
    edit: impl FnOnce(&mut Board) -> Result<T, String>,
) -> Result<(SourceChange, T), EditFailure> {
    let parsed = parse_board(path, source);
    let Some(old) = parsed.board.clone() else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    let mut board = old;
    let out = edit(&mut board).map_err(EditFailure::Model)?;
    let mut change = diff_board(source, &parsed, &board)?;
    let reread = parse_board(path, &change.source);
    let fits = reread
        .board
        .as_ref()
        .is_some_and(|b| canonical(b) == canonical(&board));
    if !fits {
        let text = print_board(&board, &parsed.uses);
        change = SourceChange {
            edits: vec![TextEdit::replace(
                Span {
                    start: 0,
                    end: source.len(),
                },
                text.clone(),
            )],
            source: text,
            reprinted: true,
        };
        debug_assert!(parse_board(path, &change.source)
            .board
            .is_some_and(|b| same_board(&b, &board)));
    }
    Ok((change, out))
}

/// Whether two boards agree on everything but their items (footprints, fiducials, copper,
/// vias, pours, keepouts, texts, graphics, dimensions) and the next id.
fn same_but_items(a: &Board, b: &Board) -> bool {
    a.name == b.name
        && a.outline == b.outline
        && a.stackup == b.stackup
        && a.stackups == b.stackups
        && a.regions == b.regions
        && a.nets == b.nets
        && a.rooms == b.rooms
        && a.limits == b.limits
        && a.rules == b.rules
        && a.net_class_rules == b.net_class_rules
        && a.sync == b.sync
}

/// Item ids that differ between two id-keyed maps: `(changed, added, removed)`.
fn map_diff<T: PartialEq>(
    a: &BTreeMap<ItemId, T>,
    b: &BTreeMap<ItemId, T>,
    eq: impl Fn(&T, &T) -> bool,
    out: &mut (Vec<ItemId>, Vec<ItemId>, Vec<ItemId>),
) {
    for (id, x) in a {
        match b.get(id) {
            Some(y) if eq(x, y) => {}
            Some(_) => out.0.push(*id),
            None => out.2.push(*id),
        }
    }
    for id in b.keys() {
        if !a.contains_key(id) {
            out.1.push(*id);
        }
    }
}

/// The items an edit changed, added and removed, or `None` when it changed anything else.
fn item_changes(old: &Board, new: &Board) -> Option<(Vec<ItemId>, Vec<ItemId>, Vec<ItemId>)> {
    if !same_but_items(old, new) {
        return None;
    }
    let mut d = (Vec::new(), Vec::new(), Vec::new());
    map_diff(&old.footprints, &new.footprints, |a, b| a == b, &mut d);
    // A footprint's schematic link and package are written outside its statement.
    let fp_ok = d.0.iter().all(|id| {
        let (a, b) = (&old.footprints[id], &new.footprints[id]);
        a.component == b.component && a.package == b.package
    }) && d.1.iter().all(|id| {
        let f = &new.footprints[id];
        f.component.is_none() && old.footprints.values().any(|o| o.package == f.package)
    }) && d.2.iter().all(|id| {
        let f = &old.footprints[id];
        f.component.is_none() && new.footprints.values().any(|o| o.package == f.package)
    });
    if !fp_ok {
        return None;
    }
    let zone_eq = |a: &Zone, b: &Zone| {
        a.id == b.id
            && a.name == b.name
            && a.net == b.net
            && a.layers == b.layers
            && a.outline == b.outline
            && a.priority == b.priority
            && a.clearance == b.clearance
            && a.min_width == b.min_width
            && a.connect == b.connect
            && a.thermal_gap == b.thermal_gap
            && a.spoke_width == b.spoke_width
            && a.islands == b.islands
            && a.locked == b.locked
    };
    map_diff(&old.fiducials, &new.fiducials, |a, b| a == b, &mut d);
    map_diff(&old.tracks, &new.tracks, |a, b| a == b, &mut d);
    map_diff(&old.vias, &new.vias, |a, b| a == b, &mut d);
    map_diff(&old.zones, &new.zones, zone_eq, &mut d);
    map_diff(&old.keepouts, &new.keepouts, |a, b| a == b, &mut d);
    map_diff(&old.texts, &new.texts, |a, b| a == b, &mut d);
    map_diff(&old.graphics, &new.graphics, |a, b| a == b, &mut d);
    map_diff(&old.dimensions, &new.dimensions, |a, b| a == b, &mut d);
    Some(d)
}

/// The minimal edits for an edit that only changed, added or removed items: each changed
/// statement is replaced, each removed one deleted, each added one goes after the last
/// statement of its kind. `None` when the general diff is needed.
fn diff_items(
    source: &str,
    parsed: &ParsedBoard,
    new: &Board,
    (changed, added, removed): &(Vec<ItemId>, Vec<ItemId>, Vec<ItemId>),
) -> Option<Vec<TextEdit>> {
    let printed_ids: BTreeSet<ItemId> = changed.iter().chain(added).copied().collect();
    let printed = statements_where(new, &|id| printed_ids.contains(&id));
    let text_of: HashMap<ItemId, (&str, Group)> = printed
        .iter()
        .filter_map(|p| {
            p.item
                .filter(|i| printed_ids.contains(i))
                .map(|i| (i, (p.text.as_str(), p.group)))
        })
        .collect();
    let mut at_of: HashMap<ItemId, Span> = HashMap::new();
    let mut last_of: HashMap<Group, Span> = HashMap::new();
    for s in &parsed.statements {
        if let Some(i) = s.item {
            at_of.insert(i, s.span);
        }
        last_of.insert(s.group, s.span);
    }
    let mut edits = Vec::new();
    for id in removed {
        edits.push(TextEdit::delete(own_range(source, *at_of.get(id)?)));
    }
    for id in changed {
        edits.push(TextEdit::replace(
            *at_of.get(id)?,
            text_of.get(id)?.0.to_owned(),
        ));
    }
    let mut inserts: BTreeMap<usize, String> = BTreeMap::new();
    for id in added {
        let (text, group) = text_of.get(id)?;
        let at = after_line(source, *last_of.get(group)?)?;
        let s = inserts.entry(at).or_default();
        s.push_str(INDENT);
        s.push_str(text);
        s.push('\n');
    }
    for (at, text) in inserts {
        edits.push(TextEdit::insert(at, text));
    }
    Some(edits)
}

/// [`edit_board`] from a reading of `source` the caller keeps: an edit that only changes items
/// prints and re-reads just their statements.
pub fn edit_parsed<T>(
    path: &str,
    source: &str,
    parsed: &ParsedBoard,
    edit: impl FnOnce(&mut Board) -> Result<T, String>,
) -> Result<(SourceChange, T), EditFailure> {
    let Some(old) = parsed.board.as_ref() else {
        return Err(EditFailure::Invalid(parsed.diagnostics.clone()));
    };
    let mut board = old.clone();
    let out = edit(&mut board).map_err(EditFailure::Model)?;
    if let Some(change) = item_changes(old, &board) {
        if let Some(edits) = diff_items(source, parsed, &board, &change) {
            if let Ok(edited) = apply_edits(source, &edits) {
                if fits_items(path, source, parsed, &edited, &board, &change) {
                    return Ok((
                        SourceChange {
                            edits,
                            source: edited,
                            reprinted: false,
                        },
                        out,
                    ));
                }
            }
        }
    }
    let mut change = diff_board(source, parsed, &board)?;
    let reread = parse_board(path, &change.source);
    let fits = reread
        .board
        .as_ref()
        .is_some_and(|b| canonical(b) == canonical(&board));
    if !fits {
        let text = print_board(&board, &parsed.uses);
        change = SourceChange {
            edits: vec![TextEdit::replace(
                Span {
                    start: 0,
                    end: source.len(),
                },
                text.clone(),
            )],
            source: text,
            reprinted: true,
        };
    }
    Ok((change, out))
}

/// Whether `edited` re-reads (window only) as the items the edit made.
fn fits_items(
    path: &str,
    source: &str,
    parsed: &ParsedBoard,
    edited: &str,
    new: &Board,
    (changed, added, removed): &(Vec<ItemId>, Vec<ItemId>, Vec<ItemId>),
) -> bool {
    let Some(w) = super::board::read_window(path, source, parsed, edited) else {
        return false;
    };
    let items = super::board::window_items(&w);
    let replaced: BTreeSet<ItemId> = super::board::window_replaced(parsed, &w)
        .into_iter()
        .collect();
    let old = parsed.board.as_ref().expect("read");
    // Every changed item reads back under its id; removed ones are gone.
    for id in changed {
        if items.get(id).copied() != super::board::item_of(new, *id).as_ref() {
            return false;
        }
    }
    if removed
        .iter()
        .any(|id| !replaced.contains(id) || items.contains_key(id))
    {
        return false;
    }
    // Added items read back in order, under new ids; window items that replaced old ones
    // unchanged are the window's context.
    let fresh: Vec<_> = items
        .iter()
        .filter(|(id, _)| !old_has(old, **id))
        .map(|(_, i)| *i)
        .collect();
    if fresh.len() != added.len() {
        return false;
    }
    let mut with_ids: Vec<super::board::BoardItem> = fresh.into_iter().cloned().collect();
    for (item, id) in with_ids.iter_mut().zip(added) {
        item.set_id(*id);
    }
    added
        .iter()
        .zip(&with_ids)
        .all(|(id, item)| super::board::item_of(new, *id).as_ref() == Some(item))
        && items.iter().all(|(id, item)| {
            !old_has(old, *id)
                || changed.contains(id)
                || super::board::item_of(old, *id).as_ref() == Some(*item)
        })
}

fn old_has(old: &Board, id: ItemId) -> bool {
    id.0 < old.next_id && super::board::item_of(old, id).is_some()
}

/// Applies an ECO plan to the layout `source` as one minimal text change: the board and its
/// `sync { }` section take the applied lines.
pub fn apply_eco(
    path: &str,
    source: &str,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
    pool: &Pool,
    options: &ApplyOptions,
) -> Result<(SourceChange, ApplyReport), EditFailure> {
    edit_board(path, source, |board| {
        crate::eco::apply(board, plan, selected, pool, options)
            .map_err(|e: ApplyError| format!("{e:?}"))
    })
}

/// Copy Room Formats on the layout `source`, as one minimal text change.
pub fn copy_room_formats(
    path: &str,
    source: &str,
    room: &str,
    targets: &[String],
    options: &RoomFormatOptions,
) -> Result<(SourceChange, RoomFormatReport), EditFailure> {
    edit_board(path, source, |board| {
        crate::room_formats::copy_room_formats(board, room, targets, options)
            .map_err(|e: RoomFormatError| format!("{e:?}"))
    })
}

/* ------------------------------------------------------------------------------------------------
 * The board catalog's model edits
 * --------------------------------------------------------------------------------------------- */

/// What a board edit acts on, by the site keys the reader gives
/// ([`ParsedBoard::sites`](super::board::ParsedBoard)) or by designator.
pub type Site = String;

/// The board catalog commands that change the model, with their inputs resolved.
#[derive(Clone, Debug, PartialEq)]
pub enum BoardEdit {
    /// `board.move`: moves and rotates items (footprints about their origin).
    Move {
        items: Vec<Site>,
        dx: Coord,
        dy: Coord,
        rotation: f64,
    },
    /// Flips footprints to the other side.
    Flip { items: Vec<Site> },
    /// Locks or unlocks items.
    Lock { items: Vec<Site>, locked: bool },
    /// Deletes items.
    Delete { items: Vec<Site> },
    /// `board.room`: adds a room around components, or replaces the room of that name.
    Room {
        name: String,
        outline: Polygon,
        members: Vec<String>,
    },
    /// `board.route`: a routed path as track segments on one layer.
    Route {
        net: Option<String>,
        layer: String,
        width: Coord,
        points: Vec<Point>,
    },
    /// `board.place-via`.
    Via {
        net: Option<String>,
        at: Point,
        diameter: Coord,
        drill: Coord,
        span: Option<(String, String)>,
        micro: bool,
    },
    /// `board.pour` (draw): a copper pour.
    Pour {
        net: Option<String>,
        layers: Vec<String>,
        outline: Polygon,
        clearance: Option<Coord>,
        connect: PadConnect,
        /// Fill order where pours overlap: the higher priority fills first.
        priority: i32,
    },
    /// `board.keepout`.
    Keepout {
        name: String,
        outline: Polygon,
        layers: Vec<String>,
        rules: KeepoutRules,
    },
    /// `board.dimension`.
    Dimension {
        layer: Layer,
        a: Point,
        b: Point,
        offset: Coord,
    },
    /// `board.rules`: adds a rule, or replaces the rule with the same id.
    Rule(crate::board_rules::RuleSpec),
    /// `board.layer-stack`: a standard stackup of `layers` copper layers and `thickness`, its
    /// dielectric layers in `dielectric` when given (else FR-4 cores and prepreg).
    LayerStack { layers: u8, thickness: Coord, dielectric: Option<Material> },
    /// `board.outline` (draw).
    Outline { outline: Polygon },
    /// `board.test-points`: a test point footprint on each net.
    TestPoint {
        net: String,
        at: Point,
        side: Side,
        diameter: Coord,
    },
}

impl BoardEdit {
    /// The catalog command this edit runs.
    pub fn command(&self) -> &'static str {
        match self {
            BoardEdit::Move { .. } => "board.move",
            BoardEdit::Flip { .. } => "board.flip",
            BoardEdit::Lock { .. } => "board.move",
            BoardEdit::Delete { .. } => "board.select",
            BoardEdit::Room { .. } => "board.room",
            BoardEdit::Route { .. } => "board.route",
            BoardEdit::Via { .. } => "board.place-via",
            BoardEdit::Pour { .. } => "board.pour",
            BoardEdit::Keepout { .. } => "board.keepout",
            BoardEdit::Dimension { .. } => "board.dimension",
            BoardEdit::Rule(_) => "board.rules",
            BoardEdit::LayerStack { .. } => "board.layer-stack",
            BoardEdit::Outline { .. } => "board.outline",
            BoardEdit::TestPoint { .. } => "board.test-points",
        }
    }
}

/// The item a site (or a footprint designator) names on `board`.
fn resolve(board: &Board, sites: &BTreeMap<ItemId, String>, site: &str) -> Result<ItemId, String> {
    if let Some((id, _)) = sites.iter().find(|(_, s)| s.as_str() == site) {
        return Ok(*id);
    }
    if let Some(f) = board.footprint_by_ref(site) {
        return Ok(f.id);
    }
    Err(format!("the board has no item `{site}`"))
}

fn net_of(board: &mut Board, net: &Option<String>) -> Result<Option<NetId>, String> {
    match net {
        None => Ok(None),
        Some(n) => board
            .net(n)
            .map(Some)
            .ok_or_else(|| format!("the board has no net `{n}`")),
    }
}

fn copper_of(board: &Board, layer: &str) -> Result<u8, String> {
    let names = board.stackup.copper_names();
    let n = names.len() as u8;
    names
        .iter()
        .position(|x| x == layer)
        .map(|i| i as u8)
        .or_else(|| (0..n).find(|i| copper_name(*i, n) == layer))
        .ok_or_else(|| format!("`{layer}` is not a copper layer of this board"))
}

fn moved(p: Point, dx: Coord, dy: Coord) -> Point {
    Point::new(p.x + dx, p.y + dy)
}

/// Applies a catalog edit to a board. `sites` maps items to the site keys of the source the
/// command's picks came from.
pub fn apply_board_edit(
    board: &mut Board,
    sites: &BTreeMap<ItemId, String>,
    edit: &BoardEdit,
) -> Result<(), String> {
    let ids = |board: &Board, items: &[Site]| -> Result<Vec<ItemId>, String> {
        items.iter().map(|s| resolve(board, sites, s)).collect()
    };
    match edit {
        BoardEdit::Move {
            items,
            dx,
            dy,
            rotation,
        } => {
            for id in ids(board, items)? {
                if let Some(f) = board.footprints.get_mut(&id) {
                    if f.locked {
                        return Err(format!("{} is locked", f.reference));
                    }
                    f.at = moved(f.at, *dx, *dy);
                    f.rotation = (f.rotation + rotation).rem_euclid(360.0);
                } else if let Some(t) = board.tracks.get_mut(&id) {
                    t.geom = match t.geom {
                        TrackGeom::Segment { a, b } => TrackGeom::Segment {
                            a: moved(a, *dx, *dy),
                            b: moved(b, *dx, *dy),
                        },
                        TrackGeom::Arc { start, mid, end } => TrackGeom::Arc {
                            start: moved(start, *dx, *dy),
                            mid: moved(mid, *dx, *dy),
                            end: moved(end, *dx, *dy),
                        },
                    };
                } else if let Some(v) = board.vias.get_mut(&id) {
                    v.at = moved(v.at, *dx, *dy);
                } else if let Some(z) = board.zones.get_mut(&id) {
                    let shift = |r: &mut Vec<Point>| {
                        for p in r {
                            *p = moved(*p, *dx, *dy);
                        }
                    };
                    shift(&mut z.outline.outer);
                    z.outline.holes.iter_mut().for_each(shift);
                    z.fill.clear();
                } else if let Some(f) = board.fiducials.get_mut(&id) {
                    f.at = moved(f.at, *dx, *dy);
                } else if let Some(t) = board.texts.get_mut(&id) {
                    t.at = moved(t.at, *dx, *dy);
                    t.angle = (t.angle + rotation).rem_euclid(360.0);
                } else {
                    return Err("that item does not move".into());
                }
            }
        }
        BoardEdit::Flip { items } => {
            for id in ids(board, items)? {
                let f = board
                    .footprints
                    .get_mut(&id)
                    .ok_or("only footprints flip")?;
                f.side = f.side.flip();
            }
        }
        BoardEdit::Lock { items, locked } => {
            for id in ids(board, items)? {
                if let Some(f) = board.footprints.get_mut(&id) {
                    f.locked = *locked;
                } else if let Some(t) = board.tracks.get_mut(&id) {
                    t.locked = *locked;
                } else if let Some(v) = board.vias.get_mut(&id) {
                    v.locked = *locked;
                } else if let Some(z) = board.zones.get_mut(&id) {
                    z.locked = *locked;
                } else if let Some(f) = board.fiducials.get_mut(&id) {
                    f.locked = *locked;
                } else {
                    return Err("that item has no lock".into());
                }
            }
        }
        BoardEdit::Delete { items } => {
            for id in ids(board, items)? {
                let gone = board.footprints.remove(&id).is_some()
                    | board.tracks.remove(&id).is_some()
                    | board.vias.remove(&id).is_some()
                    | board.zones.remove(&id).is_some()
                    | board.keepouts.remove(&id).is_some()
                    | board.fiducials.remove(&id).is_some()
                    | board.texts.remove(&id).is_some()
                    | board.graphics.remove(&id).is_some()
                    | board.dimensions.remove(&id).is_some();
                if !gone {
                    return Err("that item cannot be deleted".into());
                }
            }
        }
        BoardEdit::Room {
            name,
            outline,
            members,
        } => {
            let room = Room {
                name: name.clone(),
                outline: outline.clone(),
                members: members.clone(),
            };
            match board.rooms.iter_mut().find(|r| &r.name == name) {
                Some(r) => *r = room,
                None => board.rooms.push(room),
            }
        }
        BoardEdit::Route {
            net,
            layer,
            width,
            points,
        } => {
            if points.len() < 2 {
                return Err("a route needs at least two points".into());
            }
            let net = net_of(board, net)?;
            let layer = copper_of(board, layer)?;
            for w in points.windows(2) {
                board.add_track(net, layer, *width, w[0], w[1]);
            }
        }
        BoardEdit::Via {
            net,
            at,
            diameter,
            drill,
            span,
            micro,
        } => {
            let net = net_of(board, net)?;
            let span = match span {
                Some((a, b)) => {
                    let (a, b) = (copper_of(board, a)?, copper_of(board, b)?);
                    Some((a.min(b), a.max(b)))
                }
                None => None,
            };
            let id = board.add_via(net, *at, *diameter, *drill);
            let v = board.vias.get_mut(&id).unwrap();
            if let Some((from, to)) = span {
                v.from = from;
                v.to = to;
            }
            v.micro = *micro;
        }
        BoardEdit::Pour {
            net,
            layers,
            outline,
            clearance,
            connect,
            priority,
        } => {
            let net = net_of(board, net)?;
            let layers = layers
                .iter()
                .map(|l| copper_of(board, l))
                .collect::<Result<Vec<_>, _>>()?;
            let id = board.alloc();
            let mut z = Zone::new(id, "", net, layers, outline.clone());
            z.clearance = *clearance;
            z.connect = *connect;
            z.priority = *priority;
            board.zones.insert(id, z);
        }
        BoardEdit::Keepout {
            name,
            outline,
            layers,
            rules,
        } => {
            let layers = layers
                .iter()
                .map(|l| copper_of(board, l))
                .collect::<Result<Vec<_>, _>>()?;
            let id = board.alloc();
            board.keepouts.insert(
                id,
                Keepout {
                    id,
                    name: name.clone(),
                    outline: outline.clone(),
                    layers,
                    rules: rules.clone(),
                },
            );
        }
        BoardEdit::Dimension {
            layer,
            a,
            b,
            offset,
        } => {
            let id = board.alloc();
            board.dimensions.insert(
                id,
                Dimension {
                    id,
                    layer: *layer,
                    a: *a,
                    b: *b,
                    offset: *offset,
                    precision: 2,
                },
            );
        }
        BoardEdit::Rule(spec) => match board.rules.iter_mut().find(|r| r.id == spec.id) {
            Some(r) => *r = spec.clone(),
            None => board.rules.push(spec.clone()),
        },
        BoardEdit::LayerStack { layers, thickness, dielectric } => {
            if *layers < 2 || layers % 2 != 0 {
                return Err(
                    "a standard stackup has an even number of copper layers, 2 or more".into(),
                );
            }
            let before = board.copper_count();
            let used = board
                .tracks
                .values()
                .map(|t| t.layer)
                .chain(board.zones.values().flat_map(|z| z.layers.iter().copied()))
                .max();
            if *layers < before && used.is_some_and(|l| l > 0 && l + 1 != before) {
                return Err("inner layers carry copper; move it before removing layers".into());
            }
            let mut s = standard_stackup(*layers, *thickness, dielectric.as_ref());
            s.name = board.stackup.name.clone();
            s.finish = board.stackup.finish.clone();
            let last = |l: u8| if l + 1 == before { layers - 1 } else { l };
            for t in board.tracks.values_mut() {
                t.layer = last(t.layer);
            }
            for v in board.vias.values_mut() {
                v.from = last(v.from);
                v.to = last(v.to);
            }
            for z in board.zones.values_mut() {
                z.layers.iter_mut().for_each(|l| *l = last(*l));
            }
            board.stackup = s;
        }
        BoardEdit::Outline { outline } => board.outline = outline.clone(),
        BoardEdit::TestPoint {
            net,
            at,
            side,
            diameter,
        } => {
            let net_id = board
                .net(net)
                .ok_or_else(|| format!("the board has no net `{net}`"))?;
            let package = test_point_package(*diameter);
            let mut n = 1;
            while board.footprint_by_ref(&format!("TP{n}")).is_some() {
                n += 1;
            }
            let id = board.place(&format!("TP{n}"), package, *at, 0.0, *side);
            let f = board.footprints.get_mut(&id).unwrap();
            f.pad_nets.insert("1".into(), net_id);
            f.properties.insert("board_only".into(), "true".into());
        }
    }
    Ok(())
}

/// A test point: one round SMD pad of diameter `d`, no paste, and a courtyard 0.25 mm out.
fn test_point_package(d: Coord) -> crate::library::Package {
    use crate::library::{Package, PadDef, PadShape};
    let name = format!("TestPoint_D{}", super::values::mm_text(d));
    let mut pad = PadDef::smd("1", Point::ORIGIN, PadShape::Circle { d });
    pad.paste = false;
    let r = d / 2 + crate::geom::mm(0.25);
    Package {
        uuid: crate::library::make::uuid("package", &name),
        description: "Test point".into(),
        pads: vec![pad],
        courtyard: vec![vec![
            Point::new(-r, -r),
            Point::new(r, -r),
            Point::new(r, r),
            Point::new(-r, r),
        ]],
        silk: Vec::new(),
        fab: Vec::new(),
        model: None,
        body: None,
        smd: true,
        name,
    }
}

/// Runs a catalog edit on the layout `source` as one minimal text change.
pub fn board_command(
    path: &str,
    source: &str,
    edit: &BoardEdit,
) -> Result<SourceChange, EditFailure> {
    let parsed = parse_board(path, source);
    board_command_parsed(path, source, &parsed, edit)
}

/// [`board_command`] from a reading of `source` the caller keeps.
pub fn board_command_parsed(
    path: &str,
    source: &str,
    parsed: &ParsedBoard,
    edit: &BoardEdit,
) -> Result<SourceChange, EditFailure> {
    edit_parsed(path, source, parsed, |board| {
        apply_board_edit(board, &parsed.sites, edit)
    })
    .map(|(c, ())| c)
}

/// The standard stackup `board.layer-stack` writes: `layers` copper layers in `thickness`, with
/// every dielectric layer in `dielectric` when given, else FR-4 cores and prepreg.
pub fn standard_stackup(layers: u8, thickness: Coord, dielectric: Option<&Material>) -> Stackup {
    let mut s = Stackup::standard(layers, thickness, 1.0, 0.5);
    if let Some(m) = dielectric {
        for l in &mut s.layers {
            if matches!(l.kind, StackKind::Dielectric { .. }) {
                l.material = m.clone();
            }
        }
    }
    s
}

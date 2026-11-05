//! Re-reading a layout after a small text change.
//!
//! An edit to a 10 MB layout usually touches a few item statements (a moved footprint, a new
//! track). [`reparse_board`] finds the statements the change overlaps, reads only their new
//! text and patches the board: unchanged items keep their ids, so the caller's derived state
//! (rules, fills, DRC) updates from the [`BoardDelta`] alone.
//!
//! The patch covers item statements (footprints, fiducials, copper, vias, pours, keepouts,
//! texts, graphics, dimensions). A change that touches anything else (stackups, nets, rules,
//! packages, the sync section, the file head), renames or duplicates a footprint, or doesn't
//! read cleanly falls back to a whole read, and [`BoardDelta::whole`] says so.
//!
//! Ids: a re-read statement keeps the id of the statement it replaces when the window holds
//! the same kinds of statement in the same order; otherwise its old items are removed and the
//! new ones take fresh ids. A whole read numbers items in source order, so after an insertion
//! the two numberings differ; each is consistent with its own board.

use super::*;

/// What a re-read changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardDelta {
    /// The file was read whole: every item may have changed and ids may differ.
    pub whole: bool,
    /// Items re-read (added or modified), by their new ids.
    pub changed: Vec<ItemId>,
    /// Items whose statements are gone.
    pub removed: Vec<ItemId>,
    /// The old items the re-read statements replaced, as they were.
    pub replaced: Vec<BoardItem>,
}

/// A window's items, read against the board it belongs to.
pub(crate) struct Window {
    /// Old statement indices `first..last` the window replaces.
    first: usize,
    last: usize,
    /// Old byte range of the window, and its text's shift.
    old_start: usize,
    old_end: usize,
    delta: isize,
    /// The re-read statements, at new-source offsets, with their items.
    statements: Vec<BoardStatement>,
    items: Vec<BoardItem>,
}

/// One item read from a window.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum BoardItem {
    Footprint(Footprint),
    Fiducial(Fiducial),
    Track(Track),
    Via(Via),
    Zone(Zone),
    Keepout(Keepout),
    Text(BoardText),
    Graphic(BoardGraphic),
    Dimension(Dimension),
}

impl BoardItem {
    pub(crate) fn set_id(&mut self, id: ItemId) {
        match self {
            BoardItem::Footprint(x) => x.id = id,
            BoardItem::Fiducial(x) => x.id = id,
            BoardItem::Track(x) => x.id = id,
            BoardItem::Via(x) => x.id = id,
            BoardItem::Zone(x) => x.id = id,
            BoardItem::Keepout(x) => x.id = id,
            BoardItem::Text(x) => x.id = id,
            BoardItem::Graphic(x) => x.id = id,
            BoardItem::Dimension(x) => x.id = id,
        }
    }

    pub fn id(&self) -> ItemId {
        match self {
            BoardItem::Footprint(x) => x.id,
            BoardItem::Fiducial(x) => x.id,
            BoardItem::Track(x) => x.id,
            BoardItem::Via(x) => x.id,
            BoardItem::Zone(x) => x.id,
            BoardItem::Keepout(x) => x.id,
            BoardItem::Text(x) => x.id,
            BoardItem::Graphic(x) => x.id,
            BoardItem::Dimension(x) => x.id,
        }
    }
}

const ITEM_GROUPS: [Group; 9] = [
    Group::Footprint,
    Group::Fiducial,
    Group::Copper,
    Group::Via,
    Group::Pour,
    Group::Keepout,
    Group::Text,
    Group::Graphic,
    Group::Dimension,
];

/// Removes item `id` from whichever map holds it.
fn take_item(board: &mut Board, id: ItemId) -> Option<BoardItem> {
    if let Some(x) = board.footprints.remove(&id) {
        return Some(BoardItem::Footprint(x));
    }
    if let Some(x) = board.tracks.remove(&id) {
        return Some(BoardItem::Track(x));
    }
    if let Some(x) = board.vias.remove(&id) {
        return Some(BoardItem::Via(x));
    }
    if let Some(x) = board.zones.remove(&id) {
        return Some(BoardItem::Zone(x));
    }
    if let Some(x) = board.fiducials.remove(&id) {
        return Some(BoardItem::Fiducial(x));
    }
    if let Some(x) = board.keepouts.remove(&id) {
        return Some(BoardItem::Keepout(x));
    }
    if let Some(x) = board.texts.remove(&id) {
        return Some(BoardItem::Text(x));
    }
    if let Some(x) = board.graphics.remove(&id) {
        return Some(BoardItem::Graphic(x));
    }
    board.dimensions.remove(&id).map(BoardItem::Dimension)
}

fn put_item(board: &mut Board, item: BoardItem) {
    let id = item.id();
    match item {
        BoardItem::Footprint(x) => {
            board.footprints.insert(id, x);
        }
        BoardItem::Fiducial(x) => {
            board.fiducials.insert(id, x);
        }
        BoardItem::Track(x) => {
            board.tracks.insert(id, x);
        }
        BoardItem::Via(x) => {
            board.vias.insert(id, x);
        }
        BoardItem::Zone(x) => {
            board.zones.insert(id, x);
        }
        BoardItem::Keepout(x) => {
            board.keepouts.insert(id, x);
        }
        BoardItem::Text(x) => {
            board.texts.insert(id, x);
        }
        BoardItem::Graphic(x) => {
            board.graphics.insert(id, x);
        }
        BoardItem::Dimension(x) => {
            board.dimensions.insert(id, x);
        }
    }
}

/// The item `id` of `board`, as a window item (zone fills dropped: they are derived).
pub(crate) fn item_of(board: &Board, id: ItemId) -> Option<BoardItem> {
    let item = if let Some(x) = board.footprints.get(&id) {
        BoardItem::Footprint(x.clone())
    } else if let Some(x) = board.tracks.get(&id) {
        BoardItem::Track(x.clone())
    } else if let Some(x) = board.vias.get(&id) {
        BoardItem::Via(x.clone())
    } else if let Some(x) = board.zones.get(&id) {
        let mut z = x.clone();
        z.fill.clear();
        BoardItem::Zone(z)
    } else if let Some(x) = board.fiducials.get(&id) {
        BoardItem::Fiducial(x.clone())
    } else if let Some(x) = board.keepouts.get(&id) {
        BoardItem::Keepout(x.clone())
    } else if let Some(x) = board.texts.get(&id) {
        BoardItem::Text(x.clone())
    } else if let Some(x) = board.graphics.get(&id) {
        BoardItem::Graphic(x.clone())
    } else {
        BoardItem::Dimension(board.dimensions.get(&id)?.clone())
    };
    Some(item)
}

fn floor_boundary(s: &str, mut at: usize) -> usize {
    while at > 0 && !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The changed byte ranges of `old` and `new`: `(start, old_end, new_end)`.
fn changed_range(old: &str, new: &str) -> (usize, usize, usize) {
    let (a, b) = (old.as_bytes(), new.as_bytes());
    let max = a.len().min(b.len());
    let mut p = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    p = floor_boundary(old, p.min(max));
    let mut s = 0;
    while s < max - p && a[a.len() - 1 - s] == b[b.len() - 1 - s] {
        s += 1;
    }
    let mut old_end = a.len() - s;
    let mut new_end = b.len() - s;
    while !old.is_char_boundary(old_end) || !new.is_char_boundary(new_end) {
        old_end += 1;
        new_end += 1;
    }
    (p, old_end, new_end)
}

fn line_start(s: &str, at: usize) -> usize {
    s[..at].rfind('\n').map_or(0, |i| i + 1)
}

fn line_end(s: &str, at: usize) -> usize {
    s[at..].find('\n').map_or(s.len(), |i| at + i)
}

/// Reads the window of `new` that replaces the statements `old_src` → `new` changed, or
/// `None` when the change needs a whole read.
pub(crate) fn read_window(
    path: &str,
    old_src: &str,
    old: &ParsedBoard,
    new: &str,
) -> Option<Window> {
    let board = old.board.as_ref()?;
    let block = old.block?;
    let (p, old_end, new_end) = changed_range(old_src, new);
    // The statements the change overlaps, on whole lines, until no statement crosses the
    // window's edges.
    let statements = &old.statements;
    let mut start = line_start(old_src, p);
    let mut end = line_end(old_src, old_end.max(p));
    let (first, last) = loop {
        let first = statements.partition_point(|s| s.span.end <= start);
        let last = statements.partition_point(|s| s.span.start < end);
        let (mut s2, mut e2) = (start, end);
        if first < last {
            s2 = s2.min(line_start(old_src, statements[first].span.start));
            e2 = e2.max(line_end(old_src, statements[last - 1].span.end));
        }
        if (s2, e2) == (start, end) {
            break (first, last);
        }
        (start, end) = (s2, e2);
    };
    if start <= block.start || end >= block.end.saturating_sub(1) {
        return None;
    }
    if statements[first..last]
        .iter()
        .any(|s| !ITEM_GROUPS.contains(&s.group))
    {
        return None;
    }
    // Rules record their source offsets; a rule after a change that moves text reads whole.
    if new_end != old_end && statements[last..].iter().any(|s| s.group == Group::Rule) {
        return None;
    }
    let delta = new_end as isize - old_end as isize;
    let new_window_end = (end as isize + delta) as usize;
    if new_window_end < start || new_window_end > new.len() {
        return None;
    }
    let text = &new[start..new_window_end];
    // Read the window as a layout of its own, then move its spans to the new source.
    let name = old.name.clone().unwrap_or_default();
    let wrapper = format!(
        "{} {} {{\n",
        words::LAYOUT,
        runmat_design_syntax::json::quote(&name)
    );
    let wrapped = format!("{wrapper}{text}\n}}\n");
    let syntax = parse_source(Dialect::CircuitLayout, &wrapped);
    if !syntax.diagnostics.is_empty() {
        return None;
    }
    let items = syntax.file().items();
    let [runmat_design_syntax::ast::Item::Declaration(d)] = items.as_slice() else {
        return None;
    };
    let block_statements = d.block()?.statements();
    let shift = |s: Span| Span {
        start: s.start + start - wrapper.len(),
        end: s.end + start - wrapper.len(),
    };
    let nets: HashMap<String, NetId> = board
        .nets
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.clone(), NetId(i as u32)))
        .collect();
    let layers = Layers::of(&board.stackup);
    let ctx = Ctx {
        layers: &layers,
        nets: &nets,
        packages: &old.packages,
    };
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
            diagnostics: Vec::new(),
            packages: HashMap::new(),
        },
        sites: Sites::default(),
    };
    let mut heads = Vec::with_capacity(block_statements.len());
    for statement in &block_statements {
        let h = head(statement).ok()??;
        if !ITEM_WORDS.contains(&h.word.as_str()) {
            return None;
        }
        heads.push((h, shift(statement.span())));
    }
    let group_of = |w: &str| match w {
        words::FOOTPRINT => Group::Footprint,
        words::FIDUCIAL => Group::Fiducial,
        words::TRACK | words::ARC => Group::Copper,
        words::VIA => Group::Via,
        words::POUR => Group::Pour,
        words::KEEPOUT => Group::Keepout,
        words::TEXT => Group::Text,
        words::GRAPHIC => Group::Graphic,
        _ => Group::Dimension,
    };
    // A statement whose text is unchanged keeps its id; the rest keep theirs in order when
    // the window holds the same kinds of statement, else they take new ids.
    let new_text = |at: Span| &new[at.start..at.end];
    let mut by_text: HashMap<&str, std::collections::VecDeque<usize>> = HashMap::new();
    for (k, s) in statements[first..last].iter().enumerate() {
        by_text
            .entry(&old_src[s.span.start..s.span.end])
            .or_default()
            .push_back(k);
    }
    let mut reuse: Vec<Option<usize>> = heads
        .iter()
        .map(|(_, at)| by_text.get_mut(new_text(*at)).and_then(|q| q.pop_front()))
        .collect();
    let taken: BTreeSet<usize> = reuse.iter().flatten().copied().collect();
    let rest_old: Vec<usize> = (0..last - first).filter(|k| !taken.contains(k)).collect();
    let rest_new: Vec<usize> = (0..heads.len()).filter(|k| reuse[*k].is_none()).collect();
    if rest_old.len() == rest_new.len()
        && rest_old
            .iter()
            .zip(&rest_new)
            .all(|(o, n)| statements[first + o].group == group_of(&heads[*n].0.word))
    {
        for (o, n) in rest_old.iter().zip(&rest_new) {
            reuse[*n] = Some(*o);
        }
    }
    let mut scratch = Board::new("", Polygon::default(), board.stackup.clone());
    scratch.next_id = board.next_id;
    let mut ids = Vec::with_capacity(heads.len());
    for (k, (h, at)) in heads.iter().enumerate() {
        let id = reuse[k].and_then(|o| statements[first + o].item);
        let id = reader.read_item(&mut scratch, &ctx, h, *at, id)?;
        ids.push(id);
    }
    if !reader.out.diagnostics.is_empty() {
        return None;
    }
    let mut out_items = Vec::with_capacity(ids.len());
    for id in &ids {
        out_items.push(take_item(&mut scratch, *id)?);
    }
    // Footprints keep their schematic links; a renamed or duplicated designator changes the
    // sync section's meaning, so it reads whole.
    let replaced: BTreeSet<ItemId> = statements[first..last]
        .iter()
        .filter_map(|s| s.item)
        .collect();
    for item in &mut out_items {
        if let BoardItem::Footprint(f) = item {
            match board.footprints.get(&f.id) {
                Some(old_fp) if replaced.contains(&f.id) => {
                    if old_fp.reference != f.reference {
                        return None;
                    }
                    f.component = old_fp.component.clone();
                }
                _ => {
                    if board
                        .footprint_by_ref(&f.reference)
                        .is_some_and(|o| !replaced.contains(&o.id))
                    {
                        return None;
                    }
                }
            }
        }
    }
    let mut statements_out = reader.out.statements;
    statements_out.sort_by_key(|s| s.span.start);
    if statements_out.len() != ids.len() {
        return None;
    }
    Some(Window {
        first,
        last,
        old_start: start,
        old_end: end,
        delta,
        statements: statements_out,
        items: out_items,
    })
}

/// The items a window read, keyed by id: what an edit checks its new source against.
pub(crate) fn window_items(w: &Window) -> BTreeMap<ItemId, &BoardItem> {
    w.items.iter().map(|i| (i.id(), i)).collect()
}

/// The old statements a window replaced, by item id.
pub(crate) fn window_replaced(old: &ParsedBoard, w: &Window) -> Vec<ItemId> {
    old.statements[w.first..w.last]
        .iter()
        .filter_map(|s| s.item)
        .collect()
}

fn indexed_word(site: &str) -> Option<&str> {
    site.split_once('[').map(|(w, _)| w)
}

/// Patches `parsed` (the reading of the old source) with a window: the board, statements,
/// sites and spans of the new source.
fn apply_window(parsed: &mut ParsedBoard, w: Window) -> BoardDelta {
    let board = parsed
        .board
        .as_mut()
        .expect("a window reads against a board");
    let removed_ids: Vec<ItemId> = parsed.statements[w.first..w.last]
        .iter()
        .filter_map(|s| s.item)
        .collect();
    let mut removed: BTreeSet<ItemId> = BTreeSet::new();
    let mut replaced = Vec::new();
    for id in &removed_ids {
        replaced.extend(take_item(board, *id));
        removed.insert(*id);
    }
    let mut changed = Vec::with_capacity(w.items.len());
    for item in w.items {
        let id = item.id();
        removed.remove(&id);
        changed.push(id);
        board.next_id = board.next_id.max(id.0 + 1);
        put_item(board, item);
    }
    // Statements: the window's replace the old; the ones after move by the change's size.
    let shift = |s: &mut Span| {
        s.start = (s.start as isize + w.delta) as usize;
        s.end = (s.end as isize + w.delta) as usize;
    };
    let old_window: Vec<BoardStatement> = parsed
        .statements
        .splice(w.first..w.last, w.statements.iter().cloned())
        .collect();
    let after = w.first + w.statements.len();
    for s in &mut parsed.statements[after..] {
        shift(&mut s.span);
    }
    if let Some(b) = &mut parsed.block {
        b.end = (b.end as isize + w.delta) as usize;
    }
    let _ = (w.old_start, w.old_end);
    // Sites. Named sites are the statement's own; indexed ones count statements of a word in
    // source order, so they renumber when the window's count of a word changed.
    let count = |list: &[BoardStatement]| {
        let mut c: BTreeMap<String, usize> = BTreeMap::new();
        for s in list {
            if let Some(word) = indexed_word(&s.site) {
                *c.entry(word.to_owned()).or_default() += 1;
            }
        }
        c
    };
    let before = count(&old_window);
    let now = count(&w.statements);
    for s in &old_window {
        if parsed.spans.get(&s.site) == Some(&s.span) {
            parsed.spans.remove(&s.site);
        }
        if let Some(id) = s.item {
            parsed.sites.remove(&id);
        }
    }
    let mut counters: BTreeMap<String, usize> = BTreeMap::new();
    let renumber: BTreeSet<String> = before
        .keys()
        .chain(now.keys())
        .filter(|k| before.get(*k) != now.get(*k))
        .cloned()
        .collect();
    // Indices before the window are unchanged.
    for s in &parsed.statements[..w.first] {
        if let Some(word) = indexed_word(&s.site) {
            *counters.entry(word.to_owned()).or_default() += 1;
        }
    }
    for (k, s) in parsed.statements[w.first..].iter_mut().enumerate() {
        let in_window = k < w.statements.len();
        if let Some(word) = indexed_word(&s.site).map(str::to_owned) {
            let n = counters.entry(word.clone()).or_default();
            if in_window || renumber.contains(&word) {
                s.site = format!("{word}[{n}]");
            }
            *n += 1;
        }
        if in_window || indexed_word(&s.site).is_some_and(|w| renumber.contains(w)) {
            parsed.spans.insert(s.site.clone(), s.span);
            if let Some(id) = s.item {
                parsed.sites.insert(id, s.site.clone());
            }
        }
    }
    // A word with fewer statements leaves its last indices behind.
    for word in &renumber {
        let total = counters.get(word).copied().unwrap_or(0);
        let gone = before.get(word).copied().unwrap_or(0);
        let had = total + gone;
        for n in total..had {
            parsed.spans.remove(&format!("{word}[{n}]"));
        }
    }
    // Spans of every statement after the window moved.
    if w.delta != 0 {
        for s in &parsed.statements[after..] {
            if let Some(span) = parsed.spans.get_mut(&s.site) {
                *span = s.span;
            }
        }
    }
    // Head statements before the block don't move; spans of non-item statements after the
    // window are covered above.
    let mut changed_sorted = changed;
    changed_sorted.sort();
    BoardDelta {
        whole: false,
        changed: changed_sorted,
        removed: removed.into_iter().collect(),
        replaced,
    }
}

/// The reading of `new`, from the reading `old` of `old_src`: the changed statements re-read
/// when the change allows it (see the module docs), else the whole file.
pub fn reparse_board(
    path: &str,
    old_src: &str,
    old: ParsedBoard,
    new: &str,
) -> (ParsedBoard, BoardDelta) {
    let (parsed, delta, _) = reparse_board_keeping(path, old_src, old, new);
    (parsed, delta)
}

/// [`reparse_board`], also handing back the old reading when the file was read whole, so a
/// caller can compare the two (the session keeps the fill tiles a whole re-read didn't reach).
pub fn reparse_board_keeping(
    path: &str,
    old_src: &str,
    mut old: ParsedBoard,
    new: &str,
) -> (ParsedBoard, BoardDelta, Option<ParsedBoard>) {
    if old_src == new && old.diagnostics.is_empty() {
        return (old, BoardDelta::default(), None);
    }
    if old.diagnostics.is_empty() {
        if let Some(w) = read_window(path, old_src, &old, new) {
            let delta = apply_window(&mut old, w);
            return (old, delta, None);
        }
    }
    (
        parse_board(path, new),
        BoardDelta {
            whole: true,
            ..Default::default()
        },
        Some(old),
    )
}

use std::collections::BTreeSet;

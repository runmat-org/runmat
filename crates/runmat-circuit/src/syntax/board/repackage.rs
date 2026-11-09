//! Re-reading a layout after a change inside package blocks.
//!
//! A footprint statement names its package, and the reader copies the package into the
//! footprint, so a changed package (an ECO's footprint swap, a corrected pad) changes every
//! footprint that uses it and nothing else. [`read_packages`] re-reads the changed package
//! blocks and then only the footprint statements that name a package whose definition
//! changed. Each re-read footprint keeps its id and schematic link, so the session updates
//! rules, fills, DRC and the index from the [`BoardDelta`] alone, as it does for a moved
//! footprint.
//!
//! The change reads whole instead when it reaches outside package blocks, adds, removes or
//! renames a package, moves text ahead of a rule, or doesn't read cleanly.

use super::reparse::{apply_window, changed_range, put_item, statement_window, take_item, Window};
use super::*;
use runmat_design_syntax::ast::Statement;
use std::collections::BTreeSet;

/// The packages a change re-read, and the footprints re-read against them.
pub(crate) struct PackageChange {
    window: Window,
    packages: Vec<(String, Package)>,
    footprints: Vec<Footprint>,
}

fn reader(path: &str) -> Reader<'_> {
    Reader {
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
    }
}

/// `text` as the statements of a layout of its own, and the length of the wrapper before it.
fn wrapped_statements(name: &str, text: &str) -> Option<(Vec<Statement>, usize)> {
    let wrapper = format!(
        "{} {} {{\n",
        words::LAYOUT,
        runmat_design_syntax::json::quote(name)
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
    Some((d.block()?.statements(), wrapper.len()))
}

/// Reads the package blocks `old_src` → `new` changed and the footprints that use them, or
/// `None` when the change needs a whole read (see the module docs).
pub(crate) fn read_packages(
    path: &str,
    old_src: &str,
    old: &ParsedBoard,
    new: &str,
) -> Option<PackageChange> {
    let board = old.board.as_ref()?;
    let block = old.block?;
    let (p, old_end, new_end) = changed_range(old_src, new);
    let statements = &old.statements;
    let (start, end, first, last) = statement_window(old_src, statements, p, old_end);
    if first >= last || start <= block.start || end >= block.end.saturating_sub(1) {
        return None;
    }
    if statements[first..last]
        .iter()
        .any(|s| s.group != Group::Package)
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
    let name = old.name.clone().unwrap_or_default();
    let (blocks, wrapper) = wrapped_statements(&name, &new[start..new_window_end])?;
    // The same packages, in the same order: only their definitions changed.
    if blocks.len() != last - first {
        return None;
    }
    let mut packages = Vec::with_capacity(blocks.len());
    let mut window_statements = Vec::with_capacity(blocks.len());
    for (statement, was) in blocks.iter().zip(&statements[first..last]) {
        let Statement::Block(b) = statement else {
            return None;
        };
        if b.keyword() != words::PACKAGE {
            return None;
        }
        let key = b.name().map(|n| n.text()).unwrap_or_default();
        let site = format!("{}.{key}", words::PACKAGE);
        if site != was.site {
            return None;
        }
        let package = read_package(b, &key).ok()?;
        let span = statement.span();
        window_statements.push(BoardStatement {
            group: Group::Package,
            site,
            span: Span {
                start: span.start + start - wrapper,
                end: span.end + start - wrapper,
            },
            item: None,
        });
        packages.push((key, package));
    }
    let changed: BTreeSet<&str> = packages
        .iter()
        .filter(|(key, p)| old.packages.get(key) != Some(p))
        .map(|(key, _)| key.as_str())
        .collect();
    // The footprint statements naming a changed package, at their spans in `new`.
    let mut users: Vec<(ItemId, Span)> = Vec::new();
    for s in statements.iter().filter(|s| s.group == Group::Footprint) {
        let Some(id) = s.item else { continue };
        let f = board.footprints.get(&id)?;
        if !changed.contains(f.package.name.as_str()) {
            continue;
        }
        let moved = |at: usize| {
            if at >= end {
                (at as isize + delta) as usize
            } else {
                at
            }
        };
        users.push((
            id,
            Span {
                start: moved(s.span.start),
                end: moved(s.span.end),
            },
        ));
    }
    let footprints = read_footprints(path, &name, old, board, new, &packages, &users)?;
    Some(PackageChange {
        window: Window {
            first,
            last,
            old_start: start,
            old_end: end,
            delta,
            statements: window_statements,
            items: Vec::new(),
        },
        packages,
        footprints,
    })
}

/// Re-reads the footprint statements at `users` (spans in `new`) against the old packages with
/// `packages` in place, each under its old id with its schematic link.
fn read_footprints(
    path: &str,
    name: &str,
    old: &ParsedBoard,
    board: &Board,
    new: &str,
    packages: &[(String, Package)],
    users: &[(ItemId, Span)],
) -> Option<Vec<Footprint>> {
    if users.is_empty() {
        return Some(Vec::new());
    }
    let mut text = String::new();
    for (k, (_, at)) in users.iter().enumerate() {
        if k > 0 {
            text.push('\n');
        }
        text.push_str(new.get(at.start..at.end)?);
    }
    let (read, _) = wrapped_statements(name, &text)?;
    if read.len() != users.len() {
        return None;
    }
    let mut all = old.packages.clone();
    all.extend(packages.iter().cloned());
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
        packages: &all,
    };
    let mut reader = reader(path);
    let mut scratch = Board::new("", Polygon::default(), board.stackup.clone());
    scratch.next_id = board.next_id;
    let mut out = Vec::with_capacity(users.len());
    for (statement, (id, at)) in read.iter().zip(users) {
        let h = head(statement).ok()??;
        if h.word != words::FOOTPRINT {
            return None;
        }
        reader.read_item(&mut scratch, &ctx, &h, *at, Some(*id))?;
        let mut f = scratch.footprints.remove(id)?;
        let was = board.footprints.get(id)?;
        if was.reference != f.reference {
            return None;
        }
        f.component = was.component.clone();
        out.push(f);
    }
    reader.out.diagnostics.is_empty().then_some(out)
}

/// Patches `parsed` with a package change: the package statements and their spans, the
/// packages, and the footprints re-read against them (changed, with their old selves replaced).
pub(crate) fn apply_packages(parsed: &mut ParsedBoard, change: PackageChange) -> BoardDelta {
    let mut delta = apply_window(parsed, change.window);
    parsed.packages.extend(change.packages);
    let board = parsed
        .board
        .as_mut()
        .expect("a package change reads against a board");
    for f in change.footprints {
        let id = f.id;
        delta.replaced.extend(take_item(board, id));
        put_item(board, BoardItem::Footprint(f));
        delta.changed.push(id);
    }
    delta.changed.sort();
    delta
}

//! Signal-flow placement of one page, as an engineer draws it.
//!
//! - **Roles.** A two-pin part between a signal net and a rail is a *shunt* (a filter cap, a
//!   pull-down), drawn upright below its net (above it for a pull-up). One between two rails is
//!   a *decoupling* cap, drawn upright in a row under the IC that uses the rail (the IC whose
//!   nearby parts are numbered closest). One across a box's input and output is *feedback*,
//!   drawn above the box. Boxes, sheet symbols and two-pin parts in series are the *chain*.
//! - **Columns.** The chain's signal nets give each item a column: drivers (outputs,
//!   connectors, a sheet's outputs) feed loads (inputs), and a series part runs left pin to
//!   right pin. Feedback loops are cut where a depth-first walk from the connectors meets them,
//!   and each item sits one column right of everything that feeds it.
//! - **Rows.** Each column is ordered by its neighbours' rows (barycentre sweeps), then every
//!   item is lifted to line up its input with the pin that drives it, so a chain runs straight.
//! - **Facing.** A box whose connections mostly leave the far side is mirrored; a sheet symbol's
//!   pins sit on the side their nets go to.
//! - **Clusters.** Each group the signal nets join is laid out on its own; the groups then pack
//!   in rows, signal groups first.
//!
//! Every choice breaks ties by the netlist order, so the same design gives the same page.

use super::symbol::{Art, Place, PinSide, UnitLayout};
use super::{Cell, CellKind, NetInfo, GRID, PIN_PITCH};
use crate::geom::{Coord, Point};
use crate::library::PinType;
use std::collections::{BTreeMap, BTreeSet};

mod cluster;
use cluster::Ctx;

/// A shunt hangs this far below (or above) its net.
pub(super) const DROP: Coord = 4 * GRID;
/// Spacing of upright parts side by side.
pub(super) const UPRIGHT_PITCH: Coord = 10 * GRID;
/// The least gap between columns, and between items in a column.
pub(super) const COLUMN_GAP: Coord = 8 * GRID;
pub(super) const ROW_GAP: Coord = 6 * GRID;
/// Room for a power port and its text beyond a pin tip.
pub(super) const PORT_ROOM: Coord = 5 * GRID;
/// Most decoupling caps in one row.
pub(super) const DECAP_ROW: usize = 5;
/// Gap between clusters.
const CLUSTER_GAP: Coord = 10 * GRID;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Role {
    Chain,
    /// `pin`: the layout pin on the signal net; `up`: a pull-up (the rail above).
    Shunt { pin: usize, up: bool },
    /// `pin`: the layout pin on the positive rail.
    Decap { host: Option<usize>, pin: usize },
    Feedback { host: usize },
    Loose,
}

/// One chain terminal on a signal net.
#[derive(Clone, Copy, Debug)]
pub(super) struct Term {
    pub(super) cell: usize,
    pub(super) pin: usize,
    pub(super) class: Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    Drive,
    Sink,
    Pass,
}

pub(super) fn small(c: &Cell) -> bool {
    matches!(
        c.lay.art,
        Art::Resistor | Art::Capacitor | Art::Inductor | Art::Diode
    )
}

fn number(designator: &str) -> Option<i64> {
    let digits: String = designator
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit())
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Places every cell of a page: origins relative to the page's top-left content corner (y
/// up, so items go negative), and each sheet symbol's pin sides. Returns one place per cell.
pub(super) fn place(cells: &mut [Cell], nets: &[NetInfo]) -> Vec<Place> {
    let first = place_once(cells, nets);
    if face_partners(cells, &first) {
        place_once(cells, nets)
    } else {
        first
    }
}

/// Orders each side of the sheet symbol with the most entries by where its nets' other ends
/// sit, top down, so the wires to it run straight across instead of crossing. Returns whether
/// any order changed.
fn face_partners(cells: &mut [Cell], places: &[Place]) -> bool {
    let Some(hub) = (0..cells.len())
        .filter(|&i| matches!(cells[i].kind, CellKind::Sheet { .. }))
        .max_by_key(|&i| (cells[i].lay.pins.len(), std::cmp::Reverse(i)))
    else {
        return false;
    };
    let tip = |c: &Cell, p: usize, at: &Place| at.apply(c.lay.tip(c.lay.pins[p].side, c.lay.pins[p].at));
    let height = |net: Option<usize>, cells: &[Cell]| -> Option<Coord> {
        let ys: Vec<Coord> = (0..cells.len())
            .filter(|&j| j != hub)
            .flat_map(|j| {
                (0..cells[j].nets.len())
                    .filter(move |&p| net.is_some() && cells[j].nets[p] == net)
                    .map(move |p| (j, p))
            })
            .map(|(j, p)| tip(&cells[j], p, &places[j]).y)
            .collect();
        (!ys.is_empty()).then(|| ys.iter().sum::<Coord>() / ys.len() as Coord)
    };
    let c = &cells[hub];
    let mut order: Vec<usize> = (0..c.lay.pins.len()).collect();
    let key: Vec<(u8, Coord)> = (0..c.lay.pins.len())
        .map(|p| {
            let side = u8::from(c.lay.pins[p].side == PinSide::Right);
            (side, -height(c.nets[p], cells).unwrap_or(tip(c, p, &places[hub]).y))
        })
        .collect();
    order.sort_by_key(|&p| (key[p], p));
    if order.iter().enumerate().all(|(k, &p)| k == p) {
        return false;
    }
    let c = &mut cells[hub];
    c.lay.pins = order.iter().map(|&p| c.lay.pins[p].clone()).collect();
    c.nets = order.iter().map(|&p| c.nets[p]).collect();
    c.sheet_out = order.iter().map(|&p| c.sheet_out[p]).collect();
    super::restack_sheet(&mut c.lay);
    true
}

fn place_once(cells: &mut [Cell], nets: &[NetInfo]) -> Vec<Place> {
    let n = cells.len();
    let rail = |net: Option<usize>| net.is_some_and(|k| nets[k].rail);
    let signal = |net: Option<usize>| net.is_some_and(|k| !nets[k].rail);
    // Roles of the small parts first: shunts, feedback, decoupling.
    let mut roles = vec![Role::Chain; n];
    for i in 0..n {
        let c = &cells[i];
        if c.lay.art == Art::Hole {
            roles[i] = Role::Loose;
            continue;
        }
        if !small(c) {
            continue;
        }
        let (a, b) = (c.nets[0], c.nets[1]);
        roles[i] = match (rail(a), rail(b)) {
            (true, true) => {
                let pin = if nets[a.unwrap()].ground && !nets[b.unwrap()].ground {
                    1
                } else {
                    0
                };
                Role::Decap { host: None, pin }
            }
            (true, false) | (false, true) if signal(a) || signal(b) => {
                let (pin, r) = if rail(a) { (1, a) } else { (0, b) };
                let ground = nets[r.unwrap()].ground;
                if !ground && matches!(c.lay.art, Art::Inductor | Art::Diode) {
                    // A series inductor or diode onto a supply rail (a buck's output inductor,
                    // an OR-ing diode) carries the power on: it sits in the chain.
                    Role::Chain
                } else {
                    Role::Shunt { pin, up: !ground }
                }
            }
            (true, false) | (false, true) => Role::Loose,
            (false, false) if a.is_none() && b.is_none() => Role::Loose,
            _ => Role::Chain,
        };
    }
    // Feedback: a series part across one box's left and right pins.
    let boxes: Vec<usize> = (0..n)
        .filter(|&i| roles[i] == Role::Chain && !small(&cells[i]))
        .collect();
    for i in 0..n {
        if roles[i] != Role::Chain || !small(&cells[i]) || cells[i].lay.art == Art::Diode {
            continue;
        }
        let (a, b) = (cells[i].nets[0], cells[i].nets[1]);
        let host = boxes.iter().copied().find(|&h| {
            let side_of = |net: Option<usize>| {
                cells[h]
                    .nets
                    .iter()
                    .position(|x| net.is_some() && *x == net)
                    .map(|p| cells[h].lay.pins[p].side)
            };
            matches!(
                (side_of(a), side_of(b)),
                (Some(PinSide::Left), Some(PinSide::Right))
                    | (Some(PinSide::Right), Some(PinSide::Left))
            )
        });
        if let Some(host) = host {
            roles[i] = Role::Feedback { host };
        }
    }
    // Chain terminals per signal net.
    let mut terms: BTreeMap<usize, Vec<Term>> = BTreeMap::new();
    for i in 0..n {
        if roles[i] != Role::Chain {
            continue;
        }
        for (p, net) in cells[i].nets.iter().enumerate() {
            let Some(k) = net.filter(|k| !nets[*k].rail) else {
                continue;
            };
            let class = match &cells[i].kind {
                CellKind::Sheet { .. } => {
                    if cells[i].sheet_out[p] {
                        Class::Drive
                    } else {
                        Class::Sink
                    }
                }
                _ if cells[i].connector => Class::Drive,
                _ => match cells[i].lay.pins[p].kind {
                    PinType::Output
                    | PinType::PowerOut
                    | PinType::TriState
                    | PinType::OpenCollector
                    | PinType::OpenEmitter => Class::Drive,
                    PinType::Input | PinType::PowerIn => Class::Sink,
                    _ => Class::Pass,
                },
            };
            terms.entry(k).or_default().push(Term {
                cell: i,
                pin: p,
                class,
            });
        }
    }
    // Directed signal edges with a strength: driver to input 4, driver to passive 3, passive
    // to input 2, passive to passive (right pin to left pin) 1.
    let mut edges: BTreeSet<(std::cmp::Reverse<u8>, usize, usize)> = BTreeSet::new();
    let sheet_pair = |a: usize, b: usize| {
        let pins = |i: usize| cells[i].lay.pins.len();
        matches!(cells[a].kind, CellKind::Sheet { .. })
            && matches!(cells[b].kind, CellKind::Sheet { .. })
            && (pins(a), a) < (pins(b), b)
    };
    let mut joined: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for ts in terms.values() {
        if ts.len() > 24 {
            continue;
        }
        for a in ts {
            for b in ts {
                if a.cell == b.cell {
                    continue;
                }
                joined[a.cell].insert(b.cell);
                let side = |t: &Term| cells[t.cell].lay.pins[t.pin].side;
                let strength = match (a.class, b.class) {
                    (Class::Drive, Class::Sink) => 4,
                    (Class::Drive, Class::Pass) => 3,
                    (Class::Pass, Class::Sink) => 2,
                    (Class::Pass, Class::Pass)
                        if side(a) == PinSide::Right && side(b) == PinSide::Left =>
                    {
                        1
                    }
                    // Two sheets on a net neither drives: the smaller feeds the larger, so a
                    // top sheet's peripherals stand left of the block they serve.
                    (Class::Sink, Class::Sink) if sheet_pair(a.cell, b.cell) => 1,
                    _ => 0,
                };
                if strength > 0 {
                    edges.insert((std::cmp::Reverse(strength), a.cell, b.cell));
                }
            }
        }
    }
    let chain: Vec<usize> = (0..n).filter(|&i| roles[i] == Role::Chain).collect();
    // Power flow: each supply rail's sources (a box's power output, a series part onto the
    // rail) feed the boxes that take the rail as a supply input, so a regulator chain reads
    // left to right (battery, buck, inductor, LDO). Only a rail with a source on this page and
    // a few such inputs counts, so a logic rail feeding a whole page doesn't pull it into one
    // column chain.
    let sources = rail_sources(cells, nets, &roles);
    let mut feeds: BTreeMap<usize, usize> = BTreeMap::new();
    for (&k, from) in &sources {
        let sinks: Vec<usize> = chain
            .iter()
            .copied()
            .filter(|&i| !small(&cells[i]) && !from.contains(&i))
            .filter(|&i| {
                cells[i]
                    .nets
                    .iter()
                    .zip(&cells[i].lay.pins)
                    .any(|(x, p)| *x == Some(k) && p.kind == PinType::PowerIn)
            })
            .collect();
        if sinks.is_empty() || sinks.len() > 3 {
            continue;
        }
        for &t in &sinks {
            for &f in from {
                edges.insert((std::cmp::Reverse(1), f, t));
                joined[f].insert(t);
                joined[t].insert(f);
            }
            feeds.entry(t).or_insert(from[0]);
        }
    }
    let rank = ranks(n, &edges);
    // Decoupling hosts: boxes on the cap's positive rail. A box whose attached parts are
    // numbered within three of the cap takes it (a regulator's input and output caps); else a
    // box that takes the rail as a supply input, the one with fewest caps so far.
    let mut attached: Vec<Vec<i64>> = vec![Vec::new(); n];
    for i in 0..n {
        if let Role::Shunt { pin, .. } = roles[i] {
            if let Some(ts) = cells[i].nets[pin].and_then(|k| terms.get(&k)) {
                for t in ts {
                    attached[t.cell].extend(number(&cells[i].designator));
                }
            }
        }
        if let Role::Feedback { host } = roles[i] {
            attached[host].extend(number(&cells[i].designator));
        }
    }
    for (i, joins) in joined.iter().enumerate() {
        if roles[i] == Role::Chain && small(&cells[i]) {
            for &j in joins {
                attached[j].extend(number(&cells[i].designator));
            }
        }
    }
    let mut decaps: Vec<usize> = vec![0; n];
    for i in 0..n {
        let Role::Decap { pin, .. } = roles[i] else {
            continue;
        };
        let net = cells[i].nets[pin];
        let me = number(&cells[i].designator);
        let on_rail: Vec<usize> = boxes
            .iter()
            .copied()
            .filter(|&h| cells[h].nets.contains(&net))
            .collect();
        let near = |h: usize| {
            me.map(|m| attached[h].iter().map(|a| (a - m).abs()).min().unwrap_or(i64::MAX))
                .unwrap_or(i64::MAX)
        };
        let supply = |h: usize| {
            cells[h]
                .nets
                .iter()
                .zip(&cells[h].lay.pins)
                .any(|(x, p)| *x == net && p.kind == PinType::PowerIn)
        };
        // On a rail made on this page, a cap belongs to the source or the supply input declared
        // just before it: a regulator's output caps by its inductor, the next stage's input
        // caps by that stage.
        let driven = || {
            let from = net.and_then(|k| sources.get(&k))?;
            from.iter()
                .copied()
                .chain(on_rail.iter().copied().filter(|&h| supply(h)))
                .filter(|&h| h < i)
                .max()
        };
        let host = on_rail
            .iter()
            .copied()
            .filter(|&h| near(h) <= 3)
            .min_by_key(|&h| (near(h), decaps[h], h))
            .or_else(driven)
            .or_else(|| {
                on_rail
                    .iter()
                    .copied()
                    .min_by_key(|&h| (!supply(h), decaps[h], h))
            });
        if let Some(h) = host {
            decaps[h] += 1;
        }
        roles[i] = Role::Decap { host, pin };
    }
    // Facing: mirror boxes whose connections leave the far side; turn series parts fed from
    // the right; put each sheet pin on its nets' side.
    let mut places = vec![Place::default(); n];
    let others = |i: usize, net: usize| -> Option<f64> {
        let ts = terms.get(&net)?;
        let rs: Vec<f64> = ts
            .iter()
            .filter(|t| t.cell != i)
            .map(|t| rank[t.cell] as f64)
            .collect();
        (!rs.is_empty()).then(|| rs.iter().sum::<f64>() / rs.len() as f64)
    };
    for &i in &chain {
        let want = |p: usize| -> i32 {
            let Some(k) = cells[i].nets[p].filter(|k| !nets[*k].rail) else {
                return 0;
            };
            match others(i, k) {
                Some(r) if r > rank[i] as f64 => 1,
                Some(r) if r < rank[i] as f64 => -1,
                _ => 0,
            }
        };
        match cells[i].kind {
            CellKind::Sheet { .. } => {
                let wants: Vec<i32> = (0..cells[i].lay.pins.len()).map(want).collect();
                for (p, &w) in wants.iter().enumerate() {
                    let right = w > 0 || (w == 0 && cells[i].sheet_out[p]);
                    cells[i].lay.pins[p].side = if right {
                        PinSide::Right
                    } else {
                        PinSide::Left
                    };
                }
                super::restack_sheet(&mut cells[i].lay);
            }
            _ => {
                let score: i32 = (0..cells[i].lay.pins.len())
                    .map(|p| match cells[i].lay.pins[p].side {
                        PinSide::Left => -want(p),
                        PinSide::Right => want(p),
                        _ => 0,
                    })
                    .sum();
                if score < 0 {
                    if small(&cells[i]) {
                        places[i].rotation = 180;
                    } else {
                        places[i].mirror = true;
                    }
                }
            }
        }
    }
    for i in 0..n {
        match roles[i] {
            // Upright with the signal pin (or the positive rail) on top: pin 0 is the left pin,
            // a quarter turn clockwise puts it on top.
            Role::Shunt { pin, up } => {
                let top_is_left = (pin == 0) != up;
                places[i].rotation = if top_is_left { 270 } else { 90 };
            }
            Role::Decap { pin, .. } => places[i].rotation = if pin == 0 { 270 } else { 90 },
            _ => {}
        }
    }
    // Clusters: chain items joined by signal nets, with their attachments.
    let mut group: Vec<usize> = (0..n).collect();
    fn find(g: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while g[r] != r {
            r = g[r];
        }
        let mut y = x;
        while g[y] != r {
            let next = g[y];
            g[y] = r;
            y = next;
        }
        r
    }
    let unite = |g: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(g, a), find(g, b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            g[hi] = lo;
        }
    };
    for &i in &chain {
        for &j in &joined[i] {
            unite(&mut group, i, j);
        }
    }
    // Shunt anchors: the chain pin each shunt hangs next to, and the gap it sits in.
    let mut anchor: BTreeMap<usize, (usize, usize, usize)> = BTreeMap::new();
    // Shunts on a net no chain item is on (a divider into a port): they stand together about
    // their node, one cluster per net.
    let mut nodes: BTreeSet<usize> = BTreeSet::new();
    let mut node_of: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        match roles[i] {
            Role::Shunt { pin, .. } => {
                let Some(ts) = cells[i].nets[pin].and_then(|k| terms.get(&k)) else {
                    match cells[i].nets[pin] {
                        Some(k) => {
                            nodes.insert(i);
                            let first = *node_of.entry(k).or_insert(i);
                            unite(&mut group, i, first);
                        }
                        None => roles[i] = Role::Loose,
                    }
                    continue;
                };
                let side = |t: &Term| places[t.cell].side(cells[t.cell].lay.pins[t.pin].side);
                let pick = ts
                    .iter()
                    .filter(|t| side(t) != PinSide::Left)
                    .min_by_key(|t| (rank[t.cell], t.cell, t.pin))
                    .map(|t| (t, rank[t.cell] + 1))
                    .or_else(|| {
                        ts.iter()
                            .max_by_key(|t| (rank[t.cell], std::cmp::Reverse(t.cell)))
                            .map(|t| (t, rank[t.cell]))
                    });
                if let Some((t, gap)) = pick {
                    anchor.insert(i, (t.cell, t.pin, gap));
                    unite(&mut group, i, t.cell);
                }
            }
            Role::Feedback { host } => unite(&mut group, i, host),
            Role::Decap {
                host: Some(host), ..
            } => unite(&mut group, i, host),
            _ => {}
        }
    }
    // Loose parts (holes, unhosted caps) gather in one cluster of their own.
    let loose: Vec<usize> = (0..n)
        .filter(|&i| matches!(roles[i], Role::Loose | Role::Decap { host: None, .. }))
        .collect();
    for w in loose.windows(2) {
        unite(&mut group, w[0], w[1]);
    }
    let mut clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let r = find(&mut group, i);
        clusters.entry(r).or_default().push(i);
    }
    // Signal clusters (with sheets or connectors) first, then by size, then netlist order;
    // loose parts last.
    let mut order: Vec<Vec<usize>> = clusters.into_values().collect();
    order.sort_by_key(|m| {
        let is_loose = m.iter().all(|i| loose.contains(i));
        let lead = m
            .iter()
            .any(|&i| matches!(cells[i].kind, CellKind::Sheet { .. }) || cells[i].connector);
        let pins: usize = m.iter().map(|&i| cells[i].nets.iter().flatten().count()).sum();
        let units = m.iter().all(|&i| cells[i].nets.iter().all(Option::is_none));
        (is_loose, units, !lead, std::cmp::Reverse(pins), m[0])
    });
    let ctx = Ctx {
        cells,
        nets,
        roles: &roles,
        rank: &rank,
        terms: &terms,
        joined: &joined,
        anchor: &anchor,
        feeds: &feeds,
    };
    let mut boxes_out: Vec<[Coord; 4]> = Vec::new();
    let mut origins: Vec<Vec<(usize, Point)>> = Vec::new();
    for members in &order {
        let (placed, bbox) = if members.iter().all(|i| loose.contains(i)) {
            ctx.row(members, &mut places)
        } else if members.iter().all(|i| nodes.contains(i)) {
            ctx.node(members, &mut places)
        } else {
            ctx.cluster(members, &mut places)
        };
        origins.push(placed);
        boxes_out.push(bbox);
    }
    // Pack clusters in rows from the top left, as wide as the smallest frame (A4 to A0) whose
    // height then holds them, clear of the title block: the last row may run down beside the
    // title block, but not over it.
    let widest = boxes_out.iter().map(|b| b[2] - b[0]).max().unwrap_or(0);
    let mut chosen = Vec::new();
    for (w, h) in super::FRAMES.iter().map(|f| (f.1, f.2)) {
        let nm = |v: f64| (v * 1.0e6) as Coord;
        // Labels and texts can reach past the cluster boxes: keep 20 mm spare.
        let width = (nm(w) - 2 * super::MARGIN - nm(20.0)).max(widest);
        let full = nm(h) - 2 * super::MARGIN - nm(10.0);
        let band = full - nm(super::TITLE_ROW_MM * 6.0);
        // The title block's left edge, from the content's left (with the same 20 mm spare).
        let title_x = nm(w - super::FRAME_MM - super::TITLE_W_MM - 22.0) - super::MARGIN;
        let (offsets, used) = pack(&boxes_out, width);
        let clear = offsets.iter().zip(&boxes_out).all(|((dx, dy), b)| {
            let (right, bottom) = (b[2] + dx, b[1] + dy);
            -bottom <= band || right <= title_x
        });
        chosen = offsets;
        if used <= full && clear {
            break;
        }
    }
    for (k, (dx, dy)) in chosen.iter().enumerate() {
        for (i, o) in &origins[k] {
            places[*i].origin = Point::new(o.x + dx, o.y + dy);
        }
    }
    places
}

/// Shelf packing of cluster boxes in rows of `width`, first fit: each box goes in the first
/// row with room for it, so a small cluster fills the gap a large one left, and the clusters
/// keep their order along each row. Returns each box's offset and the height used.
fn pack(boxes: &[[Coord; 4]], width: Coord) -> (Vec<(Coord, Coord)>, Coord) {
    // Per row: the next x and the height so far.
    let mut rows: Vec<(Coord, Coord)> = Vec::new();
    let mut at: Vec<(usize, Coord)> = Vec::with_capacity(boxes.len());
    for b in boxes {
        let (w, h) = (b[2] - b[0], b[3] - b[1]);
        let row = rows
            .iter()
            .position(|(x, _)| *x + w <= width)
            .unwrap_or_else(|| {
                rows.push((0, 0));
                rows.len() - 1
            });
        let (x, row_h) = &mut rows[row];
        at.push((row, *x));
        *x += w + CLUSTER_GAP;
        *row_h = (*row_h).max(h);
    }
    let mut tops = Vec::with_capacity(rows.len());
    let mut y = 0;
    for (_, h) in &rows {
        tops.push(y);
        y -= h + CLUSTER_GAP;
    }
    let out = boxes
        .iter()
        .zip(&at)
        .map(|(b, (row, x))| (snap(x - b[0]), snap(tops[*row] - b[3])))
        .collect();
    (out, -y - CLUSTER_GAP.min(-y))
}

/// Each supply rail's sources on the page: chain items with a power output on it, and series
/// parts onto it (a buck's inductor). Ground has none.
fn rail_sources(cells: &[Cell], nets: &[NetInfo], roles: &[Role]) -> BTreeMap<usize, Vec<usize>> {
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, c) in cells.iter().enumerate() {
        if roles[i] != Role::Chain {
            continue;
        }
        for (p, net) in c.nets.iter().enumerate() {
            let Some(k) = net.filter(|k| nets[*k].rail && !nets[*k].ground) else {
                continue;
            };
            if small(c) || c.lay.pins[p].kind == PinType::PowerOut {
                let from = out.entry(k).or_default();
                if !from.contains(&i) {
                    from.push(i);
                }
            }
        }
    }
    out
}

pub(super) fn snap(v: Coord) -> Coord {
    v.div_euclid(GRID) * GRID
}

/// Columns: the longest path over a loop-free subset of the edges, taken strongest first (an
/// edge that would close a loop is dropped, so a feedback path never pushes its driver right).
fn ranks(n: usize, edges: &BTreeSet<(std::cmp::Reverse<u8>, usize, usize)>) -> Vec<usize> {
    let mut kept: Vec<Vec<usize>> = vec![Vec::new(); n];
    let reaches = |kept: &Vec<Vec<usize>>, from: usize, to: usize| {
        let mut seen = vec![false; n];
        let mut stack = vec![from];
        while let Some(v) = stack.pop() {
            if v == to {
                return true;
            }
            if !std::mem::replace(&mut seen[v], true) {
                stack.extend(kept[v].iter().copied());
            }
        }
        false
    };
    for &(_, a, b) in edges {
        if !kept[a].contains(&b) && !reaches(&kept, b, a) {
            kept[a].push(b);
        }
    }
    // Longest path by relaxation in topological order (Kahn's, lowest index first).
    let mut indeg = vec![0usize; n];
    for list in &kept {
        for &b in list {
            indeg[b] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut rank = vec![0usize; n];
    while let Some(v) = ready.pop_first() {
        for &w in &kept[v] {
            rank[w] = rank[w].max(rank[v] + 1);
            indeg[w] -= 1;
            if indeg[w] == 0 {
                ready.insert(w);
            }
        }
    }
    rank
}


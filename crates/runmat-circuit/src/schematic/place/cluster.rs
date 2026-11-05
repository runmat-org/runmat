//! One cluster laid out: columns by rank, gaps holding shunts, feedback above its box and
//! decoupling rows under their IC.

use super::*;

pub(super) struct Ctx<'a> {
    pub(super) cells: &'a [Cell],
    pub(super) nets: &'a [NetInfo],
    pub(super) roles: &'a [Role],
    pub(super) rank: &'a [usize],
    pub(super) terms: &'a BTreeMap<usize, Vec<Term>>,
    pub(super) joined: &'a [BTreeSet<usize>],
    pub(super) anchor: &'a BTreeMap<usize, (usize, usize, usize)>,
}

impl Ctx<'_> {
    /// A cell's extent around its origin as placed, with room for its texts and the ports and
    /// labels its pins get.
    fn extent(&self, i: usize, place: &Place) -> [Coord; 4] {
        let c = &self.cells[i];
        let lay: &UnitLayout = &c.lay;
        let p0 = Place {
            origin: Point::ORIGIN,
            ..*place
        };
        let mut e = p0.rect(lay.extent());
        for (k, pin) in lay.pins.iter().enumerate() {
            let side = p0.side(pin.side);
            let tip = p0.apply(lay.tip(pin.side, pin.at));
            let room = match c.nets[k] {
                Some(net) if self.nets[net].rail => PORT_ROOM,
                Some(net) if self.nets[net].labelled => {
                    2 * GRID + crate::schematic::text_width(&self.nets[net].short, crate::schematic::LABEL_TEXT)
                        + 3 * GRID
                }
                Some(_) => 2 * GRID,
                None => GRID,
            };
            let (dx, dy) = side.out();
            let far = Point::new(tip.x + dx * room, tip.y + dy * room);
            e = [
                e[0].min(far.x),
                e[1].min(far.y),
                e[2].max(far.x),
                e[3].max(far.y),
            ];
        }
        // Designator above and value below (beside an upright part).
        if place.upright() {
            [e[0], e[1], e[2] + 9 * GRID, e[3]]
        } else {
            let w = crate::schematic::text_width(&c.designator, crate::schematic::DESIGNATOR_TEXT) + 4 * GRID;
            let half = (e[2] - e[0]) / 2;
            let grow = (w / 2 - half).max(0);
            [e[0] - grow, e[1] - 3 * GRID, e[2] + grow, e[3] + 3 * GRID]
        }
    }

    /// Lays out one cluster: columns by rank, gaps holding shunts, feedback above its box and
    /// decoupling under its IC. Returns each member's origin and the cluster's box.
    pub(super) fn cluster(&self, members: &[usize], places: &mut [Place]) -> (Vec<(usize, Point)>, [Coord; 4]) {
        let chain: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&i| self.roles[i] == Role::Chain)
            .collect();
        let mut ranks: Vec<usize> = chain.iter().map(|&i| self.rank[i]).collect();
        ranks.sort();
        ranks.dedup();
        let col_of = |i: usize| ranks.binary_search(&self.rank[i]).unwrap();
        let ncols = ranks.len();
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); ncols];
        for &i in &chain {
            cols[col_of(i)].push(i);
        }
        // Barycentre sweeps: order each column by its neighbours' rows.
        let mut row: BTreeMap<usize, f64> = BTreeMap::new();
        for col in &cols {
            for (k, &i) in col.iter().enumerate() {
                row.insert(i, k as f64);
            }
        }
        for sweep in 0..4 {
            let range: Vec<usize> = if sweep % 2 == 0 {
                (1..ncols).collect()
            } else {
                (0..ncols.saturating_sub(1)).rev().collect()
            };
            for c in range {
                let mut keyed: Vec<(f64, usize)> = cols[c]
                    .iter()
                    .map(|&i| {
                        let near: Vec<f64> = self.joined[i]
                            .iter()
                            .filter(|j| row.contains_key(j) && col_of(**j) != c)
                            .map(|j| row[j])
                            .collect();
                        let b = if near.is_empty() {
                            row[&i]
                        } else {
                            near.iter().sum::<f64>() / near.len() as f64
                        };
                        (b, i)
                    })
                    .collect();
                keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
                cols[c] = keyed.iter().map(|k| k.1).collect();
                for (k, &i) in cols[c].iter().enumerate() {
                    row.insert(i, k as f64);
                }
            }
        }
        // Attachments per host.
        let feedback: BTreeMap<usize, Vec<usize>> = members.iter().fold(BTreeMap::new(), |mut m, &i| {
            if let Role::Feedback { host } = self.roles[i] {
                m.entry(host).or_insert_with(Vec::new).push(i);
            }
            m
        });
        let decaps: BTreeMap<usize, Vec<usize>> = members.iter().fold(BTreeMap::new(), |mut m, &i| {
            if let Role::Decap { host: Some(h), .. } = self.roles[i] {
                m.entry(h).or_insert_with(Vec::new).push(i);
            }
            m
        });
        // Slot of each chain item: its extent plus feedback above and decoupling below,
        // around the item's origin.
        let up_ext = |i: usize, places: &[Place]| self.extent(i, &places[i]);
        let mut slot: BTreeMap<usize, [Coord; 4]> = BTreeMap::new();
        for &i in &chain {
            let mut e = up_ext(i, places);
            let base = e;
            if let Some(fb) = feedback.get(&i) {
                let mut top = base[3];
                for &f in fb {
                    let fe = up_ext(f, places);
                    top += fe[3] - fe[1] + GRID;
                    e[0] = e[0].min(fe[0]);
                    e[2] = e[2].max(fe[2]);
                }
                e[3] = top;
            }
            if let Some(ds) = decaps.get(&i) {
                let de = up_ext(ds[0], places);
                let per = ds.len().min(DECAP_ROW) as Coord;
                let rows = ds.len().div_ceil(DECAP_ROW) as Coord;
                let w = per * UPRIGHT_PITCH;
                e[0] = e[0].min(-w / 2);
                e[2] = e[2].max(w / 2 + (de[2] - de[0]) / 2);
                e[1] = base[1] + decap_lift(&places[i]) - rows * (GRID + de[3] - de[1]);
            }
            slot.insert(i, e);
        }
        // Gaps: shunts between columns.
        let mut gaps: Vec<Vec<usize>> = vec![Vec::new(); ncols + 1];
        for &i in members {
            if let Some(&(cell, _, gap)) = self.anchor.get(&i) {
                if self.roles[i] != Role::Chain && chain.contains(&cell) {
                    let g = ranks.partition_point(|r| *r < gap);
                    gaps[g.min(ncols)].push(i);
                }
            }
        }
        let gap_w: Vec<Coord> = gaps
            .iter()
            .map(|g| {
                if g.is_empty() {
                    COLUMN_GAP
                } else {
                    COLUMN_GAP + 4 * GRID + g.len() as Coord * UPRIGHT_PITCH
                }
            })
            .collect();
        // Column x: each column as wide as its widest slot, centred.
        let mut col_x: Vec<Coord> = Vec::with_capacity(ncols);
        let mut x = gap_w[0];
        for c in 0..ncols {
            let w0 = cols[c].iter().map(|i| -slot[i][0]).max().unwrap_or(0);
            let w1 = cols[c].iter().map(|i| slot[i][2]).max().unwrap_or(0);
            col_x.push(snap(x + w0));
            x += w0 + w1 + gap_w[c + 1];
        }
        // Rows: column 0 stacked from the top; later columns lifted to line up each item's
        // first fed pin with its driver's pin, then pushed apart.
        let mut origin: BTreeMap<usize, Point> = BTreeMap::new();
        for c in 0..ncols {
            let mut wants: Vec<(Coord, usize)> = Vec::new();
            for (k, &i) in cols[c].iter().enumerate() {
                let want = (c > 0)
                    .then(|| self.lineup(i, places, &origin))
                    .flatten()
                    .unwrap_or(-(k as Coord) * PIN_PITCH);
                wants.push((want, i));
            }
            let mut floor = Coord::MAX;
            for (want, i) in wants {
                let s = slot[&i];
                let y = snap(want.min(floor.saturating_sub(s[3])));
                origin.insert(i, Point::new(col_x[c], y));
                floor = y + s[1] - ROW_GAP;
            }
        }
        for &i in &chain {
            places[i].origin = origin[&i];
        }
        let tip = |cell: usize, pin: usize, places: &[Place]| -> Point {
            let lay = &self.cells[cell].lay;
            places[cell].apply(lay.tip(lay.pins[pin].side, lay.pins[pin].at))
        };
        // Feedback above its box, centred.
        for (&h, fb) in &feedback {
            let mut top = up_ext(h, places)[3];
            for &f in fb {
                let fe = up_ext(f, places);
                let o = places[h].origin;
                places[f].origin = Point::new(o.x, snap(o.y + top - fe[1] + GRID));
                top += fe[3] - fe[1] + GRID;
            }
        }
        // Decoupling in a row under its IC.
        for (&h, ds) in &decaps {
            let o = places[h].origin;
            let bottom = up_ext(h, places)[1] + decap_lift(&places[h]);
            let de = up_ext(ds[0], places);
            let per = ds.len().min(DECAP_ROW) as Coord;
            let x0 = o.x - (per - 1) * UPRIGHT_PITCH / 2;
            for (k, &d) in ds.iter().enumerate() {
                let (row, col) = ((k / DECAP_ROW) as Coord, (k % DECAP_ROW) as Coord);
                places[d].origin = Point::new(
                    snap(x0 + col * UPRIGHT_PITCH),
                    snap(o.y + bottom - GRID - de[3] - row * (GRID + de[3] - de[1])),
                );
            }
        }
        // Shunts in their gaps: lower nets nearer the right column, upright below (or above)
        // the anchor pin.
        let mut gap_x = 0;
        for (g, list) in gaps.iter().enumerate() {
            let left = if g == 0 { 0 } else { gap_x };
            let mut keyed: Vec<(Coord, usize)> = list
                .iter()
                .map(|&i| {
                    let (cell, pin, _) = self.anchor[&i];
                    let y = tip(cell, pin, places).y;
                    let up = matches!(self.roles[i], Role::Shunt { up: true, .. });
                    (if up { -y } else { y }, i)
                })
                .collect();
            keyed.sort();
            let right_edge = if g < ncols {
                col_x[g] - cols[g].iter().map(|i| -slot[i][0]).max().unwrap_or(0)
            } else {
                left + gap_w[g]
            };
            for (k, (_, i)) in keyed.iter().enumerate() {
                let (cell, pin, _) = self.anchor[i];
                let y = tip(cell, pin, places).y;
                let up = matches!(self.roles[*i], Role::Shunt { up: true, .. });
                let lay = &self.cells[*i].lay;
                let reach = lay.half_w + lay.pin_len;
                let sx = snap(right_edge - COLUMN_GAP - k as Coord * UPRIGHT_PITCH);
                let sy = if up { y + DROP + reach } else { y - DROP - reach };
                places[*i].origin = Point::new(sx, snap(sy));
            }
            if g < ncols {
                gap_x = col_x[g] + cols[g].iter().map(|i| slot[i][2]).max().unwrap_or(0);
            }
        }
        let mut out = Vec::new();
        let mut bbox = [Coord::MAX, Coord::MAX, Coord::MIN, Coord::MIN];
        for &i in members {
            if self.roles[i] != Role::Chain
                && !matches!(self.roles[i], Role::Feedback { .. } | Role::Decap { host: Some(_), .. })
                && !self.anchor.contains_key(&i)
            {
                continue;
            }
            let e = self.extent(i, &places[i]);
            let o = places[i].origin;
            bbox = [
                bbox[0].min(o.x + e[0]),
                bbox[1].min(o.y + e[1]),
                bbox[2].max(o.x + e[2]),
                bbox[3].max(o.y + e[3]),
            ];
            out.push((i, o));
        }
        if out.is_empty() {
            bbox = [0, 0, 0, 0];
        }
        (out, bbox)
    }

    /// The origin y that lines up a cell's first fed pin with the placed pin driving it.
    fn lineup(&self, i: usize, places: &[Place], origin: &BTreeMap<usize, Point>) -> Option<Coord> {
        let lay = &self.cells[i].lay;
        for (p, pin) in lay.pins.iter().enumerate() {
            if places[i].side(pin.side) != PinSide::Left {
                continue;
            }
            let Some(ts) = self.cells[i].nets[p].and_then(|k| self.terms.get(&k)) else {
                continue;
            };
            for t in ts {
                if t.cell == i || !origin.contains_key(&t.cell) {
                    continue;
                }
                let other = &self.cells[t.cell].lay;
                let op = &other.pins[t.pin];
                let pl = Place {
                    origin: origin[&t.cell],
                    ..places[t.cell]
                };
                let at = pl.apply(other.tip(op.side, op.at));
                let mine = Place {
                    origin: Point::ORIGIN,
                    ..places[i]
                }
                .apply(lay.tip(pin.side, pin.at));
                return Some(at.y - mine.y);
            }
        }
        None
    }

    /// Loose parts in one row.
    pub(super) fn row(&self, members: &[usize], places: &mut [Place]) -> (Vec<(usize, Point)>, [Coord; 4]) {
        let mut x = 0;
        let mut out = Vec::new();
        let mut bbox = [Coord::MAX, Coord::MAX, Coord::MIN, Coord::MIN];
        for &i in members {
            if matches!(self.roles[i], Role::Decap { host: None, .. }) {
                places[i].rotation = places[i].rotation.max(90);
            }
            let e = self.extent(i, &places[i]);
            let o = Point::new(snap(x - e[0]), 0);
            places[i].origin = o;
            bbox = [
                bbox[0].min(o.x + e[0]),
                bbox[1].min(o.y + e[1]),
                bbox[2].max(o.x + e[2]),
                bbox[3].max(o.y + e[3]),
            ];
            x = o.x + e[2] + 2 * GRID;
            out.push((i, o));
        }
        (out, bbox)
    }
}

/// How far a decoupling row rises toward its IC: a boxed IC's extent keeps 3 grids under its
/// lowest port for its value, which sits beside the body, and the row keeps one grid more, so
/// the caps' port room meets the IC's ground-port room and the row sits about a port's height
/// under the IC, as an engineer draws it.
fn decap_lift(host: &Place) -> Coord {
    if host.upright() {
        0
    } else {
        4 * GRID
    }
}

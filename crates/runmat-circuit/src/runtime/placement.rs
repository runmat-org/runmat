//! `board.place` and `board.align`: footprints to new places on the board, as one
//! [`BoardEdit::Arrange`].
//!
//! - **Place** puts footprints inside the board outline, on a grid in its free room (clear of
//!   the footprints already inside), on the chosen side. From schematic places every footprint
//!   that sits outside the outline (where Update board stages new parts); Single component
//!   places the picked ones: one outside the outline goes on the grid, one already inside keeps
//!   its spot and takes the side. A locked footprint isn't moved.
//! - **Align** lines the picked components up by their extents (courtyard and pads): left,
//!   right, top or bottom edges, horizontal centers (one X) or vertical centers (one Y), or
//!   spreads them evenly between the outermost two. Board Y grows upward, so Top is the
//!   largest Y.

use super::edit::{site_of, strings, EditError, EditRequest};
use super::*;
use crate::eco::Placer;
use crate::geom::{Coord, Point, Rect};
use crate::layout::Footprint;

/// The footprints a command acts on: its `field` picks, else the selection.
fn picked<'b>(
    board: &'b Board,
    sites: &BTreeMap<ItemId, String>,
    request: &EditRequest,
    field: &str,
) -> Result<Vec<&'b Footprint>, EditError> {
    let ids = match strings(request.values.get(field)) {
        picks if !picks.is_empty() => picks,
        _ => request.selection.clone(),
    };
    let mut out = Vec::new();
    for id in &ids {
        let site = site_of(board, sites, id)?;
        let f = board
            .footprint_by_ref(&site)
            .ok_or_else(|| EditError::new(format!("{id} isn't a component on the board")))?;
        if !out.iter().any(|g: &&Footprint| g.id == f.id) {
            out.push(f);
        }
    }
    Ok(out)
}

/// Where a footprint's origin goes so its extents' center lands on `center`.
fn origin_for(f: &Footprint, center: Point) -> Point {
    let c = f.bbox().center();
    Point::new(center.x + f.at.x - c.x, center.y + f.at.y - c.y)
}

fn inside(board: &Board, f: &Footprint) -> bool {
    board.outline.contains(f.bbox().center()).unwrap_or(false)
}

pub(super) fn place(
    board: &Board,
    sites: &BTreeMap<ItemId, String>,
    request: &EditRequest,
) -> Result<BoardEdit, EditError> {
    if board.outline.outer.len() < 3 {
        return Err(EditError::new(
            "draw the board outline first; components are placed inside it",
        ));
    }
    let side = match request.values.get("side").and_then(Value::as_str) {
        Some("bottom") => Side::Bottom,
        _ => Side::Top,
    };
    let single = request.values.get("mode").and_then(Value::as_str) == Some("single");
    let mut targets: Vec<&Footprint> = if single {
        let picked = picked(board, sites, request, "components")?;
        if picked.is_empty() {
            return Err(EditError::new("pick the components to place"));
        }
        picked
    } else {
        board
            .footprints
            .values()
            .filter(|f| !f.locked && !inside(board, f))
            .collect()
    };
    if targets.is_empty() {
        return Err(EditError::new(
            "every component is on the board already; Update board brings new ones from the schematic",
        ));
    }
    if let Some(f) = targets.iter().find(|f| f.locked) {
        return Err(EditError::new(format!("{} is locked", f.reference)));
    }
    targets.sort_by(|a, b| crate::ids::natural_cmp(&a.reference, &b.reference));
    // A picked part already inside the outline keeps its spot and only takes the side; the
    // rest go on the grid.
    let (stay, targets): (Vec<&Footprint>, Vec<&Footprint>) = targets
        .into_iter()
        .partition(|f| single && inside(board, f));
    let occupied: Vec<Rect> = board
        .footprints
        .values()
        .filter(|f| !targets.iter().any(|t| t.id == f.id) && inside(board, f))
        .map(|f| f.bbox())
        .collect();
    let sizes: Vec<_> = targets
        .iter()
        .map(|f| {
            let b = f.bbox();
            (b.width(), b.height())
        })
        .collect();
    let area = board.outline.bbox();
    let spots: Vec<Point> = Placer::grid(area, &sizes, &occupied)
        .into_iter()
        .filter(|p| board.outline.contains(*p).unwrap_or(false))
        .collect();
    if spots.len() < targets.len() {
        return Err(EditError::new(format!(
            "the board has free room for {} of the {} components; enlarge the outline or move parts aside",
            spots.len(),
            targets.len()
        )));
    }
    Ok(BoardEdit::Arrange {
        moves: stay
            .iter()
            .map(|f| (f.reference.clone(), f.at))
            .chain(
                targets
                    .iter()
                    .zip(spots)
                    .map(|(f, at)| (f.reference.clone(), origin_for(f, at))),
            )
            .collect(),
        side: Some(side),
        command: "board.place",
    })
}

pub(super) fn align(
    board: &Board,
    sites: &BTreeMap<ItemId, String>,
    request: &EditRequest,
) -> Result<BoardEdit, EditError> {
    let how = request
        .values
        .get("align")
        .and_then(Value::as_str)
        .unwrap_or("left");
    let parts = picked(board, sites, request, "components")?;
    if parts.len() < 2 {
        return Err(EditError::new("pick two or more components to align"));
    }
    if let Some(f) = parts.iter().find(|f| f.locked) {
        return Err(EditError::new(format!("{} is locked", f.reference)));
    }
    let boxes: Vec<Rect> = parts.iter().map(|f| f.bbox()).collect();
    let all = boxes.iter().fold(Rect::EMPTY, |r, b| r.union(*b));
    // Each footprint's shift (dx, dy) by its box.
    let shift: Vec<(Coord, Coord)> = match how {
        "left" => boxes.iter().map(|b| (all.min.x - b.min.x, 0)).collect(),
        "right" => boxes.iter().map(|b| (all.max.x - b.max.x, 0)).collect(),
        "top" => boxes.iter().map(|b| (0, all.max.y - b.max.y)).collect(),
        "bottom" => boxes.iter().map(|b| (0, all.min.y - b.min.y)).collect(),
        "center_x" => boxes
            .iter()
            .map(|b| (all.center().x - b.center().x, 0))
            .collect(),
        "center_y" => boxes
            .iter()
            .map(|b| (0, all.center().y - b.center().y))
            .collect(),
        "distribute_x" | "distribute_y" => {
            if parts.len() < 3 {
                return Err(EditError::new("pick three or more components to distribute"));
            }
            let along = |b: &Rect| {
                if how == "distribute_x" {
                    b.center().x
                } else {
                    b.center().y
                }
            };
            let mut order: Vec<usize> = (0..boxes.len()).collect();
            order.sort_by_key(|&i| along(&boxes[i]));
            let first = along(&boxes[order[0]]);
            let last = along(&boxes[*order.last().expect("three or more")]);
            let step = (last - first) as f64 / (order.len() - 1) as f64;
            let mut out = vec![(0, 0); boxes.len()];
            for (k, &i) in order.iter().enumerate() {
                let d = first + (step * k as f64).round() as Coord - along(&boxes[i]);
                out[i] = if how == "distribute_x" { (d, 0) } else { (0, d) };
            }
            out
        }
        other => {
            return Err(EditError::new(format!(
                "`{other}` isn't an alignment; expected left, right, top, bottom, center_x, center_y, distribute_x or distribute_y"
            )))
        }
    };
    Ok(BoardEdit::Arrange {
        moves: parts
            .iter()
            .zip(shift)
            .filter(|(_, (dx, dy))| *dx != 0 || *dy != 0)
            .map(|(f, (dx, dy))| (f.reference.clone(), Point::new(f.at.x + dx, f.at.y + dy)))
            .collect(),
        side: None,
        command: "board.align",
    })
}

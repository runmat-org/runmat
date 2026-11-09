//! Tables on the sheet, snapped to its corners and zones the way the title block is: the
//! revision table in the top right corner (ASME Y14.35), parts lists, cut lists and general notes
//! on the title block's top edge, right-aligned (ISO 7573, ISO 7200), hole and bend tables in the
//! top left corner.
//! Tables in one corner stack away from it. A table placed by hand (`at`) stays there: its
//! top-left corner, or a revision table's top-right corner. A table whose corner is taken goes
//! to the free spot nearest it.

use super::marks::{line, prim, text};
use super::solver::{nearest_free, BoxR, Class};
use super::{placed_at, Ctx, Prim, Shape, Xy};
use serde_json::Value;

/// Table text and row proportions: 2.5 mm text in 7 mm rows.
const TEXT: f64 = 2.5;
const ROW: f64 = 7.0;
const PAD: f64 = 1.5;
/// The space between stacked tables.
const SPACE: f64 = 4.0;
/// The pitch of a wrapped cell's lines.
const PITCH: f64 = TEXT * 1.5;

/// A table ready to draw: columns (title, alignment), rows of cells (a group title spans the
/// table), and an optional title row. The `wrap` column keeps its set width and breaks longer
/// text at spaces onto more lines, its row growing to fit (a parts list's descriptions).
pub struct Grid {
    pub title: Option<String>,
    pub columns: Vec<(String, &'static str, Option<f64>)>,
    pub rows: Vec<Row>,
    pub wrap: Option<usize>,
}

pub enum Row {
    Cells(Vec<String>),
    Group(String),
}

impl Grid {
    fn widths(&self) -> Vec<f64> {
        self.columns
            .iter()
            .enumerate()
            .map(|(i, (title, _, fixed))| {
                let text = |n: usize| n as f64 * TEXT * 0.6 + 2.0 * PAD;
                if let (Some(w), true) = (fixed, self.wrap == Some(i)) {
                    return w.max(text(title.chars().count()));
                }
                let longest = self.rows.iter().filter_map(|r| if let Row::Cells(c) = r { c.get(i).map(|s| s.chars().count()) } else { None }).max().unwrap_or(0).max(title.chars().count());
                fixed.unwrap_or(0.0).max(text(longest))
            })
            .collect()
    }

    /// Each cell's lines: the wrap column's text broken at spaces to its width.
    fn lines(&self, widths: &[f64], cells: &[String]) -> Vec<Vec<String>> {
        cells
            .iter()
            .enumerate()
            .map(|(i, v)| match (self.wrap == Some(i), widths.get(i)) {
                (true, Some(w)) => wrap_text(v, ((w - 2.0 * PAD) / (TEXT * 0.6)).floor().max(1.0) as usize),
                _ => vec![v.clone()],
            })
            .collect()
    }

    /// A row's height: one line, plus a pitch for each further wrapped line.
    fn row_height(&self, widths: &[f64], row: &Row) -> f64 {
        match row {
            Row::Cells(c) => ROW + (self.lines(widths, c).iter().map(Vec::len).max().unwrap_or(1).max(1) - 1) as f64 * PITCH,
            Row::Group(_) => ROW,
        }
    }

    pub fn size(&self) -> Xy {
        let widths = self.widths();
        let w: f64 = widths.iter().sum();
        let body: f64 = self.rows.iter().map(|r| self.row_height(&widths, r)).sum();
        [w, (1 + usize::from(self.title.is_some())) as f64 * ROW + body]
    }

    /// The table with its top-left corner at `top_left`.
    pub fn draw(&self, id: &str, top_left: Xy) -> Vec<Prim> {
        let widths = self.widths();
        let [w, h] = self.size();
        let [x0, y0] = top_left;
        let p = |style: &'static str, shape: Shape| prim(id, "table", "Annotations", style, shape);
        let mut out = vec![p("frame", Shape::Rect { x: x0, y: y0 - h, w, h })];
        let mut row = 0;
        if let Some(t) = &self.title {
            out.push(p("note", text(x0 + w / 2.0, y0 - ROW / 2.0 - TEXT / 2.0, t, TEXT, "middle")));
            out.push(p("frame", line(vec![x0, y0 - ROW, x0 + w, y0 - ROW])));
            row = 1;
        }
        let body_top = y0 - row as f64 * ROW;
        // One row of cells from `top` down, `rh` high; each cell's lines centred in it.
        let cells = |top: f64, rh: f64, values: &[String], style: &'static str, out: &mut Vec<Prim>| {
            out.push(p("frame", line(vec![x0, top - rh, x0 + w, top - rh])));
            let mut x = x0;
            for (i, ((_, align, _), lines)) in self.columns.iter().zip(self.lines(&widths, values)).enumerate() {
                let cw = widths[i];
                let tx = match *align {
                    "end" => x + cw - PAD,
                    "middle" => x + cw / 2.0,
                    _ => x + PAD,
                };
                let first = top - rh / 2.0 + (TEXT + (lines.len().max(1) - 1) as f64 * PITCH) / 2.0 - TEXT;
                for (k, v) in lines.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                    out.push(p(style, text(tx, first - k as f64 * PITCH, v, TEXT, align)));
                }
                x += cw;
            }
        };
        let heads: Vec<String> = self.columns.iter().map(|c| c.0.to_uppercase()).collect();
        cells(body_top, ROW, &heads, "note", &mut out);
        let mut top = body_top - ROW;
        for r in &self.rows {
            let rh = self.row_height(&widths, r);
            match r {
                Row::Cells(c) => cells(top, rh, c, "note", &mut out),
                Row::Group(t) => out.push(p("muted", text(x0 + PAD, top - ROW / 2.0 - TEXT / 2.0, t, TEXT, "start"))),
            }
            top -= rh;
        }
        let mut x = x0;
        for cw in &widths[..widths.len().saturating_sub(1)] {
            x += cw;
            out.push(p("frame", line(vec![x, body_top, x, y0 - h])));
        }
        out
    }
}

/// `text` broken at spaces into lines of at most `chars` characters (a longer word keeps its
/// own line).
fn wrap_text(text: &str, chars: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ').filter(|w| !w.is_empty()) {
        match lines.last_mut() {
            Some(l) if l.chars().count() + 1 + word.chars().count() <= chars => {
                l.push(' ');
                l.push_str(word);
            }
            _ => lines.push(word.to_owned()),
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn align(v: &Value) -> &'static str {
    match v.as_str() {
        Some("right") => "end",
        Some("center") => "middle",
        _ => "start",
    }
}

fn fixed(key: &str) -> Option<f64> {
    match key {
        "item" | "quantity" => Some(12.0),
        "part_number" => Some(36.0),
        "description" => Some(56.0),
        "material" => Some(34.0),
        _ => None,
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().into_iter().flatten().map(|c| c.as_str().unwrap_or("").to_owned()).collect()
}

/// A parts list or cut list (`bom_tables` / `cut_list_tables` data) as a grid.
fn bom_grid(data: &Value) -> Grid {
    let keys: Vec<&str> = data["columns"].as_array().into_iter().flatten().map(|c| c["key"].as_str().unwrap_or("")).collect();
    let columns = data["columns"].as_array().into_iter().flatten().map(|c| (c["title"].as_str().unwrap_or("").to_owned(), align(&c["align"]), fixed(c["key"].as_str().unwrap_or("")))).collect();
    let wrap = keys.iter().position(|k| *k == "description");
    let rows = if let Some(lines) = data["lines"].as_array() {
        lines.iter().map(|l| if l["line"] == "group" { Row::Group(l["title"].as_str().unwrap_or("").to_owned()) } else { Row::Cells(strings(&l["cells"])) }).collect()
    } else {
        data["rows"].as_array().into_iter().flatten().map(|r| Row::Cells(strings(&r["cells"]))).collect()
    };
    Grid { title: None, columns, rows, wrap }
}

fn hole_grid(data: &Value) -> Grid {
    let columns = vec![("TAG".into(), "middle", Some(12.0)), ("X".into(), "end", Some(16.0)), ("Y".into(), "end", Some(16.0)), ("SIZE".into(), "start", Some(30.0))];
    let rows = data["rows"].as_array().into_iter().flatten().map(|r| Row::Cells(["tag", "x_text", "y_text", "size"].iter().map(|k| r[*k].as_str().unwrap_or("").to_owned()).collect())).collect();
    Grid { title: Some("HOLE TABLE".into()), columns, rows, wrap: None }
}

fn revision_grid(rows: &Value) -> Grid {
    let columns = vec![("REV".into(), "middle", Some(12.0)), ("DESCRIPTION".into(), "start", Some(50.0)), ("DATE".into(), "start", Some(24.0)), ("APPROVED".into(), "start", Some(30.0))];
    let rows = rows.as_array().into_iter().flatten().map(|r| Row::Cells(["rev", "description", "date", "by"].iter().map(|k| r[*k].as_str().unwrap_or("").to_owned()).collect())).collect();
    Grid { title: Some("REVISIONS".into()), columns, rows, wrap: None }
}

/// General notes: numbered in order, each note wrapping in a column as wide as the title block.
fn notes_grid(rows: &Value) -> Grid {
    let note_width = runmat_cad::drawing::title_layout::BLOCK_WIDTH - NOTE_NUMBER_WIDTH;
    let columns = vec![("NO.".into(), "middle", Some(NOTE_NUMBER_WIDTH)), ("NOTE".into(), "start", Some(note_width))];
    let rows = rows.as_array().into_iter().flatten().filter_map(Value::as_str).enumerate().map(|(i, note)| Row::Cells(vec![(i + 1).to_string(), note.to_owned()])).collect();
    Grid { title: Some("GENERAL NOTES".into()), columns, rows, wrap: Some(1) }
}

/// The notes table's number column.
const NOTE_NUMBER_WIDTH: f64 = 10.0;

fn fmt(v: f64, places: usize) -> String {
    runmat_cad::drawing::standard::metric_text(v, places)
}

fn bend_grid(flat: &Value) -> Option<Grid> {
    let rows: Vec<Row> = flat["bend_table"]
        .as_array()?
        .iter()
        .map(|r| {
            let n = |k: &str| r[k].as_f64().unwrap_or(0.0);
            Row::Cells(vec![r["tag"].as_str().unwrap_or("").into(), r["direction"].as_str().unwrap_or("").to_uppercase(), format!("{}°", fmt(n("angle"), 2)), fmt(n("inner_radius"), 2), fmt(n("allowance"), 3), fmt(n("deduction"), 3)])
        })
        .collect();
    if rows.is_empty() {
        return None;
    }
    let columns = [("TAG", 12.0), ("DIRECTION", 20.0), ("ANGLE", 16.0), ("RADIUS", 16.0), ("BA", 16.0), ("BD", 16.0)].iter().map(|(t, w)| ((*t).to_owned(), "middle", Some(*w))).collect();
    Some(Grid { title: Some("BEND TABLE".into()), columns, rows, wrap: None })
}

#[derive(Clone, Copy, PartialEq)]
enum Corner {
    TopRight,
    AboveTitle,
    TopLeft,
}

/// Places and draws every table on the sheet.
pub fn place(ctx: &mut Ctx<'_>) {
    let data = ctx.data;
    let doc = ctx.doc();
    let mut list: Vec<(String, Grid, Corner, Option<Xy>, bool)> = Vec::new();
    let mut bend_views = Vec::new();
    for t in doc["tables"].as_array().into_iter().flatten() {
        if t["sheet"].as_str() != Some(ctx.sheet.as_str()) {
            continue;
        }
        let name = t["name"].as_str().unwrap_or("").to_owned();
        let at = placed_at(t);
        let (grid, corner) = match t["kind"]["table"].as_str() {
            Some("bom") => match data["bom_tables"].get(&name) {
                Some(d) => (bom_grid(d), Corner::AboveTitle),
                None => continue,
            },
            Some("cut_list") => match data["cut_list_tables"].get(&name) {
                Some(d) => (bom_grid(d), Corner::AboveTitle),
                None => continue,
            },
            Some("hole") => match data["hole_tables"].get(&name) {
                Some(d) => (hole_grid(d), Corner::TopLeft),
                None => continue,
            },
            Some("revision") => (revision_grid(&t["kind"]["rows"]), Corner::TopRight),
            Some("notes") => (notes_grid(&t["kind"]["rows"]), Corner::AboveTitle),
            Some("bend") => {
                let view = t["kind"]["view"].as_str().unwrap_or("");
                bend_views.push(view.to_owned());
                match bend_grid(&data["hlr"][view]["flat"]) {
                    Some(g) => (g, Corner::TopLeft),
                    None => continue,
                }
            }
            _ => continue,
        };
        // A revision table is placed by its top-right corner.
        let top_right = corner == Corner::TopRight;
        list.push((format!("table:{name}"), grid, corner, at, top_right));
    }
    // A flat view with bends and no bend table of its own gets one.
    for v in doc["views"].as_array().into_iter().flatten() {
        let name = v["name"].as_str().unwrap_or("");
        if v["sheet"].as_str() != Some(ctx.sheet.as_str()) || bend_views.iter().any(|b| b == name) {
            continue;
        }
        if let Some(g) = bend_grid(&data["hlr"][name]["flat"]) {
            list.push((format!("view:{name}"), g, Corner::TopLeft, None, false));
        }
    }
    let f = ctx.frame;
    let title_top = ctx.occ.items.iter().find(|i| i.owner == "title").map_or(f.y0, |i| i.b.y1);
    let mut stacks = [f.y1, title_top, f.y1];
    for (id, grid, corner, at, top_right) in list {
        let [w, h] = grid.size();
        let top_left = match at {
            Some(p) if top_right => [p[0] - w, p[1]],
            Some(p) => p,
            None => {
                let k = corner as usize;
                let snapped = match corner {
                    Corner::TopRight => [f.x1 - w, stacks[k]],
                    Corner::AboveTitle => [f.x1 - w, stacks[k] + h],
                    Corner::TopLeft => [f.x0, stacks[k]],
                };
                let b = BoxR::new(snapped[0], snapped[1] - h, snapped[0] + w, snapped[1]);
                // A table keeps clear of the centring marks. One on the frame's right edge steps
                // left of the mark before it looks elsewhere, so it keeps its place on the title
                // block.
                let marks = super::border::centring_marks(ctx.size, &f);
                let clear = |b: &BoxR| marks.iter().all(|(p, q)| !b.grow(0.5).crossed_by(*p, *q));
                let stepped = [snapped[0] - super::border::CENTRING_REACH - SPACE / 2.0, snapped[1]];
                let stepped_box = BoxR::new(stepped[0], stepped[1] - h, stepped[0] + w, stepped[1]);
                let spot = if ctx.occ.free(&b, &id, 0.0, &[]) && clear(&b) {
                    snapped
                } else if corner != Corner::TopLeft && ctx.occ.free(&stepped_box, &id, 0.0, &[]) && clear(&stepped_box) {
                    stepped
                } else {
                    match nearest_free(&ctx.occ, &id, w, h, b.center(), 2.5, 1.0, &clear) {
                        Some(n) => [n.x0, n.y1],
                        None => {
                            ctx.finding("table_no_room", &id, format!("Table {} has no room on the sheet; it overlaps what is there", &id[id.find(':').map_or(0, |i| i + 1)..]));
                            snapped
                        }
                    }
                };
                stacks[k] = match corner {
                    Corner::AboveTitle => spot[1] + SPACE,
                    _ => spot[1] - h - SPACE,
                };
                spot
            }
        };
        ctx.out.placed.insert(id.clone(), if top_right { [top_left[0] + w, top_left[1]] } else { top_left });
        let prims = grid.draw(&id, top_left);
        ctx.occ.add(BoxR::new(top_left[0], top_left[1] - h, top_left[0] + w, top_left[1]), &id, Class::Table);
        ctx.push(prims);
    }
}

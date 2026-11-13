//! Notes as blocks of text: a note's text set in lines that wrap at spaces to a width, and the
//! sheet's general notes as a numbered list under a "NOTES:" heading, each note wrapping with a
//! hanging indent under its number (ASME Y14.100, ISO 7200).
//!
//! Text is measured the way every export sets it: each character 0.6 of the text height wide
//! (the sheet font, and Courier in the PDF), so the sheet, the PDF and the DXF break lines at
//! the same words. Each line is its own text primitive, so every export writes the same lines.
//!
//! A note keeps clear of the border, the title block, tables, views and other notes:
//! - with a `width`, it wraps to that width;
//! - placed by hand without one, it wraps at the room to its right: up to the border, the title
//!   block or the next item in its way;
//! - when it has no room where it was put, it moves to the nearest free spot and the layout says
//!   so (`note_moved`); when the sheet has no free spot for it, the layout says that instead
//!   (`note_no_room`), with what to change.

use super::marks::mark;
use super::solver::{nearest_free, BoxR, Class};
use super::{Ctx, Prim, Std, Xy};
use serde_json::Value;

/// The narrowest column a note wraps in before it counts as having no room.
const MIN_WIDTH: f64 = 30.0;
/// The column a note with no `width` and no spot of its own wraps in.
pub const DEFAULT_WIDTH: f64 = 100.0;
/// The column a leadered note with no `width` wraps in.
pub const LEADER_WIDTH: f64 = 90.0;
/// The space kept between a note and what is beside it: more than the space between its own
/// lines, so two notes never read as one.
const CLEAR: f64 = 2.5;
/// The space kept between a note and the border.
const BORDER_CLEAR: f64 = 3.0;

impl Std {
    /// The baseline-to-baseline pitch of text `size` high: ISO 3098-1 lettering type B sets
    /// lines 17/10 of the height apart; ASME Y14.2 leaves at least half the height between
    /// lines.
    pub fn line_pitch(&self, size: f64) -> f64 {
        size * if self.iso { 1.7 } else { 1.5 }
    }
}

/// `text` in lines no wider than `width` (mm) at `size`: `\n` breaks a line, and each line
/// breaks at spaces. A word longer than the width breaks between characters.
pub fn wrap(text: &str, size: f64, width: Option<f64>) -> Vec<String> {
    let per_line = width.map(|w| ((w / (size * 0.6)).floor() as usize).max(1));
    let mut out = Vec::new();
    for para in text.split('\n') {
        let Some(max) = per_line else {
            out.push(para.trim_end().to_owned());
            continue;
        };
        let mut line = String::new();
        for word in para.split(' ').filter(|w| !w.is_empty()) {
            let mut word: Vec<char> = word.chars().collect();
            let used = line.chars().count();
            if used > 0 && used + 1 + word.len() <= max {
                line.push(' ');
                line.extend(word);
                continue;
            }
            if used > 0 {
                out.push(std::mem::take(&mut line));
            }
            while word.len() > max {
                out.push(word.drain(..max).collect());
            }
            line.extend(word);
        }
        out.push(line);
    }
    out
}

/// Note text as the sheet sets it: in capitals, with the `{property}` fields filled in
/// included (ASME Y14.2 letters a drawing in uppercase; ISO 3098 sheet notes follow the same
/// practice). A unit symbol written after a number keeps its case, since both standards keep
/// SI symbols as defined: `3 mm`, `25kg`, `⌀0.2 mm`.
pub fn sheet_case(text: &str) -> String {
    let units = runmat_cad::units::Units;
    let is_unit = |w: &str| {
        let w = w.trim_end_matches(|c: char| ".,;:)".contains(c));
        !w.is_empty() && w.chars().any(char::is_alphabetic) && units.unit(w).is_some()
    };
    let number_end = |w: &str| {
        let w = w.trim_start_matches(['+', '-', '±', '(', '⌀', 'Ø', '~']);
        let end = w
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '.'))
            .map_or(w.len(), |(i, _)| i);
        (end > 0 && w[..end].chars().any(|c| c.is_ascii_digit())).then(|| w.len() - end)
    };
    let mut out = String::with_capacity(text.len());
    let mut after_number = false;
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        for (j, word) in line.split(' ').enumerate() {
            if j > 0 {
                out.push(' ');
            }
            let attached = number_end(word).map(|rest| &word[word.len() - rest..]);
            if after_number && is_unit(word) {
                out.push_str(word);
            } else if let Some(unit) = attached.filter(|u| is_unit(u)) {
                out.push_str(&word[..word.len() - unit.len()]);
                out.push_str(unit);
            } else {
                out.push_str(&word.to_uppercase());
            }
            if !word.is_empty() {
                after_number = attached == Some("");
            }
        }
    }
    out
}

/// A note's `width` in sheet millimetres, when it has one.
pub fn width_of(kind: &Value) -> Option<f64> {
    let w = serde_json::from_value::<runmat_cad::value::ValueExpr>(kind["width"].clone()).ok()?;
    runmat_cad::units::length_mm(&w).filter(|w| *w > 0.0)
}

/// Lines of text set `size` high, `pitch` apart, from a top-left corner down.
#[derive(Clone, Debug)]
pub struct TextBlock {
    pub lines: Vec<String>,
    pub size: f64,
    pub pitch: f64,
}

impl TextBlock {
    pub fn new(st: &Std, text: &str, width: Option<f64>) -> TextBlock {
        TextBlock {
            lines: wrap(&sheet_case(text), st.text, width),
            size: st.text,
            pitch: st.line_pitch(st.text),
        }
    }

    pub fn size_wh(&self) -> Xy {
        let w = self
            .lines
            .iter()
            .map(|l| Std::text_width(l, self.size))
            .fold(0.0, f64::max);
        let h = self.size * 1.25 + (self.lines.len().max(1) - 1) as f64 * self.pitch;
        [w, h]
    }

    /// The block's box with its top-left corner at `at`.
    pub fn box_at(&self, at: Xy) -> BoxR {
        let [w, h] = self.size_wh();
        BoxR::new(at[0], at[1] - h, at[0] + w, at[1])
    }

    /// The lines as text primitives, the first line's cap height at `at`'s y.
    pub fn draw(&self, id: &str, at: Xy) -> Vec<Prim> {
        self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_empty())
            .map(|(i, l)| {
                mark(
                    id,
                    "note",
                    super::marks::text(
                        at[0],
                        at[1] - self.size - i as f64 * self.pitch,
                        l,
                        self.size,
                        "start",
                    ),
                )
            })
            .collect()
    }
}

/// The sheet's general notes: "NOTES:", then each note's number with its text beside it, wrapped
/// to the column with a hanging indent.
pub struct NotesList {
    items: Vec<(String, TextBlock)>,
    size: f64,
    pitch: f64,
    indent: f64,
    width: f64,
}

impl NotesList {
    pub const HEADING: &'static str = "NOTES:";

    pub fn new(st: &Std, rows: &[String], width: f64) -> NotesList {
        let number = |i: usize| format!("{}.", i + 1);
        let indent = Std::text_width(&format!("{} ", number(rows.len().max(1) - 1)), st.text);
        let items = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (number(i), TextBlock::new(st, r, Some(width - indent))))
            .collect();
        NotesList {
            items,
            size: st.text,
            pitch: st.line_pitch(st.text),
            indent,
            width,
        }
    }

    /// The column's width and the list's height: the heading, then each note's lines, a pitch
    /// apart (the numbers and the hanging indent set the notes apart).
    pub fn size_wh(&self) -> Xy {
        let lines: usize = self.items.iter().map(|(_, b)| b.lines.len()).sum();
        let h = self.size * 1.25 + lines as f64 * self.pitch;
        [self.width, h]
    }

    /// The list with its top-left corner at `at`.
    pub fn draw(&self, id: &str, at: Xy) -> Vec<Prim> {
        let p = |shape| super::marks::prim(id, "table", "Annotations", "note", shape);
        let mut out = vec![p(super::marks::text(
            at[0],
            at[1] - self.size,
            Self::HEADING,
            self.size,
            "start",
        ))];
        let mut top = at[1] - self.pitch;
        for (n, block) in &self.items {
            out.push(p(super::marks::text(
                at[0],
                top - self.size,
                n,
                self.size,
                "start",
            )));
            for mut q in block.draw(id, [at[0] + self.indent, top]) {
                q.kind = "table";
                out.push(q);
            }
            top -= block.lines.len() as f64 * self.pitch;
        }
        out
    }
}

/// What a box would cover on the sheet, named for a finding: the title block, a table, a view,
/// another note, or the sheet's edge.
fn what_is_under(ctx: &Ctx<'_>, b: &BoxR, owner: &str) -> String {
    if !b.inside(&ctx.occ.frame) {
        return "the sheet's border".into();
    }
    let g = b.grow(CLEAR);
    ctx.occ
        .items
        .iter()
        .find(|i| i.owner != owner && g.hits(&i.b))
        .map(|i| match (i.owner.as_str(), i.class) {
            ("title", _) => "the title block".into(),
            (o, _) if o.starts_with("table:") => format!("table {}", &o[6..]),
            (o, Class::View) if o.starts_with("view:") => format!("view {}", &o[5..]),
            (o, _) if o.starts_with("annotation:") => format!("{}", &o[11..]),
            _ => "the sheet's marks".into(),
        })
        .unwrap_or_else(|| "the centring marks".into())
}

/// Whether `b` is free for a note owned by `owner`, off the border and clear of the centring
/// marks.
fn free(ctx: &Ctx<'_>, b: &BoxR, owner: &str) -> bool {
    ctx.occ.free(b, owner, CLEAR, &[]) && clear_of_border(ctx, b)
}

/// Whether `b` keeps its distance from the border and the centring marks.
fn clear_of_border(ctx: &Ctx<'_>, b: &BoxR) -> bool {
    let marks = super::border::centring_marks(ctx.size, &ctx.frame);
    b.inside(&ctx.occ.frame.grow(-BORDER_CLEAR))
        && marks.iter().all(|(p, q)| !b.grow(0.5).crossed_by(*p, *q))
}

/// A note placed by hand with no `width`: wrapped at the room to the right of `at`, narrowing to
/// the nearest item in its way. None when it has no room at `at`.
fn wrap_in_room(ctx: &Ctx<'_>, owner: &str, at: Xy, t: &str) -> Option<TextBlock> {
    let st = &ctx.std;
    let mut w = ctx.occ.frame.x1 - at[0] - BORDER_CLEAR;
    for _ in 0..12 {
        if w < MIN_WIDTH.min(Std::text_width(t, st.text)) {
            return None;
        }
        let block = TextBlock::new(st, t, Some(w));
        let b = block.box_at(at);
        if free(ctx, &b, owner) {
            return Some(block);
        }
        // Narrow to the nearest item in the way that starts right of `at`.
        let g = b.grow(CLEAR);
        let next = ctx
            .occ
            .items
            .iter()
            .filter(|i| i.owner != owner && g.hits(&i.b) && i.b.x0 > at[0] + CLEAR)
            .map(|i| i.b.x0 - at[0] - 2.0 * CLEAR)
            .fold(f64::INFINITY, f64::min);
        if !next.is_finite() || next >= w - 1e-6 {
            return None;
        }
        w = next;
    }
    None
}

/// Where a plain note (no leader) goes and how it wraps: at `at` when given and free, else the
/// free spot nearest `near`. Raises `note_moved` or `note_no_room` when it can't stay put.
pub fn fit(
    ctx: &mut Ctx<'_>,
    owner: &str,
    aid: &str,
    at: Option<Xy>,
    near: Xy,
    t: &str,
    width: Option<f64>,
) -> (Xy, TextBlock) {
    let st = ctx.std.clone();
    if let Some(at) = at {
        let here = match width {
            Some(w) => Some(TextBlock::new(&st, t, Some(w))).filter(|b| free(ctx, &b.box_at(at), owner)),
            None => wrap_in_room(ctx, owner, at, t),
        };
        if let Some(block) = here {
            return (at, block);
        }
    }
    // Free space: the note's own width, else the default column, narrowing until it fits.
    let widths: Vec<f64> = match width {
        Some(w) => vec![w],
        None => [DEFAULT_WIDTH, 75.0, 50.0]
            .into_iter()
            .map(|w| w.min(ctx.occ.frame.w() - 2.0 * BORDER_CLEAR))
            .collect(),
    };
    for w in &widths {
        let block = TextBlock::new(&st, t, Some(*w));
        let [bw, bh] = block.size_wh();
        let c = [near[0] + bw / 2.0, near[1] - bh / 2.0];
        let ok = |b: &BoxR| clear_of_border(ctx, b);
        if let Some(b) = nearest_free(&ctx.occ, owner, [bw, bh], c, 2.0, CLEAR, &ok) {
            let spot = [b.x0, b.y1];
            if let Some(at) = at {
                let under = what_is_under(ctx, &TextBlock::new(&st, t, width.or(Some(ctx.occ.frame.x1 - at[0]))).box_at(at), owner);
                ctx.finding(
                    "note_moved",
                    aid,
                    format!(
                        "Note {aid} at [{:.0}, {:.0}] has no room there: it would run into {under}. It is drawn at [{:.0}, {:.0}], {} mm wide; set its `at` to a free spot or give it a `width`. Sheet-wide notes belong in the general notes: `table notes = notes(rows = [...])`",
                        at[0], at[1], spot[0], spot[1], runmat_cad::drawing::standard::metric_text(bw, 1)
                    ),
                );
            }
            return (spot, block);
        }
    }
    let block = TextBlock::new(&st, t, Some(*widths.last().unwrap_or(&DEFAULT_WIDTH)));
    let [bw, bh] = block.size_wh();
    ctx.finding(
        "note_no_room",
        aid,
        format!(
            "Note {aid} doesn't fit on the sheet: it needs {} x {} mm of free space and none is left, so it overlaps what is there. Shorten it, give it a narrower `width`, move sheet-wide notes into the general notes (`table notes = notes(rows = [...])`), or use a larger sheet",
            runmat_cad::drawing::standard::metric_text(bw, 0),
            runmat_cad::drawing::standard::metric_text(bh, 0)
        ),
    );
    (at.unwrap_or(near), block)
}

/// Draws a plain note's block at its spot and keeps the room it takes.
pub fn draw_block(ctx: &mut Ctx<'_>, id: &str, owner: &str, at: Xy, block: &TextBlock) {
    ctx.out.placed.insert(id.to_owned(), at);
    let prims = block.draw(id, at);
    ctx.occ.add_prims(&prims, id);
    ctx.occ.add(block.box_at(at), owner, Class::Text);
    ctx.push(prims);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_at_spaces_to_the_width_and_keeps_explicit_breaks() {
        // 3.5 mm text: 2.1 mm a character, so 42 mm holds 20 characters.
        let lines = wrap("BREAK SHARP EDGES; DEBURR ALL HOLES BOTH SIDES.", 3.5, Some(42.0));
        assert_eq!(lines, ["BREAK SHARP EDGES;", "DEBURR ALL HOLES", "BOTH SIDES."]);
        assert!(lines.iter().all(|l| Std::text_width(l, 3.5) <= 42.0 + 1e-9));
        assert_eq!(wrap("A\nB C", 3.5, None), ["A", "B C"]);
        assert_eq!(wrap("6X ⌀5.5 THRU\nM5 CLEARANCE", 3.5, Some(200.0)), ["6X ⌀5.5 THRU", "M5 CLEARANCE"]);
        // A word longer than the column breaks between characters.
        assert_eq!(wrap("ABCDEFGHIJ", 3.5, Some(10.5)), ["ABCDE", "FGHIJ"]);
    }
}

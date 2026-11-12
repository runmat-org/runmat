//! Where a sheet's notes (`text` statements) go when the lock records no place for them.
//!
//! - **Beside their subject.** A note whose name names a part or a child sheet on the page
//!   (`cb1`, `cb1_load`, `load_c1` for the sheet `c1`) goes right of that subject, else left of
//!   it, else below it, where it covers nothing already drawn.
//! - **In a notes block.** Every other note goes into one column inside the sheet, in source
//!   order: right of the drawing, or below it when that keeps the page closer to the sheet's
//!   proportions.
//!
//! Notes are never stacked above the drawing, so they never push the content over the frame.

use super::{text_box, text_width, Coord, Point, GRID, NOTE_TEXT};

/// A thing a note can be beside: its name (a designator or a child sheet's instance name,
/// lowercase) and everything it draws, `[x0, y0, x1, y1]` in nm.
pub(super) struct Subject {
    pub name: String,
    pub extent: [Coord; 4],
}

/// One note's place: its baseline start, in the page's content coordinates (nm, y up).
pub(super) type NotePlace = Point;

/// The line pitch of a column of notes.
const PITCH: Coord = 2 * GRID;

/// The subject a note's name points at: the subject named exactly, or as the name's first
/// or last `_` segment run.
fn subject_of<'a>(name: &str, subjects: &'a [Subject]) -> Option<&'a Subject> {
    let n = name.to_ascii_lowercase();
    subjects.iter().find(|s| {
        n == s.name
            || n.starts_with(&format!("{}_", s.name))
            || n.ends_with(&format!("_{}", s.name))
    })
}

fn overlaps(a: &[Coord; 4], b: &[Coord; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// Places the unplaced notes (`(name, text)` in source order). `obstacles` holds everything
/// already drawn on the page (bodies with their pins, texts); `bbox` is the drawing's extent.
/// Returns each note's place, in the order given.
pub(super) fn place(
    notes: &[(&str, &str)],
    subjects: &[Subject],
    obstacles: &[[Coord; 4]],
    bbox: [Coord; 4],
) -> Vec<NotePlace> {
    let mut taken: Vec<[Coord; 4]> = obstacles.to_vec();
    let mut out: Vec<Option<NotePlace>> = vec![None; notes.len()];
    // Notes already beside each subject, for stacking.
    let mut beside: Vec<usize> = vec![0; subjects.len()];
    let lift = NOTE_TEXT * 4 / 5;
    for (k, (name, text)) in notes.iter().enumerate() {
        let Some(s) = subject_of(name, subjects) else {
            continue;
        };
        let si = subjects
            .iter()
            .position(|x| std::ptr::eq(x, s))
            .unwrap_or(0);
        let e = s.extent;
        let w = text_width(text, NOTE_TEXT);
        let row = beside[si] as Coord * PITCH;
        let candidates = [
            Point::new(e[2] + GRID, e[3] - lift - row),
            Point::new(e[0] - GRID - w, e[3] - lift - row),
            Point::new(e[0], e[1] - GRID - lift - row),
        ];
        for at in candidates {
            let b = text_box(text, NOTE_TEXT, at.x, at.y, "start");
            if !taken.iter().any(|t| overlaps(t, &b)) {
                taken.push(b);
                out[k] = Some(at);
                beside[si] += 1;
                break;
            }
        }
    }
    // The notes block, right of or below everything placed so far.
    let rest: Vec<usize> = (0..notes.len()).filter(|&k| out[k].is_none()).collect();
    if !rest.is_empty() {
        let all = taken.iter().fold(bbox, |b, t| {
            [
                b[0].min(t[0]),
                b[1].min(t[1]),
                b[2].max(t[2]),
                b[3].max(t[3]),
            ]
        });
        let block_w = rest
            .iter()
            .map(|&k| text_width(notes[k].1, NOTE_TEXT))
            .max()
            .unwrap_or(0);
        let block_h = rest.len() as Coord * PITCH;
        let (w, h) = (all[2] - all[0], all[3] - all[1]);
        // A landscape sheet is sqrt(2) wide: the arrangement needing the smaller sheet wins.
        let need = |w: Coord, h: Coord| (w as f64 / std::f64::consts::SQRT_2).max(h as f64);
        let right = need(w + 4 * GRID + block_w, h.max(block_h));
        let below = need(w.max(block_w), h + 3 * GRID + block_h);
        let first = if right <= below {
            Point::new(all[2] + 4 * GRID, all[3] - lift)
        } else {
            Point::new(all[0], all[1] - 3 * GRID - lift)
        };
        for (row, &k) in rest.iter().enumerate() {
            out[k] = Some(Point::new(first.x, first.y - row as Coord * PITCH));
        }
    }
    out.into_iter().map(|p| p.unwrap_or_default()).collect()
}

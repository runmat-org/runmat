//! A drawing built by hand, command by command, from an empty sheet: every edit reads back,
//! and the finished drawing compiles clean with the values it shows.

use super::*;
use serde_json::json;

const BRACKET: &str = include_str!("../../../tests/corpus/documents/bracket.part");
const PART: &str = "/p/bracket.part";
const PATH: &str = "/p/bracket.drawing";

const EMPTY: &str = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [] }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n}\n";

struct Bench {
    files: Files,
    text: String,
}

impl Bench {
    fn new() -> Self {
        Bench {
            files: [(PART.to_owned(), BRACKET.to_owned())].into(),
            text: EMPTY.to_owned(),
        }
    }

    fn run(&mut self, command: &str, values: Value, selection: &[&str]) -> String {
        self.try_run(command, values, selection)
            .unwrap_or_else(|e| panic!("{command}: {e}\n{}", self.text))
    }

    fn try_run(
        &mut self,
        command: &str,
        values: Value,
        selection: &[&str],
    ) -> Result<String, EditFailure> {
        let values: Values = serde_json::from_value(values).unwrap();
        let selection: Vec<String> = selection.iter().map(|s| s.to_string()).collect();
        let files = self.files.clone();
        let (changes, summary) =
            drawing_edit(PATH, &self.text, &files, command, &values, &selection)
                .expect("a drawing command")?;
        for (path, old, change) in changes {
            // Each change is a small edit of the text it was made from.
            assert!(!change.reprinted, "{command} reprinted {path}");
            let mut applied = old.clone();
            for e in change.edits.iter().rev() {
                applied.replace_range(e.span.start..e.span.end, &e.text);
            }
            assert_eq!(
                applied, change.source,
                "{command}: the edits of {path} don't make its new text"
            );
            if path == PATH {
                self.text = change.source;
            } else {
                self.files.insert(path, change.source);
            }
        }
        assert!(
            Drawing::open(PATH, &self.text, &self.files).is_ok(),
            "{command} didn't read back:\n{}",
            self.text
        );
        Ok(summary)
    }

    fn document(&self) -> DrawingDocument {
        Drawing::open(PATH, &self.text, &self.files)
            .unwrap()
            .document
    }

    fn compiled(&self) -> Value {
        let mut files = self.files.clone();
        files.insert(PATH.to_owned(), self.text.clone());
        serde_json::from_str(&crate::drawing::compile_json(PATH, &self.text, &files)).unwrap()
    }
}

fn edge(view: &str, reference: &str) -> String {
    format!("edge:{view}|{reference}")
}

const LEFT: &str = "feature.body.face.side(s1.l4)";
const RIGHT: &str = "feature.body.face.side(s1.l2)";
const NEAR: &str = "feature.body.face.side(s1.l1)";
const FAR: &str = "feature.body.face.side(s1.l3)";
const HOLE: &str = "feature.h.face.wall(a)";
const TOP: &str = "feature.body.face.end";

#[test]
fn a_drawing_is_built_by_hand_from_an_empty_sheet() {
    let mut b = Bench::new();
    // Views.
    assert_eq!(
        b.run(
            "drawing.standard-3",
            json!({ "model": "bracket.part", "projection": "third" }),
            &[]
        ),
        "Placed views front, top and right"
    );
    let d = b.document();
    let (front, top, right) = (
        d.view("front").unwrap(),
        d.view("top").unwrap(),
        d.view("right").unwrap(),
    );
    assert!(
        top.at[1] > front.at[1] && (top.at[0] - front.at[0]).abs() < 1e-9,
        "third angle: top above front"
    );
    assert!(
        right.at[0] > front.at[0],
        "third angle: right right of front"
    );
    b.run(
        "drawing.section",
        json!({ "parent": ["view:top"], "through": [edge("top", HOLE)], "label": "a" }),
        &[],
    );
    b.run("drawing.detail", json!({ "parent": ["view:front"], "center": [edge("front", TOP)], "radius": 6, "label": "B", "scale": "4:1" }), &[]);
    b.run(
        "drawing.auxiliary",
        json!({ "parent": ["view:front"], "edge": [edge("front", RIGHT)], "label": "C" }),
        &[],
    );
    // Around the middle of the front view (its 80 × 10 mm outline at 1:1 sits on its `at`).
    let f = b.document().view("front").unwrap().at;
    let corner = |dx: f64, dy: f64| format!("point:{},{}", f[0] + dx, f[1] + dy);
    b.run("drawing.broken-out", json!({ "parent": ["view:front"], "boundary": [corner(-12.0, -4.0), corner(12.0, -4.0), corner(12.0, 4.0), corner(-12.0, 4.0)], "depth": 5 }), &[]);
    b.run(
        "drawing.isometric",
        json!({ "model": "bracket.part", "scale": "1:2", "at": ["point:340,220"] }),
        &[],
    );
    b.run(
        "drawing.model-view",
        json!({ "model": "bracket.part", "orientation": "bottom", "display": "shaded" }),
        &[],
    );
    b.run(
        "drawing.projected",
        json!({ "parent": ["view:front"], "direction": "down" }),
        &[],
    );
    let d = b.document();
    assert_eq!(d.views.len(), 10, "{}", b.text);
    assert!(
        b.text.contains("view a_a = section(") && b.text.contains("label = A"),
        "{}",
        b.text
    );
    assert!(
        b.text.contains("scale = [4, 1]") && b.text.contains("radius = 6 mm"),
        "{}",
        b.text
    );
    assert_eq!(d.view("isometric").unwrap().at, [340.0, 220.0]);
    // Dimensions.
    b.run("drawing.smart-dimension", json!({ "items": [edge("front", LEFT), edge("front", RIGHT)], "tolerance": "symmetric", "plus": 0.2 }), &[]);
    b.run(
        "drawing.smart-dimension",
        json!({ "items": [edge("top", HOLE)], "tolerance": "fit", "fit": "H7" }),
        &[],
    );
    b.run(
        "drawing.smart-dimension",
        json!({ "items": [edge("top", HOLE)], "measure": "radius" }),
        &[],
    );
    b.run(
        "drawing.smart-dimension",
        json!({ "items": [edge("front", LEFT), edge("front", TOP)] }),
        &[],
    );
    b.run(
        "drawing.ordinate",
        json!({ "origin": [edge("top", LEFT)], "items": [edge("top", HOLE), edge("top", RIGHT)] }),
        &[],
    );
    b.run(
        "drawing.baseline",
        json!({ "base": [edge("top", NEAR)], "items": [edge("top", HOLE), edge("top", FAR)] }),
        &[],
    );
    b.run(
        "drawing.chain",
        json!({ "items": [edge("top", LEFT), edge("top", HOLE), edge("top", RIGHT)] }),
        &[],
    );
    let part = &b.files[PART];
    assert!(
        part.contains("fit = \"H7\"") || part.contains("fit = H7"),
        "the fit is declared in the part:\n{part}"
    );
    // Callouts, frames and symbols; the datum is declared before the frame that cites it.
    assert!(
        b.try_run(
            "drawing.gdt",
            json!({ "frame": "position|⌀0.1|A", "attach": [edge("top", HOLE)] }),
            &[]
        )
        .is_err(),
        "a frame citing an undeclared datum"
    );
    b.run(
        "drawing.datum",
        json!({ "label": "a", "attach": [edge("front", NEAR)] }),
        &[],
    );
    b.run("drawing.gdt", json!({ "frame": "position|⌀0.1(M)|A", "attach": [edge("top", HOLE)], "at": ["point:150,200"] }), &[]);
    b.run(
        "drawing.hole-callout",
        json!({ "hole": [edge("top", HOLE)] }),
        &[],
    );
    b.run(
        "drawing.datum-target",
        json!({ "label": "A1", "shape": "circle", "size": 4, "attach": [edge("front", NEAR)] }),
        &[],
    );
    b.run(
        "drawing.surface-finish",
        json!({ "ra": 0.8, "attach": [edge("front", TOP)] }),
        &[],
    );
    b.run(
        "drawing.weld-symbol",
        json!({ "type": "fillet", "size": 3, "attach": [edge("front", RIGHT)] }),
        &[],
    );
    b.run(
        "drawing.note",
        json!({ "text": "BREAK SHARP EDGES {part_number}", "at": ["point:30,60"] }),
        &[],
    );
    b.run(
        "drawing.note",
        json!({ "text": "DEBURR", "attach": [edge("front", TOP)] }),
        &[],
    );
    assert_eq!(
        b.run(
            "drawing.center-mark",
            json!({ "auto": true, "view": ["view:top"] }),
            &[]
        ),
        "Added 1 center mark"
    );
    assert!(
        b.try_run(
            "drawing.center-mark",
            json!({ "edges": [edge("top", HOLE)] }),
            &[]
        )
        .is_err(),
        "marked already"
    );
    b.run(
        "drawing.centerline",
        json!({ "edges": [edge("front", LEFT), edge("front", RIGHT)] }),
        &[],
    );
    assert!(
        b.try_run(
            "drawing.balloon",
            json!({ "attach": [edge("front", LEFT)] }),
            &[]
        )
        .is_err(),
        "balloons label assembly components"
    );
    let part = &b.files[PART];
    for declared in [
        "datum A = { on = feature.body.face.side(s1.l1) }",
        "tolerance position1",
        "modifier = mmc",
        "datum = A,",
        "ra = 0.0008 mm",
        "size size1",
        "location distance1",
    ] {
        assert!(
            part.contains(declared),
            "the part's PMI has {declared}:\n{part}"
        );
    }
    // Tables.
    b.run(
        "table.hole",
        json!({ "origin": [edge("top", LEFT)], "faces": [edge("top", HOLE)] }),
        &[],
    );
    b.run(
        "table.revision",
        json!({ "rev": "A", "description": "Initial release", "date": "2026-11-06" }),
        &[],
    );
    assert_eq!(
        b.run(
            "table.revision",
            json!({ "rev": "B", "description": "Hole to H7" }),
            &[]
        ),
        "Added revision B to revisions"
    );
    assert!(b
        .try_run("table.revision", json!({ "rev": "B" }), &[])
        .is_err());
    assert!(
        b.try_run("table.bom", json!({ "view": ["view:front"] }), &[])
            .is_err(),
        "a BOM lists an assembly"
    );
    assert!(
        b.try_run("table.bend", json!({ "view": ["view:front"] }), &[])
            .is_err(),
        "a bend table lists a flat pattern"
    );
    // Sheets and the standard.
    b.run("sheet.add", json!({ "size": "a4" }), &[]);
    b.run("drawing.standard", json!({ "standard": "asme" }), &[]);
    assert_eq!(b.document().sheets.len(), 2);
    assert!(b.text.contains("standard = asme"), "{}", b.text);

    // The finished drawing compiles clean and shows what was measured. With
    // `RUNMAT_HAND_DRAWING_OUT` set to a folder, the drawing and its part are written there (the
    // sheet's pictures paste them).
    if let Ok(dir) = std::env::var("RUNMAT_HAND_DRAWING_OUT") {
        std::fs::write(format!("{dir}/hand.drawing"), &b.text).unwrap();
        std::fs::write(format!("{dir}/bracket.part"), &b.files[PART]).unwrap();
    }
    let out = b.compiled();
    assert_eq!(out["diagnostics"], json!([]), "{}", b.text);
    let text = &out["projection"]["data"]["text"];
    let show = |id: &str| text[id].as_str().unwrap_or_default().to_owned();
    assert_eq!(show("dim"), "80 ±0.2");
    assert_eq!(show("dia"), "⌀4 H7");
    assert_eq!(show("rad"), "R2");
    assert_eq!(show("angle"), "90°");
    assert_eq!(show("ordinate"), "60");
    assert_eq!(show("ordinate2"), "80");
    assert_eq!(show("baseline"), "10 · 50");
    assert_eq!(show("chain"), "60 · 20");
    assert_eq!(show("hole"), "⌀4 THRU");
    assert!(
        show("position1").starts_with("⌖|⌀0.1"),
        "{}",
        show("position1")
    );
    assert_eq!(show("finish"), "Ra 0.8");
    assert_eq!(show("note"), "BREAK SHARP EDGES BR-100");
    assert!(
        out["projection"]["data"]["hlr"]["a_a"]["hatch"]
            .as_array()
            .is_some_and(|h| !h.is_empty()),
        "the section cuts the model"
    );
    // A click on the front view's left line picks the side face it shows edge-on; the top
    // view's hole rim picks the hole's wall.
    let hlr = &out["projection"]["data"]["hlr"];
    let picked = |view: &str| -> Vec<String> {
        hlr[view]["picks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["reference"].as_str().unwrap().to_owned())
            .collect()
    };
    assert!(
        picked("front").iter().any(|r| r == LEFT),
        "{:?}",
        picked("front")
    );
    assert!(
        picked("top").iter().any(|r| r == HOLE),
        "{:?}",
        picked("top")
    );
    // The auxiliary view looks square to the right side: it draws what the right view draws.
    let extent = |view: &str| -> [f64; 2] {
        let pts: Vec<f64> = hlr[view]["visible"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|l| l.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let (xs, ys): (Vec<f64>, Vec<f64>) = (
            pts.iter().step_by(2).copied().collect(),
            pts.iter().skip(1).step_by(2).copied().collect(),
        );
        let w = |v: &[f64]| {
            v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
        };
        [
            (w(&xs) * 10.0).round() / 10.0,
            (w(&ys) * 10.0).round() / 10.0,
        ]
    };
    assert_eq!(extent("view_c"), extent("right"));
    // The broken-out view cuts inside its boundary.
    assert!(
        hlr["front_broken"]["hatch"]
            .as_array()
            .is_some_and(|h| !h.is_empty()),
        "{}",
        hlr["front_broken"]
    );
}

#[test]
fn items_move_align_relabel_and_delete_with_what_is_drawn_from_them() {
    let mut b = Bench::new();
    b.run(
        "drawing.standard-3",
        json!({ "model": "bracket.part" }),
        &[],
    );
    b.run(
        "drawing.section",
        json!({ "parent": ["view:top"], "through": [edge("top", HOLE)], "label": "A" }),
        &[],
    );
    b.run(
        "drawing.smart-dimension",
        json!({ "items": [edge("top", LEFT), edge("top", RIGHT)] }),
        &[],
    );
    b.run(
        "drawing.note",
        json!({ "text": "NOTE", "at": ["point:30,40"] }),
        &[],
    );
    b.run(
        "table.revision",
        json!({ "rev": "A", "description": "Initial release" }),
        &[],
    );
    // Move to a point, then by a distance.
    b.run(
        "drawing.move",
        json!({ "to": ["point:50,50"] }),
        &["annotation:note"],
    );
    b.run(
        "drawing.move",
        json!({ "dx": 10, "dy": -5 }),
        &["annotation:note", "table:revisions"],
    );
    let d = b.document();
    assert_eq!(
        d.annotations.iter().find(|a| a.id == "note").unwrap().at,
        [60.0, 45.0]
    );
    // A table the sheet placed moves from where the layout put it: the revision table's
    // top-right corner at the frame's (A3: 410, 287).
    assert_eq!(d.tables[0].at, [420.0, 282.0]);
    // An annotation the sheet placed takes the point it is moved to.
    b.run(
        "drawing.move",
        json!({ "to": ["point:120,210"] }),
        &["annotation:dim"],
    );
    assert_eq!(
        b.document()
            .annotations
            .iter()
            .find(|a| a.id == "dim")
            .unwrap()
            .at,
        [120.0, 210.0]
    );
    b.run(
        "drawing.align",
        json!({ "align": "horizontal" }),
        &["annotation:dim", "annotation:note"],
    );
    assert_eq!(
        b.document()
            .annotations
            .iter()
            .find(|a| a.id == "note")
            .unwrap()
            .at,
        [60.0, 210.0]
    );
    assert!(
        b.try_run("drawing.align", json!({}), &["annotation:dim"])
            .is_err(),
        "one item has nothing to align with"
    );
    b.run("drawing.view-label", json!({ "label": "d" }), &["view:a_a"]);
    assert!(b.text.contains("label = D"), "{}", b.text);
    assert!(b
        .try_run(
            "drawing.view-label",
            json!({ "label": "E" }),
            &["view:front"]
        )
        .is_err());
    // Deleting the top view takes its section and the dimension on it; the note stays.
    assert_eq!(
        b.run("drawing.delete", json!({}), &["view:top"]),
        "Deleted 2 views, 1 annotation"
    );
    let d = b.document();
    assert_eq!(
        d.views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
        ["front", "right"]
    );
    assert_eq!(d.annotations.len(), 1);
    assert_eq!(
        b.run(
            "drawing.delete",
            json!({}),
            &["annotation:note", "table:revisions"]
        ),
        "Deleted 1 annotation, 1 table"
    );
    assert!(b.try_run("drawing.delete", json!({}), &[]).is_err());
    assert_eq!(b.compiled()["diagnostics"], json!([]));
}

#[test]
fn sheet_picks_parse_as_the_desktop_names_them() {
    assert_eq!(
        parse_pick("view:front").unwrap(),
        SheetPick::View("front".into())
    );
    assert_eq!(
        parse_pick("point:12.5,-3").unwrap(),
        SheetPick::Point([12.5, -3.0])
    );
    let SheetPick::Edge { view, text, .. } = parse_pick(&edge("top", HOLE)).unwrap() else {
        panic!()
    };
    assert_eq!((view.as_deref(), text.as_str()), (Some("top"), HOLE));
    assert!(parse_pick("point:x,1").is_err());
    assert!(parse_pick("not a reference (").is_err());
}

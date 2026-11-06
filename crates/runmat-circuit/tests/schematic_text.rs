//! Notes on a schematic sheet: `text <name> = "…"` in the `.circuit` file, drawn at the place
//! the lock records (or above the drawing), placed with the Text tool and moved like a symbol.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use serde_json::{json, Value};

const PATH: &str = "/p/reg.circuit";
const LOCK: &str = "/p/reg.circuit.lock";

const BLANK: &str = r#"runmat 0.7
std 1

circuit reg {
    root = main

    sheet main {
        component U1 = { part = "LDO-3V3-SOT23" }
        component C1 = { part = "C0603-1U-50V" }
        connect = [U1.VIN, "C1.1"]
    }
}
"#;

#[derive(Clone, Debug)]
struct Project {
    circuit: String,
    lock: Option<String>,
}

impl Project {
    fn files(&self) -> Files {
        let mut f: Files = [(PATH.to_owned(), self.circuit.clone())].into_iter().collect();
        if let Some(l) = &self.lock {
            f.insert(LOCK.to_owned(), l.clone());
        }
        f
    }

    fn step(&self, command: &str, values: Value) -> Project {
        let request = EditRequest {
            command: command.into(),
            values: values.as_object().cloned().unwrap_or_default(),
            selection: Vec::new(),
            options: CompileOptions::default(),
        };
        let out = runtime::edit(PATH, &self.circuit, &self.files(), &request).unwrap_or_else(|e| panic!("{command}: {}", e.message));
        let mut next = self.clone();
        for f in &out.files {
            if f.path == PATH {
                next.circuit = f.text.clone();
            } else if f.path == LOCK {
                next.lock = Some(f.text.clone());
            }
        }
        next
    }

    fn texts(&self) -> Vec<(String, String, f64, f64)> {
        let out = runtime::compile(PATH, &self.circuit, &self.files(), &CompileOptions::default());
        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        out.index.expect("compiles").schematic.sheets[0].texts.iter().map(|t| (t.name.clone(), t.text.clone(), t.x_mm, t.y_mm)).collect()
    }
}

#[test]
fn a_note_is_placed_where_clicked_and_moves_like_a_symbol() {
    let p = Project { circuit: BLANK.to_owned(), lock: None }.step("schematic.annotate", json!({}));
    let p = p.step("schematic.text", json!({ "text": "5 V in from the connector", "at": "point:50.8,101.6" }));
    assert!(p.circuit.contains("        connect = [U1.VIN, \"C1.1\"]\n        text note1 = \"5 V in from the connector\"\n"), "{}", p.circuit);
    let texts = p.texts();
    assert_eq!(texts.len(), 1);
    let (name, words, x, y) = &texts[0];
    assert_eq!((name.as_str(), words.as_str()), ("note1", "5 V in from the connector"));
    assert!((x - 50.8).abs() <= 0.64 && (y - 101.6).abs() <= 0.64, "{x} {y}");
    assert!(p.lock.as_deref().unwrap().contains("place main/text:note1 "));

    // A second note takes the next name; a recompile keeps both where they are.
    let q = p.step("schematic.text", json!({ "text": "Output 3.3 V" }));
    assert!(q.circuit.contains("text note2 = \"Output 3.3 V\""));
    let both = q.texts();
    assert_eq!(both[0], texts[0]);
    assert_eq!(q.texts(), both);

    // Moving the note is a lock edit.
    let m = q.step("schematic.move", json!({ "items": ["text:main/note1"], "dx": 12.7, "dy": 0 }));
    assert_eq!(m.circuit, q.circuit);
    let moved = m.texts();
    assert!(((moved[0].2 - texts[0].2) - 12.7).abs() < 1e-6, "{moved:?}");
}

#[test]
fn a_text_statement_reads_back_and_a_malformed_one_is_refused() {
    let text = BLANK.replace("connect = [U1.VIN, \"C1.1\"]", "connect = [U1.VIN, \"C1.1\"]\n        text title = \"Regulator\"");
    let p = Project { circuit: text.clone(), lock: None };
    assert_eq!(p.texts()[0].1, "Regulator");
    let bad = BLANK.replace("connect = [U1.VIN, \"C1.1\"]", "connect = [U1.VIN, \"C1.1\"]\n        text title = 3");
    let out = runtime::compile(PATH, &bad, &[(PATH.to_owned(), bad.clone())].into_iter().collect(), &CompileOptions::default());
    assert!(out.diagnostics.iter().any(|d| d.message.contains("a text is")), "{:?}", out.diagnostics);
}

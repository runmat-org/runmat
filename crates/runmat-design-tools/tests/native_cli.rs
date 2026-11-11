//! Exercise the native command consumer, including canonical workspace paths and file output.
#![cfg(feature = "occt")]

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const PLATE: &str = r#"runmat 0.7
std 1
part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body, material = "material.aluminum_6061" }
}
"#;

const PAIR: &str = "runmat 0.7\nstd 1\nassembly pair {\n    use part plate as lower\n    use part plate as upper\n}\n";

const SHEET: &str = r#"runmat 0.7
std 1
drawing sheet {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [{ label = Sheet, source = { source = sheet_number } }]
    }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(
        orientation = front,
        sheet = "1",
        model = { part = plate },
        at = [130, 120],
        display = hidden_lines
    )
}
"#;

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "runmat design cli {}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        for (name, text) in [
            ("plate.part", PLATE),
            ("pair.assembly", PAIR),
            ("sheet.drawing", SHEET),
        ] {
            std::fs::write(root.join(name), text).unwrap();
        }
        Self(std::fs::canonicalize(root).unwrap())
    }

    fn invoke(&self, args: &[&str]) -> Value {
        let output = Command::new(env!("CARGO_BIN_EXE_runmat-design"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{args:?}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "{args:?}: {error}: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn native_check_resolves_mechanical_files_in_a_workspace_with_spaces() {
    let workspace = Workspace::new();
    let file = workspace.0.join("plate.part");
    let out = workspace.invoke(&[
        "check",
        file.to_str().unwrap(),
        "--root",
        workspace.0.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(out.as_array().unwrap().len(), 1, "{out}");
    assert_eq!(out[0]["ok"], true, "{out}");
}

#[test]
fn native_run_bom_uses_the_same_workspace_namespace() {
    let workspace = Workspace::new();
    let file = workspace.0.join("pair.assembly");
    let out = workspace.invoke(&[
        "run",
        file.to_str().unwrap(),
        "assembly.bom",
        "--root",
        workspace.0.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["kind"], "table", "{out}");
    assert_eq!(out["rows"].as_array().unwrap().len(), 1, "{out}");
    assert_eq!(out["rows"][0]["cells"][6], "2", "{out}");
    assert_eq!(std::fs::read_to_string(file).unwrap(), PAIR);
}

#[test]
fn native_run_publishes_the_canonical_single_file_export() {
    let workspace = Workspace::new();
    let file = workspace.0.join("sheet.drawing");
    let directory = workspace.0.join("exports with spaces").join("nested");
    let out = workspace.invoke(&[
        "run",
        file.to_str().unwrap(),
        "drawing.export",
        "--values",
        r#"{"format":"svg","sheets":"current","sheet":"1"}"#,
        "--root",
        workspace.0.to_str().unwrap(),
        "--out",
        directory.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(out["ok"], true, "{out}");
    let files = out["files"].as_array().unwrap();
    assert_eq!(files.len(), 1, "{out}");
    let saved =
        std::fs::read_to_string(directory.join(files[0]["name"].as_str().unwrap())).unwrap();
    assert_eq!(saved, files[0]["text"].as_str().unwrap());
    assert!(saved.contains("viewBox=\"0 0 420 297\""), "{saved}");
}

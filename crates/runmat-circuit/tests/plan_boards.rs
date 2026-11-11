//! Builds an example board from a plan file with the product's board commands, the way
//! `drone_boards.rs` builds the quadcopter's: annotate, stack-up, update the board from the
//! schematic, place, pour, route chosen nets first, fan out and autoroute the rest.
//!
//! `RUNMAT_PLAN_BOARD=<folder>/<name>` names the circuit (`<name>.circuit` in `<folder>`) and
//! `RUNMAT_PLAN=<file>` the plan; the layout and lock are written to `RUNMAT_PLAN_OUT` (default:
//! the board's folder). Without `RUNMAT_PLAN_BOARD` the test does nothing. Run it in release.
//!
//! Plan lines (`#` starts a comment; lengths in board millimetres from the board's center):
//!
//! ```text
//! outline x,y x,y ...              the board outline
//! limits clearance=0.15 track=0.15 ...   fabricator limits (written as `limits = { ... }`)
//! class <name> <track mm> <clearance mm>
//! layers <n>                       a standard 1.6 mm stack-up of n copper layers
//! place <schematic path> x y rot   moves a component (`/pwr/U1`, or `J1` on the root sheet)
//! flip <schematic path>            puts a component on the bottom side
//! pour <net> <layer> <connect> x,y x,y ...   a copper pour
//! route <net> [<net> ...]          autoroutes these nets now, before the rest
//! finish                           fans out the SMD parts and autoroutes what is left
//! ```

mod drone {
    pub mod build;
}

use drone::build::Build;
use serde_json::json;
use std::path::PathBuf;

fn points(words: &[&str]) -> Vec<[f64; 2]> {
    words
        .iter()
        .map(|w| {
            let (x, y) = w.split_once(',').unwrap_or_else(|| panic!("point {w}"));
            [x.parse().unwrap(), y.parse().unwrap()]
        })
        .collect()
}

#[test]
fn build_from_plan() {
    let Ok(board) = std::env::var("RUNMAT_PLAN_BOARD") else {
        return;
    };
    let board = PathBuf::from(board);
    let dir = board.parent().unwrap().to_path_buf();
    let name = board.file_name().unwrap().to_string_lossy().to_string();
    let plan = std::fs::read_to_string(std::env::var("RUNMAT_PLAN").expect("RUNMAT_PLAN")).unwrap();
    let lines: Vec<Vec<&str>> = plan
        .lines()
        .map(|l| {
            l.split('#')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let mut start = format!("runmat 0.7\nstd 1\n\nuse circuit {name}\n\nlayout {name} {{\n");
    for w in &lines {
        match w[0] {
            "outline" => {
                let pts: Vec<String> = points(&w[1..])
                    .iter()
                    .map(|p| format!("[{}, {}]", p[0], p[1]))
                    .collect();
                start += &format!("    outline = [{}]\n", pts.join(", "));
            }
            "limits" => {
                let kv: Vec<String> = w[1..].iter().map(|s| s.replace('=', " = ")).collect();
                start += &format!("    limits = {{ {} }}\n", kv.join(", "));
            }
            "class" => {
                start += &format!(
                    "    net_class {} = {{ track_width = {} mm, clearance = {} mm }}\n",
                    w[1], w[2], w[3]
                )
            }
            _ => {}
        }
    }
    start += "}\n";

    let mut b = Build::new(&dir, &name, &start);
    if let Ok(lock) = std::fs::read_to_string(dir.join(format!("{name}.circuit.lock"))) {
        b.files.insert(format!("/p/{name}.circuit.lock"), lock);
    }
    let layers = lines
        .iter()
        .find(|w| w[0] == "layers")
        .map(|w| w[1].parse().unwrap());
    b.start(layers);
    for w in &lines {
        match w[0] {
            "place" => b.place(
                w[1],
                w[2].parse().unwrap(),
                w[3].parse().unwrap(),
                w[4].parse().unwrap(),
            ),
            "flip" => b.flip(w[1]),
            "pour" => b.pour(w[1], w[2], &points(&w[4..]), w[3]),
            "route" => {
                let nets: Vec<String> = w[1..].iter().map(|n| format!("net:{n}")).collect();
                b.run("board.autoroute", json!({ "nets": nets }), &[]);
            }
            "finish" => b.finish(&[]),
            _ => {}
        }
    }
    if std::env::var("RUNMAT_PLAN_COURTYARDS").is_ok() {
        b.courtyards();
    }
    b.report();
    let out = std::env::var_os("RUNMAT_PLAN_OUT")
        .map(PathBuf::from)
        .unwrap_or(dir);
    b.write(&out, &name);
}

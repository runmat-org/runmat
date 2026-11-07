//! Route net, from a net's right-click menu in the Desktop: the session runs it on the circuit
//! with the board (the open file is the `.circuit`, the selection the net's Outline row), the
//! router takes that net's open connections, and the other nets keep theirs.

use runmat_circuit::runtime::{CompileOptions, EditRequest, Files, Session};
use std::collections::BTreeMap;

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(format!("{}/tests/projects/io/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            (format!("/p/{name}"), text)
        })
        .collect()
}

/// Each net's unrouted connections on the compiled board.
fn unrouted(session: &mut Session, files: &Files) -> BTreeMap<String, usize> {
    let out = session.compile(CIRCUIT, &files[CIRCUIT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let mut by_net = BTreeMap::new();
    for a in &session.index().expect("compiled").board.as_ref().expect("a board").unrouted {
        *by_net.entry(a.net.clone()).or_insert(0) += 1;
    }
    by_net
}

fn route_net(session: &mut Session, files: &Files, selection: &[&str]) -> Result<runmat_circuit::runtime::EditOutput, runmat_circuit::runtime::EditError> {
    let request = EditRequest {
        command: "board.route-net".into(),
        values: Default::default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    };
    session.edit(CIRCUIT, &files[CIRCUIT], files, &request)
}

#[test]
fn route_net_routes_the_selected_nets_open_connections_only() {
    let mut files = files();
    let mut session = Session::new();
    let before = unrouted(&mut session, &files);
    let (net, open) = before.iter().max_by_key(|(_, n)| **n).map(|(k, n)| (k.clone(), *n)).expect("the io board has airwires");
    let out = route_net(&mut session, &files, &[&format!("net:{net}")]).unwrap_or_else(|e| panic!("{}", e.message));
    assert!(out.summary.starts_with("Routed "), "{}", out.summary);
    assert_eq!(out.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), [LAYOUT]);
    for f in out.files {
        files.insert(f.path, f.text);
    }
    let after = unrouted(&mut session, &files);
    assert!(after.get(&net).copied().unwrap_or(0) < open, "{net}: {open} open before, {:?} after ({})", after.get(&net), out.summary);
    for (other, n) in &before {
        if *other != net {
            assert_eq!(after.get(other), Some(n), "{other} was not selected");
        }
    }
}

#[test]
fn route_net_without_a_net_says_so() {
    let files = files();
    let mut session = Session::new();
    unrouted(&mut session, &files);
    let e = route_net(&mut session, &files, &[]).expect_err("no net selected");
    assert_eq!(e.message, "select the net to route");
}

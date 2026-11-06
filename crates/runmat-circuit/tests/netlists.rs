//! The KiCad and EDIF netlists of the flight-controller board's circuit, read back as
//! s-expressions: every pad on a net appears once, and every EDIF port reference names a port
//! its cell declares. `RUNMAT_FAB_OUT=<dir>` also writes them under `netlist/`.

use runmat_circuit::kicad::sexpr::{parse, SExpr};
use runmat_circuit::runtime::{netlist_and_pool, CompileOptions, Files};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn load() -> (runmat_circuit::elab::Netlist, runmat_circuit::library::Pool) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone");
    let files: Files = [".circuit", ".circuit.layout", ".circuit.lock"]
        .iter()
        .filter_map(|x| {
            let name = format!("flight_controller{x}");
            Some((
                format!("/p/{name}"),
                std::fs::read_to_string(dir.join(&name)).ok()?,
            ))
        })
        .collect();
    let path = "/p/flight_controller.circuit";
    netlist_and_pool(path, &files[path], &files, &CompileOptions::default())
        .unwrap_or_else(|d| panic!("{:?}", d[0].message))
}

fn write(name: &str, text: &str) {
    if let Ok(dir) = std::env::var("RUNMAT_FAB_OUT") {
        let f = PathBuf::from(dir).join("netlist").join(name);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, text).unwrap();
    }
}

fn find<'a>(e: &'a SExpr, head: &str) -> Vec<&'a SExpr> {
    e.items()
        .iter()
        .filter(|x| x.head() == Some(head))
        .collect()
}

fn first_atom(e: &SExpr, head: &str) -> String {
    find(e, head)
        .first()
        .and_then(|x| x.items().get(1))
        .and_then(|x| x.atom())
        .unwrap_or("")
        .to_owned()
}

/// Every (designator, pad) on a named net of the fitted components.
fn expected(nl: &runmat_circuit::elab::Netlist) -> BTreeSet<(String, String, String)> {
    let mut out = BTreeSet::new();
    for n in &nl.nets {
        for &(ci, pi) in &n.pins {
            let c = &nl.components[ci];
            if c.fitted() {
                for pad in &c.pins[pi].pads {
                    out.insert((n.name.clone(), c.designator.clone(), pad.clone()));
                }
            }
        }
    }
    out
}

#[test]
fn kicad_netlist_reads_back_with_every_pad() {
    let (nl, pool) = load();
    let text = runmat_circuit::netlists::kicad(&nl, &pool);
    write("flight_controller.net", &text);
    let doc = parse(&text).expect("s-expression");
    assert_eq!(doc.head(), Some("export"));
    assert_eq!(first_atom(&doc, "version"), "E");
    let comps = find(&doc, "components")[0];
    let refs: BTreeSet<String> = find(comps, "comp")
        .iter()
        .map(|c| first_atom(c, "ref"))
        .collect();
    assert_eq!(
        refs.len(),
        nl.components.iter().filter(|c| c.fitted()).count()
    );
    let mut got = BTreeSet::new();
    for net in find(find(&doc, "nets")[0], "net") {
        let name = first_atom(net, "name");
        for node in find(net, "node") {
            assert!(
                got.insert((
                    name.clone(),
                    first_atom(node, "ref"),
                    first_atom(node, "pin")
                )),
                "a pad twice"
            );
        }
    }
    assert_eq!(got, expected(&nl));
}

#[test]
fn edif_reads_back_with_declared_ports() {
    let (nl, pool) = load();
    let text = runmat_circuit::netlists::edif(&nl, &pool);
    write("flight_controller.edn", &text);
    let doc = parse(&text).expect("s-expression");
    assert_eq!(doc.head(), Some("edif"));
    assert!(text.is_ascii(), "EDIF text is ASCII");
    let id = |e: &SExpr| -> String {
        match e.atom() {
            Some(a) => a.to_owned(),
            None => e.items()[1].atom().unwrap().to_owned(),
        }
    };
    let libs = find(&doc, "library");
    assert_eq!(libs.len(), 2);
    let mut ports: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for cell in find(libs[0], "cell") {
        let view = find(cell, "view")[0];
        let iface = find(view, "interface")[0];
        ports.insert(
            id(&cell.items()[1]),
            find(iface, "port")
                .iter()
                .map(|p| id(&p.items()[1]))
                .collect(),
        );
    }
    let top = find(libs[1], "cell")[0];
    let contents = find(find(top, "view")[0], "contents")[0];
    let mut cell_of = BTreeMap::new();
    for inst in find(contents, "instance") {
        let view_ref = find(inst, "viewRef")[0];
        let cell_ref = find(view_ref, "cellRef")[0];
        cell_of.insert(
            id(&inst.items()[1]),
            cell_ref.items()[1].atom().unwrap().to_owned(),
        );
    }
    let mut joined = 0;
    for net in find(contents, "net") {
        for pr in find(find(net, "joined")[0], "portRef") {
            let port = pr.items()[1].atom().unwrap();
            let inst = find(pr, "instanceRef")[0].items()[1].atom().unwrap();
            let cell = &cell_of[inst];
            assert!(
                ports[cell].contains(port),
                "{inst}: {port} not a port of {cell}"
            );
            joined += 1;
        }
    }
    assert_eq!(joined, expected(&nl).len());
}

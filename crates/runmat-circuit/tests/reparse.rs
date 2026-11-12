//! Re-reading a layout after a text change ([`reparse_board`]) gives the board a whole read
//! gives, and falls back to a whole read for changes outside item statements.

use runmat_circuit::layout::{Board, ItemId};
use runmat_circuit::syntax::board::{parse_board, reparse_board, same_board};

const PATH: &str = "/p/io.circuit.layout";

fn source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/projects/io/io.circuit.layout"
    ))
    .unwrap()
}

/// The `n`th line that starts with `word` (after indentation).
fn line(src: &str, word: &str, n: usize) -> (usize, usize) {
    let mut at = 0;
    let mut k = 0;
    for l in src.split_inclusive('\n') {
        if l.trim_start().starts_with(word) {
            if k == n {
                return (at, at + l.len());
            }
            k += 1;
        }
        at += l.len();
    }
    panic!("no {word} line {n}");
}

/// A board's items without their ids, in a fixed order.
fn items(b: &Board) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for f in b.footprints.values() {
        let mut f = f.clone();
        f.id = ItemId(0);
        out.push(format!("{f:?}"));
    }
    for t in b.tracks.values() {
        let mut t = t.clone();
        t.id = ItemId(0);
        out.push(format!("{t:?}"));
    }
    for v in b.vias.values() {
        let mut v = v.clone();
        v.id = ItemId(0);
        out.push(format!("{v:?}"));
    }
    for z in b.zones.values() {
        let mut z = z.clone();
        z.id = ItemId(0);
        z.fill.clear();
        out.push(format!("{z:?}"));
    }
    out.sort();
    out
}

fn check(old: &str, new: &str, expect_whole: bool) {
    let parsed = parse_board(PATH, old);
    assert!(parsed.diagnostics.is_empty());
    let (re, delta) = reparse_board(PATH, old, parsed, new);
    let full = parse_board(PATH, new);
    assert_eq!(delta.whole, expect_whole, "{delta:?}");
    assert_eq!(re.diagnostics.is_empty(), full.diagnostics.is_empty());
    if let (Some(a), Some(b)) = (&re.board, &full.board) {
        // A re-read keeps ids (new statements take fresh ones); a whole read numbers in
        // source order. The items are the same.
        assert_eq!(
            items(a),
            items(b),
            "the re-read items differ from a whole read"
        );
        let (mut a, mut b) = (a.clone(), b.clone());
        for x in [&mut a, &mut b] {
            x.footprints.clear();
            x.tracks.clear();
            x.vias.clear();
            x.zones.clear();
            x.next_id = 0;
        }
        assert!(
            same_board(&a, &b),
            "the re-read board differs from a whole read"
        );
    }
    // Every statement's span and site agree with a whole read.
    let spans = |p: &runmat_circuit::syntax::board::ParsedBoard| {
        p.statements
            .iter()
            .map(|s| (s.site.clone(), s.span, s.group))
            .collect::<Vec<_>>()
    };
    assert_eq!(spans(&re), spans(&full));
    assert_eq!(re.spans, full.spans);
    assert_eq!(re.block, full.block);
    // Sites by item agree once ids are matched by statement order.
    let sites = |p: &runmat_circuit::syntax::board::ParsedBoard| {
        p.statements
            .iter()
            .filter(|s| s.item.is_some())
            .map(|s| p.sites[&s.item.unwrap()].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(sites(&re), sites(&full));
}

#[test]
fn item_changes_re_read_in_place() {
    let old = source();
    // Change a track's width.
    let (s, e) = line(&old, "track", 3);
    let l = &old[s..e];
    let w = l.find("width = ").unwrap() + "width = ".len();
    let w_end = w + l[w..].find(',').unwrap();
    let changed = format!("{}0.333{}", &l[..w], &l[w_end..]);
    let new = format!("{}{}{}", &old[..s], changed, &old[e..]);
    check(&old, &new, false);
    // Insert a track before another one.
    let (s, _) = line(&old, "track", 5);
    let new = format!(
        "{}    track = {{ layer = \"F.Cu\", width = 0.2, from = [1, 1], to = [2, 2] }}\n{}",
        &old[..s],
        &old[s..]
    );
    check(&old, &new, false);
    // Delete a via and a track.
    let (s, e) = line(&old, "via", 1);
    let new = format!("{}{}", &old[..s], &old[e..]);
    check(&old, &new, false);
    let (s, e) = line(&new, "track", 0);
    let newer = format!("{}{}", &new[..s], &new[e..]);
    check(&new, &newer, false);
    // Move a footprint.
    let (s, e) = line(&old, "footprint", 2);
    let moved = old[s..e].replacen("at = [", "at = [1", 1);
    assert_ne!(moved, old[s..e]);
    let new = format!("{}{}{}", &old[..s], moved, &old[e..]);
    check(&old, &new, false);
}

#[test]
fn other_changes_read_whole() {
    let old = source();
    // A net declaration is outside item statements.
    let (s, _) = line(&old, "net ", 0);
    let new = format!("{}    net EXTRA_NET\n{}", &old[..s], &old[s..]);
    check(&old, &new, true);
    // An error reads whole and reports it.
    let (s, e) = line(&old, "track", 0);
    let broken = old[s..e].replacen("width = ", "width = ]", 1);
    let new = format!("{}{}{}", &old[..s], broken, &old[e..]);
    check(&old, &new, true);
}

#[test]
fn package_changes_re_read_their_footprints() {
    let old = source();
    // Widen the 0402 capacitor's pads: every C_0402 footprint re-reads, nothing else does.
    let head = "package C_0402_1005Metric_IPCN = {";
    let at = old.find(head).expect("the 0402 package");
    let end = at + old[at..].find("\n    }").unwrap();
    let block = old[at..end].replace("size = [0.515, 0.57]", "size = [0.6, 0.57]");
    let new = format!("{}{}{}", &old[..at], block, &old[end..]);
    check(&old, &new, false);
    let parsed = parse_board(PATH, &old);
    let users: Vec<ItemId> = {
        let b = parsed.board.as_ref().unwrap();
        b.footprints
            .values()
            .filter(|f| f.package.name == "C_0402_1005Metric_IPCN")
            .map(|f| f.id)
            .collect()
    };
    assert!(!users.is_empty());
    let (re, delta) = reparse_board(PATH, &old, parsed, &new);
    assert_eq!(
        delta.changed, users,
        "only the package's footprints re-read"
    );
    assert_eq!(delta.replaced.len(), users.len());
    assert!(delta.removed.is_empty());
    let b = re.board.as_ref().unwrap();
    for id in &users {
        let f = &b.footprints[id];
        assert!(
            f.component.is_some(),
            "{} keeps its schematic link",
            f.reference
        );
        assert_eq!(
            &f.package, &re.packages["C_0402_1005Metric_IPCN"],
            "{} has the new package",
            f.reference
        );
        assert!(f
            .package
            .pads
            .iter()
            .all(|p| p.shape.size().0 == runmat_circuit::geom::mm(0.6)));
    }
    // A text that grows moves every statement after it: a longer description.
    let new = old.replacen(
        "description = \"C chip 0402 (1005 metric), IPC-7351B density B\"",
        "description = \"C chip 0402 (1005 metric), IPC-7351B density B, edited\"",
        1,
    );
    check(&old, &new, false);
    // Renaming a package changes which footprints it serves: a whole read.
    let renamed = old.replacen(head, "package C_0402_Renamed = {", 1);
    check(&old, &renamed, true);
}

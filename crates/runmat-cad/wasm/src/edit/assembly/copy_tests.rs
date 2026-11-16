//! Copy with mates: the copy carries its connectors and mates; an outside mate end takes the
//! next picked new reference, or keeps the original one when no pick is left.
use super::frames::{Frame, FrameKind};
use super::tests::{compile, run, top, RIG};
use serde_json::json;

fn bottom(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] < -0.99
}

/// The block mated face to face onto the plate's top.
fn block_on_plate() -> String {
    let c = compile(RIG);
    run(
        RIG,
        json!({ "command": "assembly.mate", "values": {
            "type": "coincident", "first": [c.face("plate", top)], "second": [c.face("block", bottom)]
        } }),
    )
}

#[test]
fn a_copy_takes_its_mates_onto_the_new_reference() {
    let text = block_on_plate();
    let c = compile(&text);
    let out = run(
        &text,
        json!({ "command": "assembly.copy-with-mates", "values": {
            "components": [format!("occurrence:{}", c.uuid("block"))],
            "references": [c.face("pin", top)],
        } }),
    );
    assert!(out.contains("use part block as block2 {"), "{out}");
    assert!(out.contains("connector block2_c"), "{out}");
    assert!(
        out.contains("mate coincident2 = { kind = { mate = planar }"),
        "{out}"
    );
    assert!(out.contains("connector pin_c"), "{out}");
    let after = compile(&out);
    // block2's bottom sits on the pin's top (25 mm + 30 mm); the block stays on the plate.
    let t = after.transform("block2");
    assert!((t[11] - 0.055).abs() < 1e-6, "{t:?}\n{out}");
    let t = after.transform("block");
    assert!((t[11] - 0.010).abs() < 1e-6, "{t:?}\n{out}");
}

#[test]
fn without_a_new_reference_the_copy_keeps_the_original_one() {
    let text = block_on_plate();
    let c = compile(&text);
    let out = run(
        &text,
        json!({ "command": "assembly.copy-with-mates", "values": {
            "components": [format!("occurrence:{}", c.uuid("block"))],
        } }),
    );
    let after = compile(&out);
    let t = after.transform("block2");
    assert!((t[11] - 0.010).abs() < 1e-6, "{t:?}\n{out}");
    // Too many picks are refused.
    let refused: serde_json::Value = serde_json::from_str(&crate::edit::edit_json(
        "/p/rig.assembly",
        &text,
        &super::tests::files(),
        &json!({ "command": "assembly.copy-with-mates", "values": {
            "components": [format!("occurrence:{}", c.uuid("block"))],
            "references": [c.face("pin", top), c.face("pin2", top)],
        } })
        .to_string(),
    ))
    .unwrap();
    assert_eq!(refused["ok"], false, "{refused}");
}

#![cfg(target_arch = "wasm32")]

use runmat_wasm::init_runmat;
use serde_json::Value as JsonValue;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

#[wasm_bindgen_test(async)]
async fn apply_fea_study_document_operation_creates_and_summarizes_document() {
    let runtime = init_runmat(JsValue::NULL)
        .await
        .expect("initialize wasm runtime");

    let create_input = serde_wasm_bindgen::to_value(&serde_json::json!({
        "geometry_path": "bracket.step",
        "geometry_units": "millimeter",
        "model_profile": "linear_static_structural"
    }))
    .expect("serialize create input");
    let created: JsonValue = serde_wasm_bindgen::from_value(
        runtime
            .apply_fea_study_document_operation_js(
                "create".to_string(),
                "/project/bracket_static.fea".to_string(),
                None,
                create_input,
            )
            .expect("create study document"),
    )
    .expect("deserialize create result");

    assert_eq!(created["write"], true);
    assert_eq!(created["result"]["summary"]["studyId"], "bracket_static");
    assert_eq!(created["result"]["summary"]["geometryPath"], "bracket.step");
    assert_eq!(
        created["result"]["summary"]["modelProfile"],
        "linear_static_structural"
    );
    assert_eq!(created["result"]["summary"]["runKind"], "linear_static");
    assert_eq!(created["result"]["readiness"]["readyToSolve"], false);

    let source = created["source"]
        .as_str()
        .expect("created source string")
        .to_string();
    let mesh_input = serde_wasm_bindgen::to_value(&serde_json::json!({
        "path": "/project/bracket_static.fea",
        "profile": "balanced",
        "target_size": 0.002,
        "max_elements": 50000
    }))
    .expect("serialize mesh input");
    let meshed: JsonValue = serde_wasm_bindgen::from_value(
        runtime
            .apply_fea_study_document_operation_js(
                "set_mesh".to_string(),
                "/project/bracket_static.fea".to_string(),
                Some(source),
                mesh_input,
            )
            .expect("set mesh"),
    )
    .expect("deserialize mesh result");

    assert_eq!(meshed["write"], true);
    assert_eq!(meshed["result"]["diff"]["changedSections"][0], "mesh");
    assert!(meshed["source"]
        .as_str()
        .expect("mesh source")
        .contains("mesh:"));

    let source = meshed["source"]
        .as_str()
        .expect("meshed source string")
        .to_string();
    let outputs_input = serde_wasm_bindgen::to_value(&serde_json::json!({
        "path": "/project/bracket_static.fea",
        "outputs": [
            {
                "id": "displacement_view",
                "field": "structural.displacement",
                "location": "nodes",
                "kind": "vector"
            }
        ]
    }))
    .expect("serialize outputs input");
    let with_outputs: JsonValue = serde_wasm_bindgen::from_value(
        runtime
            .apply_fea_study_document_operation_js(
                "set_outputs".to_string(),
                "/project/bracket_static.fea".to_string(),
                Some(source),
                outputs_input,
            )
            .expect("set outputs"),
    )
    .expect("deserialize outputs result");

    assert_eq!(with_outputs["write"], true);
    assert_eq!(
        with_outputs["result"]["diff"]["changedSections"][0],
        "outputs"
    );
    assert_eq!(with_outputs["result"]["counts"]["outputs"], 1);
    assert_eq!(
        with_outputs["result"]["outputs"][0]["field"],
        "structural.displacement"
    );

    let source = with_outputs["source"]
        .as_str()
        .expect("outputs source string")
        .to_string();
    let summary_input = serde_wasm_bindgen::to_value(&serde_json::json!({
        "path": "/project/bracket_static.fea"
    }))
    .expect("serialize summary input");
    let summary: JsonValue = serde_wasm_bindgen::from_value(
        runtime
            .apply_fea_study_document_operation_js(
                "get_summary".to_string(),
                "/project/bracket_static.fea".to_string(),
                Some(source),
                summary_input,
            )
            .expect("summarize study document"),
    )
    .expect("deserialize summary result");

    assert_eq!(summary["write"], false);
    assert_eq!(summary["result"]["counts"]["regions"], 0);
    assert_eq!(
        summary["result"]["readiness"]["blockers"][0],
        "Define at least one material for the structural model."
    );
}

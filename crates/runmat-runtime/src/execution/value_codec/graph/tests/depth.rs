use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nested_cells_use_the_selected_portable_depth_not_a_host_stack_cap() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let mut value = Value::Num(7.0);
    for _ in 0..129 {
        value = Value::Cell(CellArray::new(vec![value], 1, 1).unwrap());
    }
    let limits = ValueLimits {
        max_depth: 192,
        ..ValueLimits::default()
    };
    let graph = encode_payload_graph_in_context(
        &[value],
        &runtime,
        &source_map(),
        &HashMap::new(),
        &HashMap::new(),
        limits,
    )
    .unwrap();
    let decoded = decode_payload_graph_in_context(
        &graph,
        &runtime,
        &source_map(),
        &HashMap::new(),
        &HashMap::new(),
        &bindings(),
        limits,
    )
    .unwrap();
    assert_eq!(decoded.values().len(), 1);
    assert_eq!(graph.nodes.len(), 130);
}

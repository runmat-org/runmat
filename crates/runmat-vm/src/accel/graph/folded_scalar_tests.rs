use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn compiled_single_literal_fold_matches_ordinary_interpreter_value() {
    let source = "x=single(0.1); x";
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode = crate::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    // A scalar cast alone has no MIR fusion window. Inspect the canonical
    // graph of the actual compiled instructions without granting eligibility.
    let graph = build_accel_graph(&bytecode.instructions, &bytecode.var_types);
    let fold = graph
        .nodes
        .iter()
        .find(|node| node.tags.contains(&AccelGraphTag::UnaryScalarFold))
        .expect("compiler folded exact scalar");
    let folded = graph
        .value(fold.outputs[0])
        .and_then(|info| info.constant.as_ref())
        .expect("typed folded value");
    let Value::Tensor(tensor) = folded else {
        panic!("compiler scalar fold lost native Single storage");
    };
    assert_eq!(tensor.shape(), [1, 1]);
    assert_eq!(
        tensor.as_f32_slice().unwrap()[0].to_bits(),
        (0.1f64 as f32).to_bits()
    );
    let output =
        futures::executor::block_on(crate::interpret(&bytecode)).expect("ordinary VM execution");
    assert_eq!(output.last(), Some(folded));
}

use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ordinary_compilation_and_execution_without_acceleration() {
    let ast = runmat_parser::parse("answer = 2 + 3;").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode = compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    let values = futures::executor::block_on(crate::interpret(&bytecode)).unwrap();
    let answer = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "answer").then_some(*slot))
        .expect("ordinary source export remains available");
    assert!(matches!(values.get(answer), Some(runmat_value::Value::Num(value)) if *value == 5.0));
}

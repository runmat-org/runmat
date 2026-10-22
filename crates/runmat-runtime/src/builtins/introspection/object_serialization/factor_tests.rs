use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use runmat_value::{Tensor, Value};
use std::rc::Rc;

async fn selected_recipe_roundtrip(kind: &str) {
    crate::builtins::wasm_registry::register_all();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let coefficients = match kind {
        "diagonal" => vec![4.0, 0.0, 0.0, 3.0],
        "triangular" => vec![4.0, 0.0, 1.0, 3.0],
        _ => vec![4.0, 1.0, 1.0, 3.0],
    };
    let matrix = Value::Tensor(Tensor::new(coefficients, vec![2, 2]).unwrap());
    let original = crate::call_builtin_async_in_context(
        &runtime,
        "decomposition",
        &[matrix, Value::String(kind.into())],
    )
    .await
    .expect("selected public decomposition constructor")
    .with_owned(|value| value);
    let Value::Object(original) = original else {
        panic!("constructor must publish an object");
    };
    let original_data = original.numeric_factor_data().cloned();
    assert_eq!(
        original_data.is_some(),
        !matches!(kind, "diagonal" | "triangular"),
        "factor-bearing and structural recipes have distinct DATA ownership"
    );
    let saved = prepare_mat_object_element(Value::Object(original.clone()), Some(&runtime))
        .await
        .expect("selected MAT property recipe")
        .with_owned(|value| value);
    assert!(
        matches!(saved, Value::Struct(_)),
        "MAT stores a property recipe"
    );
    let (class, product) = resolve_mat_object_class(
        runmat_types::standard::DECOMPOSITION.display_name(),
        Some(&runtime),
    )
    .await
    .expect("current canonical class product");
    let restored = restore_mat_object_element(&class, &product, false, saved, Some(&runtime))
        .await
        .expect("selected MAT import reconstructs once");
    let Value::Object(restored) = restored.value() else {
        panic!("MAT import must publish an object");
    };
    assert!(restored.is_class(runmat_types::standard::DECOMPOSITION));
    assert_eq!(restored.properties(), original.properties());
    assert_eq!(restored.numeric_factor_data(), original_data.as_ref());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn selected_mat_factor_and_structural_recipes_restore_under_current_native_owner() {
    for kind in ["lu", "chol", "qr", "cod", "diagonal", "triangular"] {
        futures::executor::block_on(selected_recipe_roundtrip(kind));
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn selected_mat_factor_and_structural_recipes_restore_under_current_browser_owner() {
    for kind in ["lu", "chol", "qr", "cod", "diagonal", "triangular"] {
        selected_recipe_roundtrip(kind).await;
    }
}

use futures::executor::block_on;
use runmat_value::Value;

use crate::builtins::geometry::geometry_load_builtin;

pub(super) fn load_stl_fixture(path: &std::path::Path, x: f64) -> Value {
    std::fs::write(
        path,
        format!(
            "solid part\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex {x} 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid part\n"
        ),
    )
    .expect("STL fixture should write");
    block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("geometry should load")
}

/// A closure whose bound function has a published portable origin, so FEA
/// generator constructors can freeze it through the portable value codec.
/// Keep the returned guard alive while the closure is used.
pub(crate) fn portable_closure_fixture(
    name: &str,
    function: usize,
    captures: Vec<Value>,
) -> (Value, crate::user_functions::SourceFunctionCatalogGuard) {
    let binding = crate::user_functions::test_support::exact_portable_binding(
        function,
        "fea.callable.fixture",
    );
    let origin = binding
        .authority
        .portable_origin()
        .expect("exact_portable_binding returns a portable origin");
    let guard =
        crate::user_functions::install_source_function_catalog(Some(std::sync::Arc::new(vec![
            crate::user_functions::SourceFunctionInfo {
                source_id: runmat_types::SourceId(function),
                name: name.to_string(),
                function,
                role: runmat_types::FunctionSourceRole::Anonymous,
                binding_origin: Some(origin),
                source_file: None,
                // No local definition owner: the origin resolves to a portable binding.
                definition_owners: None,
            },
        ])))
        .expect("portable closure catalog");
    let closure = Value::Closure(runmat_value::Closure::new(
        name.to_string(),
        Some(binding),
        runmat_types::FunctionSourceRole::Anonymous,
        None,
        captures,
        None,
    ));
    (closure, guard)
}

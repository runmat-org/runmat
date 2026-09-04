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
    block_on(geometry_load_builtin(vec![Value::String(
        path.to_string_lossy().to_string(),
    )]))
    .expect("geometry should load")
}

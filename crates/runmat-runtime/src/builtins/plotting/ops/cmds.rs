// Identity-local command implementations share the plotting state and property owners.
#[path = "cmds/error.rs"]
mod error;
#[path = "cmds/axis.rs"]
mod axis;
pub use axis::axis_builtin;
pub use runmat_builtins::{AXIS_DESCRIPTOR, AXIS_INTEGER_CAPABILITIES};
#[path = "cmds/box.rs"]
mod r#box;
pub use r#box::box_builtin;
pub use runmat_builtins::BOX_DESCRIPTOR;
#[path = "cmds/cla.rs"]
mod cla;
pub use cla::cla_builtin;
pub use runmat_builtins::CLA_DESCRIPTOR;
#[path = "cmds/colorbar.rs"]
mod colorbar;
pub use colorbar::colorbar_builtin;
pub use runmat_builtins::COLORBAR_DESCRIPTOR;
#[path = "cmds/colormap.rs"]
mod colormap;
pub use colormap::colormap_builtin;
pub use runmat_builtins::{
    COLORMAP_DESCRIPTOR, COLORMAP_EXTENSIONS, COLORMAP_INTEGER_CAPABILITIES,
};
#[path = "cmds/grid.rs"]
mod grid;
pub use grid::grid_builtin;
pub use runmat_builtins::GRID_DESCRIPTOR;
#[path = "cmds/hidden.rs"]
mod hidden;
pub use hidden::hidden_builtin;
pub use runmat_builtins::HIDDEN_DESCRIPTOR;
#[path = "cmds/shading.rs"]
mod shading;
pub use runmat_builtins::SHADING_DESCRIPTOR;
pub use shading::shading_builtin;

#[cfg(test)]
#[path = "cmds/test_support.rs"]
mod test_support;
#[cfg(test)]
#[path = "cmds/tests.rs"]
mod tests;

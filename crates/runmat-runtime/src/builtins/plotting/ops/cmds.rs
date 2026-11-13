// Identity-local command implementations share the plotting state and property owners.
#[path = "cmds/error.rs"]
mod error;
#[path = "cmds/axis.rs"]
pub(crate) mod axis;
#[cfg(test)]
pub use axis::axis_builtin;
#[cfg(test)]
pub use runmat_builtins::AXIS_DESCRIPTOR;
#[path = "cmds/box.rs"]
pub(crate) mod r#box;
#[cfg(test)]
pub use r#box::box_builtin;
#[cfg(test)]
pub use runmat_builtins::BOX_DESCRIPTOR;
#[path = "cmds/cla.rs"]
pub(crate) mod cla;
#[cfg(test)]
pub use cla::cla_builtin;
#[cfg(test)]
pub use runmat_builtins::CLA_DESCRIPTOR;
#[path = "cmds/colorbar.rs"]
pub(crate) mod colorbar;
#[cfg(test)]
pub use runmat_builtins::COLORBAR_DESCRIPTOR;
#[path = "cmds/colormap.rs"]
pub(crate) mod colormap;
#[cfg(test)]
pub use colormap::colormap_builtin;
#[cfg(test)]
pub use runmat_builtins::COLORMAP_DESCRIPTOR;
#[path = "cmds/grid.rs"]
pub(crate) mod grid;
#[cfg(test)]
pub use grid::grid_builtin;
#[cfg(test)]
pub use runmat_builtins::GRID_DESCRIPTOR;
#[path = "cmds/hidden.rs"]
pub(crate) mod hidden;
#[cfg(test)]
pub use hidden::hidden_builtin;
#[cfg(test)]
pub use runmat_builtins::HIDDEN_DESCRIPTOR;
#[path = "cmds/shading.rs"]
pub(crate) mod shading;
#[cfg(test)]
pub use runmat_builtins::SHADING_DESCRIPTOR;
#[cfg(test)]
pub use shading::shading_builtin;

#[cfg(test)]
#[path = "cmds/test_support.rs"]
mod test_support;
#[cfg(test)]
#[path = "cmds/tests.rs"]
mod tests;

//! Plotting builtins backed by the runmat-plot renderer.

mod common;
pub mod context;
mod contour;
mod contourf;
mod engine;
mod gpu_helpers;
mod point;
mod state;
mod style;
pub mod web;

mod bar;
mod figure;
mod hist;
mod hold;
mod mesh;
mod meshc;
mod plot;
mod scatter;
mod scatter3;
mod stairs;
mod subplot;
mod surf;
mod surfc;

pub use web::{install_web_renderer, web_renderer_ready};

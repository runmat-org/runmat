mod model;
mod walker;

pub use model::*;
pub use walker::{
    analyze_reachability, analyze_reachability_with_functions, function_dependencies,
    selected_body_reachability, selected_function_closure,
};

#[cfg(test)]
mod tests;

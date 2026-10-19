mod output;
mod plan;
mod rows;

#[cfg(test)]
mod tests;

pub(crate) use output::finish_owned as finish_binary_owned;
pub(crate) use plan::plan as plan_binary;
pub(crate) use plan::TabularBinaryPlan;

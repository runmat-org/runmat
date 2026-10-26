//! Preparation returns a compact retained transition, not a large owned
//! problem/options result transported through every enclosing poll frame.
use super::{control::Retained, inputs, options, ScalarCooperation};
use crate::BuiltinResult;

pub(super) struct Ready {
    pub problem: Option<inputs::Problem>,
    pub options: Option<options::Options>,
}
pub(super) type Prepared = Retained<Ready>;
impl Ready {
    pub fn new(
        problem: inputs::Problem,
        options: options::Options,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Prepared> {
        Retained::new(
            Self {
                problem: Some(problem),
                options: Some(options),
            },
            cooperation,
            super::control::POLICY,
        )
    }
    pub fn problem(&self) -> &inputs::Problem {
        self.problem.as_ref().expect("prepared original problem")
    }
    pub fn options(&self) -> &options::Options {
        self.options.as_ref().expect("prepared original options")
    }
    pub fn parts_mut(&mut self) -> (&mut inputs::Problem, &mut options::Options) {
        (
            self.problem.as_mut().expect("prepared original problem"),
            self.options.as_mut().expect("prepared original options"),
        )
    }
    pub fn take(&mut self) -> (inputs::Problem, options::Options) {
        (
            self.problem
                .take()
                .expect("prepared original problem moves once"),
            self.options
                .take()
                .expect("prepared original options move once"),
        )
    }
}
pub(super) async fn prepare(
    mut raw: inputs::raw::Raw,
    population: runmat_builtins::OptimGeneticPopulation,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    let prepared = options::Prepared::new(
        std::mem::replace(&mut raw.options, inputs::raw::empty()),
        options::Activity::from_raw(&raw, population),
        cooperation,
    )
    .await?;
    let problem = inputs::prepare(raw, population, cooperation).await?;
    let options = options::Options::resolve(prepared, &problem, cooperation).await?;
    Ready::new(problem, options, cooperation)
}

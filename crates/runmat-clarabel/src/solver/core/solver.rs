use self::internal::*;
use super::callbacks::{Callback, CallbackFcnFFI, ClarabelCallbackFn};
use super::cones::Cone;
use super::{traits::*, SettingsError};
use crate::algebra::*;
use crate::solver::core::callbacks::SolverCallbacks;
use crate::solver::core::ffi::*;
#[cfg(feature = "serde")]
use crate::solver::SolverError;
use crate::timers::*;
use std::io::Write;

// ---------------------------------
// Solver status type
// ---------------------------------

/// Status of solver at termination
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Default)]
pub enum SolverStatus {
    /// Problem is not solved (solver hasn't run).
    #[default]
    Unsolved = 0,
    /// Solver terminated with a solution.
    Solved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
    PrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
    DualInfeasible,
    /// Solver terminated with a solution (reduced accuracy)
    AlmostSolved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility (reduced accuracy).
    AlmostPrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility (reduced accuracy).
    AlmostDualInfeasible,
    /// Iteration limit reached before solution or infeasibility certificate found.
    MaxIterations,
    /// Time limit reached before solution or infeasibility certificate found.
    MaxTime,
    /// Solver terminated with a numerical error
    NumericalError,
    /// Solver terminated due to lack of progress.
    InsufficientProgress,
    /// Solver terminated by user callback
    CallbackTerminated,
}

impl SolverStatus {
    pub(crate) fn is_infeasible(&self) -> bool {
        matches!(
            *self,
            |SolverStatus::PrimalInfeasible| SolverStatus::DualInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::AlmostDualInfeasible
        )
    }

    pub(crate) fn is_errored(&self) -> bool {
        // status is any of the error codes
        matches!(
            *self,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        )
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum StepDirection {
    Affine,
    Combined,
}

/// Scaling strategy used by the solver when
/// linearizing centrality conditions.
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum ScalingStrategy {
    PrimalDual,
    Dual,
}

/// An enum for reporting strategy checkpointing
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
enum StrategyCheckpoint {
    Update(ScalingStrategy), // Checkpoint is suggesting a new ScalingStrategy
    NoUpdate,                // Checkpoint recommends no change to ScalingStrategy
    Fail,                    // Checkpoint found a problem but no more ScalingStrategies to try
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// JSON file read/write trait for solver data.
/// Only available with the "serde" feature enabled.
#[cfg(feature = "serde")]
pub trait SolverJSONReadWrite<T>: Sized
where
    T: FloatT,
{
    /// write internal problem data to a JSON file
    fn save_to_file(&self, file: &mut std::fs::File) -> Result<(), std::io::Error>;
    /// load problem data from a JSON file previously saved using [`save_to_file`](self::SolverJSONReadWrite::save_to_file)
    fn load_from_file(
        file: &mut std::fs::File,
        settings: Option<crate::solver::DefaultSettings<T>>,
    ) -> Result<Self, SolverError>;
}

// ---------------------------------
// top level solver container type
// ---------------------------------

// The top-level solver.

// This trait is defined with a collection of mutually interacting associated types.
// See the [`DefaultSolver`](crate::solver::implementations::default) for an example.

pub struct Solver<T, D, V, R, K, C, I, SO, SE>
where
    I: ClarabelFFI<I>,
    SE: Settings<T>,
    T: FloatT,
{
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub prev_vars: V,
    pub info: I,
    pub solution: SO,
    pub(crate) settings: SE, // not public to avoid unchecked modifications
    pub timers: Option<Timers>,
    pub(crate) callbacks: SolverCallbacks<I, I::FFI>,
    pub(crate) phantom: std::marker::PhantomData<T>,
}

fn _print_banner(out: &mut dyn Write, is_verbose: bool) -> std::io::Result<()> {
    if !is_verbose {
        return std::io::Result::Ok(());
    }

    writeln!(
        out,
        "-------------------------------------------------------------"
    )?;
    writeln!(
        out,
        "           Clarabel.rs v{}  -  Clever Acronym                ",
        crate::VERSION
    )?;
    #[cfg(debug_assertions)]
    writeln!(
        out,
        "                  *** debug build ***                        ",
    )?;
    #[cfg(not(debug_assertions))]
    writeln!(out)?;
    writeln!(
        out,
        "                   (c) Paul Goulart                          "
    )?;
    writeln!(
        out,
        "                University of Oxford, 2022                   "
    )?;
    writeln!(
        out,
        "-------------------------------------------------------------"
    )?;
    std::io::Result::Ok(())
}

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SE: Settings<T>,
    T: FloatT,
{
    /// Create a new solver object
    pub fn set_termination_callback(&mut self, callback: impl ClarabelCallbackFn<I> + 'static) {
        self.callbacks.termination_callback = Callback::Rust(Box::new(callback));
    }

    pub fn set_termination_callback_c(
        &mut self,
        callback: CallbackFcnFFI<I::FFI>,
        data_ptr: *mut std::ffi::c_void,
    ) {
        self.callbacks.termination_callback = Callback::new_c(callback, data_ptr);
    }

    pub fn unset_termination_callback(&mut self) {
        self.callbacks.termination_callback = Callback::None;
    }

    pub fn settings(&self) -> &SE {
        &self.settings
    }

    pub fn update_settings(&mut self, settings: SE) -> Result<(), SettingsError> {
        settings.validate_as_update(&self.settings)?;
        self.settings = settings;
        Ok(())
    }
}

// RunMat extension to upstream Clarabel.rs 0.11.1: one numerical source
// body for both the original synchronous API and owned cooperative execution.
/// Normal control of the owned cooperative solve, distinct from callback errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressControl {
    /// Advance the existing numerical solve from its current state.
    Continue,
    /// Finalize the current solve with the supplied normal stop reason.
    Stop(ProgressStop),
}
/// A cooperative observer's normal stop reason, independent of callback failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressStop {
    /// The invocation's selected timing authority reached its deadline.
    TimeLimit,
    /// The observer requested termination without a callback error.
    Requested,
}
impl From<()> for ProgressControl {
    fn from(_: ()) -> Self {
        Self::Continue
    }
}

macro_rules! solve_body {
    ($solver:ident $(; $progress:ident, $context:ident)?) => {{

        // various initializations
        let mut iter: u32 = 0;
        let mut σ = T::one();
        let mut α = T::zero();
        let mut μ;

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = $solver.timers.take().unwrap();

        // solver release info, solver config
        // problem dimensions, cone types etc
        notimeit! {timers; {
            _print_banner($solver.info.print_target(), $solver.settings.core().verbose).unwrap();
            $solver.info.print_configuration(&$solver.settings, &$solver.data, &$solver.cones).unwrap();
            $solver.info.print_status_header(&$solver.settings).unwrap();
        }}

        $solver.info.reset(&mut timers);

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            $solver.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        let mut scaling = {
            if $solver.cones.allows_primal_dual_scaling() {ScalingStrategy::PrimalDual}
            else {ScalingStrategy::Dual}
        };

        loop {

            //update the residuals
            //--------------
            $solver.residuals.update(&$solver.variables, &$solver.data);

            //calculate duality gap (scaled)
            //--------------
            μ = $solver.variables.calc_mu(&$solver.residuals, &$solver.cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            $solver.info.save_scalars(μ, α, σ, iter);

            // convergence check and printing
            // --------------
            $solver.info.update(
                &mut $solver.data,
                &$solver.variables,
                &$solver.residuals,&timers);

            // RunMat cooperative extension: the original solve body is shared.
            // A cancelled/erroring owned async solve drops its entire solver.
            $(
                timers.suspend();
                let directive = $progress($context, &$solver.info).await?;
                timers.resume();
                match Into::<ProgressControl>::into(directive) {
                    ProgressControl::Continue => {},
                    ProgressControl::Stop(reason) => {
                        $solver.info.set_status(match reason {
                            ProgressStop::TimeLimit => SolverStatus::MaxTime,
                            ProgressStop::Requested => SolverStatus::CallbackTerminated,
                        });
                        break;
                    }
                }
            )?


            notimeit!{timers; {
                $solver.info.print_status(&$solver.settings).unwrap();
            }}

            // termination checks
            // --------------

            // user defined termination checks
            if $solver.callbacks.check_termination(&$solver.info) {
                $solver.info.set_status(SolverStatus::CallbackTerminated);
                break;
            }
            // internal termination checks
            let is_done = $solver.info.check_termination(&$solver.residuals, &$solver.settings, iter);

            // check for termination due to slow progress and update strategy
            if is_done{
                    match $solver.strategy_checkpoint_insufficient_progress(scaling){
                        StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                        StrategyCheckpoint::Update(s) => {scaling = s; continue}
                    }
            }  // allows continuation if new strategy provided


            // update the scalings
            // --------------
            let is_scaling_success;
            timeit!{timers => "scale cones"; {
                is_scaling_success = $solver.variables.scale_cones(&mut $solver.cones,μ,scaling);
            }}
            // check whether variables are interior points
            match $solver.strategy_checkpoint_is_scaling_success(is_scaling_success,scaling){
                StrategyCheckpoint::Fail => {break}
                StrategyCheckpoint::NoUpdate => {} // we only expect NoUpdate or Fail here
                StrategyCheckpoint::Update(_) => {unreachable!()}
            }

            //increment counter here because we only count
            //iterations that produce a KKT update
            iter += 1;

            // Update the KKT system and the constant parts of its solution.
            // Keep track of the success of each step that calls KKT
            // --------------
            //PJG: This should be a Result in Rust, but needs changes down
            //into the KKT solvers to do that.
            let mut is_kkt_solve_success : bool;
            timeit!{timers => "kkt update"; {
                is_kkt_solve_success = $solver.kktsystem.update(&$solver.data, &$solver.cones, &$solver.settings);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            $solver.step_rhs
                .affine_step_rhs(&$solver.residuals, &$solver.variables, &$solver.cones);

            timeit!{timers => "kkt solve"; {
                is_kkt_solve_success = is_kkt_solve_success &&
                $solver.kktsystem.solve(
                    &mut $solver.step_lhs,
                    &$solver.step_rhs,
                    &$solver.data,
                    &$solver.variables,
                    &mut $solver.cones,
                    StepDirection::Affine,
                    &$solver.settings,
                );
            }}  //end "kkt solve affine" timer

            // combined step only on affine step success
            if is_kkt_solve_success {

                //calculate step length and centering parameter
                // --------------
                α = $solver.get_step_length(StepDirection::Affine, scaling);
                σ = $solver.centering_parameter(α);

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                let m = if iter > 1 {T::one()} else {α};

                // calculate the combined step and length
                // --------------
                $solver.step_rhs.combined_step_rhs(
                    &$solver.residuals,
                    &$solver.variables,
                    &mut $solver.cones,
                    &mut $solver.step_lhs,
                    σ,
                    μ,
                    m
                );

                timeit!{timers => "kkt solve" ; {
                    is_kkt_solve_success =
                    $solver.kktsystem.solve(
                        &mut $solver.step_lhs,
                        &$solver.step_rhs,
                        &$solver.data,
                        &$solver.variables,
                        &mut $solver.cones,
                        StepDirection::Combined,
                        &$solver.settings,
                    );
                }} //end "kkt solve"
            }

            // check for numerical failure and update strategy
            match $solver.strategy_checkpoint_numerical_error(is_kkt_solve_success,scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }


            // compute final step length and update the current iterate
            // --------------
            α = $solver.get_step_length(StepDirection::Combined,scaling);

            // check for undersized step and update strategy
            match $solver.strategy_checkpoint_small_step(α, scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }

            // Copy previous iterate in case the next one is a dud
            $solver.info.save_prev_iterate(&$solver.variables,&mut $solver.prev_vars);

            $solver.variables.add_step(&$solver.step_lhs, α);

        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        // Check we if actually took a final step.  If not, we need
        // to recapture the scalars and print one last line
        if α.is_zero() {
            $solver.info.save_scalars(μ, α, σ, iter);
            notimeit! {timers; {$solver.info.print_status(&$solver.settings).unwrap();}}
        }

        timeit! {timers => "post-process"; {
            //check for "almost" convergence case and then extract solution
            $solver.info.post_process(&$solver.residuals, &$solver.settings);
            $solver.solution
                .post_process(&$solver.data, &mut $solver.variables, &$solver.info, &$solver.settings);
        }}

        //halt timers
        $solver.info.finalize(&mut timers);
        $solver.solution.finalize(&$solver.info);

        $solver.info.print_footer(&$solver.settings).unwrap();

        //stow the timers back into Option in the solver struct
        $solver.timers.replace(timers);
    }};
}

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

/// An interior point solver implementing a predictor-corrector scheme
//
// Only the main solver function lives in IPSolver, since this is the
// only publicly facing trait we want to give the solver.   Additional
// internal functionality for the top level solver object is implemented
// for the IPSolverUtilities trait below, upon which IPSolver depends
pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    /// Run the solver
    fn solve(&mut self);
}

impl<T, D, V, R, K, C, I, SO, SE> IPSolver<T, D, V, R, K, C, I, SO, SE>
    for Solver<T, D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: Cone<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I, SE = SE>,
    SE: Settings<T>,
{
    fn solve(&mut self) {
        solve_body!(self);
    }
}

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: Cone<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I, SE = SE>,
    SE: Settings<T>,
{
    /// Solve with an invocation-owned asynchronous progress checkpoint.
    ///
    /// The original numerical body and synchronous API are unchanged. The
    /// checkpoint runs after each residual/info update, before termination
    /// checks or the next KKT iteration, with scheduling time excluded from
    /// solver timers. It may borrow its context across the await without
    /// allocating a boxed callback future. Its error is returned unchanged.
    ///
    /// This method consumes the solver: cancelling the future or returning a
    /// checkpoint error drops the same solver and workspaces once, rather than
    /// exposing a partially progressed borrowed solver for accidental reuse.
    pub async fn solve_with_progress<X, E, F, P>(
        mut self,
        context: &mut X,
        mut progress: F,
    ) -> Result<Self, E>
    where
        F: for<'a> std::ops::AsyncFnMut(&'a mut X, &'a I) -> Result<P, E>,
        P: Into<ProgressControl>,
    {
        solve_body!(self; progress, context);
        Ok(self)
    }
}

// Encapsulate the internal helpers trait in a private module
// so it doesn't get exported
mod internal {
    use super::super::traits::*;
    use super::*;

    pub(super) trait IPSolverInternals<T, D, V, R, K, C, I, SO, SE> {
        /// Find an initial condition
        fn default_start(&mut self);

        /// Compute a centering parameter
        fn centering_parameter(&self, α: T) -> T;

        /// Compute the current step length
        fn get_step_length(&mut self, step_direction: StepDirection, scaling: ScalingStrategy)
            -> T;

        /// backtrack a step direction to the barrier
        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T;

        /// Scaling strategy checkpointing functions
        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;
    }

    impl<T, D, V, R, K, C, I, SO, SE> IPSolverInternals<T, D, V, R, K, C, I, SO, SE>
        for Solver<T, D, V, R, K, C, I, SO, SE>
    where
        T: FloatT,
        D: ProblemData<T, V = V>,
        V: Variables<T, D = D, R = R, C = C, SE = SE>,
        R: Residuals<T, D = D, V = V>,
        K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
        C: Cone<T>,
        I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
        SO: Solution<T, D = D, V = V, I = I>,
        SE: Settings<T>,
    {
        fn default_start(&mut self) {
            if self.cones.is_symmetric() {
                // set all scalings to identity (or zero for the zero cone)
                self.cones.set_identity_scaling();
                // Refactor
                self.kktsystem
                    .update(&self.data, &self.cones, &self.settings);
                // solve for primal/dual initial points via KKT
                self.kktsystem
                    .solve_initial_point(&mut self.variables, &self.data, &self.settings);
                // fix up (z,s) so that they are in the cone
                self.variables.symmetric_initialization(&mut self.cones);
            } else {
                // Assigns unit (z,s) and zeros the primal variables
                self.variables.unit_initialization(&self.cones);
            }
        }

        fn centering_parameter(&self, α: T) -> T {
            T::powi(T::one() - α, 3)
        }

        fn get_step_length(
            &mut self,
            step_direction: StepDirection,
            scaling: ScalingStrategy,
        ) -> T {
            //step length to stay within the cones
            let mut α = self.variables.calc_step_length(
                &self.step_lhs,
                &mut self.cones,
                &self.settings,
                step_direction,
            );

            // additional barrier function limits for asymmetric cones
            if !self.cones.is_symmetric()
                && step_direction == StepDirection::Combined
                && scaling == ScalingStrategy::Dual
            {
                let αinit = α;
                α = self.backtrack_step_to_barrier(αinit);
            }
            α
        }

        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T {
            let step = self.settings.core().linesearch_backtrack_step;
            let mut α = αinit;

            for _ in 0..50 {
                let barrier = self.variables.barrier(&self.step_lhs, α, &mut self.cones);
                if barrier < T::one() {
                    return α;
                } else {
                    α = step * α;
                }
            }
            α
        }

        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            if self.info.get_status() != SolverStatus::InsufficientProgress {
                // there is no problem, so nothing to do
                output = StrategyCheckpoint::NoUpdate;
            } else {
                // recover old iterate since "insufficient progress" often
                // involves actual degradation of results
                self.info
                    .reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

                // If problem is asymmetric, we can try to continue with the dual-only strategy
                if !self.cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                    self.info.set_status(SolverStatus::Unsolved);
                    output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
                } else {
                    output = StrategyCheckpoint::Fail;
                }
            }
            output
        }

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            // No update if kkt updates successfully
            if is_kkt_solve_success {
                output = StrategyCheckpoint::NoUpdate;
            }
            // If problem is asymmetric, we can try to continue with the dual-only strategy
            else if !self.cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else {
                // out of tricks.  Bail out with an error
                self.info.set_status(SolverStatus::NumericalError);
                output = StrategyCheckpoint::Fail;
            }
            output
        }

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;

            if !self.cones.is_symmetric()
                && scaling == ScalingStrategy::PrimalDual
                && α < self.settings.core().min_switch_step_length
            {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else if α <= T::max(T::zero(), self.settings.core().min_terminate_step_length) {
                self.info.set_status(SolverStatus::InsufficientProgress);
                output = StrategyCheckpoint::Fail;
            } else {
                output = StrategyCheckpoint::NoUpdate;
            }

            output
        }

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            _scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            if is_scaling_success {
                StrategyCheckpoint::NoUpdate
            } else {
                self.info.set_status(SolverStatus::NumericalError);
                StrategyCheckpoint::Fail
            }
        }
    } // end trait impl
} //end internals module

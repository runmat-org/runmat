//! Closed identities for the ordinary optimization progress charts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptimPlotKind {
    CurrentPoint,
    AnnealBestPoint,
    AnnealCurrentPoint,
    AnnealBestFunction,
    AnnealCurrentFunction,
    AnnealStopping,
    AnnealTemperature,
    PatternBestFunction,
    PatternFunctionCount,
    PatternMeshSize,
    PatternBestPoint,
    PatternMaxConstraint,
    ParticleBestFunctionValue,
    FunctionCount,
    FunctionValue,
    ConstraintViolation,
    StepSize,
    FirstOrderOptimality,
    ConstrainedFunctionValue,
    SurrogateProgress,
}
impl OptimPlotKind {
    pub const fn is_point(self) -> bool {
        matches!(
            self,
            Self::CurrentPoint
                | Self::AnnealBestPoint
                | Self::AnnealCurrentPoint
                | Self::AnnealStopping
                | Self::PatternBestPoint
        )
    }
    pub const fn value_field(self) -> Option<&'static str> {
        match self {
            Self::CurrentPoint
            | Self::AnnealBestPoint
            | Self::AnnealCurrentPoint
            | Self::AnnealStopping => None,
            Self::PatternBestPoint => None,
            Self::PatternBestFunction => Some("fval"),
            Self::PatternFunctionCount => Some("funccount"),
            Self::PatternMeshSize => Some("meshsize"),
            Self::PatternMaxConstraint => None,
            Self::SurrogateProgress => None,
            Self::AnnealBestFunction => Some("bestfval"),
            Self::AnnealCurrentFunction => Some("fval"),
            Self::AnnealTemperature => Some("temperature"),
            Self::ParticleBestFunctionValue => Some("bestfval"),
            Self::FunctionCount => Some("funccount"),
            Self::FunctionValue | Self::ConstrainedFunctionValue => Some("fval"),
            Self::ConstraintViolation => Some("constrviolation"),
            Self::StepSize => Some("stepsize"),
            Self::FirstOrderOptimality => Some("firstorderopt"),
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::CurrentPoint => "optimplotx",
            Self::AnnealBestPoint => "saplotbestx",
            Self::AnnealCurrentPoint => "saplotx",
            Self::AnnealBestFunction => "saplotbestf",
            Self::AnnealCurrentFunction => "saplotf",
            Self::AnnealStopping => "saplotstopping",
            Self::AnnealTemperature => "saplottemperature",
            Self::PatternBestFunction => "psplotbestf",
            Self::PatternFunctionCount => "psplotfuncount",
            Self::PatternMeshSize => "psplotmeshsize",
            Self::PatternBestPoint => "psplotbestx",
            Self::PatternMaxConstraint => "psplotmaxconstr",
            Self::ParticleBestFunctionValue => "pswplotbestf",
            Self::FunctionCount => "optimplotfunccount",
            Self::FunctionValue => "optimplotfval",
            Self::ConstraintViolation => "optimplotconstrviolation",
            Self::StepSize => "optimplotstepsize",
            Self::FirstOrderOptimality => "optimplotfirstorderopt",
            Self::ConstrainedFunctionValue => "optimplotfvalconstr",
            Self::SurrogateProgress => "surrogateoptplot",
        }
    }
    pub const fn title(self) -> &'static str {
        match self {
            Self::CurrentPoint | Self::AnnealCurrentPoint => "Current Point",
            Self::AnnealBestPoint => "Best Point",
            Self::AnnealBestFunction => "Best Annealing Objective",
            Self::AnnealCurrentFunction => "Current Annealing Objective",
            Self::AnnealStopping => "Stopping Criteria",
            Self::AnnealTemperature => "Mean Temperature",
            Self::PatternBestFunction => "Pattern Search Objective",
            Self::PatternFunctionCount => "Pattern Search Function Evaluations",
            Self::PatternMeshSize => "Pattern Search Mesh Size",
            Self::PatternBestPoint => "Pattern Search Point",
            Self::PatternMaxConstraint => "Nonlinear Constraint Violation",
            Self::ParticleBestFunctionValue => "Best Particle Objective",
            Self::FunctionCount => "Function Evaluations",
            Self::FunctionValue => "Function Value",
            Self::ConstraintViolation => "Constraint Violation",
            Self::StepSize => "Step Size",
            Self::FirstOrderOptimality => "First-Order Optimality",
            Self::ConstrainedFunctionValue => "Feasible Objective Value",
            Self::SurrogateProgress => "Surrogate Optimization",
        }
    }
    pub const fn x_label(self) -> &'static str {
        match self {
            Self::CurrentPoint
            | Self::AnnealBestPoint
            | Self::AnnealCurrentPoint
            | Self::PatternBestPoint => "Variable",
            Self::AnnealStopping => "Stopping criterion",
            Self::SurrogateProgress => "Function evaluations",
            _ => "Iteration",
        }
    }
    pub const fn y_label(self) -> &'static str {
        match self {
            Self::CurrentPoint
            | Self::AnnealBestPoint
            | Self::AnnealCurrentPoint
            | Self::PatternBestPoint => "Coordinate value",
            Self::AnnealStopping => "Fraction of stopping limit",
            Self::AnnealTemperature => "Temperature",
            Self::PatternMeshSize => "Mesh size",
            Self::PatternMaxConstraint => "Maximum constraint violation",
            Self::PatternFunctionCount => "Function evaluations",
            Self::PatternBestFunction => "Function value",
            Self::AnnealBestFunction | Self::AnnealCurrentFunction => "Function value",
            Self::FunctionCount => "Function evaluations",
            Self::FunctionValue
            | Self::ConstrainedFunctionValue
            | Self::ParticleBestFunctionValue => "Function value",
            Self::ConstraintViolation => "Maximum constraint violation",
            Self::StepSize => "Step size",
            Self::FirstOrderOptimality => "First-order optimality",
            Self::SurrogateProgress => "Objective function",
        }
    }
}

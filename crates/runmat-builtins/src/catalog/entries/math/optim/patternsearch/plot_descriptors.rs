use super::descriptor::parameter;
use crate::*;
pub const PATTERNSEARCH_PLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code:"RM.PATTERNSEARCH.PLOT_ARGUMENT", identifier:Some("RunMat:patternsearch:PlotArgument"),
    when:"Plot metadata, state, or output count is invalid, or the selected UI service is unavailable.",
    message:"patternsearch plot helper: invalid argument",
};
macro_rules! descriptor {
    ($name:ident,$label:literal,$metric:literal) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &[
                    parameter("optimvalues", $metric),
                    parameter("flag", "init, iter or done."),
                ],
                outputs: &[parameter(
                    "stop",
                    "Logical false after the selected progress-scene update.",
                )],
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[PATTERNSEARCH_PLOT_ERROR_INVALID_ARGUMENT],
        };
    };
}
descriptor!(
    PSPLOTBESTF_DESCRIPTOR,
    "stop = psplotbestf(optimvalues,flag)",
    "Scalar structure with finite iteration and real scalar fval."
);
descriptor!(
    PSPLOTFUNCOUNT_DESCRIPTOR,
    "stop = psplotfuncount(optimvalues,flag)",
    "Scalar structure with finite nonnegative integral iteration and funccount."
);
descriptor!(
    PSPLOTMESHSIZE_DESCRIPTOR,
    "stop = psplotmeshsize(optimvalues,flag)",
    "Scalar structure with finite nonnegative integral iteration and finite nonnegative meshsize."
);
descriptor!(PSPLOTBESTX_DESCRIPTOR,"stop = psplotbestx(optimvalues,flag)","Scalar structure with finite nonnegative integral iteration and finite full real double x coordinates.");
descriptor!(PSPLOTMAXCONSTR_DESCRIPTOR,"stop = psplotmaxconstr(optimvalues,flag)","Scalar structure with iteration and optional finite real double nonlinineq/nonlineq. Absent nonlinear constraints give zero violation.");

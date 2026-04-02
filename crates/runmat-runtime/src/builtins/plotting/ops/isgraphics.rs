use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "isgraphics",
    category = "plotting",
    summary = "Return true if the input is a valid plotting graphics handle.",
    keywords = "isgraphics,plotting,handle",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::isgraphics"
)]
pub fn isgraphics_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let Some(value) = args.first() else {
        return Ok(false);
    };
    if let Some(v) = match value {
        Value::Num(v) => Some(*v),
        _ => None,
    } {
        if !v.is_finite() || v <= 0.0 {
            return Ok(false);
        }
    }
    Ok(crate::builtins::plotting::properties::resolve_plot_handle(value, "isgraphics").is_ok())
}

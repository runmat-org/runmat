use rustmat_builtins::{builtins, Value, BuiltinFn};

/// Call a registered MATLAB builtin by name.
/// Returns an error if no builtin with that name is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    for b in builtins() {
        if b.name == name {
            let f: BuiltinFn = b.func;
            return (f)(args);
        }
    }
    Err(format!("unknown builtin `{}`", name))
} 
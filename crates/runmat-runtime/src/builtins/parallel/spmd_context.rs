use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn rank_and_size() -> (u32, u32) {
    crate::context::legacy::active()
        .and_then(|context| {
            context
                .service_ports()
                .collective()
                .map(|service| (service.context().rank.0, service.context().gang.labs.0))
        })
        .unwrap_or((1, 1))
}

macro_rules! context_builtin {
    ($function:ident, $name:literal, $member:tt) => {
        #[runtime_builtin(
                                            name = $name,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::parallel::spmd_context"
                                        )]
        fn $function() -> crate::BuiltinResult<Value> {
            Ok(Value::Num(f64::from(rank_and_size().$member)))
        }
    };
}

context_builtin!(spmd_index_builtin, "spmdIndex", 0);
context_builtin!(spmd_size_builtin, "spmdSize", 1);
context_builtin!(labindex_builtin, "labindex", 0);
context_builtin!(numlabs_builtin, "numlabs", 1);

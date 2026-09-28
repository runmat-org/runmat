use runmat_builtins::{
    LABINDEX_DESCRIPTOR, LABINDEX_TOO_MANY_OUTPUTS, NUMLABS_DESCRIPTOR, NUMLABS_TOO_MANY_OUTPUTS,
    SPMD_INDEX_DESCRIPTOR, SPMD_INDEX_TOO_MANY_OUTPUTS, SPMD_SIZE_DESCRIPTOR,
    SPMD_SIZE_TOO_MANY_OUTPUTS,
};
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
    ($function:ident, $name:literal, $member:tt, $descriptor:ident, $error:ident) => {
        #[runtime_builtin(
                                            name = $name,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::parallel::spmd_context"
                                        )]
        fn $function() -> crate::BuiltinResult<Value> {
            crate::output_count::validate_fixed_outputs(&$descriptor, &$error, $name)?;
            Ok(Value::Num(f64::from(rank_and_size().$member)))
        }
    };
}

context_builtin!(
    spmd_index_builtin,
    "spmdIndex",
    0,
    SPMD_INDEX_DESCRIPTOR,
    SPMD_INDEX_TOO_MANY_OUTPUTS
);
context_builtin!(
    spmd_size_builtin,
    "spmdSize",
    1,
    SPMD_SIZE_DESCRIPTOR,
    SPMD_SIZE_TOO_MANY_OUTPUTS
);
context_builtin!(
    labindex_builtin,
    "labindex",
    0,
    LABINDEX_DESCRIPTOR,
    LABINDEX_TOO_MANY_OUTPUTS
);
context_builtin!(
    numlabs_builtin,
    "numlabs",
    1,
    NUMLABS_DESCRIPTOR,
    NUMLABS_TOO_MANY_OUTPUTS
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_context_is_one_based_for_all_four_bindings() {
        let readers: [fn() -> crate::BuiltinResult<Value>; 4] = [
            spmd_index_builtin,
            spmd_size_builtin,
            labindex_builtin,
            numlabs_builtin,
        ];
        for read in readers {
            assert_eq!(read().unwrap(), Value::Num(1.0));
        }
    }

    #[test]
    fn every_binding_rejects_excess_outputs_before_context_read() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let cases: [(fn() -> crate::BuiltinResult<Value>, Option<&str>); 4] = [
            (spmd_index_builtin, SPMD_INDEX_TOO_MANY_OUTPUTS.identifier),
            (spmd_size_builtin, SPMD_SIZE_TOO_MANY_OUTPUTS.identifier),
            (labindex_builtin, LABINDEX_TOO_MANY_OUTPUTS.identifier),
            (numlabs_builtin, NUMLABS_TOO_MANY_OUTPUTS.identifier),
        ];
        for (read, expected) in cases {
            assert_eq!(read().unwrap_err().identifier(), expected);
        }
    }
}

use once_cell::sync::OnceCell;
use runmat_builtins::Value;

fn env_flag(name: &'static str) -> bool {
    static DEBUG_VARS: OnceCell<bool> = OnceCell::new();
    static DEBUG_INDEX: OnceCell<bool> = OnceCell::new();
    match name {
        "RUNMAT_DEBUG_VARS" => {
            *DEBUG_VARS.get_or_init(|| std::env::var(name).as_deref() == Ok("1"))
        }
        "RUNMAT_DEBUG_INDEX" => {
            *DEBUG_INDEX.get_or_init(|| std::env::var(name).as_deref() == Ok("1"))
        }
        _ => std::env::var(name).as_deref() == Ok("1"),
    }
}

fn store_var_filter() -> &'static str {
    static FILTER: OnceCell<String> = OnceCell::new();
    FILTER.get_or_init(|| std::env::var("RUNMAT_DEBUG_STORE_VAR").unwrap_or_default())
}

pub fn trace_load_var(pc: usize, var_index: usize, value: &Value) {
    if env_flag("RUNMAT_DEBUG_VARS") {
        match value {
            Value::Tensor(t) => {
                eprintln!("[vm] LoadVar var={} Tensor shape={:?}", var_index, t.shape)
            }
            Value::GpuTensor(h) => eprintln!(
                "[vm] LoadVar var={} GpuTensor shape={:?}",
                var_index, h.shape
            ),
            _ => {}
        }
    }
    if env_flag("RUNMAT_DEBUG_INDEX") {
        match value {
            Value::GpuTensor(h) => log::debug!(
                "[vm] LoadVar GPU tensor pc={} var={} shape={:?}",
                pc,
                var_index,
                h.shape
            ),
            Value::Tensor(t) => log::debug!(
                "[vm] LoadVar tensor pc={} var={} shape={:?}",
                pc,
                var_index,
                t.shape
            ),
            _ => {}
        }
    }
}

pub fn trace_store_var(pc: usize, var_index: usize, value: &Value) {
    if env_flag("RUNMAT_DEBUG_VARS") {
        match value {
            Value::Tensor(t) => {
                eprintln!("[vm] StoreVar var={} Tensor shape={:?}", var_index, t.shape)
            }
            Value::GpuTensor(h) => eprintln!(
                "[vm] StoreVar var={} GpuTensor shape={:?}",
                var_index, h.shape
            ),
            _ => {}
        }
    }

    let filter = store_var_filter();
    let log_this = if filter.trim().is_empty() {
        false
    } else if filter.trim().eq_ignore_ascii_case("*") {
        true
    } else if let Ok(target) = filter.trim().parse::<usize>() {
        target == var_index
    } else {
        false
    };
    if log_this {
        log::debug!(
            "[vm] StoreVar value pc={} var={} value={:?}",
            pc,
            var_index,
            value
        );
    }

    if env_flag("RUNMAT_DEBUG_INDEX") {
        match value {
            Value::GpuTensor(h) => log::debug!(
                "[vm] StoreVar GPU tensor pc={} var={} shape={:?}",
                pc,
                var_index,
                h.shape
            ),
            Value::Tensor(t) => log::debug!(
                "[vm] StoreVar tensor pc={} var={} shape={:?}",
                pc,
                var_index,
                t.shape
            ),
            _ => {}
        }
    }
}

pub fn trace_call_builtin(pc: usize, name: &str, arg_count: usize, stack: &[Value]) {
    if env_flag("RUNMAT_DEBUG_STACK") {
        log::debug!(
            "[vm] CallBuiltin pc={} name={} argc={} stack_len={} top={:?}",
            pc,
            name,
            arg_count,
            stack.len(),
            stack.last()
        );
    }
}

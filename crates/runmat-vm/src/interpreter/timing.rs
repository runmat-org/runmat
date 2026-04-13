#[cfg(not(target_arch = "wasm32"))]
use runmat_time::Instant;
#[cfg(target_arch = "wasm32")]
type Instant = ();

#[cfg(not(target_arch = "wasm32"))]
pub struct InterpreterTiming {
    enabled: bool,
    host_span_start: Option<(Instant, usize)>,
    host_span_last_pc: Option<usize>,
    host_span_instrs: u64,
    seq: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl InterpreterTiming {
    pub fn new() -> Self {
        let enabled = std::env::var("RUNMAT_INTERPRETER_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
            .unwrap_or(false);
        Self {
            enabled,
            host_span_start: None,
            host_span_last_pc: None,
            host_span_instrs: 0,
            seq: 0,
        }
    }

    pub fn note_host_instr(&mut self, pc: usize) {
        if !self.enabled {
            return;
        }
        if self.host_span_start.is_none() {
            self.host_span_start = Some((Instant::now(), pc));
            self.host_span_instrs = 0;
        }
        self.host_span_instrs += 1;
        self.host_span_last_pc = Some(pc);
    }

    pub fn flush_host_span(&mut self, reason: &str, detail: Option<&str>) {
        if !self.enabled {
            return;
        }
        let Some((start, start_pc)) = self.host_span_start.take() else {
            return;
        };
        let duration = start.elapsed();
        let end_pc = self.host_span_last_pc.unwrap_or(start_pc);
        let instrs = self.host_span_instrs.max(1);
        if let Some(extra) = detail {
            log::debug!(
                "interpreter_host_span seq={} reason={} detail={} pc_span=[{}..{}] instrs={} duration_ns={}",
                self.seq,
                reason,
                extra,
                start_pc,
                end_pc,
                instrs,
                duration.as_nanos()
            );
        } else {
            log::debug!(
                "interpreter_host_span seq={} reason={} pc_span=[{}..{}] instrs={} duration_ns={}",
                self.seq,
                reason,
                start_pc,
                end_pc,
                instrs,
                duration.as_nanos()
            );
        }
        self.seq += 1;
        self.host_span_last_pc = None;
        self.host_span_instrs = 0;
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for InterpreterTiming {
    fn drop(&mut self) {
        self.flush_host_span("drop", None);
    }
}

#[cfg(target_arch = "wasm32")]
pub struct InterpreterTiming;

#[cfg(target_arch = "wasm32")]
impl InterpreterTiming {
    pub fn new() -> Self {
        Self
    }

    pub fn note_host_instr(&mut self, _pc: usize) {}

    pub fn flush_host_span(&mut self, _reason: &str, _detail: Option<&str>) {}
}

#[cfg(target_arch = "wasm32")]
impl Drop for InterpreterTiming {
    fn drop(&mut self) {}
}

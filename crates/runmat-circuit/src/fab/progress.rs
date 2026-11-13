//! Progress while the fabrication set is made, for a caller that shows it (the Desktop's
//! Fabrication outputs lists each output as it starts).
//!
//! A caller runs the export inside [`reporting`]; the export calls [`step`] as each stage
//! starts. Outside [`reporting`], [`step`] does nothing, so the CLI and the server pay nothing.
//!
//! The steps, in order: `board` (zones filled), then each requested output by its settings id
//! (`gerber`, `drill`, `drill_map`, `drawing`, `assembly_drawing`, `pick_place`, `bom`,
//! `ipc2581`, `odb`, `ipc356`), then `preflight` when the caller asked for it. A step's start
//! means every earlier step is done.

use std::cell::RefCell;

type Sink = Box<dyn FnMut(&str)>;

thread_local! {
    static SINK: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// Runs `f` with `sink` called with each step's id as it starts; the previous sink comes back
/// afterwards, so reports nest.
pub fn reporting<R>(sink: impl FnMut(&str) + 'static, f: impl FnOnce() -> R) -> R {
    let previous = SINK.with(|s| s.borrow_mut().replace(Box::new(sink)));
    let out = f();
    SINK.with(|s| *s.borrow_mut() = previous);
    out
}

/// Reports that the step `id` starts.
pub fn step(id: &str) {
    SINK.with(|s| {
        if let Some(sink) = s.borrow_mut().as_mut() {
            sink(id);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn steps_reach_the_sink_only_while_reporting() {
        step("before");
        let seen = Rc::new(RefCell::new(Vec::<String>::new()));
        let into = seen.clone();
        let out = reporting(
            move |id| into.borrow_mut().push(id.to_owned()),
            || {
                step("board");
                step("gerber");
                7
            },
        );
        step("after");
        assert_eq!(out, 7);
        assert_eq!(*seen.borrow(), ["board", "gerber"]);
    }
}

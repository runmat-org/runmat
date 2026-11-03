//! Generated native entries must resume the exact member and short-circuit site.

use super::*;
use crate::{ExecutableSource, ProcedureInvocation, ProcedureTarget};
use futures::executor::block_on;

#[path = "native_sequence_truth/input.rs"]
mod input;
use input::{install_input, pending_once};
#[path = "native_sequence_truth/member.rs"]
mod member;
#[path = "native_sequence_truth/site_shape.rs"]
mod site_shape;
#[path = "native_sequence_truth/truth.rs"]
mod truth;
#[path = "native_sequence_truth/capture.rs"]
mod capture;
#[path = "native_sequence_truth/constructor_source.rs"]
mod constructor_source;
#[path = "native_sequence_truth/super_retained.rs"]
mod super_retained;

const MEMBER_CLASS: &str = r#"
classdef PendingMember < handle
    methods
        function obj = PendingMember()
            global pending_member_witness;
            pending_member_witness = obj;
        end
        function out = subsref(obj, S)
            global member_reads;
            member_reads = member_reads + 1;
            input('member: ');
            out = 7;
        end
    end
end
"#;

const SOURCE: &str = r#"
function output = readMember()
    global member_reads;
    member_reads = 0;
    object = PendingMember();
    output = object.item;
end
function output = readDynamicMember()
    global member_reads;
    member_reads = 0;
    object = PendingMember();
    name = 'item';
    output = object.(name);
end
function output = shortCircuit()
    global rhs_calls;
    rhs_calls = 0;
    skipped = 0 && rhs();
    output = 1 && rhs();
    if skipped || rhs_calls ~= 1
        error('RunMat:NativeReplay', 'short-circuit replayed its right side');
    end
end
function output = rhs()
    global rhs_calls;
    rhs_calls = rhs_calls + 1;
    output = input('truth: ');
end
function [left, right] = captureSequence()
    values = {7, 3};
    [left, right] = deal(values{:});
end
function [left, right, extra] = captureMismatch()
    values = {7, 3};
    [left, right, extra] = deal(values{:});
end
"#;

fn invoke(function: &str, requested_outputs: usize) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(function.into()),
        arguments: crate::ProcedureArguments::Source(Vec::new()),
        requested_outputs,
    }
}

fn with_native_unit(
    test: impl FnOnce(&mut RunMatSession, &crate::ExecutableUnit, Arc<AtomicUsize>),
) {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = super::super::cwd_lock();
        let directory = tempfile::TempDir::new().expect("native continuation source directory");
        std::fs::write(directory.path().join("PendingMember.m"), MEMBER_CLASS)
            .expect("write member overload");
        std::fs::write(directory.path().join("native_sequence_truth.m"), SOURCE)
            .expect("write compiled function source");
        let _cwd = super::super::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("native session");
        let unit = block_on(
            session.compile_executable_unit(
                ExecutableSource::frozen(
                    "native-sequence-truth@1",
                    "native_sequence_truth.m",
                    directory
                        .path()
                        .join("native_sequence_truth.m")
                        .to_string_lossy(),
                    SOURCE,
                ),
                None,
            ),
        )
        .expect("compile generated native entry fixture");
        assert!(unit.bytecode().instructions.is_empty());
        for name in [
            "readMember",
            "readDynamicMember",
            "shortCircuit",
            "captureSequence",
            "captureMismatch",
        ] {
            assert!(unit.functions().names.contains_key(name), "missing {name}");
        }
        site_shape::assert_compiled_sites(&unit);
        test(&mut session, &unit, Arc::new(AtomicUsize::new(0)));
    });
}

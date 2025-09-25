use assert_cmd::prelude::*;
use std::process::Command;

#[test]
fn runs_help() {
    let mut cmd = Command::cargo_bin("runmatfunc").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

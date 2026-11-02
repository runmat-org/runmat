use runmat_execution_artifact::target_test::{
    NativePairedProductHeader, NativePairedProductRole, NativeTargetTestRequest,
};
use runmat_process_host::ipc::frame::{self, FrameLimits};

use crate::input::LinkedAotProcessInput;

const FRAME_LIMITS: FrameLimits = FrameLimits {
    max_message_bytes: runmat_execution_artifact::target_test::MAX_NATIVE_TARGET_TEST_WIRE_BYTES
        as u32,
};

pub(crate) fn run(
    linked: LinkedAotProcessInput,
    header: NativePairedProductHeader,
) -> Result<(), String> {
    if header.role != NativePairedProductRole::Test {
        return Err("paired native image is not a test product".into());
    }
    let mut control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(|error| format!("target-test control bootstrap failed: {error}"))?;
    let bytes = frame::read_payload_blocking(&mut control.reader, FRAME_LIMITS)
        .map_err(|error| format!("target-test request frame is invalid: {error}"))?;
    let request = NativeTargetTestRequest::from_canonical_bytes(&bytes, &header.binding)
        .map_err(|error| format!("target-test request is invalid: {error}"))?;
    let result = crate::execute::execute_linked_test(linked, &header, &request)?;
    let bytes = result
        .to_canonical_bytes(&request, &header.binding)
        .map_err(|error| format!("target-test result is invalid: {error}"))?;
    frame::write_payload_blocking(&mut control.writer, &bytes, FRAME_LIMITS)
        .map_err(|error| format!("target-test result frame failed: {error}"))
}

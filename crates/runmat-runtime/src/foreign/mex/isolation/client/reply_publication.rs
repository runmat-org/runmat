use super::*;
use runmat_process_host::ipc::PendingFrameWrite;

pub(super) fn callback_reply_frame(
    response: MexCallbackResult,
    limits: FrameLimits,
) -> Result<PendingFrameWrite, MexWireError> {
    let message = DriverMessage::CallbackResult(response);
    message
        .validate()
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    let payload = serde_json::to_vec(&message)
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    PendingFrameWrite::new(payload, limits).map_err(transport_error)
}

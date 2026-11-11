import { REMOTE_DRIVER_COMPLETION_CONTROL_BYTES } from "./remote-driver-capacity.js";
export { REMOTE_DRIVER_COMPLETION_CONTROL_BYTES } from "./remote-driver-capacity.js";

/** Request the child limit plus its separate parent control component before authorization. */
export function remoteDriverMemoryBytes(childMemoryBytes: number): number {
  if (!Number.isSafeInteger(childMemoryBytes) || childMemoryBytes < 0 ||
      childMemoryBytes > Number.MAX_SAFE_INTEGER - REMOTE_DRIVER_COMPLETION_CONTROL_BYTES) {
    throw new RangeError("Remote driver child memory must fit the portable parent capacity");
  }
  return childMemoryBytes + REMOTE_DRIVER_COMPLETION_CONTROL_BYTES;
}

import "fake-indexeddb/auto";
import { installNodeSyncXhr } from "./node-sync-xhr.js";

if (typeof process !== "undefined" && process.versions?.node) {
  installNodeSyncXhr();
}

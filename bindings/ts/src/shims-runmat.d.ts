declare module "../pkg/runmat_wasm.js" {
  export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;
  export function initRunMat(options: Record<string, unknown>): Promise<any>;
  export function registerFsProvider(provider: Record<string, unknown>): void;
  const init: (module?: InitInput | Promise<InitInput>) => Promise<unknown>;
  export default init;
}

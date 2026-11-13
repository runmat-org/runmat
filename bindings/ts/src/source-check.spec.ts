import { describe, expect, it, vi } from 'vitest';
import { checkNativeSource, type SourceCheckReport, type SourceCheckRequest } from './source-check.js';

describe('ordinary source check boundary', () => {
  it('forwards project/function/snapshot settings without interpreting compiler results', async () => {
    const request: SourceCheckRequest = { target: { kind: 'function', path: '/src/helper.m', function: 'helper' },
      userSettingsJson: '{"runtime":{"language":{"compat":"strict"}}}', denyWarnings: true,
      inputManifest: { snapshotId: 'captured', entries: [] } };
    const report: SourceCheckReport = { schema_version: 1, document_kind: 'function', path: '/src/helper.m',
      outcome: 'failed', language_compatibility: 'strict', project_revision: null, documents: [],
      summary: { errors: 1, warnings: 0, warnings_denied: true } };
    const native = { checkSource: vi.fn(async () => report) };
    expect(await checkNativeSource(native, request)).toBe(report);
    expect(native.checkSource).toHaveBeenCalledWith(request);
  });
  it('rejects stale WASM and preserves native failures rather than returning empty diagnostics', async () => {
    const request: SourceCheckRequest = { target: { kind: 'project', path: '/runmat.toml' } };
    await expect(checkNativeSource({}, request)).rejects.toThrow('rebuild');
    const failure = Error('captured source revision changed');
    await expect(checkNativeSource({ checkSource: async () => { throw failure; } }, request)).rejects.toBe(failure);
  });
});

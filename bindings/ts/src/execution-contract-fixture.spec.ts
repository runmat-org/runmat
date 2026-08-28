import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

interface ExecutionContractFixture {
  schema: string;
  program_revision_cbor_hex: string;
  program_revision_identity: string;
  envelope_cbor_hex: string;
  task_id: string;
}

describe("portable execution contract fixture", () => {
  it("is consumable without target-specific integer or byte rewriting", () => {
    const path = resolve(
      process.cwd(),
      "../../crates/runmat-execution/tests/fixtures/execution_contract_v1.json"
    );
    const fixture = JSON.parse(
      readFileSync(path, "utf8")
    ) as ExecutionContractFixture;

    expect(fixture.schema).toBe("runmat-execution-contract-fixture/v1");
    expect(decodeHex(fixture.program_revision_cbor_hex).byteLength).toBe(211);
    expect(decodeHex(fixture.envelope_cbor_hex)).toEqual(
      new Uint8Array([
        0xa6, 0x00, 0x01, 0x01, 0x00, 0x02, 0x07, 0x03, 0x00, 0x04, 0x18,
        0x2a, 0x05, 0x43, 0x01, 0x02, 0x03
      ])
    );
    expect(fixture.program_revision_identity).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(fixture.task_id).toMatch(/^task_[0-9a-f]{32}$/);
  });
});

function decodeHex(value: string): Uint8Array {
  if (value.length % 2 !== 0 || !/^[0-9a-f]*$/.test(value)) {
    throw new Error("execution fixture contains invalid canonical hexadecimal");
  }
  return Uint8Array.from(
    value.match(/.{2}/g)?.map((byte) => Number.parseInt(byte, 16)) ?? []
  );
}

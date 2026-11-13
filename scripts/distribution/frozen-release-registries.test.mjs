import assert from "node:assert/strict";
import test from "node:test";
import { expectedPackages } from "./frozen-release-registries.mjs";

const metadata = () => ({ workspace_members: ["one", "two"], packages: [
  { id: "one", name: "runmat-one", version: "0.7.1", publish: null },
  { id: "two", name: "runmat-private", version: "0.7.1", publish: [] },
  { id: "external", name: "external", version: "1.0.0", publish: null },
] });
test("derives every expected publishable version from the frozen workspace", () => {
  assert.deepEqual(expectedPackages(metadata(), "0.7.1").map(pkg => pkg.name), ["runmat-one"]);
  const more = metadata(); more.packages[1].publish = ["crates-io"];
  assert.equal(expectedPackages(more, "0.7.1").length, 2);
});
test("rejects wrong versions, duplicate identities, unsafe names and empty inventories", () => {
  for (const edit of [m => { m.packages[0].version = "0.7.0"; }, m => { m.packages[0].name = "../other"; },
    m => { m.packages[0].publish = []; }, m => { m.workspace_members.push("extra"); m.packages.push({ ...m.packages[0], id: "extra" }); }]) {
    const input = metadata(); edit(input); assert.throws(() => expectedPackages(input, "0.7.1"));
  }
});

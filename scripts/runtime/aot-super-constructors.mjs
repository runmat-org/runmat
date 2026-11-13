// Packaged-product cases run through the embedded-AOT CLI and its emitted,
// linked executable. The callback-linked AOT runtime tests cover a different
// boundary and cannot establish that generated machine code entered here.
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export function verifyAotSuperConstructors({ directory, compile, invoke, packageOnly = false }) {
    function source(name, lines, parent = directory) {
        writeFileSync(join(parent, `${name}.m`), `${lines.join("\n")}\n`, { encoding: "utf8", flag: "wx" });
    }
    function execute(name, lines, expected, minimumRetained) {
        if (packageOnly && name !== "aot_super_package") return;
        const planPath = join(directory, `${name}.plan.json`);
        const executable = compile(name, `${lines.join("\n")}\n`, ["--link-plan-json", planPath]);
        const plan = JSON.parse(readFileSync(planPath, "utf8"));
        assert.equal(plan.schema_version, 4, `${name} did not emit the typed link-plan schema`);
        assert.ok(
            Array.isArray(plan.retained_function_ids) && plan.retained_function_ids.length >= minimumRetained,
            `${name} omitted its typed class-method retention closure`
        );
        const result = invoke(executable);
        assert.equal(result.status, 0, `linked ${name} failed:\n${result.stderr}`);
        assert.equal(result.stdout.trim(), expected, `linked ${name} changed retained superclass behavior`);
    }

    source("AotProbeBase", [
        "classdef AotProbeBase < handle",
        "  properties",
        "    baseValue = 12 / 3",
        "    witnessedValue = 0",
        "  end",
        "  methods",
        "    function obj = AotProbeBase(v)",
        "      global aot_super_witness;",
        "      aot_super_witness = obj;",
        "      obj.witnessedValue = v;",
        "    end",
        "  end",
        "end"
    ]);
    source("AotProbeDerived", [
        "classdef AotProbeDerived < AotProbeBase",
        "  properties",
        "    derivedValue = 2 + 3",
        "  end",
        "  methods",
        "    function obj = AotProbeDerived(v)",
        "      obj@AotProbeBase(v);",
        "      obj.derivedValue = obj.derivedValue + v;",
        "    end",
        "  end",
        "end"
    ]);
    execute("aot_super_plain", [
        "function result = aot_super_plain()",
        "  global aot_super_witness;",
        "  obj = AotProbeDerived(7);",
        "  aot_super_witness.derivedValue = 13;",
        "  result = obj.baseValue * 100 + obj.witnessedValue * 10 + obj.derivedValue;",
        "end"
    ], "483", 3);

    const packageDirectory = join(directory, "+pkg");
    mkdirSync(packageDirectory);
    source("AotPkgBase", [
        "classdef AotPkgBase < handle",
        "  properties",
        "    baseValue = 9 / 3",
        "    witnessedValue = 0",
        "  end",
        "  methods",
        "    function obj = AotPkgBase(v)",
        "      global aot_pkg_super_witness;",
        "      aot_pkg_super_witness = obj;",
        "      obj.witnessedValue = v;",
        "    end",
        "  end",
        "end"
    ], packageDirectory);
    source("AotPkgDerived", [
        "classdef AotPkgDerived < pkg.AotPkgBase",
        "  properties",
        "    derivedValue = 1 + 3",
        "  end",
        "  methods",
        "    function obj = AotPkgDerived(v)",
        "      obj@pkg.AotPkgBase(v);",
        "      obj.derivedValue = obj.derivedValue + v;",
        "    end",
        "  end",
        "end"
    ], packageDirectory);
    execute("aot_super_package", [
        "function result = aot_super_package()",
        "  global aot_pkg_super_witness;",
        "  obj = pkg.AotPkgDerived(3);",
        "  aot_pkg_super_witness.derivedValue = 8;",
        "  result = obj.baseValue * 100 + obj.witnessedValue * 10 + obj.derivedValue;",
        "end"
    ], "338", 3);

    source("AotImplicitBase", [
        "classdef AotImplicitBase < handle",
        "  properties",
        "    baseValue = 6",
        "  end",
        "end"
    ]);
    source("AotImplicitDerived", [
        "classdef AotImplicitDerived < AotImplicitBase",
        "  properties",
        "    derivedValue = 2",
        "  end",
        "  methods",
        "    function obj = AotImplicitDerived()",
        "      obj@AotImplicitBase();",
        "    end",
        "  end",
        "end"
    ]);
    execute("aot_super_implicit", [
        "function result = aot_super_implicit()",
        "  obj = AotImplicitDerived();",
        "  result = obj.baseValue * 10 + obj.derivedValue;",
        "end"
    ], "62", 2);
}

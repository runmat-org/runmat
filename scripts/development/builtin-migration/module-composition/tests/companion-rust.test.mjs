import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { renderModuleCompositionCompanion, renderModuleCompositionProduct } from "../generate.mjs";

const ALWAYS = { kind: "always" };

function child(module, sourcePath, visibility, order) {
  return {
    module, source_kind: sourcePath.endsWith("/mod.rs") ? "directory" : "file",
    source_path: sourcePath, role: "identity", visibility,
    declaration_condition: ALWAYS, declaration_order: order,
    macro_use: false, reexports: [], aggregation_sources: [],
  };
}

function write(root, relativePath, source) {
  const target = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, source);
}

test("generated companion exports preserve public Rust paths", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-rust-exports-"));
  try {
    const base = "catalog/entries";
    const alpha = child("alpha", `crates/runmat-builtins/src/${base}/alpha/mod.rs`, "catalog", 0);
    alpha.reexports = [{ kind: "glob", visibility: "public", condition: ALWAYS, doc_hidden: false }];
    const beta = child("beta", `crates/runmat-builtins/src/${base}/beta/mod.rs`, "catalog", 1);
    beta.reexports = [{ kind: "glob", visibility: "public", condition: ALWAYS, doc_hidden: false }];
    const product = {
      product_id: "catalog-root", crate_role: "catalog",
      path: `crates/runmat-builtins/src/${base}/mod.rs`, module_path: "crate::catalog::entries",
      state: "present", aggregations: [], aggregation_exports: [],
      children: [alpha, beta, {
        ...child("exports", `crates/runmat-builtins/src/${base}/exports.rs`, "private", 2),
        role: "support", companion_section: "reexports",
      }],
    };
    write(root, `${base}/mod.rs`, renderModuleCompositionProduct(product));
    write(root, `${base}/exports.rs`, renderModuleCompositionCompanion(product, "reexports"));
    write(root, `${base}/alpha/mod.rs`, "pub struct Alpha;\n");
    write(root, `${base}/beta/mod.rs`, "pub struct Beta;\n");
    write(root, "lib.rs", "pub mod catalog { pub mod entries; }\npub use catalog::entries::{Alpha, Beta};\n");
    const compiled = spawnSync("rustc", ["--edition=2021", "--crate-type=lib", "--emit=metadata",
      "-o", path.join(root, "out.rmeta"), path.join(root, "lib.rs")], { encoding: "utf8" });
    assert.equal(compiled.status, 0, compiled.stderr);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("declaration and reexport companions compose without losing the exports module", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-rust-both-companions-"));
  try {
    const base = "catalog/entries/io";
    const alpha = child("alpha", `crates/runmat-builtins/src/${base}/alpha.rs`, "private", 0);
    alpha.reexports = [{ kind: "named", visibility: "public", condition: ALWAYS,
      doc_hidden: false, items: [{ name: "Alpha", alias: null }] }];
    const product = {
      product_id: "catalog-io", crate_role: "catalog",
      path: `crates/runmat-builtins/src/${base}/mod.rs`,
      module_path: "crate::catalog::entries::io", state: "present",
      aggregations: [], aggregation_exports: [],
      children: [
        alpha,
        { ...child("declarations", `crates/runmat-builtins/src/${base}/declarations.rs`, "private", 2),
          role: "support", companion_section: "declarations" },
        { ...child("exports", `crates/runmat-builtins/src/${base}/exports.rs`, "private", 1),
          role: "support", companion_section: "reexports" },
      ],
    };
    write(root, `${base}/mod.rs`, renderModuleCompositionProduct(product));
    write(root, `${base}/declarations.rs`, renderModuleCompositionCompanion(product, "declarations"));
    write(root, `${base}/exports.rs`, renderModuleCompositionCompanion(product, "reexports"));
    write(root, `${base}/alpha.rs`, "pub struct Alpha;\n");
    write(root, "lib.rs", "pub mod catalog { pub mod entries { pub mod io; } }\npub use catalog::entries::io::Alpha;\n");
    const compiled = spawnSync("rustc", ["--edition=2021", "--crate-type=lib", "--emit=metadata",
      "-o", path.join(root, "out.rmeta"), path.join(root, "lib.rs")], { encoding: "utf8" });
    assert.equal(compiled.status, 0, compiled.stderr);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("included declaration companion resolves children at the reviewed parent paths", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-rust-declarations-"));
  try {
    const base = "catalog/entries/io/repl_fs";
    const product = {
      product_id: "catalog-io-repl-fs", crate_role: "catalog",
      path: `crates/runmat-builtins/src/${base}/mod.rs`,
      module_path: "crate::catalog::entries::io::repl_fs",
      state: "present", aggregations: [], aggregation_exports: [],
      children: [
        child("alpha", `crates/runmat-builtins/src/${base}/alpha.rs`, "catalog_io", 0),
        child("beta", `crates/runmat-builtins/src/${base}/beta/mod.rs`, "private", 1),
        { ...child("declarations", `crates/runmat-builtins/src/${base}/declarations.rs`, "private", 3),
          role: "support", companion_section: "declarations" },
        { ...child("registry", `crates/runmat-builtins/src/${base}/registry.rs`, "private", 2),
          role: "support" },
      ],
    };
    write(root, `${base}/mod.rs`, renderModuleCompositionProduct(product));
    write(root, `${base}/declarations.rs`, renderModuleCompositionCompanion(product, "declarations"));
    write(root, `${base}/alpha.rs`, "pub fn value() -> usize { 7 }\n");
    write(root, `${base}/beta/mod.rs`, "mod nested; pub fn value() -> usize { nested::VALUE }\n");
    write(root, `${base}/beta/nested.rs`, "pub const VALUE: usize = 7;\n");
    write(root, `${base}/registry.rs`, "mod groups; pub fn value() -> usize { groups::VALUE }\n");
    write(root, `${base}/registry/groups.rs`, "pub const VALUE: usize = 7;\n");
    write(root, "lib.rs", "pub mod catalog { pub mod entries { pub mod io { pub mod repl_fs; pub fn value() -> usize { repl_fs::alpha::value() } } } }\n");
    const compiled = spawnSync("rustc", ["--edition=2021", "--crate-type=lib", "--emit=metadata",
      "-o", path.join(root, "out.rmeta"), path.join(root, "lib.rs")], { encoding: "utf8" });
    assert.equal(compiled.status, 0, compiled.stderr);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("a noncanonical file child keeps its nested module lookup before and after inclusion", () => {
  const base = "catalog/entries/io/repl_fs";
  const widget = child("widget", `crates/runmat-builtins/src/${base}/alternate/widget.rs`, "private", 0);
  widget.reexports = [{ kind: "named", visibility: "public", condition: ALWAYS, doc_hidden: false,
    items: [{ name: "value", alias: null }] }];
  const product = {
    product_id: "catalog-io-repl-fs", crate_role: "catalog",
    path: `crates/runmat-builtins/src/${base}/mod.rs`,
    module_path: "crate::catalog::entries::io::repl_fs",
    state: "present", aggregations: [], aggregation_exports: [],
    children: [
      { ...child("declarations", `crates/runmat-builtins/src/${base}/declarations.rs`, "private", 1),
        role: "support", companion_section: "declarations" },
      widget,
    ],
  };
  for (const generated of [false, true]) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-rust-noncanonical-"));
    try {
      write(root, `${base}/mod.rs`, generated
        ? renderModuleCompositionProduct(product)
        : '#[path = "alternate/widget.rs"]\nmod widget;\npub use self::widget::value;\n');
      if (generated) write(root, `${base}/declarations.rs`, renderModuleCompositionCompanion(product, "declarations"));
      write(root, `${base}/alternate/widget.rs`, "mod inner; pub fn value() -> usize { inner::VALUE }\n");
      write(root, `${base}/alternate/inner.rs`, "pub const VALUE: usize = 7;\n");
      write(root, "main.rs", "mod catalog { pub mod entries { pub mod io { pub mod repl_fs; } } }\nfn main() { assert_eq!(catalog::entries::io::repl_fs::value(), 7); }\n");
      const binary = path.join(root, "test");
      const compiled = spawnSync("rustc", ["--edition=2021", "-o", binary,
        path.join(root, "main.rs")], { encoding: "utf8" });
      assert.equal(compiled.status, 0, compiled.stderr);
      const executed = spawnSync(binary, [], { encoding: "utf8" });
      assert.equal(executed.status, 0, executed.stderr);
    } finally { fs.rmSync(root, { recursive: true, force: true }); }
  }
});

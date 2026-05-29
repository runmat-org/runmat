---
title: "String, I/O & Filesystem Built-ins"
repo: "runmat-org/runmat"
branch: "dev"
source_url: "https://app.devin.ai/org/runmat-org/wiki/runmat-org/runmat?branch=dev#6.3"
wiki_hash: "#6.3"
section: "6.3"
category: "Runtime & Built-in Functions"
category_hash: "#6"
page_order: 24
last_updated: "May 28, 2026, 9:18:58 PM"
diagram_count: 0
source_files:
  - label: "crates/runmat-core/src/value_metadata.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-core/src/value_metadata.rs"
  - label: "crates/runmat-core/tests/printf_semantics.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-core/tests/printf_semantics.rs"
  - label: "crates/runmat-runtime/src/builtins/builtins-json/datetime.json"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/datetime.json"
  - label: "crates/runmat-runtime/src/builtins/cells/core/cell.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cell.rs"
  - label: "crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs"
  - label: "crates/runmat-runtime/src/builtins/cells/core/cellstr.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cellstr.rs"
  - label: "crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs"
  - label: "crates/runmat-runtime/src/builtins/common/format.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/format.rs"
  - label: "crates/runmat-runtime/src/builtins/common/tensor.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/tensor.rs"
  - label: "crates/runmat-runtime/src/builtins/datetime/mod.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/datetime/mod.rs"
  - label: "crates/runmat-runtime/src/builtins/duration/mod.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/duration/mod.rs"
  - label: "crates/runmat-runtime/src/builtins/introspection/who.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/introspection/who.rs"
  - label: "crates/runmat-runtime/src/builtins/introspection/whos.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/introspection/whos.rs"
  - label: "crates/runmat-runtime/src/builtins/io/disp.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fclose.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fclose.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/feof.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/feof.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fgetl.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fgetl.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fgets.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fgets.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fileread.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fileread.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fopen.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fopen.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fread.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fread.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/frewind.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/frewind.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs"
  - label: "crates/runmat-runtime/src/builtins/io/filetext/registry.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/registry.rs"
  - label: "crates/runmat-runtime/src/builtins/io/mat/load.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/load.rs"
  - label: "crates/runmat-runtime/src/builtins/io/mat/save.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/save.rs"
  - label: "crates/runmat-runtime/src/builtins/io/net/close.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/close.rs"
  - label: "crates/runmat-runtime/src/builtins/io/net/read.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/read.rs"
  - label: "crates/runmat-runtime/src/builtins/io/net/readline.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/readline.rs"
  - label: "crates/runmat-runtime/src/builtins/io/net/write.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/write.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/movefile.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/movefile.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/rmdir.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/rmdir.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/rmpath.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/rmpath.rs"
  - label: "crates/runmat-runtime/src/builtins/io/repl_fs/savepath.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/savepath.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/csvread.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvread.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs"
  - label: "crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/char.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/char.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/compose.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/compose.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/num2str.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/num2str.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/sprintf.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/sprintf.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/str2double.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/str2double.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/string.empty.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.empty.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/string.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/strings.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/strings.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/core/strlength.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/strlength.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/lower.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/lower.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/replace.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/replace.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/strip.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strip.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/strrep.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strrep.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs"
  - label: "crates/runmat-runtime/src/builtins/strings/transform/upper.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/upper.rs"
  - label: "crates/runmat-runtime/src/console.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/console.rs"
  - label: "crates/runmat-vm/src/object/class_def.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-vm/src/object/class_def.rs"
  - label: "crates/runmat-wasm/src/runtime/filesystem/handle.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-wasm/src/runtime/filesystem/handle.rs"
  - label: "crates/runmat-wasm/src/runtime/filesystem/provider.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-wasm/src/runtime/filesystem/provider.rs"
headings:
  - level: 1
    text: "String, I/O & Filesystem Built-ins"
    id: "6.3-string-io-filesystem-built-ins"
  - level: 2
    text: "String Built-ins & Formatting"
    id: "6.3-string-built-ins-formatting"
  - level: 3
    text: "Core Conversion & Formatting"
    id: "6.3-core-conversion-formatting"
  - level: 3
    text: "String Manipulation Logic"
    id: "6.3-string-manipulation-logic"
  - level: 2
    text: "File I/O Subsystem"
    id: "6.3-file-io-subsystem"
  - level: 3
    text: "Low-Level File Access"
    id: "6.3-low-level-file-access"
  - level: 3
    text: "Tabular Data (CSV/DLM)"
    id: "6.3-tabular-data-csvdlm"
  - level: 3
    text: "MAT-file Persistence"
    id: "6.3-mat-file-persistence"
  - level: 2
    text: "Filesystem & Environment Operations"
    id: "6.3-filesystem-environment-operations"
  - level: 3
    text: "Path Expansion"
    id: "6.3-path-expansion"
  - level: 2
    text: "Network I/O"
    id: "6.3-network-io"
  - level: 2
    text: "Datetime & Duration Objects"
    id: "6.3-datetime-duration-objects"
---
# String, I/O & Filesystem Built-ins

<details>
<summary>Relevant source files</summary>

- [crates/runmat-core/src/value_metadata.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-core/src/value_metadata.rs)
- [crates/runmat-core/tests/printf_semantics.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-core/tests/printf_semantics.rs)
- [crates/runmat-runtime/src/builtins/builtins-json/datetime.json](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/datetime.json)
- [crates/runmat-runtime/src/builtins/cells/core/cell.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cell.rs)
- [crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs)
- [crates/runmat-runtime/src/builtins/cells/core/cellstr.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/cellstr.rs)
- [crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs)
- [crates/runmat-runtime/src/builtins/common/format.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/format.rs)
- [crates/runmat-runtime/src/builtins/common/tensor.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/tensor.rs)
- [crates/runmat-runtime/src/builtins/datetime/mod.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/datetime/mod.rs)
- [crates/runmat-runtime/src/builtins/duration/mod.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/duration/mod.rs)
- [crates/runmat-runtime/src/builtins/introspection/who.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/introspection/who.rs)
- [crates/runmat-runtime/src/builtins/introspection/whos.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/introspection/whos.rs)
- [crates/runmat-runtime/src/builtins/io/disp.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fclose.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fclose.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/feof.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/feof.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fgetl.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fgetl.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fgets.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fgets.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fileread.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fileread.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fopen.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fopen.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fread.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fread.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/frewind.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/frewind.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs)
- [crates/runmat-runtime/src/builtins/io/filetext/registry.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/registry.rs)
- [crates/runmat-runtime/src/builtins/io/mat/load.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/load.rs)
- [crates/runmat-runtime/src/builtins/io/mat/save.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/save.rs)
- [crates/runmat-runtime/src/builtins/io/net/close.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/close.rs)
- [crates/runmat-runtime/src/builtins/io/net/read.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/read.rs)
- [crates/runmat-runtime/src/builtins/io/net/readline.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/readline.rs)
- [crates/runmat-runtime/src/builtins/io/net/write.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/write.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/movefile.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/movefile.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/rmdir.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/rmdir.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/rmpath.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/rmpath.rs)
- [crates/runmat-runtime/src/builtins/io/repl_fs/savepath.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/repl_fs/savepath.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/csvread.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvread.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs)
- [crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs)
- [crates/runmat-runtime/src/builtins/strings/core/char.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/char.rs)
- [crates/runmat-runtime/src/builtins/strings/core/compose.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/compose.rs)
- [crates/runmat-runtime/src/builtins/strings/core/num2str.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/num2str.rs)
- [crates/runmat-runtime/src/builtins/strings/core/sprintf.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/sprintf.rs)
- [crates/runmat-runtime/src/builtins/strings/core/str2double.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/str2double.rs)
- [crates/runmat-runtime/src/builtins/strings/core/string.empty.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.empty.rs)
- [crates/runmat-runtime/src/builtins/strings/core/string.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.rs)
- [crates/runmat-runtime/src/builtins/strings/core/strings.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/strings.rs)
- [crates/runmat-runtime/src/builtins/strings/core/strlength.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/strlength.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/lower.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/lower.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/replace.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/replace.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/strip.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strip.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/strrep.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strrep.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs)
- [crates/runmat-runtime/src/builtins/strings/transform/upper.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/transform/upper.rs)
- [crates/runmat-runtime/src/console.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/console.rs)
- [crates/runmat-vm/src/object/class_def.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-vm/src/object/class_def.rs)
- [crates/runmat-wasm/src/runtime/filesystem/handle.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-wasm/src/runtime/filesystem/handle.rs)
- [crates/runmat-wasm/src/runtime/filesystem/provider.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-wasm/src/runtime/filesystem/provider.rs)

</details>

This section covers the implementation and architecture of RunMat's string manipulation, file I/O, and filesystem management subsystems. These built-ins bridge the gap between the high-performance numeric VM and external data sources, providing MATLAB-compatible interfaces for text processing, structured data persistence (MAT/CSV/JSON), and virtualized filesystem operations.

## String Built-ins & Formatting

RunMat implements string built-ins by wrapping Rust's standard string handling with MATLAB-specific semantics, such as 1-based indexing and specific whitespace trimming rules.

### Core Conversion & Formatting

The `string` and `char` built-ins handle conversion from numeric and logical types to text. While these operations typically occur on the CPU, the system includes `GpuOpKind::Custom("conversion")` logic to automatically gather GPU tensors to host memory before formatting [crates/runmat-runtime/src/builtins/strings/core/string.rs #105-118](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.rs#L105-L118)

Formatting is primarily driven by `sprintf`, which supports standard MATLAB format specifiers. It is utilized by `fprintf` and `disp` for output rendering [crates/runmat-runtime/src/builtins/io/disp.rs #142-148](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs#L142-L148)

### String Manipulation Logic

| Built-in | Implementation Role | Source |
| --- | --- | --- |
| string | Converts arrays to MATLAB string objects; handles UTF-8 encoding. | crates/runmat-runtime/src/builtins/strings/core/string.rs#132-141 |
| sprintf | Core formatting engine for strings and numeric data. | crates/runmat-runtime/src/builtins/strings/core/sprintf.rs#1-20 |
| strtrim | Removes leading/trailing whitespace from character arrays. | crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs#1-15 |
| strip | Modern string array whitespace/character removal. | crates/runmat-runtime/src/builtins/strings/transform/strip.rs#1-15 |

String Conversion Data Flow The following diagram illustrates how various data types are unified into the string representation.

Sources: [crates/runmat-runtime/src/builtins/strings/core/string.rs #142-179](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/strings/core/string.rs#L142-L179) [crates/runmat-runtime/src/builtins/io/disp.rs #134-148](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs#L134-L148)

## File I/O Subsystem

RunMat provides a robust file I/O layer that supports both low-level file descriptors and high-level tabular data import/export.

### Low-Level File Access

Low-level I/O mimics the C-style interface of MATLAB (`fopen`, `fread`, `fwrite`, `fclose`).

- File Registry: A central registry manages `fid` (File Identifiers). Standard IDs 1 and 2 are reserved for `stdout` and `stderr` [crates/runmat-core/tests/printf_semantics.rs #72-81](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-core/tests/printf_semantics.rs#L72-L81)
- Virtualized Access: All file operations pass through `runmat-filesystem`, allowing the same code to run on native OS files or browser-based virtual filesystems (OPFS/IndexedDB) [crates/runmat-runtime/src/builtins/io/filetext/fopen.rs #1-20](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fopen.rs#L1-L20)

### Tabular Data (CSV/DLM)

The `csvread`, `dlmread`, and `readmatrix` functions implement high-level parsing.

- Range Support: Supports A1-style range strings and numeric range vectors [crates/runmat-runtime/src/builtins/io/tabular/csvread.rs #64-93](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvread.rs#L64-L93)
- Delimiter Logic: `dlmread` accepts character delimiters or numeric ASCII codes [crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs #5-7](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs#L5-L7)

### MAT-file Persistence

RunMat implements a custom MAT-file codec for `load` and `save` operations.

- `load`: Supports selective variable loading and regular expression filtering [crates/runmat-runtime/src/builtins/io/mat/load.rs #96-122](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/load.rs#L96-L122)
- `save`: Can persist the entire workspace or specific variables. It supports the `-struct` flag to save fields of a struct as individual variables [crates/runmat-runtime/src/builtins/io/mat/save.rs #106-132](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/save.rs#L106-L132)

File I/O Entity Mapping

Sources: [crates/runmat-runtime/src/builtins/io/filetext/fopen.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/filetext/fopen.rs) [crates/runmat-runtime/src/builtins/io/mat/load.rs #16-19](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/mat/load.rs#L16-L19) [crates/runmat-runtime/src/builtins/io/tabular/csvread.rs #15](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/csvread.rs#L15-L15)

## Filesystem & Environment Operations

Filesystem built-ins provide directory management and path manipulation. These operations are essential for script portability across different host environments.

| Function | Description | Source |
| --- | --- | --- |
| cd | Changes the current working directory in the VFS. | crates/runmat-runtime/src/builtins/io/repl_fs/mod.rs |
| ls / dir | Lists files; returns a struct array with metadata (name, date, bytes). | crates/runmat-runtime/src/builtins/io/repl_fs/mod.rs |
| mkdir | Creates a new directory via the active filesystem provider. | crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs#1-10 |
| copyfile | Copies files, handling path expansion (e.g., ~ home directory). | crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs#1-15 |
| whos | Introspects variables in the current workspace or a MAT-file. | crates/runmat-runtime/src/builtins/introspection/whos.rs#105-126 |

### Path Expansion

Built-ins like `dlmwrite` and `copyfile` use `expand_user_path` to resolve platform-specific paths and MATLAB-specific conventions before passing them to the low-level `vfs` [crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs #22](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs#L22-L22)

## Network I/O

RunMat supports basic network operations via a socket-like interface, often used for instrument control or distributed logging.

- `write` / `read`: Functions in `crates/runmat-runtime/src/builtins/io/net/` provide asynchronous network access [crates/runmat-runtime/src/builtins/io/net/write.rs #1-10](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/write.rs#L1-L10)
- Cleanup: `net/close.rs` ensures sockets are properly released to the host OS [crates/runmat-runtime/src/builtins/io/net/close.rs #1-5](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/net/close.rs#L1-L5)

## Datetime & Duration Objects

RunMat treats `datetime` and `duration` as first-class objects implemented via the `runmat-vm` object system.

- Internal Storage: `datetime` objects store a serial date number (`__serial`) and a format string [crates/runmat-runtime/src/builtins/datetime/mod.rs #18-20](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/datetime/mod.rs#L18-L20)
- Display Logic: `disp` detects these classes and calls specialized rendering functions like `datetime_display_text` to provide human-readable output [crates/runmat-runtime/src/builtins/io/disp.rs #156-179](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs#L156-L179)
- Broadcasting: Duration arithmetic (e.g., adding a duration array to a datetime array) follows MATLAB's standard broadcasting rules [crates/runmat-runtime/src/builtins/duration/mod.rs #70-85](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/duration/mod.rs#L70-L85)

Sources: [crates/runmat-runtime/src/builtins/datetime/mod.rs #1-50](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/datetime/mod.rs#L1-L50) [crates/runmat-runtime/src/builtins/duration/mod.rs #1-47](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/duration/mod.rs#L1-L47) [crates/runmat-runtime/src/builtins/io/disp.rs #150-179](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/io/disp.rs#L150-L179)

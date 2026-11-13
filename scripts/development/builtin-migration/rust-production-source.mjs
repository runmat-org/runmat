function blankRange(characters, start, end) {
  for (let index = start; index < end; index += 1) {
    if (characters[index] !== "\n" && characters[index] !== "\r") characters[index] = " ";
  }
}

// Preserve source offsets, newlines, and Rust punctuation while removing text
// that cannot own an item. Architecture guards need lexical boundaries, not a
// second Rust parser.
export function maskRustTriviaAndLiterals(text) {
  const characters = text.split("");
  for (let index = 0; index < text.length;) {
    if (text.startsWith("//", index)) {
      const end = text.indexOf("\n", index + 2);
      const limit = end < 0 ? text.length : end;
      blankRange(characters, index, limit);
      index = limit;
      continue;
    }
    if (text.startsWith("/*", index)) {
      let depth = 1;
      let cursor = index + 2;
      while (cursor < text.length && depth > 0) {
        if (text.startsWith("/*", cursor)) { depth += 1; cursor += 2; }
        else if (text.startsWith("*/", cursor)) { depth -= 1; cursor += 2; }
        else cursor += 1;
      }
      blankRange(characters, index, cursor);
      index = cursor;
      continue;
    }
    const raw = text.slice(index).match(/^(?:br|cr|r)(#*)"/);
    if (raw) {
      const terminator = `"${raw[1]}`;
      const contentStart = index + raw[0].length;
      const close = text.indexOf(terminator, contentStart);
      const end = close < 0 ? text.length : close + terminator.length;
      blankRange(characters, index, end);
      index = end;
      continue;
    }
    if (text[index] === '"') {
      let cursor = index + 1;
      while (cursor < text.length) {
        if (text[cursor] === "\\") cursor += 2;
        else if (text[cursor++] === '"') break;
      }
      blankRange(characters, index, cursor);
      index = cursor;
      continue;
    }
    if (text[index] === "'") {
      if (/[A-Za-z_]/.test(text[index + 1] ?? "") && text[index + 2] !== "'") {
        index += 1;
        continue;
      }
      let cursor = index + 1;
      while (cursor < text.length && text[cursor] !== "\n" && text[cursor] !== "\r") {
        if (text[cursor] === "\\") cursor += 2;
        else if (text[cursor++] === "'") break;
      }
      if (text[cursor - 1] === "'") {
        blankRange(characters, index, cursor);
        index = cursor;
        continue;
      }
    }
    index += 1;
  }
  return characters.join("");
}

function matchingDelimiter(text, start, open, close) {
  let depth = 0;
  for (let cursor = start; cursor < text.length; cursor += 1) {
    if (text[cursor] === open) depth += 1;
    else if (text[cursor] === close && --depth === 0) return cursor + 1;
  }
  return text.length;
}

function outerAttributeEnd(text, start) {
  if (text[start] !== "#") return null;
  let cursor = start + 1;
  while (/\s/.test(text[cursor] ?? "")) cursor += 1;
  if (text[cursor] === "!" || text[cursor] !== "[") return null;
  return matchingDelimiter(text, cursor, "[", "]");
}

function cfgRequiresTest(expression) {
  const compact = expression.replace(/\s/g, "");
  if (compact === "test") return true;
  for (const operator of ["all", "any"]) {
    const prefix = `${operator}(`;
    if (!compact.startsWith(prefix) || !compact.endsWith(")")) continue;
    const arguments_ = [];
    let depth = 0;
    let start = prefix.length;
    for (let cursor = start; cursor < compact.length - 1; cursor += 1) {
      if (compact[cursor] === "(") depth += 1;
      else if (compact[cursor] === ")") depth -= 1;
      else if (compact[cursor] === "," && depth === 0) {
        arguments_.push(compact.slice(start, cursor));
        start = cursor + 1;
      }
    }
    arguments_.push(compact.slice(start, -1));
    return operator === "all"
      ? arguments_.some(cfgRequiresTest)
      : arguments_.every(cfgRequiresTest);
  }
  return false;
}

function attributeIsTestOnly(text, start, end) {
  const body = text.slice(text.indexOf("[", start) + 1, end - 1);
  const match = body.match(/^\s*cfg\s*\((.*)\)\s*$/s);
  return match ? cfgRequiresTest(match[1]) : false;
}

function testItemEnd(text, start) {
  let cursor = start;
  while (/\s/.test(text[cursor] ?? "")) cursor += 1;
  while (text[cursor] === "#") {
    const end = outerAttributeEnd(text, cursor);
    if (end === null) break;
    cursor = end;
    while (/\s/.test(text[cursor] ?? "")) cursor += 1;
  }
  const head = text.slice(cursor, cursor + 160);
  const semicolonItem = /^(?:pub(?:\s*\([^)]*\))?\s+)?(?:use|static|type|const\s+(?!fn\b)|extern\s+crate)\b/.test(head);
  let parentheses = 0;
  let brackets = 0;
  let braces = 0;
  for (; cursor < text.length; cursor += 1) {
    const character = text[cursor];
    if (character === "(") parentheses += 1;
    else if (character === ")") parentheses -= 1;
    else if (character === "[") brackets += 1;
    else if (character === "]") brackets -= 1;
    else if (character === "{" && parentheses === 0 && brackets === 0) {
      if (semicolonItem) braces += 1;
      else return matchingDelimiter(text, cursor, "{", "}");
    } else if (character === "}" && semicolonItem && braces > 0) braces -= 1;
    else if (character === ";" && parentheses === 0 && brackets === 0 && braces === 0) {
      return cursor + 1;
    }
  }
  return text.length;
}

export function productionRust(text) {
  const code = maskRustTriviaAndLiterals(text);
  const characters = text.split("");
  for (let cursor = 0; cursor < code.length; cursor += 1) {
    if (code[cursor] !== "#") continue;
    const attributeEnd = outerAttributeEnd(code, cursor);
    if (attributeEnd === null || !attributeIsTestOnly(code, cursor, attributeEnd)) continue;
    const itemEnd = testItemEnd(code, attributeEnd);
    blankRange(characters, cursor, itemEnd);
    cursor = itemEnd - 1;
  }
  return characters.join("");
}

export function rustFunctionRegions(text) {
  const code = maskRustTriviaAndLiterals(text);
  const regions = [];
  const declarations = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)[^;{]*\{/g;
  for (const match of code.matchAll(declarations)) {
    const start = match.index;
    const bodyStart = start + match[0].lastIndexOf("{");
    let depth = 0;
    for (let cursor = bodyStart; cursor < code.length; cursor += 1) {
      if (code[cursor] === "{") depth += 1;
      else if (code[cursor] === "}") depth -= 1;
      if (depth === 0) {
        regions.push({ name: match[1], start, end: cursor + 1 });
        break;
      }
    }
  }
  return regions;
}

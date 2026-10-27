// Runtime boundary checks derived from the generated canonical contracts.
const ts = require('typescript');
const { aliasGuard, aliasGuardHelpers } = require('./wire-integers.cjs');

function renderGuards(body, roots) {
  const source = ts.createSourceFile('contracts.ts', body, ts.ScriptTarget.Latest, true);
  const declarations = new Map(source.statements.filter((s) => s.name).map((s) => [s.name.text, s]));
  const reachable = new Map();
  const integers = new Set();
  const object = (members, value, parameters) => {
    const checks = members.map((member) => {
      if (!ts.isPropertySignature(member) || !member.type) throw new Error('Unsupported guard property');
      const key = JSON.stringify(member.name.text);
      const field = `${value}[${key}]`;
      const check = expression(member.type, field, parameters);
      return member.questionToken ? `(${field} === undefined || ${check})` : check;
    });
    return `(isRecord(${value}) && ${checks.join(' && ') || 'true'})`;
  };
  const visit = (name) => {
    if (reachable.has(name)) return;
    const declaration = declarations.get(name);
    if (!declaration) throw new Error(`Unresolved guard contract ${name}`);
    reachable.set(name, null);
    const parameters = new Map((declaration.typeParameters ?? []).map((parameter) => [parameter.name.text, `check${parameter.name.text}`]));
    const check = ts.isInterfaceDeclaration(declaration)
      ? object(declaration.members, 'value', parameters)
      : expression(declaration.type, 'value', parameters);
    const arguments = [...parameters.values()].map((parameter) => `, ${parameter}: (value: unknown) => boolean`).join('');
    reachable.set(name, `function check${name}(value: unknown${arguments}): boolean { return ${check}; }`);
  };
  const expression = (type, value, parameters = new Map()) => {
    switch (type.kind) {
      case ts.SyntaxKind.StringKeyword: return `typeof ${value} === 'string'`;
      case ts.SyntaxKind.NumberKeyword: return `(typeof ${value} === 'number' && Number.isFinite(${value}))`;
      case ts.SyntaxKind.BooleanKeyword: return `typeof ${value} === 'boolean'`;
    }
    if (ts.isParenthesizedTypeNode(type)) return expression(type.type, value, parameters);
    if (ts.isLiteralTypeNode(type)) return `${value} === ${type.literal.getText(source)}`;
    if (ts.isUnionTypeNode(type)) return `(${type.types.map((t) => expression(t, value, parameters)).join(' || ')})`;
    if (ts.isIntersectionTypeNode(type)) return `(${type.types.map((t) => expression(t, value, parameters)).join(' && ')})`;
    if (ts.isTypeLiteralNode(type)) return object(type.members, value, parameters);
    if (ts.isArrayTypeNode(type)) return `(Array.isArray(${value}) && ${value}.every((entry: unknown) => ${expression(type.elementType, 'entry', parameters)}))`;
    if (ts.isTupleTypeNode(type)) return `(Array.isArray(${value}) && ${value}.length === ${type.elements.length} && ${type.elements.map((t, i) => expression(t, `${value}[${i}]`, parameters)).join(' && ')})`;
    if (ts.isTypeReferenceNode(type) && type.typeName.getText(source) === 'Record'
      && type.typeArguments?.length === 2 && type.typeArguments[0].kind === ts.SyntaxKind.StringKeyword) {
      return `(isRecord(${value}) && Object.values(${value}).every((entry: unknown) => ${expression(type.typeArguments[1], 'entry', parameters)}))`;
    }
    if (ts.isTypeReferenceNode(type)) {
      const name = type.typeName.getText(source);
      if (name === 'Uint8Array' && !type.typeArguments?.length) return `${value} instanceof Uint8Array`;
      const integer = !type.typeArguments?.length && aliasGuard(name, value);
      if (integer) { integers.add(name); return integer; }
      if (parameters.has(name)) return `${parameters.get(name)}(${value})`;
      const declaration = declarations.get(name);
      if ((declaration?.typeParameters?.length ?? 0) !== (type.typeArguments?.length ?? 0)) throw new Error(`Guard type arity mismatch: ${type.getText(source)}`);
      visit(name);
      const arguments = (type.typeArguments ?? []).map((argument) => `, (entry: unknown) => ${expression(argument, 'entry', parameters)}`).join('');
      return `check${name}(${value}${arguments})`;
    }
    throw new Error(`Unsupported guard type: ${type.getText(source)}`);
  };
  roots.forEach(visit);
  return [
    '// @generated from canonical FEA semantic contracts. Do not edit by hand.',
    `import type { ${roots.join(', ')} } from './fea-semantic-contracts.js';`,
    'function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null && !Array.isArray(value); }',
    ...aliasGuardHelpers(integers),
    ...reachable.values(),
    ...roots.map((name) => `export function is${name}(value: unknown): value is ${name} { return check${name}(value); }`),
    '',
  ].join('\n');
}

module.exports = { renderGuards };

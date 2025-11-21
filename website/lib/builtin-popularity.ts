import type { DisplayCategory } from './builtin-utils';

// Popular/common MATLAB functions by JSON category
// These are the functions most commonly used in MATLAB documentation, tutorials, and examples
const POPULAR_FUNCTIONS: Record<string, string[]> = {
  'cells/core': ['cell', 'cell2mat', 'cellfun', 'cellstr'],
  'containers/map': ['containers.Map'],
  'structs/core': ['struct', 'fieldnames', 'getfield', 'setfield'],
  'array/creation': ['zeros', 'ones', 'eye', 'nan', 'inf', 'colon', 'linspace', 'logspace'],
  'array/shape': ['reshape', 'squeeze', 'permute', 'cat', 'horzcat', 'vertcat'],
  'array/introspection': ['size', 'length', 'numel', 'ndims'],
  'array/indexing': ['end', 'find', 'ind2sub', 'sub2ind'],
  'array/sorting_sets': ['sort', 'sortrows', 'unique', 'ismember', 'diff'],
  'math/elementwise': ['abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan'],
  'math/reduction': ['sum', 'mean', 'std', 'min', 'max', 'any', 'all'],
  'math/trigonometry': ['sin', 'cos', 'tan', 'asin', 'acos', 'atan'],
  'math/linalg/ops': ['dot', 'cross', 'norm'],
  'math/linalg/solve': ['inv', 'mldivide', 'mrdivide'],
  'math/linalg/factor': ['lu', 'qr', 'chol', 'svd'],
  'logical/rel': ['eq', 'ne', 'lt', 'le', 'gt', 'ge'],
  'logical/bit': ['and', 'or', 'xor', 'not'],
  'stats/summary': ['mean', 'std', 'median', 'var'],
  'io/filetext': ['fopen', 'fclose', 'fread', 'fwrite', 'fprintf'],
  'io/tabular': ['readtable', 'writetable', 'readmatrix', 'writematrix'],
  'io/mat': ['load', 'save'],
  'general': ['if', 'for', 'while', 'function', 'return', 'break', 'continue'],
  'introspection': ['class', 'isa', 'which', 'exist'],
  'diagnostics': ['assert', 'error', 'warning'],
  'timing': ['tic', 'toc'],
  'image/filters': ['filter2', 'fspecial', 'imfilter'],
};

// Get popular functions for a category
export function getPopularFunctions(category: DisplayCategory): string[] {
  return POPULAR_FUNCTIONS[category] || [];
}

// Legacy export for backward compatibility
export const POPULAR_FUNCTIONS_BY_CATEGORY = new Proxy({} as Record<DisplayCategory, string[]>, {
  get: (_, prop: string) => getPopularFunctions(prop),
});

// Map display category to JSON categories for filtering
// Since display category IS the JSON category now, just return it
export function getJsonCategoriesForDisplayCategory(displayCategory: DisplayCategory): string[] {
  return [displayCategory];
}


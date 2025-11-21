import type { Builtin } from './builtins';
export type { BuiltinBadge } from './badge-utils';
export { getBuiltinBadges } from './badge-utils';

// DisplayCategory is now the JSON category path directly (e.g., "array/creation")
export type DisplayCategory = string;

export function getDisplayCategory(builtin: Builtin): DisplayCategory {
  const categories = builtin.category || [];
  
  // Return the first category from JSON (functions typically have one primary category)
  // If multiple categories exist, use the first one
  if (categories.length > 0) {
    return categories[0];
  }
  
  // Fallback for functions without categories
  return 'general';
}


import type { Builtin } from './builtins';

export type BuiltinBadge = 'GPU' | 'Fusion' | 'BLAS/LAPACK';

// DisplayCategory is now the JSON category path directly (e.g., "array/creation")
export type DisplayCategory = string;

export function getBuiltinBadges(builtin: Builtin): BuiltinBadge[] {
  const badges: BuiltinBadge[] = [];
  const keywords = builtin.keywords || [];
  const categories = builtin.category || [];
  const keywordStr = keywords.join(' ').toLowerCase();
  const categoryStr = categories.join(' ').toLowerCase();

  // GPU badge: keyword contains "gpu" OR category includes "acceleration/gpu"
  if (keywordStr.includes('gpu') || categoryStr.includes('acceleration/gpu')) {
    badges.push('GPU');
  }

  // Fusion badge: keyword contains "fusion"
  if (keywordStr.includes('fusion')) {
    badges.push('Fusion');
  }

  // BLAS/LAPACK badge: category includes "math/linalg/*" OR keyword contains "blas"/"lapack"
  if (categories.some(c => c.startsWith('math/linalg/')) || 
      keywordStr.includes('blas') || keywordStr.includes('lapack')) {
    badges.push('BLAS/LAPACK');
  }

  return badges;
}

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


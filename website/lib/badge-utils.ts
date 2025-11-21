import type { Builtin } from './builtins';

export type BuiltinBadge = 'GPU' | 'Fusion' | 'BLAS/LAPACK';

export function getBuiltinBadges(builtin: Builtin): BuiltinBadge[] {
  const badges: BuiltinBadge[] = [];
  const keywords = builtin.keywords || [];
  const categories = builtin.category || [];
  const keywordStr = keywords.join(' ').toLowerCase();
  const categoryStr = categories.join(' ').toLowerCase();

  if (keywordStr.includes('gpu') || categoryStr.includes('acceleration/gpu')) {
    badges.push('GPU');
  }

  if (keywordStr.includes('fusion')) {
    badges.push('Fusion');
  }

  if (
    categories.some((c) => c.startsWith('math/linalg/')) ||
    keywordStr.includes('blas') ||
    keywordStr.includes('lapack')
  ) {
    badges.push('BLAS/LAPACK');
  }

  return badges;
}

export function builtinHasGpuSupport(builtin: Builtin): boolean {
  return getBuiltinBadges(builtin).includes('GPU');
}


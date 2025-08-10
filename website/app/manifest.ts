import { MetadataRoute } from 'next'

export const dynamic = 'force-static'

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'RunMat - Modern MATLAB Runtime',
    short_name: 'RunMat',
    description: 'Fast, free, modern MATLAB runtime with Jupyter kernel, BLAS/LAPACK, beautiful plotting, and JIT compilation',
    start_url: '/',
    display: 'standalone',
    background_color: '#0b1220',
    theme_color: '#3ea7fd',
    icons: [
      {
        src: '/favicon.ico',
        sizes: 'any',
        type: 'image/x-icon',
      },
      {
        src: '/favicon.svg',
        sizes: 'any',
        type: 'image/svg+xml',
      },
    ],
  }
}

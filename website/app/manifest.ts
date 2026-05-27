import { MetadataRoute } from 'next'

export const dynamic = 'force-static'

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'RunMat - Modern MATLAB Runtime',
    short_name: 'RunMat',
    description: 'Pre-release MATLAB-style runtime with BLAS/LAPACK, JIT compilation, GPU acceleration, and plotting',
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

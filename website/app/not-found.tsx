import Link from 'next/link'
 
export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="max-w-md w-full space-y-8 text-center">
        <div>
          <h2 className="mt-6 text-3xl font-bold text-foreground">
            Page not found
          </h2>
          <p className="mt-2 text-[0.938rem] text-foreground">
            Sorry, we couldn&apos;t find the page you&apos;re looking for.
          </p>
        </div>
        <div>
          <Link 
            href="/"
            className="group relative w-full flex justify-center py-2 px-4 border-0 text-sm font-semibold rounded-none text-white bg-[hsl(var(--brand))] hover:bg-[hsl(var(--brand))]/90 shadow-none"
          >
            Return Home
          </Link>
        </div>
      </div>
    </div>
  )
}
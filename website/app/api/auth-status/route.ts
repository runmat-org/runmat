import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const AUTH_COOKIE_NAME = "runmat_auth";

export async function GET() {
  const authenticated = (await cookies()).get(AUTH_COOKIE_NAME)?.value === "1";
  return NextResponse.json({ authenticated });
}

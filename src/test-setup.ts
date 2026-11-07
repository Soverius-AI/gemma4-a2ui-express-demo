import { vi } from "vitest";

// Code that resolves URLs against the page reads location.origin; give
// Node-based unit tests the dev server's origin.
vi.stubGlobal("location", new URL("http://localhost:4210/"));

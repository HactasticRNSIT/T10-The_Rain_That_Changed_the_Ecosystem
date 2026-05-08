import { createServerFn } from "@tanstack/react-start";
import { compute, type EcoInputs } from "./resilience-engine";

export const predictEco = createServerFn({ method: "POST" })
  .inputValidator((d: EcoInputs) => d)
  .handler(async ({ data }) => compute(data));

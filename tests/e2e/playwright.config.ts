import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: ".",
  use: { headless: true, baseURL: "http://127.0.0.1:8080" },
  webServer: {
    command: "uv run python -m lenet1_physical.scripts.run_twin_dev",
    url: "http://127.0.0.1:8080/healthz",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});

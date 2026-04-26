import { test, expect } from "@playwright/test";

test("twin page loads with scene canvas", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#scene")).toBeVisible();
});

test("ws frames arrive and seq increments", async ({ page }) => {
  await page.goto("/");
  const seq = page.locator("#seq");
  await expect(seq).not.toHaveText("--", { timeout: 5_000 });
  const first = await seq.textContent();
  await page.waitForTimeout(700);
  const second = await seq.textContent();
  expect(Number(second)).toBeGreaterThan(Number(first));
});

test("clicking Sample issues POST /sample", async ({ page }) => {
  let posted = false;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().endsWith("/sample")) posted = true;
  });
  await page.goto("/");
  await page.click("#btn-sample");
  await expect.poll(() => posted, { timeout: 3_000 }).toBe(true);
});

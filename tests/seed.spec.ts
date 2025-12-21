import { test, expect } from '@playwright/test';

test('seed test - navigate to login page', async ({ page }) => {
  await page.goto('http://leaftaps.com/opentaps');
  await expect(page).toHaveTitle(/Leaftaps/);
});

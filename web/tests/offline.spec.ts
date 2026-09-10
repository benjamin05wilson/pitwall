import { test, expect } from '@playwright/test'

test.beforeEach(async ({ context }) => {
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.hostname !== '127.0.0.1') throw new Error(`Offline demo attempted external request: ${url.origin}`)
    return route.continue()
  })
})

test('no weights: recommendation, candidates and frontier survive heatmap failure', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Optional heatmap unavailable.', { exact: false })).toBeVisible()
  await expect(page.locator('.strat-label')).toBeVisible()
  await expect(page.locator('tbody tr')).toHaveCount(3)
  await expect(page.getByText('Risk / reward frontier')).toBeVisible()
  await expect(page.locator('.pill').filter({ hasText: /Python/ })).toBeVisible()
  await page.screenshot({ path: 'test-results/offline-strategy.png', fullPage: true })
  await page.getByRole('button', { name: 'robust', exact: true }).click()
  await expect(page.locator('.reason')).toContainText('Optimised for robust')
  await expect(page.locator('tbody tr')).toHaveCount(3)
})

test('input change clears stale results; failed core request is visible', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.strat-label')).toBeVisible()
  await page.route('**/api/optimize', route => route.fulfill({ status: 503, body: '{}' }))
  await page.getByRole('button', { name: 'expected', exact: true }).click()
  await expect(page.locator('.strat-label')).toHaveCount(0)
  await expect(page.getByRole('alert')).toContainText('Recommendation failed')
  await expect(page.locator('tbody tr')).toHaveCount(0)
})

test('late obsolete response cannot overwrite the new objective', async ({ page }) => {
  await page.route('**/api/optimize', async route => {
    const response = await route.fetch()
    if (route.request().postDataJSON().objective === 'podium') await new Promise(r => setTimeout(r, 1500))
    await route.fulfill({ response })
  })
  await page.goto('/')
  await page.waitForRequest('**/api/optimize')
  await page.getByRole('button', { name: 'win', exact: true }).click()
  await expect(page.locator('.reason')).toContainText('Optimised for win')
  await page.waitForTimeout(1800)
  await expect(page.locator('.reason')).toContainText('Optimised for win')
})

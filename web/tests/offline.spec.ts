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
  // Optional recording dwell is presentation time, never a correctness wait.
  if (process.env.PITWALL_RECORD === '1') await page.waitForTimeout(4000)
  await page.getByRole('button', { name: 'robust', exact: true }).click()
  await expect(page.locator('.reason')).toContainText('Optimised for robust')
  await expect(page.locator('tbody tr')).toHaveCount(3)
  if (process.env.PITWALL_RECORD === '1') await page.waitForTimeout(4000)
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

test('late obsolete response cannot overwrite the new objective', async ({ page, request }) => {
  // Use a real bounded API result, then control transport ordering explicitly.
  // Model/runtime speed must not determine whether the race is exercised.
  const seedResponse = await request.post('/api/optimize', { data: { scenarios: 2 } })
  expect(seedResponse.ok()).toBeTruthy()
  const seed = await seedResponse.json()
  let releaseOld!: () => void
  let observedOld!: () => void
  const oldArrived = new Promise<void>(resolve => { observedOld = resolve })
  const oldReleased = new Promise<void>(resolve => { releaseOld = resolve })
  await page.route('**/api/optimize', async route => {
    const objective = route.request().postDataJSON().objective
    if (objective === 'podium') { observedOld(); await oldReleased }
    await route.fulfill({ json: { ...seed, objective } })
  })
  await page.goto('/')
  await oldArrived
  await page.getByRole('button', { name: 'win', exact: true }).click()
  await expect(page.locator('.reason')).toContainText('Optimised for win')
  const oldResponse = page.waitForResponse(response => response.url().endsWith('/api/optimize') && response.request().postDataJSON().objective === 'podium')
  releaseOld()
  await (await oldResponse).finished()
  // Flush response processing and React rendering after the obsolete result.
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  await expect(page.locator('.reason')).toContainText('Optimised for win')
  await expect(page.locator('.reason')).not.toContainText('Optimised for podium')
})

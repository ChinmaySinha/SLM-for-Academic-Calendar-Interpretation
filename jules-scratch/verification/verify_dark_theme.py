from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Go to the local Flask app
    page.goto("http://127.0.0.1:5001/")

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/dark_theme_verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)

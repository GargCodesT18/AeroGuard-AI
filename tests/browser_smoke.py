"""Run against the local Flask server with Playwright and Microsoft Edge."""

import os
import re

from playwright.sync_api import expect, sync_playwright

BASE_URL = os.getenv("AEROGUARD_TEST_URL", "http://127.0.0.1:8000")


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: (errors.append(dialog.message), dialog.dismiss()))

        def load(path):
            response = page.goto(f"{BASE_URL}/{path}", wait_until="domcontentloaded")
            assert response is not None and response.ok, (path, response.status if response else None)
            expect(page).to_have_title(re.compile("AeroGuard AI"))

        load("")
        expect(page.locator("#dataset-select")).to_have_value("FD001")
        page.locator("#dataset-select").select_option("FD003")
        expect(page.locator("#engine-select option")).to_have_count(100)

        load("Live_Prediction_Panel.html")
        for dataset, count in (("FD001", 100), ("FD002", 259), ("FD003", 100), ("FD004", 248)):
            page.locator(f"#pill-{dataset}").click()
            expect(page.locator("#engine-select option")).to_have_count(count)
            with page.expect_response(re.compile(r"/api/engine-history\?")) as history:
                page.locator("#run-btn").click()
            assert history.value.ok
            expect(page.locator("#engine-id-display")).to_contain_text(dataset)
            expect(page.locator("#uncertainty")).to_contain_text("validation-scaled")
            page.wait_for_function("Chart.getChart('history-chart') && Chart.getChart('sensor-chart')")

        load("Model_Insights.html")
        expect(page.locator('[data-kpi="val-rmse"]')).to_have_text(re.compile(r"^\d+\.\d+$"))
        expect(page.locator('[data-kpi="latent-dim"]')).to_have_text("64")
        expect(page.get_by_text("Confidence Score", exact=True)).to_have_count(0)

        load("Model_Details_Architecture.html")
        expect(page.locator("#uncertainty-svg path")).to_have_count(2)
        expect(page.get_by_text("[50, 24]", exact=True)).to_be_visible()
        expect(page.get_by_text("64 Units", exact=True)).to_have_count(2)
        expect(page.get_by_text("Linear", exact=True)).to_be_visible()
        expect(page.get_by_text("RUL (cycles)", exact=True)).to_be_visible()
        expect(page.locator("#spec-epochs")).to_have_text("20 Epochs")

        load("Data_Explorer.html")
        expect(page.locator("#explorer-count")).to_contain_text("1–25 of 100")
        page.locator("#explorer-next").click()
        expect(page.locator("#explorer-count")).to_contain_text("26–50 of 100")

        load("API_Deployment_Demo.html")
        expect(page.locator("#api-demo-response-block")).to_contain_text("predictionInterval")
        expect(page.locator("#api-interval-value")).to_contain_text("cycles")
        expect(page.locator("#api-measured-latency")).to_have_text(re.compile(r"^\d+\.\d ms$"))
        expect(page.get_by_text("Not benchmarked", exact=True)).to_be_visible()

        load("Model_Pipeline_Diagram.html")
        expect(page.get_by_text("Prediction intervals (90% target)", exact=True)).to_be_visible()
        assert not errors, errors
        browser.close()
    print("Playwright passed for all seven dashboard pages, four datasets and live uncertainty")


if __name__ == "__main__":
    main()

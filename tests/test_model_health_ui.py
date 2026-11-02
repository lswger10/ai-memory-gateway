from test_dashboard_model_settings import browser, dashboard


def test_saved_profile_can_be_tested_and_enabled_with_one_explicit_call(dashboard):
    page, responses, writes = dashboard
    def check(route):
        writes.append(("/api/model-health", route.request.post_data_json))
        responses["/api/model-profiles"]["profiles"][2]["test_status"] = "passed"
        route.fulfill(json={"profile_id": "untested", "profile_revision": 1, "status": "verified"})
    page.route("**/api/model-health", check)
    page.locator('[data-health-profile-id="untested"]').click()
    page.wait_for_function("document.querySelector('#models-msg').textContent.includes('测试通过')")
    assert writes == [("/api/model-health", {"profile_id": "untested", "profile_revision": 1,
        "confirm_provider_charges": True})]
    assert "passed" in page.locator('[data-health-profile-id="untested"]').locator('..').inner_text()


def test_new_profile_does_not_inherit_provider_route_headers(dashboard):
    page, _, _ = dashboard
    page.evaluate("editModelProfile('claude')")
    page.fill('#model-profile-headers', '{"X-OfoxAI-Provider-Type":"bedrock"}')
    page.evaluate("editModelProfile()")
    assert page.input_value('#model-profile-headers') == ''

from locust import HttpUser, task, between


class MiBiciUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_demand(self):
        payload = {
            "departures": 15,
            "arrivals": 12,
            "latitud": 20.6736,
            "longitud": -103.344,
            "capacidad": 30,
            "day_of_week": 2,
            "day_of_month": 19,
            "month": 11,
            "is_weekend": 0,
            "day_of_week_sin": 0.9749,
            "day_of_week_cos": -0.2225,
            "month_sin": -0.5,
            "month_cos": 0.8660,
            "departures_lag_1": 14,
            "departures_lag_7": 16,
            "departures_mean_prev_7d": 15.2,
            "departures_mean_prev_28d": 14.8,
            "arrivals_lag_1": 13,
            "arrivals_lag_7": 11,
            "arrivals_mean_prev_7d": 12.5,
            "arrivals_mean_prev_28d": 12.1,
        }

        with self.client.post(
            "/predict/demand",
            json=payload,
            name="/predict/demand",
            catch_response=True,
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text}"
                )
                return

            try:
                data = response.json()

                if "predicted_trips_next_day" not in data:
                    response.failure("Missing prediction field")
                elif "model_name" not in data:
                    response.failure("Missing model_name")
                elif "model_version" not in data:
                    response.failure("Missing model_version")
                else:
                    response.success()

            except ValueError:
                response.failure("Invalid JSON response")
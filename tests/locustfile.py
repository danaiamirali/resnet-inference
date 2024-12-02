from locust import HttpUser, task, between

class ImageClassifierUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        with open("cat.jpg", "rb") as image:
            # Send a multipart form-data request
            response = self.client.post(
                "/predict",
                files={"file": ("cat.jpg", image, "image/jpeg")},
            )
            print(response.text)  # Debug response


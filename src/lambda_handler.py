"""AWS Lambda handler using Mangum.

Wraps the FastAPI app for Lambda + API Gateway deployment.
"""

from mangum import Mangum

from src.main import app

# Create Lambda handler
handler = Mangum(app, lifespan="off")

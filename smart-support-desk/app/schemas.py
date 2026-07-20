from pydantic import BaseModel


class TicketIn(BaseModel):
    """The request body users send when they want to classify a ticket."""

    text: str


class TicketOut(BaseModel):
    """The response body our API sends back after making a prediction."""

    text: str
    predicted_category: str
